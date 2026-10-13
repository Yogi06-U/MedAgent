"""Phase 3 tests: Safety Agent guardrails, seeded fault benchmark, gate enforcement, PDF, and audit."""

import os
import sqlite3
import pytest
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver

from medagent.schemas import (
    CaseInput,
    ImagingReport,
    LabReport,
    LabResult,
    LabPattern,
    RagReport,
    Evidence,
    ClinicalDraft,
    Differential,
    DoctorDecision,
)
from medagent.agents.safety import (
    run_safety_agent,
    sanitize_language,
    check_negation,
    check_urgency,
)
from medagent.graph import build_graph
from medagent.audit import get_audit_trail


def test_safety_language_sanitization():
    """Seeded fault: drug doses and over-certain language must be sanitized."""
    raw_text = (
        "Patient definitely has pneumonia. Administer amoxicillin 500 mg tid for 7 days. "
        "Heart failure is ruled out and diagnosis proves bacterial origin."
    )
    cleaned, issues = sanitize_language(raw_text)

    # Must remove dosing sentence
    assert "500 mg" not in cleaned
    assert "amoxicillin 500 mg" not in cleaned
    assert any(i.code == "LANGUAGE_DOSING_REMOVED" for i in issues)

    # Must soften over-certain wording
    assert "definitely" not in cleaned
    assert "ruled out" not in cleaned
    assert any(i.code == "LANGUAGE_CERTAINTY_SOFTENED" for i in issues)


def test_safety_grounding_purges_bad_citations():
    """Seeded fault: hallucinated citation IDs must be removed."""
    draft = ClinicalDraft(
        summary="Clinical presentation of pneumonia.",
        differentials=[
            Differential(
                condition="Pneumonia",
                likelihood="high",
                rationale="Focal infiltrate and fever",
                supporting_findings=["Consolidation", "Fever"],
                evidence_ids=["S1", "S_FAKE_999", "S_NONEXISTENT"],
            )
        ],
        mode="rule_engine",
    )
    rag_rep = RagReport(
        status="ok",
        evidence=[
            Evidence(
                id="S1",
                source="sample: pneumonia.md",
                section="Diagnosis",
                passage="Pneumonia diagnosis criteria.",
                score=0.9,
                query="pneumonia",
            )
        ],
    )
    case_in = CaseInput(case_id="case_grounding", clinical_text="Cough and fever.")
    safety_rep = run_safety_agent(case_in, draft, rag_report=rag_rep)

    # Must purge fake citation IDs
    assert draft.differentials[0].evidence_ids == ["S1"]
    assert any(i.code == "GROUNDING_HALLUCINATED_CITATION" for i in safety_rep.issues)


def test_safety_contraindications_blockers():
    """Seeded faults: contraindications must trigger blocker severity."""
    case_in = CaseInput(
        case_id="case_contra",
        clinical_text="72yo female taking lisinopril and spironolactone. Contrast CT suggested.",
    )
    lab_rep = LabReport(
        status="ok",
        results=[
            LabResult(test="Serum Creatinine", raw_name="Cr", value=2.6, unit="mg/dL", flag="high"),
            LabResult(test="Potassium", raw_name="K", value=6.1, unit="mEq/L", flag="critical high"),
            LabResult(test="eGFR", raw_name="eGFR", value=22.0, unit="mL/min", flag="low"),
        ],
    )
    draft = ClinicalDraft(
        summary="Renal impairment and hyperkalemia.",
        differentials=[
            Differential(
                condition="Severe Hyperkalemia",
                likelihood="high",
                rationale="High K",
                supporting_findings=["K 6.1"],
                evidence_ids=[],
            )
        ],
        suggested_workup=["Contrast CT abdomen"],
        mode="rule_engine",
    )

    safety_rep = run_safety_agent(case_in, draft, lab_report=lab_rep)
    assert safety_rep.verdict == "blocked"
    codes = [i.code for i in safety_rep.issues]
    assert "CONTRAINDICATION_CONTRAST_RENAL" in codes
    assert "CONTRAINDICATION_RAAS_HYPERKALEMIA" in codes


def test_safety_negation_vs_positive_urgency():
    """Verify negation handling distinguishes 'no chest pain' from acute chest pain."""
    case_negated = CaseInput(
        case_id="neg_case",
        clinical_text="45yo male presenting with fatigue. Denies chest pain, has no shortness of breath, without hemoptysis.",
    )
    is_urgent_neg, reasons_neg = check_urgency(case_negated, None, None)
    assert is_urgent_neg is False
    assert len(reasons_neg) == 0

    case_positive = CaseInput(
        case_id="pos_case",
        clinical_text="60yo male presenting with severe acute crushing chest pain radiating to left jaw.",
    )
    is_urgent_pos, reasons_pos = check_urgency(case_positive, None, None)
    assert is_urgent_pos is True
    assert any("Chest pain" in r for r in reasons_pos)


def test_safety_blocker_reassuring_during_emergency():
    """Seeded fault: downplaying acute emergency triggers blocker."""
    case_in = CaseInput(
        case_id="downplay_case",
        clinical_text="Patient with acute severe crushing chest pain.",
    )
    draft = ClinicalDraft(
        summary="Patient presentation appears benign and suitable for routine follow-up at home.",
        differentials=[
            Differential(
                condition="Atypical Chest Pain",
                likelihood="low",
                rationale="Non-cardiac features",
                supporting_findings=["Chest pain"],
            )
        ],
        mode="rule_engine",
    )
    safety_rep = run_safety_agent(case_in, draft)
    assert safety_rep.verdict == "blocked"
    assert any(i.code == "BLOCKER_REASSURING_DURING_EMERGENCY" for i in safety_rep.issues)


def test_gate_rules_blocked_approve_converted_to_reject():
    """Gate enforcement: if safety verdict is blocked, approve is auto-converted to reject."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    mem_saver = SqliteSaver(conn)
    graph = build_graph(checkpointer=mem_saver)

    case_in = CaseInput(
        case_id="blocked_gate_case",
        clinical_text="Acute crushing chest pain, taking spironolactone with K 6.4 mEq/L and contrast recommended.",
        lab_text="Potassium 6.4 mEq/L\nCreatinine 3.2 mg/dL",
    )

    config = {"configurable": {"thread_id": "blocked_gate_case"}}
    initial_state = {
        "case_input": case_in,
        "plan": None,
        "imaging_report": None,
        "lab_report": None,
        "rag_report": None,
        "clinical_draft": None,
        "safety_report": None,
        "safety_retries": 1,  # Retried already
        "feedback_for_reasoning": None,
        "doctor_decision": None,
        "final_report": None,
        "error": None,
    }

    # Run until interrupt
    graph.invoke(initial_state, config=config)
    state = graph.get_state(config)
    assert "physician_gate_node" in state.next

    safety = state.values.get("safety_report")
    assert safety.verdict == "blocked"

    # Physician attempts direct approve on blocked case
    bad_approve = DoctorDecision(
        decision="approve",
        reviewer="Dr. Rushed",
        notes="Looks okay to release.",
    )

    final_state = graph.invoke(Command(resume=bad_approve.model_dump()), config=config)
    final_report = final_state["final_report"]

    # Must be converted to reject!
    assert final_report.decision == "reject"
    assert final_report.released is False
    assert final_report.pdf_path is None
    assert "AUTO-CONVERTED" in final_state["doctor_decision"].notes


def test_gate_rules_rejection_and_edit():
    """Gate enforcement: edit summary is permitted; rejection releases no report."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    mem_saver = SqliteSaver(conn)
    graph = build_graph(checkpointer=mem_saver)

    case_in = CaseInput(
        case_id="edit_gate_case",
        clinical_text="Fever and productive cough.",
        lab_text="WBC 14.5 x10^3/uL",
    )
    config = {"configurable": {"thread_id": "edit_gate_case"}}
    initial_state = {
        "case_input": case_in,
        "plan": None,
        "imaging_report": None,
        "lab_report": None,
        "rag_report": None,
        "clinical_draft": None,
        "safety_report": None,
        "safety_retries": 1,
        "feedback_for_reasoning": None,
        "doctor_decision": None,
        "final_report": None,
        "error": None,
    }

    graph.invoke(initial_state, config=config)

    # Physician edits summary
    edit_decision = DoctorDecision(
        decision="edit",
        reviewer="Dr. Specialist",
        notes="Adjusted summary for bacterial etiology.",
        edited_summary="Physician edited summary: Community acquired bacterial pneumonia confirmed.",
    )
    res = graph.invoke(Command(resume=edit_decision.model_dump()), config=config)
    final_rep = res["final_report"]

    assert final_rep.decision == "edit"
    assert final_rep.released is True
    assert final_rep.pdf_path is not None
    assert os.path.exists(final_rep.pdf_path)
    assert "Community acquired bacterial pneumonia confirmed" in final_rep.markdown

    # Verify audit record
    audit_entries = get_audit_trail(case_id="edit_gate_case")
    assert any(e["decision"] == "edit" for e in audit_entries)
