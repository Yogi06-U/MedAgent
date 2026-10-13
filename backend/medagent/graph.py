"""LangGraph orchestrator for MedAgent multimodal multi-agent pipeline."""

import asyncio
import logging
import sqlite3
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from medagent.config import settings
from medagent.schemas import (
    CaseInput,
    ImagingReport,
    LabReport,
    RagReport,
    ClinicalDraft,
    SafetyReport,
    DoctorDecision,
    FinalReport,
)
from medagent.agents.imaging import run_imaging_agent
from medagent.agents.lab import run_lab_agent
from medagent.agents.rag import run_rag_agent, run_second_pass_rag
from medagent.agents.reasoning import run_reasoning_agent
from medagent.agents.safety import run_safety_agent
from medagent.report import generate_markdown_report, generate_pdf_report
from medagent.audit import log_event

logger = logging.getLogger("medagent.graph")

# Event pub-sub for SSE
_event_subscribers: Dict[str, List[asyncio.Queue]] = {}
_case_events_history: Dict[str, List[Dict[str, Any]]] = {}


def register_event_queue(case_id: str, queue: asyncio.Queue) -> None:
    """Register an SSE listener for a given case."""
    if case_id not in _event_subscribers:
        _event_subscribers[case_id] = []
    _event_subscribers[case_id].append(queue)


def unregister_event_queue(case_id: str, queue: asyncio.Queue) -> None:
    """Unregister an SSE listener."""
    if case_id in _event_subscribers and queue in _event_subscribers[case_id]:
        _event_subscribers[case_id].remove(queue)


def get_event_history(case_id: str) -> List[Dict[str, Any]]:
    """Get all past events for a case."""
    return _case_events_history.get(case_id, [])


def emit_event(case_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    """Emit an SSE event to all connected listeners and save to history."""
    evt = {
        "event": event_type,
        "case_id": case_id,
        "timestamp": time.time(),
        "data": payload,
    }
    if case_id not in _case_events_history:
        _case_events_history[case_id] = []
    _case_events_history[case_id].append(evt)

    queues = _event_subscribers.get(case_id, [])
    for q in list(queues):
        try:
            q.put_nowait(evt)
        except Exception:
            pass


class PlanOutput(TypedDict):
    has_image: bool
    has_labs: bool
    steps: List[str]


class MedAgentState(TypedDict):
    case_input: CaseInput
    plan: Optional[PlanOutput]
    imaging_report: Optional[ImagingReport]
    lab_report: Optional[LabReport]
    rag_report: Optional[RagReport]
    clinical_draft: Optional[ClinicalDraft]
    safety_report: Optional[SafetyReport]
    safety_retries: int
    feedback_for_reasoning: Optional[str]
    doctor_decision: Optional[DoctorDecision]
    final_report: Optional[FinalReport]
    error: Optional[str]


def plan_node(state: MedAgentState) -> Dict[str, Any]:
    """Orchestrator intake planning node."""
    case = state["case_input"]
    t0 = time.time()
    emit_event(case.case_id, "node_started", {"node": "plan"})

    has_image = bool(case.image_paths and len(case.image_paths) > 0)
    has_labs = bool((case.lab_paths and len(case.lab_paths) > 0) or (case.lab_text and case.lab_text.strip()))

    steps = ["plan"]
    if has_image:
        steps.append("imaging")
    if has_labs:
        steps.append("lab")
    steps.extend(["rag", "reasoning", "safety", "physician_gate", "finalize"])

    plan_data: PlanOutput = {
        "has_image": has_image,
        "has_labs": has_labs,
        "steps": steps,
    }

    emit_event(
        case.case_id,
        "node_finished",
        {"node": "plan", "duration": time.time() - t0, "payload": plan_data},
    )
    return {"plan": plan_data}


def route_from_plan(state: MedAgentState) -> List[str]:
    """Conditional routing based on intake plan."""
    plan = state["plan"]
    if not plan:
        return ["rag_node"]

    destinations: List[str] = []
    if plan["has_image"]:
        destinations.append("imaging_node")
    if plan["has_labs"]:
        destinations.append("lab_node")

    if not destinations:
        return ["rag_node"]
    return destinations


def imaging_node(state: MedAgentState) -> Dict[str, Any]:
    """Execute Imaging Agent."""
    case = state["case_input"]
    t0 = time.time()
    emit_event(case.case_id, "node_started", {"node": "imaging"})
    try:
        report = run_imaging_agent(case)
    except Exception as e:
        logger.error("Imaging agent error: %s", str(e), exc_info=True)
        report = ImagingReport(
            status="error",
            message=f"Imaging agent encountered an unexpected error: {str(e)}",
        )
    emit_event(
        case.case_id,
        "node_finished",
        {"node": "imaging", "duration": time.time() - t0, "payload": report.model_dump()},
    )
    return {"imaging_report": report}


def lab_node(state: MedAgentState) -> Dict[str, Any]:
    """Execute Lab Agent."""
    case = state["case_input"]
    t0 = time.time()
    emit_event(case.case_id, "node_started", {"node": "lab"})
    try:
        report = run_lab_agent(case)
    except Exception as e:
        logger.error("Lab agent error: %s", str(e), exc_info=True)
        report = LabReport(
            status="error",
            message=f"Lab agent encountered an unexpected error: {str(e)}",
        )
    emit_event(
        case.case_id,
        "node_finished",
        {"node": "lab", "duration": time.time() - t0, "payload": report.model_dump()},
    )
    return {"lab_report": report}


def rag_node(state: MedAgentState) -> Dict[str, Any]:
    """Execute RAG Agent."""
    case = state["case_input"]
    t0 = time.time()
    emit_event(case.case_id, "node_started", {"node": "rag"})
    try:
        report = run_rag_agent(
            case,
            imaging_report=state.get("imaging_report"),
            lab_report=state.get("lab_report"),
        )
    except Exception as e:
        logger.error("RAG agent error: %s", str(e), exc_info=True)
        report = RagReport(
            status="error",
            backend="error",
            message=f"RAG agent encountered an error: {str(e)}",
        )
    emit_event(
        case.case_id,
        "node_finished",
        {"node": "rag", "duration": time.time() - t0, "payload": report.model_dump()},
    )
    return {"rag_report": report}


def reasoning_node(state: MedAgentState) -> Dict[str, Any]:
    """Execute Clinical Reasoning Agent and rationale-guided second retrieval."""
    case = state["case_input"]
    t0 = time.time()
    emit_event(case.case_id, "node_started", {"node": "reasoning"})

    draft = run_reasoning_agent(
        case,
        imaging_report=state.get("imaging_report"),
        lab_report=state.get("lab_report"),
        rag_report=state.get("rag_report"),
        feedback=state.get("feedback_for_reasoning"),
    )

    # Rationale-guided second pass: for any differential lacking evidence, query the store
    rag_rep = state.get("rag_report")
    if rag_rep:
        existing_evidence = list(rag_rep.evidence)
        new_evidence_added = False
        for diff in draft.differentials:
            if not diff.evidence_ids:
                second_pass_items = run_second_pass_rag(diff.condition)
                for item in second_pass_items:
                    existing_evidence.append(item)
                    diff.evidence_ids.append(item.id)
                    new_evidence_added = True
        if new_evidence_added:
            rag_rep.evidence = existing_evidence

    emit_event(
        case.case_id,
        "node_finished",
        {"node": "reasoning", "duration": time.time() - t0, "payload": draft.model_dump()},
    )
    return {"clinical_draft": draft, "rag_report": rag_rep, "feedback_for_reasoning": None}


def safety_node(state: MedAgentState) -> Dict[str, Any]:
    """Execute Safety Agent."""
    case = state["case_input"]
    draft = state["clinical_draft"]
    t0 = time.time()
    emit_event(case.case_id, "node_started", {"node": "safety"})

    safety_rep = run_safety_agent(
        case,
        draft,
        imaging_report=state.get("imaging_report"),
        lab_report=state.get("lab_report"),
        rag_report=state.get("rag_report"),
    )

    retries = state.get("safety_retries", 0)
    feedback = None

    if safety_rep.verdict == "blocked" and retries < 1:
        blocker_msgs = [i.message for i in safety_rep.issues if i.severity == "blocker"]
        feedback = "Safety Agent Blocked Draft: " + "; ".join(blocker_msgs)

    emit_event(
        case.case_id,
        "node_finished",
        {"node": "safety", "duration": time.time() - t0, "payload": safety_rep.model_dump()},
    )

    return {
        "safety_report": safety_rep,
        "safety_retries": retries + (1 if feedback else 0),
        "feedback_for_reasoning": feedback,
    }


def route_from_safety(state: MedAgentState) -> str:
    """Check if safety blocked and should retry reasoning."""
    if state.get("feedback_for_reasoning"):
        return "reasoning_node"
    return "physician_gate_node"


def physician_gate_node(state: MedAgentState) -> Dict[str, Any]:
    """Human-in-the-loop gate: interrupts graph execution until decision received."""
    case = state["case_input"]
    emit_event(
        case.case_id,
        "awaiting_physician",
        {
            "case_id": case.case_id,
            "draft": state["clinical_draft"].model_dump() if state.get("clinical_draft") else None,
            "safety": state["safety_report"].model_dump() if state.get("safety_report") else None,
        },
    )

    # Interrupt execution; resumes with DoctorDecision dict
    raw_decision = interrupt({
        "message": "Clinical review required before release.",
        "case_id": case.case_id,
    })

    if isinstance(raw_decision, dict):
        decision = DoctorDecision.model_validate(raw_decision)
    elif isinstance(raw_decision, DoctorDecision):
        decision = raw_decision
    else:
        decision = DoctorDecision(
            decision="reject",
            reviewer="System Fallback",
            notes=f"Unrecognized decision type: {type(raw_decision)}",
        )

    # Safety enforcement: If safety verdict is blocked, approve is converted to reject
    safety_rep = state.get("safety_report")
    if safety_rep and safety_rep.verdict == "blocked" and decision.decision == "approve":
        logger.warning("Safety verdict was BLOCKED: converting approve to reject.")
        decision.decision = "reject"
        decision.notes = (
            f"[AUTO-CONVERTED] Draft was BLOCKED by Safety Agent. Approval disallowed. "
            f"Original notes: {decision.notes}"
        )

    return {"doctor_decision": decision}


def finalize_node(state: MedAgentState) -> Dict[str, Any]:
    """Finalize report, generate PDF if approved/edited, and record audit log."""
    case = state["case_input"]
    decision = state["doctor_decision"]
    draft = state["clinical_draft"]
    t0 = time.time()
    emit_event(case.case_id, "node_started", {"node": "finalize"})

    released = decision.decision in ("approve", "edit")
    pdf_path = None

    markdown_report = generate_markdown_report(
        case,
        draft,
        decision,
        imaging_report=state.get("imaging_report"),
        lab_report=state.get("lab_report"),
    )

    if released:
        try:
            pdf_path = generate_pdf_report(
                case,
                draft,
                decision,
                imaging_report=state.get("imaging_report"),
                lab_report=state.get("lab_report"),
            )
        except Exception as e:
            logger.error("Failed to generate PDF report: %s", str(e), exc_info=True)
            pdf_path = None

    final_report = FinalReport(
        case_id=case.case_id,
        released=released,
        reviewer=decision.reviewer,
        decision=decision.decision,
        markdown=markdown_report,
        pdf_path=pdf_path,
    )

    # Log audit event
    log_event(
        case_id=case.case_id,
        event_type=f"case_{decision.decision}",
        reviewer=decision.reviewer,
        decision=decision.decision,
        details={
            "released": released,
            "has_pdf": bool(pdf_path),
            "safety_verdict": state["safety_report"].verdict if state.get("safety_report") else "unknown",
        },
    )

    emit_event(
        case.case_id,
        "node_finished",
        {"node": "finalize", "duration": time.time() - t0, "payload": final_report.model_dump()},
    )
    emit_event(
        case.case_id,
        "finished",
        {"case_id": case.case_id, "final_report": final_report.model_dump()},
    )

    return {"final_report": final_report}


def build_graph(checkpointer: Optional[Any] = None) -> Any:
    """Build and compile the LangGraph workflow."""
    builder = StateGraph(MedAgentState)

    builder.add_node("plan_node", plan_node)
    builder.add_node("imaging_node", imaging_node)
    builder.add_node("lab_node", lab_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("reasoning_node", reasoning_node)
    builder.add_node("safety_node", safety_node)
    builder.add_node("physician_gate_node", physician_gate_node)
    builder.add_node("finalize_node", finalize_node)

    builder.add_edge(START, "plan_node")
    builder.add_conditional_edges(
        "plan_node",
        route_from_plan,
        ["imaging_node", "lab_node", "rag_node"],
    )

    builder.add_edge("imaging_node", "rag_node")
    builder.add_edge("lab_node", "rag_node")
    builder.add_edge("rag_node", "reasoning_node")
    builder.add_edge("reasoning_node", "safety_node")

    builder.add_conditional_edges(
        "safety_node",
        route_from_safety,
        {"reasoning_node": "reasoning_node", "physician_gate_node": "physician_gate_node"},
    )

    builder.add_edge("physician_gate_node", "finalize_node")
    builder.add_edge("finalize_node", END)

    return builder.compile(checkpointer=checkpointer)


# Global SQLite checkpointer and compiled workflow
_db_conn = sqlite3.connect(str(settings.DB_PATH), check_same_thread=False)
graph_checkpointer = SqliteSaver(_db_conn)
graph_app = build_graph(checkpointer=graph_checkpointer)
