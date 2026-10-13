"""Phase 1 Foundation tests: schemas, graph interrupt/resume, and API endpoints."""

import os
import pytest
from fastapi.testclient import TestClient
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
from medagent.api import app
from medagent.graph import build_graph
from langgraph.types import Command
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


def test_pydantic_v2_schemas():
    """Verify strict validation and typing of schemas."""
    case = CaseInput(
        case_id="case_test_01",
        clinical_text="Cough and fever for 3 days.",
        patient_age=45,
        patient_sex="male",
    )
    assert case.case_id == "case_test_01"
    assert case.patient_age == 45
    assert case.patient_sex == "male"
    assert case.image_paths == []

    decision = DoctorDecision(
        decision="approve",
        reviewer="Dr. Ellis",
        notes="Validated findings",
    )
    assert decision.decision == "approve"
    assert decision.reviewer == "Dr. Ellis"


def test_graph_end_to_end_interrupt_and_resume():
    """Verify graph reaches physician interrupt and resumes upon decision."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    mem_saver = SqliteSaver(conn)
    graph = build_graph(checkpointer=mem_saver)

    case_input = CaseInput(
        case_id="test_case_int",
        clinical_text="Fever, productive cough, crackles.",
        patient_age=52,
        patient_sex="female",
        image_paths=["fake_cxr.png"],
        lab_text="WBC 14.5",
    )

    config = {"configurable": {"thread_id": "test_case_int"}}
    initial_state = {
        "case_input": case_input,
        "plan": None,
        "imaging_report": None,
        "lab_report": None,
        "rag_report": None,
        "clinical_draft": None,
        "safety_report": None,
        "safety_retries": 0,
        "feedback_for_reasoning": None,
        "doctor_decision": None,
        "final_report": None,
        "error": None,
    }

    # Step 1: Run until interrupt
    res = graph.invoke(initial_state, config=config)
    state = graph.get_state(config)
    assert "physician_gate_node" in state.next

    # Step 2: Resume with DoctorDecision
    decision = DoctorDecision(
        decision="approve",
        reviewer="Dr. House",
        notes="Clinical correlation confirms bacterial pneumonia.",
    )
    final_res = graph.invoke(Command(resume=decision.model_dump()), config=config)
    assert "final_report" in final_res
    final_report = final_res["final_report"]
    assert final_report.released is True
    assert final_report.reviewer == "Dr. House"
    assert final_report.decision == "approve"
    assert final_report.pdf_path is not None
    assert os.path.exists(final_report.pdf_path)


def test_api_health_and_models_status():
    """Verify FastAPI /api/health and /api/models/status endpoints."""
    client = TestClient(app)

    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    status_resp = client.get("/api/models/status")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert "torch" in data
    assert "imaging_model" in data
    assert "llm_engine" in data
