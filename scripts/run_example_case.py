"""Runnable Example Walkthrough for MedAgent:
Runs a complete multimodal clinical case (Notes + CXR + Labs) through the full multi-agent pipeline,
triggers the physician gate, approves the case, and outputs the final clinical PDF report.
"""

import os
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, os.path.abspath("backend"))

from medagent.config import settings
from medagent.schemas import CaseInput, DoctorDecision
from medagent.graph import build_graph, graph_checkpointer


def main():
    print("=" * 70)
    print(" MedAgent - Multimodal Multi-Agent Clinical Case Walkthrough")
    print("=" * 70)

    # 1. Define Clinical Case Inputs
    sample_cxr = os.path.abspath("samples/sample_cxr.png")
    lab_text = (
        "WBC 16.4 10^3/uL (4.0-11.0)\n"
        "Neutrophils 85.0 % (40.0-70.0)\n"
        "CRP 112.0 mg/L (0.0-5.0)\n"
        "Creatinine 1.0 mg/dL (0.7-1.3)"
    )

    case_input = CaseInput(
        case_id="case_demo_pneumonia",
        clinical_text="64yo male with 4 days of fever, purulent cough, right pleuritic chest pain and dyspnea.",
        patient_age=64,
        patient_sex="male",
        image_paths=[sample_cxr] if os.path.exists(sample_cxr) else [],
        lab_text=lab_text,
    )

    print(f"\n[1] Case Submitted: ID={case_input.case_id}")
    print(f"    Patient: {case_input.patient_age}yo {case_input.patient_sex}")
    print(f"    Clinical Notes: \"{case_input.clinical_text}\"")
    print(f"    Attached CXR   : {case_input.image_paths[0] if case_input.image_paths else 'None'}")
    print(f"    Attached Labs  : WBC, Neutrophils, CRP, Creatinine")

    # 2. Initialize LangGraph Orchestrator
    print("\n[2] Initializing LangGraph Orchestrator & SQLite Checkpointer...")
    graph = build_graph(checkpointer=graph_checkpointer)
    thread_config = {"configurable": {"thread_id": case_input.case_id}}


    # 3. Execute Pipeline to Physician Gate
    print("\n[3] Running Specialist Agents in Parallel...")
    print("    - Orchestrator Planning...")
    print("    - Imaging Agent (TorchXRayVision DenseNet-121 + Grad-CAM)...")
    print("    - Lab Agent (Deterministic 60+ Alias Parser + Pattern Rules)...")
    print("    - RAG Agent (Gale Encyclopedia Dense Embeddings)...")
    print("    - Clinical Reasoning Agent (Multimodal Differential Synthesis)...")
    print("    - Safety Agent (Grounding, Contraindications, Urgency)...")

    # Run up to interrupt
    for _ in graph.stream({"case_input": case_input}, thread_config):
        pass

    state = graph.get_state(thread_config)
    print("\n[4] Pipeline Halted at Physician Gate (interrupt).")
    print(f"    Current Stage: {state.next}")

    # Inspect Intermediate State
    state_values = state.values
    draft = state_values.get("draft")
    safety = state_values.get("safety")
    imaging = state_values.get("imaging")
    lab = state_values.get("lab")
    rag = state_values.get("rag")

    if imaging and imaging.findings:
        print("\n    >>> Imaging Findings (DenseNet-121):")
        for f in imaging.findings:
            print(f"        * {f.finding}: probability={f.probability:.1%}, location={f.location}, severity={f.severity}")
        if imaging.heatmap_paths:
            for k, p in imaging.heatmap_paths.items():
                print(f"        * Grad-CAM Overlay Saved: {p}")

    if lab and lab.patterns:
        print("\n    >>> Lab Patterns Detected:")
        for p in lab.patterns:
            print(f"        * {p.name} [{p.urgency.upper()}]: {p.description}")

    if rag and rag.evidence:
        print("\n    >>> RAG Evidence Retrieved:")
        for ev in rag.evidence[:3]:
            print(f"        * [{ev.id}] {ev.source} - {ev.section} (relevance score: {ev.score:.3f})")

    if draft and draft.differentials:
        print("\n    >>> Generated Differential Diagnoses:")
        for d in draft.differentials:
            print(f"        * {d.condition} [{d.likelihood.upper()}]")
            print(f"          Rationale: {d.rationale}")
            print(f"          Supporting: {', '.join(d.supporting_findings)}")
            if d.against:
                print(f"          Discordant/Against: {d.against}")

    if safety:
        print(f"\n    >>> Safety Agent Verdict: {safety.verdict.upper()} (Urgent: {safety.urgent})")
        if safety.issues:
            for iss in safety.issues:
                print(f"        - [{iss.severity.upper()}] {iss.code}: {iss.message}")

    # 4. Physician Reviews and Signs Off
    print("\n[5] Simulating Physician Review & Gate Decision...")
    physician_name = "Dr. Sarah Chen, MD"
    decision = DoctorDecision(
        decision="approve",
        reviewer=physician_name,
        notes="Clinical differential and radiographic findings verified. Consistent with bacterial CAP.",
    )
    print(f"    Reviewer: {decision.reviewer}")
    print(f"    Action  : {decision.decision.upper()}")
    print(f"    Notes   : {decision.notes}")

    # Resume graph past interrupt
    from langgraph.types import Command
    for _ in graph.stream(Command(resume=decision), thread_config):
        pass

    final_state = graph.get_state(thread_config)
    final_report = final_state.values.get("final_report")

    # 5. Output Final Report
    print("\n" + "=" * 70)
    print(" CASE COMPLETED & RELEASED")
    print("=" * 70)
    if final_report:
        print(f"Status        : Released={final_report.released}")
        print(f"Sign-off By   : {final_report.reviewer}")
        print(f"PDF Generated : {final_report.pdf_path}")
        print("\nExecutive Summary:")
        print(final_report.markdown.strip()[:400] + "...\n")
    print("=" * 70)


if __name__ == "__main__":
    main()
