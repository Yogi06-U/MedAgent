"""Phase 2 tests: Imaging Agent (DenseNet + Grad-CAM), Lab Agent, RAG Agent, Reasoning Agent."""

import os
from pathlib import Path
import numpy as np
import pytest

from medagent.schemas import CaseInput, ImagingReport, LabReport, RagReport, ClinicalDraft
from medagent.agents.imaging import (
    load_radiograph,
    compute_gradcam,
    determine_location,
    calculate_severity,
    generate_heatmap_overlay,
    run_imaging_agent,
    get_densenet_model,
)
from medagent.agents.lab import (
    normalize_unit_and_value,
    parse_lab_line,
    evaluate_patterns,
    run_lab_agent,
    get_reference_ranges,
)
from medagent.agents.rag import (
    chunk_document,
    build_or_load_knowledge_index,
    construct_queries,
    run_rag_agent,
    run_second_pass_rag,
)
from medagent.agents.reasoning import run_rule_engine, run_reasoning_agent


def test_imaging_agent_and_gradcam():
    """Verify DenseNet loading, inference, Grad-CAM, location and overlay generation."""
    sample_img = "samples/sample_cxr.png"
    assert os.path.exists(sample_img)

    # 1. Quality checks and loading
    arr, quality_notes, err = load_radiograph(sample_img)
    assert err is None
    assert arr is not None
    assert arr.shape == (512, 512)

    # 2. Location logic test with synthetic CAM
    # Right lower zone: image-left (patient's right: x < 112), vertical lower zone (y > 143)
    cam_rlz = np.zeros((224, 224), dtype=np.float32)
    cam_rlz[160:210, 20:80] = 0.95  # image-left (patient's right), lower zone
    loc = determine_location(cam_rlz, "Consolidation")
    assert "right" in loc
    assert "lower" in loc

    # Bilateral test
    cam_bi = np.zeros((224, 224), dtype=np.float32)
    cam_bi[100:150, 40:80] = 0.95
    cam_bi[100:150, 140:180] = 0.95
    loc_bi = determine_location(cam_bi, "Infiltration")
    assert "bilateral" in loc_bi

    # Cardiomegaly special condition
    loc_cardio = determine_location(cam_rlz, "Cardiomegaly")
    assert loc_cardio == "cardiac / mediastinal silhouette"

    # 3. Severity heuristic test
    sev_high = calculate_severity(0.85, np.ones((224, 224)) * 0.9)
    assert sev_high == "high"
    sev_mod = calculate_severity(0.65, np.zeros((224, 224)))
    assert sev_mod == "moderate"
    sev_low = calculate_severity(0.40, np.zeros((224, 224)))
    assert sev_low == "low"

    # 4. End-to-end Imaging Agent
    case_in = CaseInput(
        case_id="cxr_test_case",
        clinical_text="Suspected pneumonia with productive cough",
        image_paths=[sample_img],
    )
    report = run_imaging_agent(case_in)
    assert report.status in ("ok", "unavailable")
    if report.status == "ok":
        assert len(report.findings) <= 6
        assert all(f.probability >= 0.35 for f in report.findings)
        # Check overlay generated
        for finding, hpath in report.heatmap_paths.items():
            assert os.path.exists(hpath)
            # Copy first overlay to screenshots for visual proof
            overlay_dest = "screenshots/phase2_overlay.png"
            with open(hpath, "rb") as src, open(overlay_dest, "wb") as dst:
                dst.write(src.read())
            break


def test_lab_agent_normalization_and_patterns():
    """Verify alias mapping, unit conversions, reference ranges, and pattern rules."""
    ref_db = get_reference_ranges()
    assert len(ref_db) >= 60

    # 1. Unit conversions
    val, unit = normalize_unit_and_value("glucose", 10.0, "mmol/L")
    assert abs(val - 180.2) < 0.5
    assert unit == "mg/dL"

    val, unit = normalize_unit_and_value("creatinine", 176.8, "umol/L")
    assert abs(val - 2.0) < 0.1
    assert unit == "mg/dL"

    val, unit = normalize_unit_and_value("troponin_i", 80.0, "ng/L")
    assert val == 0.08
    assert unit == "ng/mL"

    val, unit = normalize_unit_and_value("wbc", 12500, "/cumm")
    assert val == 12.5
    assert unit == "x10^3/uL"

    val, unit = normalize_unit_and_value("hba1c", 53.0, "mmol/mol")
    assert abs(val - 7.0) < 0.2
    assert unit == "%"

    # 2. Line parsing with override range
    res, unparsed = parse_lab_line("Hemoglobin: 10.2 g/dL (12.0 - 16.0)", ref_db, "female")
    assert res is not None
    assert res.test == "Hemoglobin"
    assert res.value == 10.2
    assert res.flag == "low"
    assert res.ref_low == 12.0

    # 3. Pattern detection: Pneumonia pattern
    case_infective = CaseInput(
        case_id="lab_test_infective",
        clinical_text="Fever, cough",
        lab_text="WBC 16.5 x10^3/uL\nNeutrophils 84 %\nCRP 95 mg/L\nProcalcitonin 1.8 ng/mL",
    )
    rep = run_lab_agent(case_infective)
    assert rep.status == "ok"
    assert len(rep.results) >= 4
    pattern_names = [p.name for p in rep.patterns]
    assert "Leukocytosis with Neutrophilia" in pattern_names
    assert "Raised Inflammatory Markers" in pattern_names

    # 4. Pattern detection: AKI & Hyperkalemia (urgent)
    case_renal = CaseInput(
        case_id="lab_test_renal",
        clinical_text="Oliguria, lethargy",
        lab_text="Potassium 6.2 mEq/L\nCreatinine 3.5 mg/dL\nBUN 55 mg/dL\neGFR 14 mL/min/1.73m2\nBicarbonate 16 mEq/L",
    )
    rep_renal = run_lab_agent(case_renal)
    assert rep_renal.status == "ok"
    renal_patterns = {p.name: p.urgency for p in rep_renal.patterns}
    assert "Hyperkalaemia" in renal_patterns
    assert renal_patterns["Hyperkalaemia"] == "urgent"
    assert "Renal Impairment" in renal_patterns
    assert "Metabolic Acidosis" in renal_patterns

    # 5. Neural Lab Agent Deep Learning Inferences
    assert len(rep_renal.inferred_diseases) >= 2
    inferred_names = [d.disease for d in rep_renal.inferred_diseases]
    assert any("Kidney" in n for n in inferred_names)
    assert any("Hyperkalemia" in n for n in inferred_names)
    top_pred = rep_renal.inferred_diseases[0]
    assert top_pred.confidence_score is not None
    assert top_pred.confidence_score >= 0.50


def test_rag_agent_chunking_and_retrieval():
    """Verify document chunker, no duplicate chunks, query construction, and retrieval."""
    sample_doc = Path("backend/medagent/knowledge/pneumonia.md")
    assert sample_doc.exists()

    chunks = chunk_document(sample_doc)
    assert len(chunks) >= 1
    # Verify unique IDs
    chunk_ids = [c["id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))

    # Query construction
    case_in = CaseInput(
        case_id="rag_test",
        clinical_text="68yo male with fever, purulent sputum, and dyspnea.",
    )
    queries = construct_queries(case_in)
    assert len(queries) <= 6

    # End-to-end RAG retrieval
    rag_rep = run_rag_agent(case_in)
    assert rag_rep.status == "ok"
    assert len(rag_rep.evidence) <= 8
    # Unique citation IDs (S1, S2, ...)
    assert all(ev.id.startswith("S") for ev in rag_rep.evidence)

    # Second pass retrieval
    second_pass = run_second_pass_rag("Hyponatremia")
    assert len(second_pass) >= 1


def test_reasoning_agent_rule_engine():
    """Verify rule engine evaluates conditions with multi-source likelihood criteria."""
    case_in = CaseInput(
        case_id="reason_test_pneumonia",
        clinical_text="68yo male with 4-day productive cough, fever 38.8C, and pleuritic chest pain. Crackles at right base.",
        lab_text="WBC 16.2 x10^3/uL\nNeutrophils 82 %\nCRP 128 mg/L",
    )
    lab_rep = run_lab_agent(case_in)
    rag_rep = run_rag_agent(case_in, lab_report=lab_rep)

    draft = run_reasoning_agent(case_in, lab_report=lab_rep, rag_report=rag_rep)
    assert draft.mode == "rule_engine"
    assert 2 <= len(draft.differentials) <= 5
    top_diff = draft.differentials[0]
    assert "Pneumonia" in top_diff.condition
    # Supported by >= 2 sources (notes + labs) -> likelihood should be high or moderate
    assert top_diff.likelihood in ("high", "moderate")
    assert len(top_diff.supporting_findings) >= 2
    assert len(draft.summary.split(".")) <= 5
