"""Evaluation harness for Clinical Reasoning Agent:
Evaluates differential diagnosis concordance, Top-1 and Top-3 accuracy, and grounding validity
across a benchmark suite of 25 clinical vignettes covering all 25 supported conditions.
"""

import os
import json
import time
from typing import List, Dict, Any

from medagent.schemas import (
    CaseInput,
    ImagingReport,
    ImagingFinding,
    LabReport,
    LabResult,
    LabPattern,
    RagReport,
    Evidence,
)
from medagent.agents.reasoning import run_reasoning_agent, CONDITIONS_CATALOG


BENCHMARK_VIGNETTES: List[Dict[str, Any]] = [
    {
        "id": "V01",
        "expected_condition": "Community-Acquired Pneumonia",
        "case": CaseInput(case_id="V01", clinical_text="64yo male with 4 days of fever, purulent cough, sputum, right pleuritic chest pain and dyspnea.", patient_age=64, patient_sex="male"),
        "imaging": ImagingReport(status="ok", findings=[ImagingFinding(finding="Consolidation", probability=0.88, location="right lower zone", severity="high", confidence="high")]),
        "lab": LabReport(status="ok", results=[], patterns=[LabPattern(name="Leukocytosis with Neutrophilia", description="High WBC", supporting_tests=["WBC Count"], urgency="soon")]),
    },
    {
        "id": "V02",
        "expected_condition": "Acute Kidney Injury",
        "case": CaseInput(case_id="V02", clinical_text="72yo female with severe vomiting, oliguria, lethargy, decreased urine output and acute dehydration.", patient_age=72, patient_sex="female"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[LabPattern(name="Renal Impairment", description="Creatinine elevated", supporting_tests=["Serum Creatinine", "BUN"], urgency="urgent")]),
    },
    {
        "id": "V03",
        "expected_condition": "Microcytic Iron Deficiency Anemia",
        "case": CaseInput(case_id="V03", clinical_text="42yo female with 6 months of heavy menorrhagia, fatigue, pallor, pica and brittle nails.", patient_age=42, patient_sex="female"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[LabPattern(name="Microcytic Anaemia", description="Low Hb, low MCV", supporting_tests=["Hemoglobin", "MCV", "Ferritin"], urgency="soon")]),
    },
    {
        "id": "V04",
        "expected_condition": "Congestive Heart Failure / Pulmonary Edema",
        "case": CaseInput(case_id="V04", clinical_text="75yo male with worsening orthopnea, paroxysmal nocturnal dyspnea, edema, crackles and JVD.", patient_age=75, patient_sex="male"),
        "imaging": ImagingReport(status="ok", findings=[
            ImagingFinding(finding="Cardiomegaly", probability=0.85, location="cardiac silhouette", severity="moderate", confidence="high"),
            ImagingFinding(finding="Edema", probability=0.82, location="bilateral perihilar", severity="high", confidence="high")
        ]),
        "lab": LabReport(status="skipped", results=[], patterns=[]),
    },
    {
        "id": "V05",
        "expected_condition": "Pulmonary Embolism",
        "case": CaseInput(case_id="V05", clinical_text="55yo female post-orthopedic surgery with sudden pleuritic chest pain, tachypnea, hemoptysis, and DVT.", patient_age=55, patient_sex="female"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="skipped", results=[], patterns=[]),
    },
    {
        "id": "V06",
        "expected_condition": "Diabetic Ketoacidosis",
        "case": CaseInput(case_id="V06", clinical_text="21yo male with type 1 diabetes presenting with vomiting, abdominal pain, Kussmaul breathing and fruity breath.", patient_age=21, patient_sex="male"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Hyperglycaemia with Low Bicarbonate (DKA Pattern)", description="DKA triad", supporting_tests=["Glucose", "Bicarbonate"], urgency="urgent"),
            LabPattern(name="Metabolic Acidosis", description="Low HCO3", supporting_tests=["Bicarbonate"], urgency="urgent")
        ]),
    },
    {
        "id": "V07",
        "expected_condition": "Chronic Kidney Disease",
        "case": CaseInput(case_id="V07", clinical_text="68yo female with 10-year history of hypertensive chronic kidney disease, nocturia, pruritus, eGFR 22.", patient_age=68, patient_sex="female"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[LabPattern(name="Renal Impairment", description="Chronically elevated creatinine", supporting_tests=["Serum Creatinine"], urgency="soon")]),
    },
    {
        "id": "V08",
        "expected_condition": "Acute Myocardial Infarction / ACS",
        "case": CaseInput(case_id="V08", clinical_text="58yo male with acute crushing chest pressure radiating to the arm and jaw with diaphoresis.", patient_age=58, patient_sex="male"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[LabPattern(name="Raised Troponin", description="Elevated high sensitivity troponin", supporting_tests=["Troponin I"], urgency="urgent")]),
    },
    {
        "id": "V09",
        "expected_condition": "Sepsis / Severe Systemic Infection",
        "case": CaseInput(case_id="V09", clinical_text="80yo nursing home resident with confusion, fever 39.2C, severe hypotension 80/50, and tachycardia 125.", patient_age=80, patient_sex="female"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Raised Lactate", description="Lactate > 4 mmol/L", supporting_tests=["Lactate"], urgency="urgent"),
            LabPattern(name="Raised Inflammatory Markers", description="CRP markedly elevated", supporting_tests=["C-Reactive Protein"], urgency="urgent")
        ]),
    },
    {
        "id": "V10",
        "expected_condition": "Acute Hepatocellular Injury",
        "case": CaseInput(case_id="V10", clinical_text="35yo male with right upper quadrant abdominal tenderness, acute jaundice, hepatomegaly and dark urine.", patient_age=35, patient_sex="male"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Hepatocellular Pattern", description="Marked transaminitis ALT > 1000", supporting_tests=["ALT (SGPT)", "AST (SGOT)"], urgency="urgent"),
            LabPattern(name="Hyperbilirubinaemia", description="Elevated total bilirubin", supporting_tests=["Total Bilirubin"], urgency="soon")
        ]),
    },
    {
        "id": "V11",
        "expected_condition": "Cholestasis / Obstructive Biliary Disease",
        "case": CaseInput(case_id="V11", clinical_text="62yo female with dark urine, pale stool, icterus, intense pruritus and gallstones.", patient_age=62, patient_sex="female"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Cholestatic Pattern", description="Alkaline phosphatase and GGT markedly elevated", supporting_tests=["Alkaline Phosphatase", "GGT"], urgency="soon"),
            LabPattern(name="Hyperbilirubinaemia", description="Elevated direct bilirubin", supporting_tests=["Direct Bilirubin"], urgency="soon")
        ]),
    },
    {
        "id": "V12",
        "expected_condition": "Acute Pancreatitis",
        "case": CaseInput(case_id="V12", clinical_text="48yo male with severe epigastric pain radiating to the back, vomiting with pain, and alcohol intake history.", patient_age=48, patient_sex="male"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Raised Inflammatory Markers", description="Inflammatory response", supporting_tests=["C-Reactive Protein"], urgency="soon")
        ]),
    },
    {
        "id": "V13",
        "expected_condition": "Cardiomegaly",
        "case": CaseInput(case_id="V13", clinical_text="66yo male with 20-year history of hypertension, fatigue, and progressive dyspnea on exertion.", patient_age=66, patient_sex="male"),
        "imaging": ImagingReport(status="ok", findings=[
            ImagingFinding(finding="Cardiomegaly", probability=0.88, location="cardiac silhouette", severity="high", confidence="high")
        ]),
        "lab": LabReport(status="skipped", results=[], patterns=[]),
    },
    {
        "id": "V14",
        "expected_condition": "Pneumothorax",
        "case": CaseInput(case_id="V14", clinical_text="24yo tall thin male with sudden severe unilateral pleuritic chest pain, hyperresonance and acute dyspnea.", patient_age=24, patient_sex="male"),
        "imaging": ImagingReport(status="ok", findings=[
            ImagingFinding(finding="Pneumothorax", probability=0.89, location="right hemithorax", severity="high", confidence="high")
        ]),
        "lab": LabReport(status="skipped", results=[], patterns=[]),
    },
    {
        "id": "V15",
        "expected_condition": "Macrocytic / Megaloblastic Anemia",
        "case": CaseInput(case_id="V15", clinical_text="67yo vegan male presenting with subacute peripheral neuropathy, paresthesias, ataxia and glossitis.", patient_age=67, patient_sex="male"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Macrocytic Anaemia", description="MCV 115 fL with low hemoglobin", supporting_tests=["MCV", "Hemoglobin"], urgency="soon"),
            LabPattern(name="Vitamin Deficiencies", description="Serum B12 low", supporting_tests=["Vitamin B12"], urgency="soon")
        ]),
    },
    {
        "id": "V16",
        "expected_condition": "Severe Hyperkalemia",
        "case": CaseInput(case_id="V16", clinical_text="60yo male with end-stage renal disease missing hemodialysis, palpitations, paresthesias, and flaccid weakness.", patient_age=60, patient_sex="male"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Severe Hyperkalaemia", description="Serum potassium 6.8 mmol/L", supporting_tests=["Potassium"], urgency="urgent")
        ]),
    },
    {
        "id": "V17",
        "expected_condition": "Hypokalemia",
        "case": CaseInput(case_id="V17", clinical_text="50yo female on high-dose furosemide presenting with cramps, muscle weakness, vomiting and polyuria.", patient_age=50, patient_sex="female"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Hypokalaemia", description="Potassium 2.8 mmol/L", supporting_tests=["Potassium"], urgency="urgent")
        ]),
    },
    {
        "id": "V18",
        "expected_condition": "Hyponatremia",
        "case": CaseInput(case_id="V18", clinical_text="79yo female presenting with acute confusion, lethargy, headache, nausea, and unsteady gait.", patient_age=79, patient_sex="female"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Hyponatraemia", description="Sodium 118 mmol/L", supporting_tests=["Sodium"], urgency="urgent")
        ]),
    },
    {
        "id": "V19",
        "expected_condition": "Hypercalcemia",
        "case": CaseInput(case_id="V19", clinical_text="65yo male with bone pain, kidney stones, constipation, polyuria, and depression.", patient_age=65, patient_sex="male"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Hypercalcaemia", description="Corrected calcium 3.1 mmol/L", supporting_tests=["Serum Calcium"], urgency="urgent")
        ]),
    },
    {
        "id": "V20",
        "expected_condition": "COPD Acute Exacerbation",
        "case": CaseInput(case_id="V20", clinical_text="70yo smoker with chronic copd, wheezing, barrel chest, increased sputum, and worsening dyspnea.", patient_age=70, patient_sex="male"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="skipped", results=[], patterns=[]),
    },
    {
        "id": "V21",
        "expected_condition": "Pleural Effusion",
        "case": CaseInput(case_id="V21", clinical_text="61yo male with dullness at the left lung base, decreased breath sounds, pleuritic pain and dyspnea.", patient_age=61, patient_sex="male"),
        "imaging": ImagingReport(status="ok", findings=[
            ImagingFinding(finding="Effusion", probability=0.86, location="left lower zone", severity="high", confidence="high")
        ]),
        "lab": LabReport(status="skipped", results=[], patterns=[]),
    },
    {
        "id": "V22",
        "expected_condition": "Type 2 Diabetes Mellitus",
        "case": CaseInput(case_id="V22", clinical_text="51yo male with polyuria, polydipsia, polyphagia, blurred vision, and weight loss.", patient_age=51, patient_sex="male"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Diabetic Range Glycemia", description="HbA1c 9.5% and elevated glucose", supporting_tests=["HbA1c", "Fasting Glucose"], urgency="routine")
        ]),
    },
    {
        "id": "V23",
        "expected_condition": "Hypoglycemia",
        "case": CaseInput(case_id="V23", clinical_text="44yo diabetic female taking glipizide found diaphoresis, tremor, sweating, hunger and tachycardia.", patient_age=44, patient_sex="female"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Hypoglycaemia", description="Blood glucose 2.2 mmol/L", supporting_tests=["Random Glucose"], urgency="urgent")
        ]),
    },
    {
        "id": "V24",
        "expected_condition": "Metabolic Acidosis",
        "case": CaseInput(case_id="V24", clinical_text="52yo male with tachypnea, Kussmaul breathing, lethargy, vomiting and confusion.", patient_age=52, patient_sex="male"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Metabolic Acidosis", description="Low bicarbonate 12 mmol/L", supporting_tests=["Bicarbonate / CO2"], urgency="urgent")
        ]),
    },
    {
        "id": "V25",
        "expected_condition": "Normocytic Anemia of Chronic Disease",
        "case": CaseInput(case_id="V25", clinical_text="63yo female with long-standing rheumatoid arthritis, autoimmune flare, chronic fatigue and weight loss.", patient_age=63, patient_sex="female"),
        "imaging": ImagingReport(status="skipped", findings=[]),
        "lab": LabReport(status="ok", results=[], patterns=[
            LabPattern(name="Normocytic Anaemia", description="Normocytic normochromic anemia", supporting_tests=["Hemoglobin", "MCV"], urgency="routine"),
            LabPattern(name="Raised Inflammatory Markers", description="High ESR and CRP", supporting_tests=["ESR", "C-Reactive Protein"], urgency="soon")
        ]),
    },
]


def evaluate_reasoning_agent() -> dict:
    """Run reasoning agent on 25 benchmark vignettes and compute diagnostic concordance."""
    total_cases = len(BENCHMARK_VIGNETTES)
    top1_matches = 0
    top3_matches = 0
    all_differentials_grounded = 0
    latencies = []
    case_results = []

    rag_dummy = RagReport(
        status="ok",
        queries=["eval benchmark query"],
        evidence=[Evidence(id="S1", source="Gale Encyclopedia of Medicine", section="Clinical Overview", passage="Benchmark reference passage.", score=0.92, query="clinical")],
        backend="local",
        message="Evaluator dummy RAG",
    )

    for item in BENCHMARK_VIGNETTES:
        cid = item["id"]
        expected = item["expected_condition"]
        case_input = item["case"]
        imaging_rep = item["imaging"]
        lab_rep = item["lab"]

        t0 = time.perf_counter()
        draft = run_reasoning_agent(case_input, imaging_rep, lab_rep, rag_dummy)
        dur = (time.perf_counter() - t0) * 1000.0
        latencies.append(dur)

        conditions = [d.condition for d in draft.differentials]

        # Top 1 Match
        is_top1 = False
        if conditions and (expected.lower() == conditions[0].lower() or expected.lower() in conditions[0].lower() or conditions[0].lower() in expected.lower()):
            is_top1 = True
            top1_matches += 1

        # Top 3 Match
        is_top3 = False
        top3_list = conditions[:3]
        if any(expected.lower() == c.lower() or expected.lower() in c.lower() or c.lower() in expected.lower() for c in top3_list):
            is_top3 = True
            top3_matches += 1

        # Check evidence grounding
        valid_findings = all(len(d.supporting_findings) > 0 for d in draft.differentials)
        if valid_findings and len(draft.differentials) > 0:
            all_differentials_grounded += 1

        case_results.append({
            "vignette_id": cid,
            "expected_condition": expected,
            "predicted_top1": conditions[0] if conditions else None,
            "all_predicted": conditions,
            "top1_match": is_top1,
            "top3_match": is_top3,
            "grounded": valid_findings,
            "mode": draft.mode,
            "latency_ms": round(dur, 2),
        })

    top1_acc = top1_matches / total_cases
    top3_acc = top3_matches / total_cases
    grounding_rate = all_differentials_grounded / total_cases

    results = {
        "total_vignettes": total_cases,
        "engine_mode": draft.mode,
        "top1_matches": top1_matches,
        "top1_accuracy": round(top1_acc, 4),
        "top3_matches": top3_matches,
        "top3_accuracy": round(top3_acc, 4),
        "grounding_validity_rate": round(grounding_rate, 4),
        "mean_latency_ms": round(float(sum(latencies) / len(latencies)), 2),
        "case_results": case_results,
    }

    with open("eval/reasoning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = evaluate_reasoning_agent()
    print("=" * 60)
    print("CLINICAL REASONING AGENT EVALUATION (25 Vignettes)")
    print("=" * 60)
    print(f"Engine Mode               : {res['engine_mode']}")
    print(f"Total Vignettes Evaluated : {res['total_vignettes']}")
    print(f"Top-1 Diagnostic Accuracy : {res['top1_accuracy']:.1%} ({res['top1_matches']}/{res['total_vignettes']})")
    print(f"Top-3 Diagnostic Accuracy : {res['top3_accuracy']:.1%} ({res['top3_matches']}/{res['total_vignettes']})")
    print(f"Evidence Grounding Rate   : {res['grounding_validity_rate']:.1%}")
    print(f"Mean Latency per Case     : {res['mean_latency_ms']:.2f} ms")
