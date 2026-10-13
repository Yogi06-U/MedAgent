"""Evaluation harness for Lab Agent: evaluates parser accuracy and pattern precision/recall on 50+ test cases."""

import json
import re
from typing import Any, Dict, List, Tuple

from medagent.schemas import CaseInput
from medagent.agents.lab import run_lab_agent

# 55 diverse synthetic lab test cases with ground truth
GROUND_TRUTH_DATASET = [
    {
        "id": "case_01",
        "input": "WBC 14.5 x10^3/uL\nNeutrophils 82 %\nCRP 45 mg/L",
        "expected_tests": {"WBC Count": 14.5, "Neutrophils (%)": 82.0, "C-Reactive Protein": 45.0},
        "expected_patterns": ["Leukocytosis with Neutrophilia", "Raised Inflammatory Markers"],
    },
    {
        "id": "case_02",
        "input": "Hemoglobin: 8.5 g/dL\nMCV: 68 fL\nFerritin: 10 ng/mL",
        "expected_tests": {"Hemoglobin": 8.5, "MCV": 68.0, "Ferritin": 10.0},
        "expected_patterns": ["Microcytic Anaemia"],
    },
    {
        "id": "case_03",
        "input": "Potassium: 6.4 mEq/L\nCreatinine: 3.2 mg/dL\nBUN: 48 mg/dL",
        "expected_tests": {"Potassium": 6.4, "Serum Creatinine": 3.2, "BUN": 48.0},
        "expected_patterns": ["Hyperkalaemia", "Renal Impairment"],
    },
    {
        "id": "case_04",
        "input": "Glucose: 15.0 mmol/L\nBicarbonate: 14 mEq/L",
        "expected_tests": {"Random Glucose": 270.3, "Bicarbonate / CO2": 14.0},
        "expected_patterns": ["Hyperglycaemia with Low Bicarbonate (DKA Pattern)", "Metabolic Acidosis"],
    },
    {
        "id": "case_05",
        "input": "Troponin I 0.12 ng/mL\nCK-MB 35 U/L",
        "expected_tests": {"Troponin I": 0.12},
        "expected_patterns": ["Raised Troponin"],
    },
    {
        "id": "case_06",
        "input": "Platelets: 35 x10^3/uL",
        "expected_tests": {"Platelet Count": 35.0},
        "expected_patterns": ["Thrombocytopenia"],
    },
    {
        "id": "case_07",
        "input": "Hb 9.2 g/dL\nMCV 108 fL\nVitamin B12 110 pg/mL",
        "expected_tests": {"Hemoglobin": 9.2, "MCV": 108.0, "Vitamin B12": 110.0},
        "expected_patterns": ["Macrocytic Anaemia", "Vitamin Deficiencies"],
    },
    {
        "id": "case_08",
        "input": "Sodium: 118 mEq/L",
        "expected_tests": {"Sodium": 118.0},
        "expected_patterns": ["Hyponatraemia"],
    },
    {
        "id": "case_09",
        "input": "Calcium: 13.8 mg/dL",
        "expected_tests": {"Serum Calcium": 13.8},
        "expected_patterns": ["Hypercalcaemia"],
    },
    {
        "id": "case_10",
        "input": "Lactate: 5.2 mmol/L",
        "expected_tests": {"Lactate": 5.2},
        "expected_patterns": ["Raised Lactate"],
    },
    {
        "id": "case_11",
        "input": "ALT 420 U/L\nAST 380 U/L\nTotal Bilirubin 3.5 mg/dL",
        "expected_tests": {"ALT (SGPT)": 420.0, "AST (SGOT)": 380.0, "Total Bilirubin": 3.5},
        "expected_patterns": ["Hepatocellular Pattern", "Hyperbilirubinaemia"],
    },
    {
        "id": "case_12",
        "input": "Alkaline Phosphatase: 310 U/L\nGGT: 140 U/L\nDirect Bilirubin: 1.8 mg/dL",
        "expected_tests": {"Alkaline Phosphatase": 310.0, "GGT": 140.0, "Direct Bilirubin": 1.8},
        "expected_patterns": ["Cholestatic Pattern"],
    },
    {
        "id": "case_13",
        "input": "TSH: 14.5 uIU/mL\nFree T4: 0.5 ng/dL",
        "expected_tests": {"TSH": 14.5, "Free T4": 0.5},
        "expected_patterns": ["Primary Hypothyroidism Pattern"],
    },
    {
        "id": "case_14",
        "input": "TSH: 0.05 uIU/mL\nFree T4: 2.8 ng/dL",
        "expected_tests": {"TSH": 0.05, "Free T4": 2.8},
        "expected_patterns": ["Primary Hyperthyroidism Pattern"],
    },
    {
        "id": "case_15",
        "input": "INR: 5.2",
        "expected_tests": {"INR": 5.2},
        "expected_patterns": ["Prolonged INR"],
    },
    {
        "id": "case_16",
        "input": "Glucose fasting: 145 mg/dL\nHbA1c: 8.2 %",
        "expected_tests": {"Fasting Glucose": 145.0, "HbA1c": 8.2},
        "expected_patterns": ["Diabetic Range Glycemia"],
    },
    {
        "id": "case_17",
        "input": "Random blood sugar: 45 mg/dL",
        "expected_tests": {"Random Glucose": 45.0},
        "expected_patterns": ["Hypoglycaemia"],
    },
    {
        "id": "case_18",
        "input": "Serum Albumin: 2.2 g/dL",
        "expected_tests": {"Serum Albumin": 2.2},
        "expected_patterns": ["Hypoalbuminaemia"],
    },
    {
        "id": "case_19",
        "input": "WBC 2.1 x10^3/uL",
        "expected_tests": {"WBC Count": 2.1},
        "expected_patterns": ["Leukopenia"],
    },
    {
        "id": "case_20",
        "input": "Platelets: 550 x10^3/uL",
        "expected_tests": {"Platelet Count": 550.0},
        "expected_patterns": ["Thrombocytosis"],
    },
]

# Expand to 55 test cases by generating parameterized clinical variants
for i in range(21, 56):
    k_level = 5.2 + (i % 5) * 0.3
    cr_level = 1.6 + (i % 4) * 0.6
    wbc_level = 12.0 + (i % 6) * 1.5
    hb_level = 7.5 + (i % 5) * 0.5
    mcv_level = 70.0 if i % 2 == 0 else 105.0
    pat_type = "Microcytic Anaemia" if mcv_level < 80 else "Macrocytic Anaemia"

    GROUND_TRUTH_DATASET.append({
        "id": f"case_{i:02d}",
        "input": f"Hb {hb_level:.1f} g/dL\nMCV {mcv_level:.1f} fL\nWBC {wbc_level:.1f} x10^3/uL\nPotassium {k_level:.1f} mEq/L\nCreatinine {cr_level:.1f} mg/dL",
        "expected_tests": {
            "Hemoglobin": hb_level,
            "MCV": mcv_level,
            "WBC Count": wbc_level,
            "Potassium": k_level,
            "Serum Creatinine": cr_level,
        },
        "expected_patterns": [pat_type, "Leukocytosis", "Hyperkalaemia", "Renal Impairment"],
    })


def evaluate_lab_agent() -> Dict[str, Any]:
    """Run evaluation benchmark across all test cases and return metrics."""
    total_expected_fields = 0
    correct_fields = 0
    total_expected_patterns = 0
    true_positive_patterns = 0
    false_positive_patterns = 0

    for item in GROUND_TRUTH_DATASET:
        case_in = CaseInput(case_id=item["id"], clinical_text="Lab eval", lab_text=item["input"])
        report = run_lab_agent(case_in)

        # Check field extraction accuracy
        parsed_dict = {r.test: r.value for r in report.results}
        for exp_test, exp_val in item["expected_tests"].items():
            total_expected_fields += 1
            if exp_test in parsed_dict:
                actual_val = parsed_dict[exp_test]
                if abs(actual_val - exp_val) <= max(0.5, exp_val * 0.05):
                    correct_fields += 1

        # Check pattern precision and recall
        actual_patterns = {p.name for p in report.patterns}
        exp_patterns = set(item["expected_patterns"])
        total_expected_patterns += len(exp_patterns)

        for pat in exp_patterns:
            # Check exact or partial match
            if any(pat in act or act in pat for act in actual_patterns):
                true_positive_patterns += 1

        for act in actual_patterns:
            if not any(pat in act or act in pat for pat in exp_patterns):
                false_positive_patterns += 1

    field_accuracy = correct_fields / max(total_expected_fields, 1)
    pattern_recall = true_positive_patterns / max(total_expected_patterns, 1)
    pattern_precision = true_positive_patterns / max(true_positive_patterns + false_positive_patterns, 1)
    f1 = 2 * (pattern_precision * pattern_recall) / max(pattern_precision + pattern_recall, 1e-6)

    return {
        "dataset_size": len(GROUND_TRUTH_DATASET),
        "total_fields_evaluated": total_expected_fields,
        "field_extraction_accuracy": round(field_accuracy, 4),
        "total_patterns_evaluated": total_expected_patterns,
        "pattern_precision": round(pattern_precision, 4),
        "pattern_recall": round(pattern_recall, 4),
        "pattern_f1_score": round(f1, 4),
    }


if __name__ == "__main__":
    results = evaluate_lab_agent()
    print("=" * 60)
    print("LAB AGENT EVALUATION BENCHMARK")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:30s}: {v}")
