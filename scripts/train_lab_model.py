"""
Training pipeline for ClinicalLabNet: Multi-Label Clinical Laboratory Disease Classifier.
Generates an extensive, clinically calibrated multi-analyte dataset (12,000 cases),
trains the PyTorch neural network with BCEWithLogitsLoss, evaluates metrics,
and serializes weights to data/weights/clinical_lab_model.pt.
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from medagent.models.lab_net import (
    ClinicalLabNet,
    LAB_DISEASE_CLASSES,
    FEATURE_KEYS,
    FEATURE_SCALING,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_lab_model")

# Set random seeds for reproducible clinical training
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PANEL_TYPES = {
    "full": FEATURE_KEYS,
    "cbc": [
        "hemoglobin", "hematocrit", "wbc", "platelets", "mcv", "mch", "mchc", "rdw",
        "neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils"
    ],
    "bmp": [
        "fasting_glucose", "random_glucose", "blood_urea_nitrogen", "urea", "creatinine",
        "egfr", "sodium", "potassium", "chloride", "bicarbonate", "calcium"
    ],
    "cmp": [
        "fasting_glucose", "random_glucose", "blood_urea_nitrogen", "urea", "creatinine",
        "egfr", "sodium", "potassium", "chloride", "bicarbonate", "calcium",
        "alt", "ast", "alp", "total_bilirubin", "albumin"
    ],
    "hepatic": [
        "alt", "ast", "alp", "ggt", "total_bilirubin", "direct_bilirubin", "albumin", "inr"
    ],
    "cardiac": [
        "troponin_i", "troponin_t", "crp", "lactate", "fasting_glucose", "potassium",
        "creatinine", "blood_urea_nitrogen"
    ],
    "endocrine_metabolic": [
        "fasting_glucose", "random_glucose", "hba1c", "tsh", "free_t4", "total_cholesterol",
        "triglycerides", "ldl_cholesterol", "hdl_cholesterol"
    ],
    "inflammatory_sepsis": [
        "wbc", "neutrophils", "platelets", "lactate", "crp", "esr", "procalcitonin",
        "blood_urea_nitrogen", "creatinine", "bicarbonate"
    ],
    "iron_anemia": [
        "hemoglobin", "hematocrit", "mcv", "mch", "mchc", "rdw", "ferritin",
        "serum_iron", "vitamin_b12", "folate"
    ],
    "pancreatic": [
        "lipase", "amylase", "wbc", "calcium", "triglycerides", "alt", "ast", "total_bilirubin"
    ],
}


def sample_baseline_normal() -> Dict[str, float]:
    """Generate healthy physiological values with biological variance."""
    v: Dict[str, float] = {}
    v["hemoglobin"] = float(np.random.normal(14.2, 1.1))
    v["hematocrit"] = v["hemoglobin"] * 3.0 + float(np.random.normal(0, 0.8))
    v["wbc"] = float(np.clip(np.random.normal(7.2, 1.3), 4.5, 10.5))
    v["platelets"] = float(np.clip(np.random.normal(260.0, 40.0), 160.0, 390.0))
    v["mcv"] = float(np.clip(np.random.normal(89.0, 3.5), 82.0, 97.0))
    v["mch"] = float(np.clip(np.random.normal(30.0, 1.5), 27.0, 33.0))
    v["mchc"] = float(np.clip(np.random.normal(33.8, 0.8), 32.0, 35.5))
    v["rdw"] = float(np.clip(np.random.normal(12.8, 0.7), 11.5, 14.5))
    v["neutrophils"] = float(np.clip(np.random.normal(58.0, 6.0), 45.0, 70.0))
    v["lymphocytes"] = float(np.clip(100.0 - v["neutrophils"] - 8.0, 20.0, 40.0))
    v["monocytes"] = float(np.clip(np.random.normal(5.5, 1.2), 3.0, 9.0))
    v["eosinophils"] = float(np.clip(np.random.normal(2.2, 0.8), 1.0, 4.5))
    v["basophils"] = float(np.clip(np.random.normal(0.6, 0.2), 0.2, 1.2))

    v["fasting_glucose"] = float(np.clip(np.random.normal(88.0, 7.0), 72.0, 99.0))
    v["random_glucose"] = float(np.clip(np.random.normal(105.0, 12.0), 80.0, 135.0))
    v["hba1c"] = float(np.clip(np.random.normal(5.2, 0.25), 4.6, 5.6))

    v["blood_urea_nitrogen"] = float(np.clip(np.random.normal(13.5, 2.5), 8.0, 19.0))
    v["urea"] = v["blood_urea_nitrogen"] * 2.14
    v["creatinine"] = float(np.clip(np.random.normal(0.9, 0.12), 0.65, 1.15))
    v["egfr"] = float(np.clip(np.random.normal(98.0, 10.0), 85.0, 125.0))

    v["sodium"] = float(np.clip(np.random.normal(140.0, 1.8), 136.0, 144.0))
    v["potassium"] = float(np.clip(np.random.normal(4.2, 0.25), 3.7, 4.8))
    v["chloride"] = float(np.clip(np.random.normal(101.5, 2.0), 97.0, 106.0))
    v["bicarbonate"] = float(np.clip(np.random.normal(25.0, 1.5), 23.0, 28.0))
    v["calcium"] = float(np.clip(np.random.normal(9.4, 0.35), 8.8, 10.1))
    v["magnesium"] = float(np.clip(np.random.normal(2.05, 0.15), 1.8, 2.4))
    v["phosphate"] = float(np.clip(np.random.normal(3.5, 0.35), 2.8, 4.3))
    v["lactate"] = float(np.clip(np.random.normal(1.0, 0.25), 0.6, 1.6))

    v["alt"] = float(np.clip(np.random.normal(24.0, 6.0), 12.0, 38.0))
    v["ast"] = float(np.clip(np.random.normal(22.0, 5.5), 12.0, 35.0))
    v["alp"] = float(np.clip(np.random.normal(72.0, 14.0), 45.0, 105.0))
    v["ggt"] = float(np.clip(np.random.normal(26.0, 8.0), 12.0, 42.0))
    v["total_bilirubin"] = float(np.clip(np.random.normal(0.6, 0.18), 0.2, 1.0))
    v["direct_bilirubin"] = float(np.clip(np.random.normal(0.15, 0.05), 0.05, 0.25))
    v["albumin"] = float(np.clip(np.random.normal(4.3, 0.25), 3.8, 4.8))

    v["tsh"] = float(np.clip(np.random.normal(2.0, 0.7), 0.6, 3.8))
    v["free_t4"] = float(np.clip(np.random.normal(1.25, 0.15), 0.95, 1.65))

    v["troponin_i"] = float(np.clip(np.random.exponential(0.006), 0.001, 0.014))
    v["troponin_t"] = float(np.clip(np.random.exponential(0.004), 0.001, 0.009))
    v["crp"] = float(np.clip(np.random.exponential(1.5), 0.2, 4.8))
    v["esr"] = float(np.clip(np.random.normal(8.0, 2.5), 2.0, 14.0))
    v["procalcitonin"] = float(np.clip(np.random.exponential(0.03), 0.01, 0.08))

    v["ferritin"] = float(np.clip(np.random.normal(120.0, 35.0), 50.0, 220.0))
    v["serum_iron"] = float(np.clip(np.random.normal(88.0, 18.0), 60.0, 130.0))
    v["vitamin_b12"] = float(np.clip(np.random.normal(460.0, 90.0), 280.0, 750.0))
    v["folate"] = float(np.clip(np.random.normal(10.5, 2.5), 5.5, 17.0))
    v["vitamin_d"] = float(np.clip(np.random.normal(36.0, 7.0), 25.0, 55.0))
    v["inr"] = float(np.clip(np.random.normal(1.02, 0.05), 0.9, 1.15))

    v["lipase"] = float(np.clip(np.random.normal(32.0, 9.0), 15.0, 52.0))
    v["amylase"] = float(np.clip(np.random.normal(54.0, 12.0), 28.0, 85.0))

    v["total_cholesterol"] = float(np.clip(np.random.normal(172.0, 16.0), 135.0, 199.0))
    v["triglycerides"] = float(np.clip(np.random.normal(118.0, 18.0), 65.0, 149.0))
    v["ldl_cholesterol"] = float(np.clip(np.random.normal(96.0, 14.0), 60.0, 115.0))
    v["hdl_cholesterol"] = float(np.clip(np.random.normal(54.0, 7.0), 42.0, 75.0))
    return v


def inject_pathology(v: Dict[str, float], condition_name: str) -> None:
    """Modulate analyte values to mirror concrete disease pathophysiology."""
    if condition_name == "Acute Kidney Injury / Renal Failure":
        v["creatinine"] = float(np.random.uniform(2.1, 7.5))
        v["blood_urea_nitrogen"] = float(np.random.uniform(42.0, 125.0))
        v["urea"] = v["blood_urea_nitrogen"] * 2.14
        v["egfr"] = float(np.random.uniform(7.0, 32.0))
        v["potassium"] = float(np.clip(v["potassium"] + np.random.uniform(0.6, 2.0), 4.6, 7.2))
        v["bicarbonate"] = float(np.random.uniform(14.0, 21.0))
        v["phosphate"] = float(np.random.uniform(4.8, 8.5))

    elif condition_name == "Severe Hyperkalemia":
        v["potassium"] = float(np.random.uniform(5.8, 8.6))

    elif condition_name == "Hypokalemia":
        v["potassium"] = float(np.random.uniform(1.8, 3.3))

    elif condition_name == "Diabetic Ketoacidosis (DKA)":
        v["fasting_glucose"] = float(np.random.uniform(320.0, 780.0))
        v["random_glucose"] = v["fasting_glucose"] + float(np.random.uniform(-10.0, 30.0))
        v["bicarbonate"] = float(np.random.uniform(5.0, 16.0))
        v["sodium"] = float(np.random.uniform(124.0, 134.0))  # pseudohyponatremia
        v["potassium"] = float(np.random.uniform(4.5, 6.8))
        v["blood_urea_nitrogen"] = float(np.random.uniform(28.0, 65.0))
        v["creatinine"] = float(np.random.uniform(1.4, 2.8))

    elif condition_name == "Type 2 Diabetes Mellitus / Severe Hyperglycemia":
        v["fasting_glucose"] = float(np.random.uniform(138.0, 290.0))
        v["random_glucose"] = float(np.random.uniform(195.0, 380.0))
        v["hba1c"] = float(np.random.uniform(6.8, 13.5))
        v["bicarbonate"] = float(np.random.uniform(22.0, 28.0))

    elif condition_name == "Severe Hypoglycemia":
        v["fasting_glucose"] = float(np.random.uniform(22.0, 52.0))
        v["random_glucose"] = v["fasting_glucose"]

    elif condition_name == "High Anion Gap Metabolic Acidosis":
        v["bicarbonate"] = float(np.random.uniform(6.0, 16.0))
        v["chloride"] = float(np.random.uniform(96.0, 103.0))
        v["sodium"] = float(np.random.uniform(136.0, 144.0))
        v["lactate"] = float(np.random.uniform(3.2, 9.5))

    elif condition_name == "Hyponatremia":
        v["sodium"] = float(np.random.uniform(112.0, 132.0))

    elif condition_name == "Hypernatremia / Severe Dehydration":
        v["sodium"] = float(np.random.uniform(148.0, 168.0))
        v["blood_urea_nitrogen"] = float(np.random.uniform(32.0, 75.0))
        v["creatinine"] = float(np.random.uniform(1.1, 1.8))  # BUN/Cr > 25

    elif condition_name == "Acute Bacterial Infection / Sepsis Risk":
        v["wbc"] = float(np.random.uniform(14.5, 36.0))
        v["neutrophils"] = float(np.random.uniform(78.0, 95.0))
        v["lymphocytes"] = float(np.clip(100.0 - v["neutrophils"] - 5.0, 3.0, 16.0))
        v["crp"] = float(np.random.uniform(45.0, 320.0))
        v["procalcitonin"] = float(np.random.uniform(1.2, 28.0))
        v["lactate"] = float(np.random.uniform(2.3, 8.5))
        v["esr"] = float(np.random.uniform(35.0, 110.0))

    elif condition_name == "Microcytic Iron Deficiency Anemia":
        v["hemoglobin"] = float(np.random.uniform(6.2, 10.4))
        v["hematocrit"] = v["hemoglobin"] * 3.0 + float(np.random.normal(0, 0.5))
        v["mcv"] = float(np.random.uniform(58.0, 77.0))
        v["mch"] = float(np.random.uniform(18.0, 25.5))
        v["mchc"] = float(np.random.uniform(28.0, 31.8))
        v["rdw"] = float(np.random.uniform(16.0, 24.0))
        v["ferritin"] = float(np.random.uniform(3.0, 22.0))
        v["serum_iron"] = float(np.random.uniform(12.0, 42.0))

    elif condition_name == "Macrocytic / Megaloblastic Anemia":
        v["hemoglobin"] = float(np.random.uniform(6.5, 10.8))
        v["hematocrit"] = v["hemoglobin"] * 3.0
        v["mcv"] = float(np.random.uniform(103.0, 132.0))
        v["mch"] = float(np.random.uniform(33.5, 42.0))
        if random.random() < 0.7:
            v["vitamin_b12"] = float(np.random.uniform(60.0, 180.0))
        else:
            v["folate"] = float(np.random.uniform(1.0, 2.7))

    elif condition_name == "Severe Thrombocytopenia":
        v["platelets"] = float(np.random.uniform(6.0, 68.0))

    elif condition_name == "Acute Hepatocellular Injury / Transaminitis":
        v["alt"] = float(np.random.uniform(190.0, 3500.0))
        v["ast"] = float(np.random.uniform(160.0, 2900.0))
        v["total_bilirubin"] = float(np.random.uniform(1.4, 7.5))
        v["direct_bilirubin"] = v["total_bilirubin"] * float(np.random.uniform(0.5, 0.8))

    elif condition_name == "Cholestasis / Obstructive Biliary Disease":
        v["alp"] = float(np.random.uniform(195.0, 850.0))
        v["ggt"] = float(np.random.uniform(95.0, 580.0))
        v["total_bilirubin"] = float(np.random.uniform(2.8, 18.0))
        v["direct_bilirubin"] = v["total_bilirubin"] * float(np.random.uniform(0.65, 0.9))

    elif condition_name == "Acute Pancreatitis":
        v["lipase"] = float(np.random.uniform(190.0, 2800.0))
        v["amylase"] = float(np.random.uniform(160.0, 1200.0))
        v["wbc"] = float(np.random.uniform(11.5, 19.5))
        v["crp"] = float(np.random.uniform(20.0, 140.0))

    elif condition_name == "Acute Myocardial Infarction / ACS":
        v["troponin_i"] = float(np.random.uniform(0.08, 22.0))
        v["troponin_t"] = float(np.random.uniform(0.05, 12.0))
        v["crp"] = float(np.random.uniform(8.0, 65.0))

    elif condition_name == "Hypercalcemia":
        v["calcium"] = float(np.random.uniform(10.8, 16.5))

    elif condition_name == "Atherogenic Dyslipidemia":
        v["triglycerides"] = float(np.random.uniform(210.0, 850.0))
        v["ldl_cholesterol"] = float(np.random.uniform(145.0, 280.0))
        v["total_cholesterol"] = float(np.random.uniform(230.0, 390.0))
        v["hdl_cholesterol"] = float(np.random.uniform(22.0, 38.0))


def build_clinical_sample() -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs a realistic patient profile:
    - Normal baseline with real physiological noise.
    - Pathologies assigned according to clinical co-occurrence rules.
    - Realistic laboratory order panel masking (CBC, BMP, CMP, Full, etc.).
    - Returns:
      x: 108-dim vector (54 normalized values + 54 presence masks).
      y: 20-dim binary multi-label ground truth vector.
    """
    v = sample_baseline_normal()
    y = np.zeros(len(LAB_DISEASE_CLASSES), dtype=np.float32)

    # Co-occurrence probability distributions
    scenario = random.choices(
        [
            "normal",
            "single_disease",
            "renal_metabolic",
            "cardio_renal",
            "sepsis_aki",
            "diabetic_dka",
            "hepato_biliary",
            "anemia_cytopenia",
        ],
        weights=[0.20, 0.40, 0.08, 0.08, 0.08, 0.06, 0.05, 0.05],
        k=1
    )[0]

    assigned_conditions: List[str] = []

    if scenario == "normal":
        assigned_conditions = ["Normal / Unremarkable Laboratory Profile"]

    elif scenario == "single_disease":
        non_normal = [c for c in LAB_DISEASE_CLASSES if c != "Normal / Unremarkable Laboratory Profile"]
        chosen = random.choice(non_normal)
        assigned_conditions = [chosen]

    elif scenario == "renal_metabolic":
        assigned_conditions = ["Acute Kidney Injury / Renal Failure"]
        if random.random() < 0.65:
            assigned_conditions.append("Severe Hyperkalemia")
        if random.random() < 0.45:
            assigned_conditions.append("High Anion Gap Metabolic Acidosis")

    elif scenario == "cardio_renal":
        assigned_conditions = ["Acute Myocardial Infarction / ACS"]
        if random.random() < 0.50:
            assigned_conditions.append("Acute Kidney Injury / Renal Failure")
        if random.random() < 0.40:
            assigned_conditions.append("Atherogenic Dyslipidemia")

    elif scenario == "sepsis_aki":
        assigned_conditions = ["Acute Bacterial Infection / Sepsis Risk"]
        if random.random() < 0.70:
            assigned_conditions.append("High Anion Gap Metabolic Acidosis")
        if random.random() < 0.55:
            assigned_conditions.append("Acute Kidney Injury / Renal Failure")
        if random.random() < 0.35:
            assigned_conditions.append("Severe Thrombocytopenia")

    elif scenario == "diabetic_dka":
        if random.random() < 0.6:
            assigned_conditions = ["Diabetic Ketoacidosis (DKA)", "High Anion Gap Metabolic Acidosis"]
            if random.random() < 0.4:
                assigned_conditions.append("Severe Hyperkalemia")
        else:
            assigned_conditions = ["Type 2 Diabetes Mellitus / Severe Hyperglycemia"]
            if random.random() < 0.5:
                assigned_conditions.append("Atherogenic Dyslipidemia")

    elif scenario == "hepato_biliary":
        if random.random() < 0.5:
            assigned_conditions = ["Acute Hepatocellular Injury / Transaminitis"]
        else:
            assigned_conditions = ["Cholestasis / Obstructive Biliary Disease"]
            if random.random() < 0.4:
                assigned_conditions.append("Acute Pancreatitis")

    elif scenario == "anemia_cytopenia":
        if random.random() < 0.6:
            assigned_conditions = ["Microcytic Iron Deficiency Anemia"]
        else:
            assigned_conditions = ["Macrocytic / Megaloblastic Anemia"]
        if random.random() < 0.4:
            assigned_conditions.append("Severe Thrombocytopenia")

    for cond in assigned_conditions:
        inject_pathology(v, cond)

    # Select panel type to simulate real-world laboratory orders
    if scenario == "normal":
        relevant_panels = ["full", "cbc", "bmp", "cmp", "hepatic", "cardiac", "endocrine_metabolic", "iron_anemia"]
    else:
        relevant_panels = ["full"]
        if any("Anemia" in c or "Thrombocytopenia" in c for c in assigned_conditions):
            relevant_panels.extend(["cbc", "iron_anemia", "full"])
        if any("Kidney" in c or "Hyperkalemia" in c or "Hypokalemia" in c or "Acidosis" in c for c in assigned_conditions):
            relevant_panels.extend(["bmp", "cmp", "full"])
        if any("Sepsis" in c for c in assigned_conditions):
            relevant_panels.extend(["inflammatory_sepsis", "full"])
        if any("Hepatocellular" in c or "Cholestasis" in c for c in assigned_conditions):
            relevant_panels.extend(["cmp", "hepatic", "full"])
        if any("Pancreatitis" in c for c in assigned_conditions):
            relevant_panels.extend(["pancreatic", "cmp"])
        if any("Myocardial" in c for c in assigned_conditions):
            relevant_panels.extend(["cardiac", "full"])
        if any("Diabetes" in c or "DKA" in c or "Dyslipidemia" in c for c in assigned_conditions):
            relevant_panels.extend(["endocrine_metabolic", "bmp", "cmp"])

    chosen_panel_name = random.choice(relevant_panels)
    available_tests = set(PANEL_TYPES[chosen_panel_name])

    # Simulate realistic laboratory test dropout (some labs omit specific tests from a panel)
    if random.random() < 0.65:
        drop_rate = random.uniform(0.10, 0.40)
        available_tests = {t for t in available_tests if random.random() > drop_rate}
        if not available_tests:
            available_tests = set(PANEL_TYPES[chosen_panel_name][:3])

    # Occasionally add 1 to 4 random bonus tests
    if chosen_panel_name != "full" and random.random() < 0.4:
        bonus_tests = random.sample(FEATURE_KEYS, k=random.randint(1, 4))
        available_tests.update(bonus_tests)

    # Encode into 108-dim representation
    val_vector = np.zeros(len(FEATURE_KEYS), dtype=np.float32)
    mask_vector = np.zeros(len(FEATURE_KEYS), dtype=np.float32)

    for i, k in enumerate(FEATURE_KEYS):
        if k in available_tests and k in v:
            mean_val, std_val = FEATURE_SCALING.get(k, (0.0, 1.0))
            val_vector[i] = float((v[k] - mean_val) / max(std_val, 1e-4))
            mask_vector[i] = 1.0
        else:
            val_vector[i] = 0.0
            mask_vector[i] = 0.0

    x = np.concatenate([val_vector, mask_vector], axis=0)

    # Ground truth labels: verify conditions are detectable given available tests
    if scenario == "normal":
        y[LAB_DISEASE_CLASSES.index("Normal / Unremarkable Laboratory Profile")] = 1.0
    else:
        for cond in assigned_conditions:
            if cond == "Normal / Unremarkable Laboratory Profile":
                continue
            idx = LAB_DISEASE_CLASSES.index(cond)
            if cond in ("Severe Hyperkalemia", "Hypokalemia") and "potassium" in available_tests:
                y[idx] = 1.0
            elif cond == "Acute Kidney Injury / Renal Failure" and ("creatinine" in available_tests or "egfr" in available_tests or "blood_urea_nitrogen" in available_tests):
                y[idx] = 1.0
            elif cond == "Diabetic Ketoacidosis (DKA)" and ("fasting_glucose" in available_tests or "random_glucose" in available_tests) and ("bicarbonate" in available_tests or "potassium" in available_tests):
                y[idx] = 1.0
            elif cond == "Type 2 Diabetes Mellitus / Severe Hyperglycemia" and ("fasting_glucose" in available_tests or "random_glucose" in available_tests or "hba1c" in available_tests):
                y[idx] = 1.0
            elif cond == "Severe Hypoglycemia" and ("fasting_glucose" in available_tests or "random_glucose" in available_tests):
                y[idx] = 1.0
            elif cond == "High Anion Gap Metabolic Acidosis" and "bicarbonate" in available_tests:
                y[idx] = 1.0
            elif cond in ("Hyponatremia", "Hypernatremia / Severe Dehydration") and "sodium" in available_tests:
                y[idx] = 1.0
            elif cond == "Acute Bacterial Infection / Sepsis Risk" and ("wbc" in available_tests or "procalcitonin" in available_tests or "crp" in available_tests or "lactate" in available_tests):
                y[idx] = 1.0
            elif cond == "Microcytic Iron Deficiency Anemia" and ("hemoglobin" in available_tests or "hematocrit" in available_tests or "ferritin" in available_tests) and ("mcv" in available_tests or "ferritin" in available_tests or "rdw" in available_tests):
                y[idx] = 1.0
            elif cond == "Macrocytic / Megaloblastic Anemia" and ("hemoglobin" in available_tests or "hematocrit" in available_tests) and ("mcv" in available_tests or "vitamin_b12" in available_tests or "folate" in available_tests):
                y[idx] = 1.0
            elif cond == "Severe Thrombocytopenia" and "platelets" in available_tests:
                y[idx] = 1.0
            elif cond == "Acute Hepatocellular Injury / Transaminitis" and ("alt" in available_tests or "ast" in available_tests):
                y[idx] = 1.0
            elif cond == "Cholestasis / Obstructive Biliary Disease" and ("alp" in available_tests or "total_bilirubin" in available_tests or "ggt" in available_tests):
                y[idx] = 1.0
            elif cond == "Acute Pancreatitis" and ("lipase" in available_tests or "amylase" in available_tests):
                y[idx] = 1.0
            elif cond == "Acute Myocardial Infarction / ACS" and ("troponin_i" in available_tests or "troponin_t" in available_tests):
                y[idx] = 1.0
            elif cond == "Hypercalcemia" and "calcium" in available_tests:
                y[idx] = 1.0
            elif cond == "Atherogenic Dyslipidemia" and ("triglycerides" in available_tests or "ldl_cholesterol" in available_tests or "total_cholesterol" in available_tests):
                y[idx] = 1.0

    return x, y


class ClinicalLabDataset(Dataset):
    """In-memory dataset for laboratory neural training."""
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x = torch.from_numpy(x_data).float()
        self.y = torch.from_numpy(y_data).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Calculate multi-label evaluation metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_b, y_b in dataloader:
            x_b = x_b.to(device)
            logits = model(x_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y_b.numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # Binary accuracy per label with 0.5 threshold
    bin_preds = (preds >= 0.5).astype(np.float32)

    # Class-wise metrics
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for c in range(targets.shape[1]):
        t_c = targets[:, c]
        p_c = bin_preds[:, c]

        tp = np.sum((t_c == 1) & (p_c == 1))
        fp = np.sum((t_c == 0) & (p_c == 1))
        fn = np.sum((t_c == 1) & (p_c == 0))
        tn = np.sum((t_c == 0) & (p_c == 0))

        acc = (tp + tn) / max(len(t_c), 1)
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 1.0
        rec = tp / max(tp + fn, 1) if (tp + fn) > 0 else 1.0
        f1 = (2 * prec * rec) / max(prec + rec, 1e-6)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "macro_accuracy": float(np.mean(accuracies)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "exact_match_ratio": float(np.mean(np.all(bin_preds == targets, axis=1))),
    }


def main():
    logger.info("Initializing Clinical Laboratory Model Dataset Generation...")
    TOTAL_SAMPLES = 12000
    x_all = np.zeros((TOTAL_SAMPLES, len(FEATURE_KEYS) * 2), dtype=np.float32)
    y_all = np.zeros((TOTAL_SAMPLES, len(LAB_DISEASE_CLASSES)), dtype=np.float32)

    for i in range(TOTAL_SAMPLES):
        x_i, y_i = build_clinical_sample()
        x_all[i] = x_i
        y_all[i] = y_i

    # Train / Val / Test split (80% / 10% / 10%)
    indices = np.random.permutation(TOTAL_SAMPLES)
    n_train = int(0.80 * TOTAL_SAMPLES)
    n_val = int(0.10 * TOTAL_SAMPLES)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = ClinicalLabDataset(x_all[train_idx], y_all[train_idx])
    val_ds = ClinicalLabDataset(x_all[val_idx], y_all[val_idx])
    test_ds = ClinicalLabDataset(x_all[test_idx], y_all[test_idx])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    logger.info("Dataset generated: %d train, %d val, %d test samples.", len(train_ds), len(val_ds), len(test_ds))

    # Calculate positive weights for class balancing in BCEWithLogitsLoss
    pos_counts = np.sum(y_all[train_idx], axis=0)
    neg_counts = len(train_idx) - pos_counts
    pos_weights = torch.tensor(
        np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, 15.0),
        dtype=torch.float32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    model = ClinicalLabNet(
        in_features=len(FEATURE_KEYS) * 2,
        num_classes=len(LAB_DISEASE_CLASSES),
        hidden_dims=[256, 128, 64],
        dropout=0.15,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = 0.0
    best_state = None

    logger.info("Starting training of ClinicalLabNet for %d epochs...", epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(x_b)

        scheduler.step()
        epoch_loss = running_loss / len(train_ds)
        val_metrics = evaluate_model(model, val_loader, device)

        logger.info(
            "Epoch %02d/%d - Loss: %.4f | Val F1: %.4f | Val Precision: %.4f | Val Recall: %.4f | Val Acc: %.4f",
            epoch, epochs, epoch_loss, val_metrics["macro_f1"], val_metrics["macro_precision"],
            val_metrics["macro_recall"], val_metrics["macro_accuracy"]
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    logger.info("Training complete. Evaluating best checkpoint on held-out test set...")
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, test_loader, device)
    logger.info("================ FINAL TEST EVALUATION ================")
    logger.info("Macro Accuracy:  %.2f%%", test_metrics["macro_accuracy"] * 100)
    logger.info("Macro Precision: %.2f%%", test_metrics["macro_precision"] * 100)
    logger.info("Macro Recall:    %.2f%%", test_metrics["macro_recall"] * 100)
    logger.info("Macro F1-Score:  %.4f", test_metrics["macro_f1"])
    logger.info("Exact Match:     %.2f%%", test_metrics["exact_match_ratio"] * 100)
    logger.info("=======================================================")

    # Ensure weights directory exists
    weights_dir = BASE_DIR / "data" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_file = weights_dir / "clinical_lab_model.pt"
    meta_file = weights_dir / "clinical_lab_model_meta.json"

    # Save checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_features": len(FEATURE_KEYS) * 2,
            "num_classes": len(LAB_DISEASE_CLASSES),
            "classes": LAB_DISEASE_CLASSES,
            "feature_keys": FEATURE_KEYS,
            "feature_scaling": FEATURE_SCALING,
            "hidden_dims": [256, 128, 64],
            "metrics": test_metrics,
        },
        weights_file
    )
    logger.info("Trained model saved to: %s (%d bytes)", weights_file, weights_file.stat().st_size)

    # Save metadata JSON for runtime diagnostics
    meta_data = {
        "architecture": "ClinicalLabNet (PyTorch Multi-Label Deep Neural Network)",
        "in_features": len(FEATURE_KEYS) * 2,
        "num_classes": len(LAB_DISEASE_CLASSES),
        "classes": LAB_DISEASE_CLASSES,
        "parameters": sum(p.numel() for p in model.parameters()),
        "metrics": test_metrics,
        "epochs": epochs,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)
    logger.info("Metadata saved to: %s", meta_file)


if __name__ == "__main__":
    main()
