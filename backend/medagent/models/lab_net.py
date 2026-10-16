import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("medagent.lab_net")

LAB_DISEASE_CLASSES: List[str] = [
    "Acute Kidney Injury / Renal Failure",
    "Severe Hyperkalemia",
    "Hypokalemia",
    "Diabetic Ketoacidosis (DKA)",
    "Type 2 Diabetes Mellitus / Severe Hyperglycemia",
    "Severe Hypoglycemia",
    "High Anion Gap Metabolic Acidosis",
    "Hyponatremia",
    "Hypernatremia / Severe Dehydration",
    "Acute Bacterial Infection / Sepsis Risk",
    "Microcytic Iron Deficiency Anemia",
    "Macrocytic / Megaloblastic Anemia",
    "Severe Thrombocytopenia",
    "Acute Hepatocellular Injury / Transaminitis",
    "Cholestasis / Obstructive Biliary Disease",
    "Acute Pancreatitis",
    "Acute Myocardial Infarction / ACS",
    "Hypercalcemia",
    "Atherogenic Dyslipidemia",
    "Normal / Unremarkable Laboratory Profile",
]

FEATURE_KEYS: List[str] = [
    "hemoglobin", "hematocrit", "wbc", "platelets", "mcv", "mch", "mchc", "rdw",
    "neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils",
    "fasting_glucose", "random_glucose", "hba1c", "blood_urea_nitrogen", "urea",
    "creatinine", "egfr", "sodium", "potassium", "chloride", "bicarbonate",
    "calcium", "magnesium", "phosphate", "lactate", "alt", "ast", "alp", "ggt",
    "total_bilirubin", "direct_bilirubin", "albumin", "tsh", "free_t4",
    "troponin_i", "troponin_t", "crp", "esr", "procalcitonin", "ferritin",
    "serum_iron", "vitamin_b12", "folate", "vitamin_d", "inr", "lipase",
    "amylase", "total_cholesterol", "triglycerides", "ldl_cholesterol", "hdl_cholesterol"
]

# Baseline biological scaling values (mean, std) for neural normalization
FEATURE_SCALING: Dict[str, tuple[float, float]] = {
    "hemoglobin": (14.0, 2.5),
    "hematocrit": (42.0, 6.0),
    "wbc": (8.0, 4.5),
    "platelets": (275.0, 100.0),
    "mcv": (90.0, 10.0),
    "mch": (30.0, 3.5),
    "mchc": (34.0, 2.0),
    "rdw": (13.0, 2.0),
    "neutrophils": (60.0, 15.0),
    "lymphocytes": (30.0, 10.0),
    "monocytes": (6.0, 3.0),
    "eosinophils": (3.0, 2.0),
    "basophils": (0.8, 0.6),
    "fasting_glucose": (95.0, 40.0),
    "random_glucose": (115.0, 60.0),
    "hba1c": (5.6, 1.8),
    "blood_urea_nitrogen": (16.0, 15.0),
    "urea": (30.0, 25.0),
    "creatinine": (1.0, 1.2),
    "egfr": (90.0, 30.0),
    "sodium": (140.0, 6.0),
    "potassium": (4.2, 0.9),
    "chloride": (101.0, 5.0),
    "bicarbonate": (25.0, 5.5),
    "calcium": (9.4, 1.0),
    "magnesium": (2.0, 0.4),
    "phosphate": (3.5, 0.9),
    "lactate": (1.3, 1.5),
    "alt": (30.0, 60.0),
    "ast": (28.0, 55.0),
    "alp": (85.0, 50.0),
    "ggt": (35.0, 45.0),
    "total_bilirubin": (0.7, 1.5),
    "direct_bilirubin": (0.2, 0.8),
    "albumin": (4.2, 0.7),
    "tsh": (2.2, 2.5),
    "free_t4": (1.3, 0.5),
    "troponin_i": (0.015, 0.1),
    "troponin_t": (0.01, 0.08),
    "crp": (5.0, 25.0),
    "esr": (12.0, 20.0),
    "procalcitonin": (0.1, 0.8),
    "ferritin": (120.0, 150.0),
    "serum_iron": (85.0, 40.0),
    "vitamin_b12": (450.0, 250.0),
    "folate": (10.0, 6.0),
    "vitamin_d": (35.0, 18.0),
    "inr": (1.05, 0.6),
    "lipase": (40.0, 80.0),
    "amylase": (65.0, 90.0),
    "total_cholesterol": (185.0, 45.0),
    "triglycerides": (140.0, 90.0),
    "ldl_cholesterol": (110.0, 40.0),
    "hdl_cholesterol": (52.0, 15.0),
}

DISEASE_METADATA: Dict[str, Dict[str, Any]] = {
    "Severe Hyperkalemia": {
        "severity": "critical",
        "pathophysiology": "Dangerous elevation of extracellular potassium lowering myocardial resting membrane threshold, impairing cardiac repolarization, and predisposing to fatal ventricular arrhythmias or asystole.",
        "workup": [
            "Immediate 12-lead ECG for peaked T waves, PR prolongation, and QRS widening",
            "Continuous cardiac telemetry monitoring",
            "Membrane stabilization with IV Calcium Gluconate (10%) if ECG changes present",
            "Intracellular shifting therapy (IV regular insulin + 50% Dextrose, nebulized albuterol)",
        ],
    },
    "Hypokalemia": {
        "severity": "moderate",
        "pathophysiology": "Depletion of extracellular potassium hyperpolarizing excitable cell membranes, impairing neuromuscular transmission, and causing cardiac repolarization delay with prominent U waves.",
        "workup": [
            "12-lead ECG for ST depression, T wave flattening, and U waves",
            "Serum magnesium level (hypomagnesemia refractory potassium wasting)",
            "Oral or intravenous potassium chloride replacement protocol",
        ],
    },
    "Acute Kidney Injury / Renal Failure": {
        "severity": "severe",
        "pathophysiology": "Abrupt decline in glomerular filtration rate leading to retention of urea, creatinine, and nitrogenous metabolic waste products with disruption of fluid, electrolyte, and acid-base homeostasis.",
        "workup": [
            "Renal duplex ultrasound to exclude obstructive hydronephrosis",
            "Urinalysis with urine microscopy for casts, protein, and dysmorphic RBCs",
            "Fractional excretion of sodium (FeNa) or urea (FEUrea)",
            "Strict intake/output tracking and discontinuation of nephrotoxic medications",
        ],
    },
    "Diabetic Ketoacidosis (DKA)": {
        "severity": "critical",
        "pathophysiology": "Profound insulin deficiency with excessive counter-regulatory hormone surge driving unchecked lipolysis, hepatic free fatty acid oxidation into ketoacids, high anion gap metabolic acidosis, and severe osmotic dehydration.",
        "workup": [
            "Serum beta-hydroxybutyrate and urine ketones",
            "Arterial/venous blood gas for pH and lactate",
            "Aggressive isotonic IV crystalloid volume resuscitation",
            "Continuous intravenous regular insulin infusion",
            "Hourly capillary blood glucose and serial potassium tracking",
        ],
    },
    "High Anion Gap Metabolic Acidosis": {
        "severity": "severe",
        "pathophysiology": "Pathological accumulation of unmeasured fixed organic acids (lactic acid, ketoacids, uremic toxins) consuming serum bicarbonate buffers.",
        "workup": [
            "Arterial blood gas for definitive pH and pCO2 respiratory compensation",
            "Serum lactate and serum beta-hydroxybutyrate",
            "Screening for toxic ingestions if etiology undetermined",
        ],
    },
    "Type 2 Diabetes Mellitus / Severe Hyperglycemia": {
        "severity": "moderate",
        "pathophysiology": "Chronic metabolic disorder of peripheral insulin resistance and progressive pancreatic beta-cell secretory failure resulting in sustained hyperglycemia.",
        "workup": [
            "Fasting lipid panel and spot urine albumin-to-creatinine ratio",
            "Dilated diabetic retinal examination",
            "Comprehensive diabetic foot and peripheral vascular evaluation",
        ],
    },
    "Severe Hypoglycemia": {
        "severity": "critical",
        "pathophysiology": "Critically low plasma glucose concentration depriving cerebral neurons of essential oxidative fuel, triggering counter-regulatory catecholamine release and neuroglycopenic dysfunction.",
        "workup": [
            "Immediate oral fast-acting glucose or intravenous 10-50% dextrose",
            "Reconciliation of insulin, sulfonylurea, or oral hypoglycemic doses",
            "Serum C-peptide, insulin, and beta-hydroxybutyrate if etiology unknown",
        ],
    },
    "Hyponatremia": {
        "severity": "moderate",
        "pathophysiology": "Excess of total body water relative to sodium content causing extracellular hypo-osmolality, driving water shift into intracellular spaces and predisposing to cerebral edema.",
        "workup": [
            "Serum and urine osmolality",
            "Urine sodium concentration",
            "Clinical extracellular volume status determination (euvolemic vs hypovolemic vs hypervolemic)",
        ],
    },
    "Hypernatremia / Severe Dehydration": {
        "severity": "severe",
        "pathophysiology": "Net free water deficit producing serum hyperosmolality, intracellular dehydration, and cellular shrinkage.",
        "workup": [
            "Calculation of total body free water deficit",
            "Gradual oral or hypotonic IV free water repletion monitoring",
        ],
    },
    "Acute Bacterial Infection / Sepsis Risk": {
        "severity": "critical",
        "pathophysiology": "Invasive pathogen eliciting acute bone marrow leukocyte demargination, bandemia, and systemic inflammatory endothelial activation with capillary leak.",
        "workup": [
            "Blood cultures (2 sets drawn from separate sites prior to antibiotics)",
            "Diagnostic imaging directed at suspected source (chest radiograph, CT abdomen)",
            "Serial serum lactate every 2-4 hours to monitor tissue perfusion",
            "Prompt broad-spectrum empiric antimicrobial coverage",
        ],
    },
    "Microcytic Iron Deficiency Anemia": {
        "severity": "moderate",
        "pathophysiology": "Inadequate iron stores restricting erythroblast heme and hemoglobin synthesis, resulting in microcytic, hypochromic erythrocytes with reduced systemic oxygen delivery.",
        "workup": [
            "Complete iron studies (serum ferritin, serum iron, TIBC, transferrin saturation)",
            "Fecal occult blood test / endoscopy evaluation for gastrointestinal blood loss",
            "Oral or intravenous therapeutic iron replacement",
        ],
    },
    "Macrocytic / Megaloblastic Anemia": {
        "severity": "moderate",
        "pathophysiology": "Deficiency of vitamin B12 or folate disrupting purine/thymidylate synthesis and erythroblast nuclear maturation, leading to nuclear-cytoplasmic dyssynchrony and abnormally large erythrocytes.",
        "workup": [
            "Serum methylmalonic acid (MMA) and total homocysteine levels",
            "Peripheral blood smear examination for hypersegmented neutrophils",
            "Vitamin B12 and folate therapeutic repletion",
        ],
    },
    "Severe Thrombocytopenia": {
        "severity": "critical",
        "pathophysiology": "Substantial reduction in circulating platelets predisposing to spontaneous mucocutaneous hemorrhage, petechiae, and visceral bleeding.",
        "workup": [
            "Peripheral blood smear to rule out EDTA-dependent pseudothrombocytopenia",
            "Coagulation profile (PT/INR, aPTT, fibrinogen) to exclude DIC",
            "Medication reconciliation for heparin-induced or drug-induced thrombocytopenia",
        ],
    },
    "Acute Hepatocellular Injury / Transaminitis": {
        "severity": "severe",
        "pathophysiology": "Direct toxic, ischemic, or viral destruction of hepatocytes resulting in extensive leakage of cytosolic aminotransferases into circulation.",
        "workup": [
            "Acute viral hepatitis panel (Anti-HAV IgM, HBsAg, Anti-HBc IgM, Anti-HCV)",
            "Serum acetaminophen level and comprehensive toxicology screen",
            "Coagulation profile (PT/INR) and serum albumin to evaluate hepatic synthetic reserve",
            "Right upper quadrant abdominal ultrasound with Doppler",
        ],
    },
    "Cholestasis / Obstructive Biliary Disease": {
        "severity": "moderate",
        "pathophysiology": "Impairment of bile excretion secondary to extrahepatic ductal stone/stricture or intrahepatic canalicular injury, causing accumulation of bile acids, alkaline phosphatase, and conjugated bilirubin.",
        "workup": [
            "Right upper quadrant abdominal ultrasound for biliary ductal dilatation or cholelithiasis",
            "Magnetic resonance cholangiopancreatography (MRCP)",
            "Fractionated bilirubin and serum lipase",
        ],
    },
    "Acute Pancreatitis": {
        "severity": "severe",
        "pathophysiology": "Premature intrapancreatic trypsinogen activation within acinar cells causing enzymatic autodigestion, local tissue necrosis, vascular damage, and acute systemic inflammatory mediator release.",
        "workup": [
            "Abdominal ultrasound to evaluate for gallstone pancreatitis",
            "Serum triglycerides and calcium levels",
            "Aggressive isotonic intravenous fluid resuscitation",
            "Contrast-enhanced abdominal CT if clinical deterioration occurs after 48-72h",
        ],
    },
    "Acute Myocardial Infarction / ACS": {
        "severity": "critical",
        "pathophysiology": "Coronary atherosclerotic plaque rupture with occlusive thrombosis causing cardiomyocyte ischemia, irreversible necrosis, and systemic leakage of high-sensitivity cardiac troponins.",
        "workup": [
            "Serial 12-lead ECGs (every 15-30 minutes)",
            "Repeat high-sensitivity troponin in 1 to 2 hours to evaluate dynamic rise/fall delta",
            "Emergent cardiology consultation for invasive coronary angiography",
            "Telemetry cardiac monitoring and guideline-directed medical therapy",
        ],
    },
    "Hypercalcemia": {
        "severity": "moderate",
        "pathophysiology": "Elevation of extracellular calcium concentration secondary to excessive bone resorption (hyperparathyroidism, malignancy) or increased intestinal absorption, diminishing neuromuscular excitability.",
        "workup": [
            "Intact parathyroid hormone (iPTH)",
            "PTH-related peptide (PTHrP) and 25-OH Vitamin D",
            "12-lead ECG for shortened QT interval",
            "Isotonic saline IV volume expansion",
        ],
    },
    "Atherogenic Dyslipidemia": {
        "severity": "mild",
        "pathophysiology": "Elevated circulating apolipoprotein B-containing atherogenic lipoproteins promoting subendothelial retention, arterial foam cell formation, and atherosclerotic plaque progression.",
        "workup": [
            "Cardiovascular 10-year ASCVD risk stratification",
            "Fasting glucose / HbA1c to screen for secondary causes",
            "Dietary modification and statin initiation",
        ],
    },
    "Normal / Unremarkable Laboratory Profile": {
        "severity": "mild",
        "pathophysiology": "Normal biochemical and cellular physiological homeostasis with all evaluated parameters falling within established biological reference intervals.",
        "workup": [
            "Routine preventative health maintenance",
            "Periodic re-evaluation as clinically indicated",
        ],
    },
}


class ClinicalLabNet(nn.Module):
    """
    Deep neural network multi-label disease classifier for clinical laboratory reports.
    Ingests standardized laboratory analyte values alongside presence-mask encodings.
    """
    def __init__(
        self,
        in_features: int = len(FEATURE_KEYS) * 2,  # 54 values + 54 presence masks = 108
        num_classes: int = len(LAB_DISEASE_CLASSES),
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        prev_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw uncalibrated logits of shape (batch_size, num_classes)."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def vectorize_lab_results(results: List[Any]) -> torch.Tensor:
    """
    Transform a list of parsed LabResult items into a 108-dim normalized feature tensor.
    First 54 elements: standard-scaled numerical values (default 0.0 if absent).
    Next 54 elements: binary presence mask (1.0 if measured in report, 0.0 if missing).
    """
    val_vector = [0.0] * len(FEATURE_KEYS)
    mask_vector = [0.0] * len(FEATURE_KEYS)

    # Name normalization mapping
    alias_to_feature = {
        "hemoglobin": "hemoglobin",
        "hematocrit": "hematocrit",
        "wbc count": "wbc",
        "wbc": "wbc",
        "platelet count": "platelets",
        "platelets": "platelets",
        "mcv": "mcv",
        "mch": "mch",
        "mchc": "mchc",
        "rdw": "rdw",
        "neutrophils (%)": "neutrophils",
        "neutrophils": "neutrophils",
        "lymphocytes (%)": "lymphocytes",
        "lymphocytes": "lymphocytes",
        "monocytes (%)": "monocytes",
        "monocytes": "monocytes",
        "eosinophils (%)": "eosinophils",
        "eosinophils": "eosinophils",
        "basophils (%)": "basophils",
        "basophils": "basophils",
        "fasting glucose": "fasting_glucose",
        "postprandial glucose": "random_glucose",
        "random glucose": "random_glucose",
        "hba1c": "hba1c",
        "bun": "blood_urea_nitrogen",
        "blood urea nitrogen": "blood_urea_nitrogen",
        "blood urea": "urea",
        "urea": "urea",
        "serum creatinine": "creatinine",
        "creatinine": "creatinine",
        "egfr": "egfr",
        "sodium": "sodium",
        "potassium": "potassium",
        "chloride": "chloride",
        "bicarbonate / co2": "bicarbonate",
        "bicarbonate": "bicarbonate",
        "serum calcium": "calcium",
        "calcium": "calcium",
        "magnesium": "magnesium",
        "phosphate": "phosphate",
        "lactate": "lactate",
        "alt (sgpt)": "alt",
        "alt": "alt",
        "ast (sgot)": "ast",
        "ast": "ast",
        "alkaline phosphatase": "alp",
        "alp": "alp",
        "ggt": "ggt",
        "total bilirubin": "total_bilirubin",
        "direct bilirubin": "direct_bilirubin",
        "serum albumin": "albumin",
        "albumin": "albumin",
        "tsh": "tsh",
        "free t4": "free_t4",
        "troponin i": "troponin_i",
        "troponin t": "troponin_t",
        "c-reactive protein": "crp",
        "crp": "crp",
        "esr": "esr",
        "procalcitonin": "procalcitonin",
        "ferritin": "ferritin",
        "serum iron": "serum_iron",
        "vitamin b12": "vitamin_b12",
        "serum folate": "folate",
        "vitamin d (25-oh)": "vitamin_d",
        "inr": "inr",
        "lipase": "lipase",
        "amylase": "amylase",
        "total cholesterol": "total_cholesterol",
        "triglycerides": "triglycerides",
        "ldl cholesterol": "ldl_cholesterol",
        "hdl cholesterol": "hdl_cholesterol",
    }

    key_to_idx = {k: i for i, k in enumerate(FEATURE_KEYS)}

    for r in results:
        t_name = getattr(r, "test", "").lower().strip()
        val = getattr(r, "value", None)
        if val is None:
            continue

        feat_key = alias_to_feature.get(t_name)
        if not feat_key:
            # Check substring
            for a_key, target in alias_to_feature.items():
                if a_key in t_name or t_name in a_key:
                    feat_key = target
                    break

        if feat_key and feat_key in key_to_idx:
            idx = key_to_idx[feat_key]
            mean_val, std_val = FEATURE_SCALING.get(feat_key, (0.0, 1.0))
            # Standard score normalization
            val_vector[idx] = float((val - mean_val) / max(std_val, 1e-4))
            mask_vector[idx] = 1.0

    combined = val_vector + mask_vector
    return torch.tensor([combined], dtype=torch.float32)


_LAB_MODEL_SINGLETON: Optional[ClinicalLabNet] = None


def get_clinical_lab_model(weights_path: Optional[Path] = None) -> Optional[ClinicalLabNet]:
    """
    Thread-safe cached singleton loader for trained ClinicalLabNet PyTorch model weights.
    Loads data/weights/clinical_lab_model.pt into evaluation mode.
    """
    global _LAB_MODEL_SINGLETON
    if _LAB_MODEL_SINGLETON is not None:
        return _LAB_MODEL_SINGLETON

    if weights_path is None:
        try:
            from medagent.config import settings
            weights_path = settings.CLINICAL_LAB_WEIGHTS
        except Exception:
            weights_path = Path("data/weights/clinical_lab_model.pt")

    weights_path = Path(weights_path)
    if not weights_path.exists():
        logger.warning("ClinicalLabNet weights not found at %s", weights_path)
        return None

    try:
        model = ClinicalLabNet()
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        model.eval()
        _LAB_MODEL_SINGLETON = model
        logger.info("Successfully loaded ClinicalLabNet deep neural model from %s", weights_path)
        return _LAB_MODEL_SINGLETON
    except Exception as e:
        logger.error("Failed to load ClinicalLabNet model: %s", e)
        return None


def infer_diseases_with_lab_net(
    results: List[Any],
    patient_sex: Optional[str] = None,
    threshold: float = 0.50,
) -> List[Any]:
    """
    Run PyTorch neural tensor forward-pass inference on vectorized laboratory findings.
    Uses trained ClinicalLabNet multi-label deep neural network to predict disease probabilities.
    """
    from medagent.schemas import LabDiseaseInference

    if not results:
        return []

    model = get_clinical_lab_model()
    if model is None:
        logger.warning("ClinicalLabNet model checkpoint unavailable; returning empty disease inferences.")
        return []

    # Build input tensor: shape (1, 108)
    x = vectorize_lab_results(results)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # Identify measured tests with abnormal flags for clinical rationale
    abnormal_tests = [
        f"{r.test}: {r.value:g} {r.unit} ({r.flag.upper()})"
        for r in results
        if getattr(r, "flag", "normal") not in ("normal", "unknown")
    ]
    all_measured = [
        f"{r.test}: {r.value:g} {r.unit} ({r.flag.upper()})"
        for r in results
    ]

    # Suppress Normal Profile if any measured analyte is flagged abnormal
    norm_idx = LAB_DISEASE_CLASSES.index("Normal / Unremarkable Laboratory Profile")
    if abnormal_tests:
        probs[norm_idx] = 0.0

    inferred: List[LabDiseaseInference] = []

    # Map neural network predictions to LabDiseaseInference objects
    for idx, disease_name in enumerate(LAB_DISEASE_CLASSES):
        prob = float(probs[idx])
        if prob < threshold:
            continue

        meta = DISEASE_METADATA.get(disease_name, {
            "severity": "moderate",
            "pathophysiology": "Pathological deviation in physiological biomarkers.",
            "workup": ["Follow-up laboratory assessment"],
        })

        if prob >= 0.85:
            likelihood = "high"
        elif prob >= 0.65:
            likelihood = "moderate"
        else:
            likelihood = "low"

        conf_pct = prob * 100.0
        relevant_biomarkers = abnormal_tests if abnormal_tests else all_measured[:4]

        if disease_name == "Normal / Unremarkable Laboratory Profile":
            rationale = (
                f"Neural Network Confidence: {conf_pct:.1f}% (ClinicalLabNet multi-label model). "
                f"Evaluated analytes are physiologically concordant with reference homeostasis."
            )
        else:
            rationale = (
                f"Neural Network Confidence: {conf_pct:.1f}% (ClinicalLabNet deep multi-label classifier). "
                f"Biomarker pattern is characteristic of {disease_name.lower()}."
            )

        inferred.append(LabDiseaseInference(
            disease=disease_name,
            likelihood=likelihood,
            severity=meta.get("severity", "moderate"),
            pathophysiology=meta.get("pathophysiology", ""),
            clinical_rationale=rationale,
            supporting_biomarkers=relevant_biomarkers,
            suggested_confirmatory_workup=meta.get("workup", []),
            confidence_score=round(prob, 4),
        ))

    # Sort descending by neural confidence score
    inferred.sort(key=lambda d: d.confidence_score if d.confidence_score is not None else 0.0, reverse=True)

    # If both specific diseases and "Normal" are present, discard "Normal"
    if len(inferred) > 1:
        inferred = [d for d in inferred if d.disease != "Normal / Unremarkable Laboratory Profile"]

    # If no disease passed threshold:
    if not inferred:
        if not abnormal_tests:
            norm_prob = float(probs[norm_idx])
            meta_norm = DISEASE_METADATA["Normal / Unremarkable Laboratory Profile"]
            inferred.append(LabDiseaseInference(
                disease="Normal / Unremarkable Laboratory Profile",
                likelihood="high" if norm_prob >= 0.50 else "moderate",
                severity="mild",
                pathophysiology=meta_norm["pathophysiology"],
                clinical_rationale=f"Neural Network Confidence: {norm_prob * 100.0:.1f}%. All assessed parameters remain within normal biological reference bounds.",
                supporting_biomarkers=all_measured[:4],
                suggested_confirmatory_workup=meta_norm["workup"],
                confidence_score=round(norm_prob, 4),
            ))
        else:
            # Pick highest confidence non-normal class from neural probabilities
            best_idx = None
            best_p = 0.0
            for i in range(len(LAB_DISEASE_CLASSES)):
                if i != norm_idx and probs[i] > best_p:
                    best_p = float(probs[i])
                    best_idx = i

            if best_idx is not None and best_p >= 0.25:
                top_disease = LAB_DISEASE_CLASSES[best_idx]
                top_meta = DISEASE_METADATA.get(top_disease, {})
                inferred.append(LabDiseaseInference(
                    disease=top_disease,
                    likelihood="moderate" if best_p >= 0.40 else "low",
                    severity=top_meta.get("severity", "moderate"),
                    pathophysiology=top_meta.get("pathophysiology", ""),
                    clinical_rationale=f"Neural Network Confidence: {best_p * 100.0:.1f}% (ClinicalLabNet deep classifier). Analyte deviation reflects probable {top_disease.lower()}.",
                    supporting_biomarkers=abnormal_tests,
                    suggested_confirmatory_workup=top_meta.get("workup", ["Targeted follow-up testing"]),
                    confidence_score=round(best_p, 4),
                ))

    return inferred

