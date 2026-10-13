"""Lab Agent: parses laboratory reports across formats, normalizes units, and detects clinical patterns."""

import csv
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from medagent.config import settings
from medagent.schemas import CaseInput, LabReport, LabResult, LabPattern

logger = logging.getLogger("medagent.lab")

# Cache for reference ranges
_REFERENCE_RANGES: Optional[Dict[str, Any]] = None


def get_reference_ranges() -> Dict[str, Any]:
    """Load reference intervals and alias dictionary from JSON configuration."""
    global _REFERENCE_RANGES
    if _REFERENCE_RANGES is not None:
        return _REFERENCE_RANGES

    ref_file = settings.REFERENCE_RANGES_FILE
    if not ref_file.exists():
        logger.warning("Reference ranges file not found at %s", ref_file)
        return {}

    with open(ref_file, "r", encoding="utf-8") as f:
        _REFERENCE_RANGES = json.load(f)
    return _REFERENCE_RANGES


def find_test_match(raw_name: str, ref_db: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Match raw test name against canonical aliases."""
    normalized_name = raw_name.lower().strip()
    normalized_name = re.sub(r"[\s_:\-\.]+", " ", normalized_name)

    for test_key, data in ref_db.items():
        if normalized_name == test_key.lower():
            return test_key, data
        if normalized_name == data.get("canonical", "").lower():
            return test_key, data
        for alias in data.get("aliases", []):
            if normalized_name == alias.lower():
                return test_key, data

    # Partial / substring match
    for test_key, data in ref_db.items():
        for alias in data.get("aliases", []):
            if len(alias) >= 3 and alias.lower() in normalized_name:
                return test_key, data

    return None


def normalize_unit_and_value(test_key: str, val: float, unit_str: str) -> Tuple[float, str]:
    """
    Standardize clinical laboratory units.
    Converts SI units (mmol/L, umol/L) to conventional units (mg/dL) where expected,
    and cell counts (/cumm, /uL, lakhs/cumm) to x10^3/uL.
    """
    unit_lower = unit_str.lower().strip()

    # WBC and Platelet counts -> x10^3/uL
    if test_key in ("wbc", "platelets"):
        if "lakh" in unit_lower:
            # 1.5 lakhs/cumm -> 150 x10^3/uL
            return val * 100.0, "x10^3/uL"
        elif "/cumm" in unit_lower or "/ul" in unit_lower or "/mm3" in unit_lower:
            if val > 1000.0:
                return round(val / 1000.0, 2), "x10^3/uL"
            return val, "x10^3/uL"
        elif "x10^3" in unit_lower or "k/ul" in unit_lower or "10^3" in unit_lower:
            return val, "x10^3/uL"
        elif "x10^9" in unit_lower or "10^9/l" in unit_lower:
            # 10^9/L == 10^3/uL
            return val, "x10^3/uL"

    # Glucose: mmol/L -> mg/dL (multiply by 18.0182)
    if "glucose" in test_key:
        if "mmol/l" in unit_lower:
            return round(val * 18.0182, 1), "mg/dL"

    # Creatinine: umol/L -> mg/dL (divide by 88.4)
    if test_key == "creatinine":
        if "umol/l" in unit_lower or "µmol/l" in unit_lower:
            return round(val / 88.4, 2), "mg/dL"

    # BUN / Urea: mmol/L -> mg/dL
    if test_key in ("blood_urea_nitrogen", "urea"):
        if "mmol/l" in unit_lower:
            factor = 2.8 if test_key == "blood_urea_nitrogen" else 6.0
            return round(val * factor, 1), "mg/dL"

    # Bilirubin: umol/L -> mg/dL (divide by 17.1)
    if "bilirubin" in test_key:
        if "umol/l" in unit_lower or "µmol/l" in unit_lower:
            return round(val / 17.1, 2), "mg/dL"

    # Calcium: mmol/L -> mg/dL (multiply by 4.008)
    if test_key == "calcium":
        if "mmol/l" in unit_lower:
            return round(val * 4.008, 2), "mg/dL"

    # Cholesterol / Lipids: mmol/L -> mg/dL
    if "cholesterol" in test_key or test_key == "triglycerides":
        if "mmol/l" in unit_lower:
            factor = 88.57 if test_key == "triglycerides" else 38.67
            return round(val * factor, 1), "mg/dL"

    # Troponin: ng/L or pg/mL -> ng/mL (divide by 1000)
    if "troponin" in test_key:
        if "ng/l" in unit_lower or "pg/ml" in unit_lower:
            return round(val / 1000.0, 4), "ng/mL"

    # HbA1c: mmol/mol -> %
    if test_key == "hba1c":
        if "mmol/mol" in unit_lower:
            pct = (0.09148 * val) + 2.152
            return round(pct, 1), "%"

    return val, unit_str


def parse_reference_range(range_str: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract low and high bounds from printed reference interval string."""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:-|–|to)\s*([0-9]+(?:\.[0-9]+)?)", range_str)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            pass
    return None, None


def determine_flag(
    value: float,
    ref_low: Optional[float],
    ref_high: Optional[float],
    crit_low: Optional[float],
    crit_high: Optional[float],
) -> str:
    """Classify result value against normal intervals and critical panic limits."""
    if crit_low is not None and value < crit_low:
        return "critical low"
    if crit_high is not None and value > crit_high:
        return "critical high"

    if ref_low is not None and value < ref_low:
        return "low"
    if ref_high is not None and value > ref_high:
        return "high"
    if ref_low is not None or ref_high is not None:
        return "normal"

    return "unknown"


def parse_lab_line(
    line: str,
    ref_db: Dict[str, Any],
    patient_sex: Optional[str] = None,
) -> Tuple[Optional[LabResult], Optional[str]]:
    """
    Parse a single line into a LabResult.
    Expected patterns: 'name: value unit [range]', 'name value unit range', etc.
    """
    cleaned = line.strip()
    if not cleaned or cleaned.startswith("#"):
        return None, None

    # Regex capturing: (Test Name) [:=]? (Value) (Optional Unit) (Optional [Range])
    pattern = re.compile(
        r"^(?P<name>[a-zA-Z0-9_\-\s\(\)\/\+\%]+?)"
        r"[:\t\s=]+"
        r"(?P<val>[0-9]+(?:\.[0-9]+)?)"
        r"(?:\s*(?P<unit>[a-zA-Z\^\d\%\/µ]+))?"
        r"(?:\s*[\(\[\{]?(?P<range>[0-9\.\s\-–to]+)[\)\]\}]?)?$",
        re.IGNORECASE,
    )

    m = pattern.match(cleaned)
    if not m:
        # Check CSV or delimiter like comma or semicolon
        parts = re.split(r"[,;]\s*", cleaned)
        if len(parts) >= 2:
            raw_name = parts[0].strip()
            try:
                raw_val = float(re.findall(r"[0-9]+(?:\.[0-9]+)?", parts[1])[0])
                raw_unit = parts[2].strip() if len(parts) > 2 else ""
                printed_range = parts[3].strip() if len(parts) > 3 else ""
            except (IndexError, ValueError):
                return None, cleaned
        else:
            return None, cleaned
    else:
        raw_name = m.group("name").strip()
        try:
            raw_val = float(m.group("val"))
        except ValueError:
            return None, cleaned
        raw_unit = m.group("unit") or ""
        printed_range = m.group("range") or ""

    match_info = find_test_match(raw_name, ref_db)
    if not match_info:
        # Unknown test -> unparsed line
        return None, cleaned

    test_key, ref_info = match_info
    norm_val, norm_unit = normalize_unit_and_value(test_key, raw_val, raw_unit or ref_info.get("unit", ""))

    # Determine reference range: printed range overrides table
    p_low, p_high = parse_reference_range(printed_range)
    if p_low is not None and p_high is not None:
        ref_low, ref_high = p_low, p_high
    else:
        # Check sex-specific range
        sex_key = patient_sex if patient_sex in ("male", "female") else "default"
        interval = ref_info.get(sex_key) or ref_info.get("default", {})
        ref_low = interval.get("low")
        ref_high = interval.get("high")

    crit_low = ref_info.get("critical_low")
    crit_high = ref_info.get("critical_high")

    flag = determine_flag(norm_val, ref_low, ref_high, crit_low, crit_high)

    result = LabResult(
        test=ref_info.get("canonical", test_key),
        raw_name=raw_name,
        value=norm_val,
        unit=norm_unit,
        ref_low=ref_low,
        ref_high=ref_high,
        flag=flag,
    )
    return result, None


def extract_text_from_file(file_path: str) -> Tuple[str, List[str]]:
    """Extract text lines from CSV/TSV, PDF, or text file."""
    path = Path(file_path)
    if not path.exists():
        return "", [f"File not found: {file_path}"]

    ext = path.suffix.lower()
    text = ""
    notes: List[str] = []

    if ext in (".txt", ".csv", ".tsv"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            notes.append(f"Error reading {path.name}: {str(e)}")

    elif ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pages_text = [page.extract_text() or "" for page in pdf.pages]
                text = "\n".join(pages_text)
        except ImportError:
            notes.append("pdfplumber not available; cannot parse PDF lab report.")
        except Exception as e:
            notes.append(f"PDF extraction failed for {path.name}: {str(e)}")

    elif ext in (".png", ".jpg", ".jpeg"):
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(path)
            text = pytesseract.image_to_string(img)
        except ImportError:
            notes.append(f"Tesseract OCR not installed in environment; image {path.name} cannot be parsed.")
        except Exception as e:
            notes.append(f"OCR extraction failed for {path.name}: {str(e)}")

    return text, notes


def evaluate_patterns(results: List[LabResult]) -> List[LabPattern]:
    """
    Synthesize multi-test clinical patterns with assigned urgency levels:
    routine | soon | urgent.
    """
    res_map: Dict[str, LabResult] = {r.test: r for r in results}
    patterns: List[LabPattern] = []

    # Helper lookup
    def get_val(name: str) -> Optional[float]:
        return res_map[name].value if name in res_map else None

    def get_flag(name: str) -> Optional[str]:
        return res_map[name].flag if name in res_map else None

    hb = get_val("Hemoglobin")
    mcv = get_val("MCV")
    wbc = get_val("WBC Count")
    neut = get_val("Neutrophils (%)")
    plt = get_val("Platelet Count")
    crp = get_val("C-Reactive Protein")
    esr = get_val("ESR")
    pct = get_val("Procalcitonin")
    gluc_fast = get_val("Fasting Glucose")
    gluc_rand = get_val("Random Glucose")
    a1c = get_val("HbA1c")
    k = get_val("Potassium")
    na = get_val("Sodium")
    creat = get_val("Serum Creatinine")
    bun = get_val("BUN")
    egfr = get_val("eGFR")
    bicarb = get_val("Bicarbonate / CO2")
    ca = get_val("Serum Calcium")
    lactate = get_val("Lactate")
    alt = get_val("ALT (SGPT)")
    ast = get_val("AST (SGOT)")
    alp = get_val("Alkaline Phosphatase")
    ggt = get_val("GGT")
    tbili = get_val("Total Bilirubin")
    dbili = get_val("Direct Bilirubin")
    alb = get_val("Serum Albumin")
    tsh = get_val("TSH")
    ft4 = get_val("Free T4")
    trop_i = get_val("Troponin I")
    trop_t = get_val("Troponin T")
    inr = get_val("INR")
    ferritin = get_val("Ferritin")
    b12 = get_val("Vitamin B12")
    folate = get_val("Serum Folate")
    vit_d = get_val("Vitamin D (25-OH)")
    tg = get_val("Triglycerides")
    ldl = get_val("LDL Cholesterol")

    # 1. Anemias
    if hb is not None and get_flag("Hemoglobin") in ("low", "critical low"):
        if mcv is not None and mcv < 80.0:
            patterns.append(LabPattern(
                name="Microcytic Anaemia",
                description="Low hemoglobin with low MCV (<80 fL), typical of iron deficiency or thalassemia trait.",
                supporting_tests=["Hemoglobin", "MCV"] + (["Ferritin"] if ferritin else []),
                urgency="soon",
            ))
        elif mcv is not None and mcv > 100.0:
            patterns.append(LabPattern(
                name="Macrocytic Anaemia",
                description="Low hemoglobin with elevated MCV (>100 fL), suspicious for vitamin B12 or folate deficiency.",
                supporting_tests=["Hemoglobin", "MCV"] + (["Vitamin B12"] if b12 else []),
                urgency="soon",
            ))
        else:
            patterns.append(LabPattern(
                name="Normocytic Anaemia",
                description="Low hemoglobin with normal red cell indices, common in anemia of chronic disease, bleeding, or renal failure.",
                supporting_tests=["Hemoglobin"] + (["MCV"] if mcv else []),
                urgency="routine",
            ))

    # 2. White Blood Cell Patterns
    if wbc is not None:
        if wbc > 11.0:
            supp = ["WBC Count"]
            if neut is not None and neut > 75.0:
                supp.append("Neutrophils (%)")
                patterns.append(LabPattern(
                    name="Leukocytosis with Neutrophilia",
                    description="Elevated total leukocyte count with neutrophil predominance, suggesting active bacterial infection or severe acute inflammation.",
                    supporting_tests=supp,
                    urgency="soon",
                ))
            else:
                patterns.append(LabPattern(
                    name="Leukocytosis",
                    description="Elevated white blood cell count indicating active inflammatory, infectious, or reactive marrow response.",
                    supporting_tests=supp,
                    urgency="routine",
                ))
        elif wbc < 4.0:
            patterns.append(LabPattern(
                name="Leukopenia",
                description="Depressed white cell count indicating impaired immune defense or viral suppression.",
                supporting_tests=["WBC Count"],
                urgency="urgent" if wbc < 2.0 else "soon",
            ))

    # 3. Platelet Patterns
    if plt is not None:
        if plt < 150.0:
            patterns.append(LabPattern(
                name="Thrombocytopenia",
                description="Low platelet count increasing risk of mucosal or systemic bleeding.",
                supporting_tests=["Platelet Count"],
                urgency="urgent" if plt < 50.0 else "soon",
            ))
        elif plt > 450.0:
            patterns.append(LabPattern(
                name="Thrombocytosis",
                description="Elevated platelet count, frequently secondary to iron deficiency, inflammation, or myeloproliferative disorder.",
                supporting_tests=["Platelet Count"],
                urgency="routine",
            ))

    # 4. Inflammatory Markers
    inflam_tests = []
    if crp is not None and crp > 10.0:
        inflam_tests.append("C-Reactive Protein")
    if esr is not None and esr > 20.0:
        inflam_tests.append("ESR")
    if pct is not None and pct > 0.5:
        inflam_tests.append("Procalcitonin")

    if inflam_tests:
        patterns.append(LabPattern(
            name="Raised Inflammatory Markers",
            description="Systemic inflammatory activation evidenced by elevated acute phase reactants.",
            supporting_tests=inflam_tests,
            urgency="soon" if (pct and pct > 2.0) else "routine",
        ))

    # 5. Glycemic Patterns & DKA
    glucose = gluc_fast or gluc_rand
    if glucose is not None:
        if glucose < 70.0:
            patterns.append(LabPattern(
                name="Hypoglycaemia",
                description="Blood glucose below 70 mg/dL requiring immediate glucose stabilization.",
                supporting_tests=[res_map[list(res_map.keys())[0]].test],
                urgency="urgent" if glucose < 54.0 else "soon",
            ))
        elif glucose >= 250.0 and bicarb is not None and bicarb < 18.0:
            patterns.append(LabPattern(
                name="Hyperglycaemia with Low Bicarbonate (DKA Pattern)",
                description="Severe hyperglycemia with high anion gap metabolic acidosis consistent with diabetic ketoacidosis or hyperosmolar state.",
                supporting_tests=["Glucose", "Bicarbonate / CO2"],
                urgency="urgent",
            ))
        elif (gluc_fast and gluc_fast >= 126.0) or (gluc_rand and gluc_rand >= 200.0) or (a1c and a1c >= 6.5):
            patterns.append(LabPattern(
                name="Diabetic Range Glycemia",
                description="Laboratory indices meeting diagnostic threshold for diabetes mellitus.",
                supporting_tests=[t for t in ["Fasting Glucose", "Random Glucose", "HbA1c"] if t in res_map],
                urgency="routine",
            ))
        elif (gluc_fast and 100.0 <= gluc_fast <= 125.0) or (a1c and 5.7 <= a1c <= 6.4):
            patterns.append(LabPattern(
                name="Pre-diabetic Range",
                description="Impaired fasting glucose or pre-diabetic HbA1c level.",
                supporting_tests=[t for t in ["Fasting Glucose", "HbA1c"] if t in res_map],
                urgency="routine",
            ))

    # 6. Renal Impairment
    renal_tests = []
    if creat is not None and get_flag("Serum Creatinine") in ("high", "critical high"):
        renal_tests.append("Serum Creatinine")
    if bun is not None and get_flag("BUN") in ("high", "critical high"):
        renal_tests.append("BUN")
    if egfr is not None and egfr < 60.0:
        renal_tests.append("eGFR")

    if renal_tests:
        patterns.append(LabPattern(
            name="Renal Impairment",
            description="Elevated nitrogenous wastes and/or reduced filtration rate indicating acute or chronic renal dysfunction.",
            supporting_tests=renal_tests,
            urgency="urgent" if (creat and creat > 3.0) or (egfr and egfr < 15.0) else "soon",
        ))

    # 7. Electrolytes: Potassium & Sodium
    if k is not None:
        if k > 5.0:
            patterns.append(LabPattern(
                name="Hyperkalaemia",
                description="Elevated serum potassium conferring risk of fatal cardiac arrhythmias.",
                supporting_tests=["Potassium"],
                urgency="urgent" if k >= 6.0 else "soon",
            ))
        elif k < 3.5:
            patterns.append(LabPattern(
                name="Hypokalaemia",
                description="Low serum potassium posing risk for muscle weakness and cardiac dysrhythmias.",
                supporting_tests=["Potassium"],
                urgency="urgent" if k < 2.8 else "soon",
            ))

    if na is not None:
        if na < 135.0:
            patterns.append(LabPattern(
                name="Hyponatraemia",
                description="Subnormal serum sodium requiring cautious osmolality evaluation and rate-controlled correction.",
                supporting_tests=["Sodium"],
                urgency="urgent" if na < 120.0 else "soon",
            ))
        elif na > 145.0:
            patterns.append(LabPattern(
                name="Hypernatraemia",
                description="Elevated serum sodium indicating free water deficit or unreplaced fluid losses.",
                supporting_tests=["Sodium"],
                urgency="urgent" if na > 160.0 else "soon",
            ))

    # 8. Calcium
    if ca is not None and ca > 10.2:
        patterns.append(LabPattern(
            name="Hypercalcaemia",
            description="Elevated serum calcium suspicious for hyperparathyroidism or malignancy.",
            supporting_tests=["Serum Calcium"],
            urgency="urgent" if ca > 13.0 else "soon",
        ))

    # 9. Acid-Base & Lactate
    if bicarb is not None and bicarb < 22.0:
        patterns.append(LabPattern(
            name="Metabolic Acidosis",
            description="Decreased serum bicarbonate / total CO2 indicating systemic acid accumulation or bicarbonate loss.",
            supporting_tests=["Bicarbonate / CO2"],
            urgency="urgent" if bicarb < 15.0 else "soon",
        ))

    if lactate is not None and lactate > 2.0:
        patterns.append(LabPattern(
            name="Raised Lactate",
            description="Serum lactate elevation signaling tissue hypoperfusion, severe sepsis, or anaerobic metabolism.",
            supporting_tests=["Lactate"],
            urgency="urgent" if lactate >= 4.0 else "soon",
        ))

    # 10. Hepatic & Biliary
    if (alt is not None and alt > 150.0) or (ast is not None and ast > 150.0):
        patterns.append(LabPattern(
            name="Hepatocellular Pattern",
            description="Marked transaminase elevation reflecting acute hepatocellular injury or necrosis.",
            supporting_tests=[t for t in ["ALT (SGPT)", "AST (SGOT)"] if t in res_map],
            urgency="soon",
        ))

    if (alp is not None and alp > 150.0) or (ggt is not None and ggt > 50.0) or (dbili is not None and dbili > 0.3):
        patterns.append(LabPattern(
            name="Cholestatic Pattern",
            description="Elevation of alkaline phosphatase, GGT, or direct bilirubin indicating biliary stasis or obstruction.",
            supporting_tests=[t for t in ["Alkaline Phosphatase", "GGT", "Direct Bilirubin"] if t in res_map],
            urgency="soon",
        ))

    if tbili is not None and tbili > 1.2:
        patterns.append(LabPattern(
            name="Hyperbilirubinaemia",
            description="Elevated total bilirubin with visible or subclinical jaundice.",
            supporting_tests=["Total Bilirubin"],
            urgency="soon" if tbili > 3.0 else "routine",
        ))

    if alb is not None and alb < 3.5:
        patterns.append(LabPattern(
            name="Hypoalbuminaemia",
            description="Low serum albumin reflecting systemic inflammation, malnutrition, nephrosis, or reduced hepatic synthesis.",
            supporting_tests=["Serum Albumin"],
            urgency="soon" if alb < 2.5 else "routine",
        ))

    # 11. Thyroid
    if tsh is not None:
        if tsh > 4.5 and ft4 is not None and ft4 < 0.8:
            patterns.append(LabPattern(
                name="Primary Hypothyroidism Pattern",
                description="Elevated TSH with low free thyroxine confirming primary thyroid insufficiency.",
                supporting_tests=["TSH", "Free T4"],
                urgency="routine",
            ))
        elif tsh < 0.4 and ft4 is not None and ft4 > 1.8:
            patterns.append(LabPattern(
                name="Primary Hyperthyroidism Pattern",
                description="Suppressed TSH with elevated free thyroxine indicating hyperthyroidism / thyrotoxicosis.",
                supporting_tests=["TSH", "Free T4"],
                urgency="soon",
            ))

    # 12. Cardiac & Coagulation
    troponin_val = trop_i or trop_t
    if troponin_val is not None:
        trop_flag = get_flag("Troponin I") or get_flag("Troponin T")
        if trop_flag in ("high", "critical high"):
            patterns.append(LabPattern(
                name="Raised Troponin",
                description="Cardiac troponin elevation above reference limits signifying acute or evolving myocardial necrosis.",
                supporting_tests=[t for t in ["Troponin I", "Troponin T"] if t in res_map],
                urgency="urgent",
            ))

    if inr is not None and inr > 1.2:
        patterns.append(LabPattern(
            name="Prolonged INR",
            description="Elevated International Normalized Ratio indicating coagulopathy, anticoagulation, or synthetic liver impairment.",
            supporting_tests=["INR"],
            urgency="urgent" if inr >= 4.5 else "soon",
        ))

    # 13. Dyslipidemia
    if (tg is not None and tg > 150.0) or (ldl is not None and ldl > 100.0):
        patterns.append(LabPattern(
            name="Dyslipidaemia",
            description="Atherogenic lipid profile with elevated triglycerides or LDL.",
            supporting_tests=[t for t in ["Triglycerides", "LDL Cholesterol"] if t in res_map],
            urgency="routine",
        ))

    # 14. Vitamin Deficiencies
    vit_tests = []
    if b12 is not None and b12 < 200.0:
        vit_tests.append("Vitamin B12")
    if folate is not None and folate < 3.0:
        vit_tests.append("Serum Folate")
    if vit_d is not None and vit_d < 30.0:
        vit_tests.append("Vitamin D (25-OH)")

    if vit_tests:
        patterns.append(LabPattern(
            name="Vitamin Deficiencies",
            description=f"Nutritional deficiency identified in: {', '.join(vit_tests)}.",
            supporting_tests=vit_tests,
            urgency="routine",
        ))

    return patterns


def run_lab_agent(case_input: CaseInput) -> LabReport:
    """Execute complete deterministic laboratory evaluation pipeline."""
    has_text = bool(case_input.lab_text and case_input.lab_text.strip())
    has_files = bool(case_input.lab_paths and len(case_input.lab_paths) > 0)

    if not has_text and not has_files:
        return LabReport(
            status="skipped",
            results=[],
            patterns=[],
            unparsed_lines=[],
            message="No laboratory data provided for this case.",
        )

    ref_db = get_reference_ranges()
    combined_lines: List[str] = []
    notes: List[str] = []

    if has_text:
        combined_lines.extend(case_input.lab_text.splitlines())

    if has_files:
        for fpath in case_input.lab_paths:
            file_text, file_notes = extract_text_from_file(fpath)
            notes.extend(file_notes)
            if file_text:
                combined_lines.extend(file_text.splitlines())

    parsed_results: List[LabResult] = []
    unparsed: List[str] = []
    seen_tests = set()

    for line in combined_lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        result, unparsed_line = parse_lab_line(line_clean, ref_db, case_input.patient_sex)
        if result:
            # Deduplicate or replace
            if result.test not in seen_tests:
                parsed_results.append(result)
                seen_tests.add(result.test)
        elif unparsed_line:
            unparsed.append(unparsed_line)

    # Sort results: abnormal/critical first, then normal
    flag_priority = {
        "critical high": 0,
        "critical low": 1,
        "high": 2,
        "low": 3,
        "normal": 4,
        "unknown": 5,
    }
    parsed_results.sort(key=lambda r: flag_priority.get(r.flag, 99))

    patterns = evaluate_patterns(parsed_results)

    status_val = "ok" if parsed_results else ("error" if not unparsed else "ok")
    msg = f"Processed {len(parsed_results)} laboratory tests, synthesized {len(patterns)} clinical patterns."
    if notes:
        msg += " " + " ".join(notes)

    return LabReport(
        status=status_val,
        results=parsed_results,
        patterns=patterns,
        unparsed_lines=unparsed,
        message=msg,
    )
