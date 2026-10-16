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
from medagent.schemas import CaseInput, LabReport, LabResult, LabPattern, LabDiseaseInference

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
    Parse a single line or table row into a LabResult.
    Handles 'name: value unit [range]', 'name value unit range flag', CSV/TSV cells, and tables.
    """
    cleaned = line.strip()
    if not cleaned or cleaned.startswith("#"):
        return None, None

    raw_name = ""
    raw_val = None
    raw_unit = ""
    printed_range = ""

    # Strategy 1: Check delimiter-separated cells (from PDF tables or CSV/TSV)
    parts = [p.strip() for p in re.split(r"\s*[:;,|\t]\s*", cleaned) if p.strip()]
    if len(parts) >= 2:
        name_cand = parts[0]
        for idx in range(1, len(parts)):
            num_match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", parts[idx])
            if num_match:
                try:
                    raw_val = float(num_match.group(0))
                    raw_name = name_cand
                    raw_unit = parts[idx + 1] if len(parts) > idx + 1 else ""
                    printed_range = parts[idx + 2] if len(parts) > idx + 2 else ""
                    break
                except ValueError:
                    pass

    # Strategy 2: If not resolved, use structured regex
    if raw_val is None or not raw_name:
        pattern = re.compile(
            r"^(?P<name>[a-zA-Z0-9_\-\s\(\)\/\+\%]+?)"
            r"[:\t\s=]+"
            r"(?P<val>[0-9]+(?:\.[0-9]+)?)"
            r"(?:\s*(?P<unit>[a-zA-Z\^\d\%\/µ]+))?"
            r"(?:\s*[\(\[\{]?(?P<range>[0-9\.\s\-–to]+)[\)\]\}]?)?"
            r"(?:\s*(?P<flag_note>[a-zA-Z\*]+))?$",
            re.IGNORECASE,
        )
        m = pattern.match(cleaned)
        if m:
            raw_name = m.group("name").strip()
            try:
                raw_val = float(m.group("val"))
            except ValueError:
                return None, cleaned
            raw_unit = m.group("unit") or ""
            printed_range = m.group("range") or ""
        else:
            return None, cleaned

    # Check for range embedded in line if not yet found
    if not printed_range:
        m_rng = re.search(r"([0-9]+(?:\.[0-9]+)?\s*[-–to]+\s*[0-9]+(?:\.[0-9]+)?)", cleaned)
        if m_rng and m_rng.group(1) != str(raw_val):
            printed_range = m_rng.group(1)

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
    """
    Extract structured text lines from CSV/TSV, PDF, or text file.
    Supports table extraction from multi-column PDF lab reports via pdfplumber.
    """
    path = Path(file_path)
    if not path.exists():
        return "", [f"File not found: {file_path}"]

    ext = path.suffix.lower()
    extracted_lines: List[str] = []
    notes: List[str] = []

    if ext in (".txt", ".csv", ".tsv"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                if ext in (".csv", ".tsv"):
                    delimiter = "\t" if ext == ".tsv" else ","
                    reader = csv.reader(f, delimiter=delimiter)
                    for row in reader:
                        clean_cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                        if len(clean_cells) >= 2:
                            extracted_lines.append(" : ".join(clean_cells))
                else:
                    extracted_lines.extend(f.read().splitlines())
        except Exception as e:
            notes.append(f"Error reading {path.name}: {str(e)}")

    elif ext == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    # 1. Extract structured tables from PDF lab report
                    tables = page.extract_tables()
                    for table in tables or []:
                        for row in table or []:
                            if not row:
                                continue
                            clean_cells = [str(c).strip().replace("\n", " ") for c in row if c is not None and str(c).strip()]
                            if len(clean_cells) >= 2:
                                extracted_lines.append(" : ".join(clean_cells))
                    # 2. Extract raw textual lines as well
                    page_text = page.extract_text()
                    if page_text:
                        extracted_lines.extend(page_text.splitlines())
        except ImportError:
            notes.append("pdfplumber not available; cannot parse PDF lab report.")
        except Exception as e:
            notes.append(f"PDF extraction failed for {path.name}: {str(e)}")

    elif ext in (".png", ".jpg", ".jpeg"):
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(path)
            ocr_text = pytesseract.image_to_string(img)
            extracted_lines.extend(ocr_text.splitlines())
        except ImportError:
            notes.append(f"Tesseract OCR not installed in environment; image {path.name} cannot be parsed.")
        except Exception as e:
            notes.append(f"OCR extraction failed for {path.name}: {str(e)}")

    return "\n".join(extracted_lines), notes


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


def infer_laboratory_diseases(results: List[LabResult], patient_sex: Optional[str] = None) -> List[LabDiseaseInference]:
    """
    Infer concrete clinical diseases and syndromes from standardized laboratory findings
    using evidence-based diagnostic criteria (KDIGO, ADA, WHO, AHA).
    """
    if not results:
        return []

    res_map: Dict[str, LabResult] = {r.test: r for r in results}

    def get_val(name: str) -> Optional[float]:
        return res_map[name].value if name in res_map else None

    def get_flag(name: str) -> Optional[str]:
        return res_map[name].flag if name in res_map else None

    def fmt_biomarker(name: str) -> str:
        r = res_map[name]
        return f"{r.test}: {r.value:g} {r.unit} ({r.flag.upper()})"

    inferred: List[LabDiseaseInference] = []

    # Values extraction
    hb = get_val("Hemoglobin")
    mcv = get_val("MCV")
    wbc = get_val("WBC Count")
    neut = get_val("Neutrophils (%)")
    plt = get_val("Platelet Count")
    crp = get_val("C-Reactive Protein")
    esr = get_val("ESR")
    pct = get_val("Procalcitonin")
    fbs = get_val("Fasting Glucose")
    rbs = get_val("Random Glucose")
    a1c = get_val("HbA1c")
    k = get_val("Potassium")
    na = get_val("Sodium")
    cl = get_val("Chloride")
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
    tsh = get_val("TSH")
    ft4 = get_val("Free T4")
    trop_i = get_val("Troponin I")
    trop_t = get_val("Troponin T")
    ferritin = get_val("Ferritin")
    b12 = get_val("Vitamin B12")
    folate = get_val("Serum Folate")
    tg = get_val("Triglycerides")
    ldl = get_val("LDL Cholesterol")
    lipase = get_val("Lipase")
    amylase = get_val("Amylase")

    # 1. Critical Hyperkalemia
    if k is not None and k >= 5.5:
        is_crit = k >= 6.0
        inferred.append(LabDiseaseInference(
            disease="Severe Hyperkalemia",
            likelihood="high",
            severity="critical" if is_crit else "severe",
            pathophysiology="Dangerous elevation of extracellular potassium lowering myocardial resting membrane threshold, impairing cardiac repolarization, and predisposing to fatal ventricular arrhythmias or asystole.",
            clinical_rationale=f"Serum potassium elevated to {k:.2f} mEq/L, crossing the safety threshold (3.5–5.0 mEq/L) and posing acute cardiotoxicity risk.",
            supporting_biomarkers=[fmt_biomarker("Potassium")],
            suggested_confirmatory_workup=[
                "Immediate 12-lead ECG for peaked T waves, PR prolongation, and QRS widening",
                "Continuous cardiac telemetry monitoring",
                "Membrane stabilization with IV Calcium Gluconate (10%) if ECG changes present",
                "Intracellular shifting therapy (IV regular insulin + 50% Dextrose, nebulized albuterol)",
            ],
        ))

    # 2. Hypokalemia
    if k is not None and k < 3.5:
        is_crit = k < 2.8
        inferred.append(LabDiseaseInference(
            disease="Hypokalemia",
            likelihood="high",
            severity="critical" if is_crit else ("severe" if k < 3.0 else "moderate"),
            pathophysiology="Depletion of extracellular potassium hyperpolarizing excitable cell membranes, impairing neuromuscular transmission, and causing cardiac repolarization delay with prominent U waves.",
            clinical_rationale=f"Serum potassium decreased to {k:.2f} mEq/L below physiological minimum (3.5 mEq/L).",
            supporting_biomarkers=[fmt_biomarker("Potassium")],
            suggested_confirmatory_workup=[
                "12-lead ECG for ST depression, T wave flattening, and U waves",
                "Serum magnesium level (hypomagnesemia refractory potassium wasting)",
                "Oral or intravenous potassium chloride replacement protocol",
            ],
        ))

    # 3. Acute Kidney Injury (AKI) / Advanced Renal Failure
    if creat is not None and (creat >= 1.4 or get_flag("Serum Creatinine") in ("high", "critical high") or (egfr is not None and egfr < 60.0)):
        is_crit = creat >= 4.0 or (egfr is not None and egfr < 15.0)
        is_sev = creat >= 2.0 or (egfr is not None and egfr < 30.0)
        supp = [fmt_biomarker("Serum Creatinine")]
        if egfr is not None:
            supp.append(fmt_biomarker("eGFR"))
        if bun is not None:
            supp.append(fmt_biomarker("BUN"))

        inferred.append(LabDiseaseInference(
            disease="Acute Kidney Injury / Renal Impairment",
            likelihood="high",
            severity="critical" if is_crit else ("severe" if is_sev else "moderate"),
            pathophysiology="Abrupt decline in glomerular filtration rate leading to retention of urea, creatinine, and nitrogenous metabolic waste products with disruption of fluid, electrolyte, and acid-base homeostasis.",
            clinical_rationale=f"Serum creatinine elevated to {creat:.2f} mg/dL" + (f" with severely reduced eGFR of {egfr:.1f} mL/min/1.73m2" if egfr else "") + ", demonstrating renal parenchymal filtration failure.",
            supporting_biomarkers=supp,
            suggested_confirmatory_workup=[
                "Renal duplex ultrasound to exclude obstructive hydronephrosis",
                "Urinalysis with urine microscopy for casts, protein, and dysmorphic RBCs",
                "Fractional excretion of sodium (FeNa) or urea (FEUrea)",
                "Strict intake/output tracking and discontinuation of nephrotoxic medications",
            ],
        ))

    # 4. Prerenal Azotemia / Dehydration
    if bun is not None and creat is not None and creat > 0:
        ratio = bun / creat
        if ratio >= 20.0 and bun > 20.0:
            inferred.append(LabDiseaseInference(
                disease="Prerenal Azotemia / Dehydration",
                likelihood="high" if ratio >= 25.0 else "moderate",
                severity="moderate",
                pathophysiology="Decreased effective renal perfusion pressure increasing proximal tubular reabsorption of urea relative to creatinine without intrinsic tubular structural necrosis.",
                clinical_rationale=f"Elevated BUN-to-Creatinine ratio of {ratio:.1f}:1 (BUN {bun:.1f} mg/dL, Creatinine {creat:.2f} mg/dL), characteristic of volume contraction or decreased cardiac output.",
                supporting_biomarkers=[fmt_biomarker("BUN"), fmt_biomarker("Serum Creatinine"), f"BUN/Creatinine Ratio: {ratio:.1f}:1 (normal < 20:1)"],
                suggested_confirmatory_workup=[
                    "Clinical volume status assessment (mucous membranes, skin turgor, orthostatic vitals)",
                    "Diagnostic fluid challenge with isotonic crystalloids",
                    "Urine specific gravity and fractional excretion of sodium",
                ],
            ))

    # 5. Diabetic Ketoacidosis (DKA)
    gluc = fbs or rbs
    anion_gap = (na - (cl + bicarb)) if (na is not None and cl is not None and bicarb is not None) else None
    if gluc is not None and gluc >= 250.0 and ((bicarb is not None and bicarb <= 18.0) or (anion_gap is not None and anion_gap >= 14.0)):
        supp = [fmt_biomarker("Fasting Glucose" if fbs else "Random Glucose")]
        if bicarb is not None:
            supp.append(fmt_biomarker("Bicarbonate / CO2"))
        if anion_gap is not None:
            supp.append(f"Calculated Anion Gap: {anion_gap:.1f} mEq/L (HIGH)")

        inferred.append(LabDiseaseInference(
            disease="Diabetic Ketoacidosis (DKA)",
            likelihood="high",
            severity="critical",
            pathophysiology="Profound insulin deficiency with excessive counter-regulatory hormone surge driving unchecked lipolysis, hepatic free fatty acid oxidation into ketoacids, high anion gap metabolic acidosis, and severe osmotic dehydration.",
            clinical_rationale=f"Marked hyperglycemia ({gluc:.1f} mg/dL) coupled with metabolic acidosis" + (f" (Bicarbonate {bicarb:.1f} mEq/L)" if bicarb else "") + (f" and high anion gap ({anion_gap:.1f} mEq/L)" if anion_gap else "") + ".",
            supporting_biomarkers=supp,
            suggested_confirmatory_workup=[
                "Serum beta-hydroxybutyrate and urine ketones",
                "Arterial/venous blood gas for pH and lactate",
                "Aggressive isotonic IV crystalloid volume resuscitation",
                "Continuous intravenous regular insulin infusion",
                "Hourly capillary blood glucose and serial potassium tracking",
            ],
        ))

    # 6. High Anion Gap Metabolic Acidosis (HAGMA)
    if (bicarb is not None and bicarb < 20.0) or (anion_gap is not None and anion_gap > 12.0):
        if not any(d.disease == "Diabetic Ketoacidosis (DKA)" for d in inferred):
            supp = []
            if bicarb is not None:
                supp.append(fmt_biomarker("Bicarbonate / CO2"))
            if anion_gap is not None:
                supp.append(f"Calculated Anion Gap: {anion_gap:.1f} mEq/L (HIGH)")
            if lactate is not None and lactate > 2.0:
                supp.append(fmt_biomarker("Lactate"))

            inferred.append(LabDiseaseInference(
                disease="Metabolic Acidosis (High Anion Gap)" if (anion_gap and anion_gap > 12.0) else "Metabolic Acidosis",
                likelihood="high",
                severity="severe" if (bicarb and bicarb < 15.0) else "moderate",
                pathophysiology="Pathological accumulation of unmeasured fixed organic acids (lactic acid, ketoacids, uremic toxins) consuming serum bicarbonate buffers.",
                clinical_rationale="Low serum bicarbonate" + (f" ({bicarb:.1f} mEq/L)" if bicarb else "") + (f" with elevated serum anion gap of {anion_gap:.1f} mEq/L" if anion_gap else "") + ".",
                supporting_biomarkers=supp,
                suggested_confirmatory_workup=[
                    "Arterial blood gas for definitive pH and pCO2 respiratory compensation",
                    "Serum lactate and serum beta-hydroxybutyrate",
                    "Screening for toxic ingestions if etiology undetermined",
                ],
            ))

    # 7. Type 2 Diabetes Mellitus / Uncontrolled Hyperglycemia
    if not any(d.disease == "Diabetic Ketoacidosis (DKA)" for d in inferred):
        if (a1c is not None and a1c >= 6.5) or (fbs is not None and fbs >= 126.0) or (rbs is not None and rbs >= 200.0):
            supp = []
            if a1c is not None:
                supp.append(fmt_biomarker("HbA1c"))
            if fbs is not None:
                supp.append(fmt_biomarker("Fasting Glucose"))
            if rbs is not None:
                supp.append(fmt_biomarker("Random Glucose"))

            inferred.append(LabDiseaseInference(
                disease="Type 2 Diabetes Mellitus / Hyperglycemia",
                likelihood="high",
                severity="moderate" if (a1c and a1c >= 9.0 or (gluc and gluc >= 250)) else "mild",
                pathophysiology="Chronic metabolic disorder of peripheral insulin resistance and progressive pancreatic beta-cell secretory failure resulting in sustained hyperglycemia.",
                clinical_rationale="Diabetic range glycemia established by " + (f"HbA1c {a1c:.1f}% (threshold >= 6.5%)" if a1c else f"blood glucose {gluc:.1f} mg/dL") + ".",
                supporting_biomarkers=supp,
                suggested_confirmatory_workup=[
                    "Fasting lipid panel and spot urine albumin-to-creatinine ratio",
                    "Dilated diabetic retinal examination",
                    "Comprehensive diabetic foot and peripheral vascular evaluation",
                ],
            ))

    # 8. Severe Hypoglycemia
    if gluc is not None and gluc < 70.0:
        is_crit = gluc < 54.0
        inferred.append(LabDiseaseInference(
            disease="Severe Hypoglycemia",
            likelihood="high",
            severity="critical" if is_crit else "severe",
            pathophysiology="Critically low plasma glucose concentration depriving cerebral neurons of essential oxidative fuel, triggering counter-regulatory catecholamine release and neuroglycopenic dysfunction.",
            clinical_rationale=f"Plasma glucose critically depressed to {gluc:.1f} mg/dL (alert limit < 70 mg/dL).",
            supporting_biomarkers=[fmt_biomarker("Fasting Glucose" if fbs else "Random Glucose")],
            suggested_confirmatory_workup=[
                "Immediate oral fast-acting glucose or intravenous 10-50% dextrose",
                "Reconciliation of insulin, sulfonylurea, or oral hypoglycemic doses",
                "Serum C-peptide, insulin, and beta-hydroxybutyrate if etiology unknown",
            ],
        ))

    # 9. Hyponatremia
    if na is not None and na < 135.0:
        is_crit = na < 120.0
        inferred.append(LabDiseaseInference(
            disease="Hyponatremia",
            likelihood="high",
            severity="critical" if is_crit else ("severe" if na < 125.0 else "moderate"),
            pathophysiology="Excess of total body water relative to sodium content causing extracellular hypo-osmolality, driving water shift into intracellular spaces and predisposing to cerebral edema.",
            clinical_rationale=f"Serum sodium depressed to {na:.1f} mEq/L (reference interval 135–145 mEq/L).",
            supporting_biomarkers=[fmt_biomarker("Sodium")],
            suggested_confirmatory_workup=[
                "Serum and urine osmolality",
                "Urine sodium concentration",
                "Clinical extracellular volume status determination (euvolemic vs hypovolemic vs hypervolemic)",
            ],
        ))

    # 10. Hypernatremia
    if na is not None and na > 145.0:
        is_crit = na > 155.0
        inferred.append(LabDiseaseInference(
            disease="Hypernatremia / Dehydration",
            likelihood="high",
            severity="critical" if is_crit else "severe",
            pathophysiology="Net free water deficit producing serum hyperosmolality, intracellular dehydration, and cellular shrinkage.",
            clinical_rationale=f"Serum sodium elevated to {na:.1f} mEq/L.",
            supporting_biomarkers=[fmt_biomarker("Sodium")],
            suggested_confirmatory_workup=[
                "Calculation of total body free water deficit",
                "Gradual oral or hypotonic IV free water repletion monitoring",
            ],
        ))

    # 11. Acute Bacterial Infection / Sepsis Risk
    if wbc is not None and (wbc > 12.0 or wbc < 4.0):
        has_shift = (neut is not None and neut > 80.0) or (crp is not None and crp > 50.0) or (pct is not None and pct > 0.5) or (lactate is not None and lactate > 2.0)
        if has_shift:
            supp = [fmt_biomarker("WBC Count")]
            if neut is not None:
                supp.append(fmt_biomarker("Neutrophils (%)"))
            if crp is not None:
                supp.append(fmt_biomarker("C-Reactive Protein"))
            if pct is not None:
                supp.append(fmt_biomarker("Procalcitonin"))
            if lactate is not None:
                supp.append(fmt_biomarker("Lactate"))

            is_crit = (lactate is not None and lactate >= 2.0) or (pct is not None and pct >= 2.0)
            inferred.append(LabDiseaseInference(
                disease="Acute Bacterial Infection / Sepsis Risk",
                likelihood="high",
                severity="critical" if is_crit else "severe",
                pathophysiology="Invasive pathogen eliciting acute bone marrow leukocyte demargination, bandemia, and systemic inflammatory endothelial activation with capillary leak.",
                clinical_rationale=f"Leukocytosis ({wbc:.1f} x10^3/uL) with left shift and markedly elevated inflammatory/sepsis biomarkers.",
                supporting_biomarkers=supp,
                suggested_confirmatory_workup=[
                    "Blood cultures (2 sets drawn from separate sites prior to antibiotics)",
                    "Diagnostic imaging directed at suspected source (chest radiograph, CT abdomen)",
                    "Serial serum lactate every 2-4 hours to monitor tissue perfusion",
                    "Prompt broad-spectrum empiric antimicrobial coverage",
                ],
            ))

    # 12. Microcytic Iron Deficiency Anemia
    if hb is not None and get_flag("Hemoglobin") in ("low", "critical low") and mcv is not None and mcv < 80.0:
        supp = [fmt_biomarker("Hemoglobin"), fmt_biomarker("MCV")]
        if ferritin is not None:
            supp.append(fmt_biomarker("Ferritin"))

        inferred.append(LabDiseaseInference(
            disease="Microcytic Iron Deficiency Anemia",
            likelihood="high" if (ferritin is not None and ferritin < 30.0) else "moderate",
            severity="severe" if hb < 8.0 else ("moderate" if hb < 10.0 else "mild"),
            pathophysiology="Inadequate iron stores restricting erythroblast heme and hemoglobin synthesis, resulting in microcytic, hypochromic erythrocytes with reduced systemic oxygen delivery.",
            clinical_rationale=f"Low hemoglobin ({hb:.1f} g/dL) with microcytosis (MCV {mcv:.1f} fL)" + (f" and depleted ferritin ({ferritin:.1f} ng/mL)" if ferritin else "") + ".",
            supporting_biomarkers=supp,
            suggested_confirmatory_workup=[
                "Complete iron studies (serum ferritin, serum iron, TIBC, transferrin saturation)",
                "Fecal occult blood test / endoscopy evaluation for gastrointestinal blood loss",
                "Oral or intravenous therapeutic iron replacement",
            ],
        ))

    # 13. Macrocytic / Megaloblastic Anemia
    if hb is not None and get_flag("Hemoglobin") in ("low", "critical low") and ((mcv is not None and mcv > 100.0) or (b12 is not None and b12 < 200.0) or (folate is not None and folate < 3.0)):
        supp = [fmt_biomarker("Hemoglobin")]
        if mcv is not None:
            supp.append(fmt_biomarker("MCV"))
        if b12 is not None:
            supp.append(fmt_biomarker("Vitamin B12"))
        if folate is not None:
            supp.append(fmt_biomarker("Serum Folate"))

        inferred.append(LabDiseaseInference(
            disease="Macrocytic / Megaloblastic Anemia",
            likelihood="high",
            severity="severe" if hb < 8.0 else "moderate",
            pathophysiology="Deficiency of vitamin B12 or folate disrupting purine/thymidylate synthesis and erythroblast nuclear maturation, leading to nuclear-cytoplasmic dyssynchrony and abnormally large erythrocytes.",
            clinical_rationale=f"Low hemoglobin ({hb:.1f} g/dL) with macrocytosis" + (f" (MCV {mcv:.1f} fL)" if mcv else "") + ".",
            supporting_biomarkers=supp,
            suggested_confirmatory_workup=[
                "Serum methylmalonic acid (MMA) and total homocysteine levels",
                "Peripheral blood smear examination for hypersegmented neutrophils",
                "Vitamin B12 and folate therapeutic repletion",
            ],
        ))

    # 14. Normocytic Anemia of Chronic Disease / Inflammation
    if hb is not None and get_flag("Hemoglobin") in ("low", "critical low") and mcv is not None and 80.0 <= mcv <= 100.0:
        if not any("Anemia" in d.disease for d in inferred):
            supp = [fmt_biomarker("Hemoglobin"), fmt_biomarker("MCV")]
            if crp is not None:
                supp.append(fmt_biomarker("C-Reactive Protein"))
            if esr is not None:
                supp.append(fmt_biomarker("ESR"))

            inferred.append(LabDiseaseInference(
                disease="Normocytic Anemia of Chronic Disease / Inflammation",
                likelihood="moderate",
                severity="moderate" if hb < 10.0 else "mild",
                pathophysiology="Chronic inflammation-induced hepcidin elevation sequestering iron inside reticuloendothelial macrophages, blunting renal erythropoietin response, and shortening red blood cell survival.",
                clinical_rationale=f"Normocytic anemia (Hb {hb:.1f} g/dL, MCV {mcv:.1f} fL) with active inflammatory biomarker elevation.",
                supporting_biomarkers=supp,
                suggested_confirmatory_workup=[
                    "Reticulocyte production index",
                    "Soluble transferrin receptor-ferritin index",
                    "Evaluation and management of underlying chronic inflammatory or systemic disorder",
                ],
            ))

    # 15. Severe Thrombocytopenia
    if plt is not None and plt < 150.0:
        is_crit = plt < 50.0
        inferred.append(LabDiseaseInference(
            disease="Thrombocytopenia",
            likelihood="high",
            severity="critical" if is_crit else ("severe" if plt < 100.0 else "moderate"),
            pathophysiology="Substantial reduction in circulating platelets predisposing to spontaneous mucocutaneous hemorrhage, petechiae, and visceral bleeding.",
            clinical_rationale=f"Platelet count severely reduced to {plt:.1f} x10^3/uL (reference interval 150–450 x10^3/uL).",
            supporting_biomarkers=[fmt_biomarker("Platelet Count")],
            suggested_confirmatory_workup=[
                "Peripheral blood smear to rule out EDTA-dependent pseudothrombocytopenia",
                "Coagulation profile (PT/INR, aPTT, fibrinogen) to exclude DIC",
                "Medication reconciliation for heparin-induced or drug-induced thrombocytopenia",
            ],
        ))

    # 16. Acute Hepatocellular Injury / Transaminitis
    if (alt is not None and alt >= 150.0) or (ast is not None and ast >= 135.0):
        is_crit = (alt is not None and alt >= 1000.0) or (ast is not None and ast >= 1000.0)
        supp = []
        if alt is not None:
            supp.append(fmt_biomarker("ALT (SGPT)"))
        if ast is not None:
            supp.append(fmt_biomarker("AST (SGOT)"))

        inferred.append(LabDiseaseInference(
            disease="Acute Hepatocellular Injury / Transaminitis",
            likelihood="high",
            severity="critical" if is_crit else ("severe" if (alt and alt >= 500) else "moderate"),
            pathophysiology="Direct toxic, ischemic, or viral destruction of hepatocytes resulting in extensive leakage of cytosolic aminotransferases into circulation.",
            clinical_rationale="Marked transaminitis exceeding 3x upper reference limit" + (f" (ALT {alt:.1f} U/L, AST {ast:.1f} U/L)" if alt and ast else "") + ".",
            supporting_biomarkers=supp,
            suggested_confirmatory_workup=[
                "Acute viral hepatitis panel (Anti-HAV IgM, HBsAg, Anti-HBc IgM, Anti-HCV)",
                "Serum acetaminophen level and comprehensive toxicology screen",
                "Coagulation profile (PT/INR) and serum albumin to evaluate hepatic synthetic reserve",
                "Right upper quadrant abdominal ultrasound with Doppler",
            ],
        ))

    # 17. Cholestasis / Obstructive Biliary Disease
    if ((alp is not None and alp > 147.0) or (ggt is not None and ggt > 50.0)) and ((tbili is not None and tbili > 1.2) or (dbili is not None and dbili > 0.3)):
        supp = []
        if alp is not None:
            supp.append(fmt_biomarker("Alkaline Phosphatase"))
        if tbili is not None:
            supp.append(fmt_biomarker("Total Bilirubin"))
        if dbili is not None:
            supp.append(fmt_biomarker("Direct Bilirubin"))

        inferred.append(LabDiseaseInference(
            disease="Cholestasis / Obstructive Biliary Disease",
            likelihood="high",
            severity="severe" if (tbili and tbili >= 5.0) else "moderate",
            pathophysiology="Impairment of bile excretion secondary to extrahepatic ductal stone/stricture or intrahepatic canalicular injury, causing accumulation of bile acids, alkaline phosphatase, and conjugated bilirubin.",
            clinical_rationale="Concomitant elevation of cholestatic enzymes (ALP) and bilirubin.",
            supporting_biomarkers=supp,
            suggested_confirmatory_workup=[
                "Right upper quadrant abdominal ultrasound for biliary ductal dilatation or cholelithiasis",
                "Magnetic resonance cholangiopancreatography (MRCP)",
                "Fractionated bilirubin and serum lipase",
            ],
        ))

    # 18. Acute Pancreatitis
    if (lipase is not None and lipase >= 180.0) or (amylase is not None and amylase >= 300.0):
        supp = []
        if lipase is not None:
            supp.append(fmt_biomarker("Lipase"))
        if amylase is not None:
            supp.append(fmt_biomarker("Amylase"))

        inferred.append(LabDiseaseInference(
            disease="Acute Pancreatitis",
            likelihood="high",
            severity="severe",
            pathophysiology="Premature intrapancreatic trypsinogen activation within acinar cells causing enzymatic autodigestion, local tissue necrosis, vascular damage, and acute systemic inflammatory mediator release.",
            clinical_rationale="Pancreatic digestive enzyme elevation exceeding 3x upper limit of normal.",
            supporting_biomarkers=supp,
            suggested_confirmatory_workup=[
                "Abdominal ultrasound to evaluate for gallstone pancreatitis",
                "Serum triglycerides and calcium levels",
                "Aggressive isotonic intravenous fluid resuscitation",
                "Contrast-enhanced abdominal CT if clinical deterioration occurs after 48-72h",
            ],
        ))

    # 19. Acute Myocardial Infarction / Myocardial Necrosis
    trop = trop_i or trop_t
    if trop is not None and ((trop_i is not None and trop_i > 0.04) or (trop_t is not None and trop_t > 0.01) or get_flag("Troponin I") in ("high", "critical high") or get_flag("Troponin T") in ("high", "critical high")):
        inferred.append(LabDiseaseInference(
            disease="Acute Myocardial Infarction / ACS",
            likelihood="high",
            severity="critical",
            pathophysiology="Coronary atherosclerotic plaque rupture with occlusive thrombosis causing cardiomyocyte ischemia, irreversible necrosis, and systemic leakage of high-sensitivity cardiac troponins.",
            clinical_rationale=f"Elevated high-sensitivity cardiac troponin ({trop:.4f} ng/mL), establishing acute myocardial injury.",
            supporting_biomarkers=[fmt_biomarker("Troponin I" if trop_i else "Troponin T")],
            suggested_confirmatory_workup=[
                "Serial 12-lead ECGs (every 15-30 minutes)",
                "Repeat high-sensitivity troponin in 1 to 2 hours to evaluate dynamic rise/fall delta",
                "Emergent cardiology consultation for invasive coronary angiography",
                "Telemetry cardiac monitoring and guideline-directed medical therapy",
            ],
        ))

    # 20. Hypercalcemia
    if ca is not None and ca > 10.2:
        is_crit = ca > 13.0
        inferred.append(LabDiseaseInference(
            disease="Hypercalcemia",
            likelihood="high",
            severity="critical" if is_crit else ("severe" if ca > 12.0 else "moderate"),
            pathophysiology="Elevation of extracellular calcium concentration secondary to excessive bone resorption (hyperparathyroidism, malignancy) or increased intestinal absorption, diminishing neuromuscular excitability.",
            clinical_rationale=f"Serum calcium elevated to {ca:.2f} mg/dL (reference limit 10.2 mg/dL).",
            supporting_biomarkers=[fmt_biomarker("Serum Calcium")],
            suggested_confirmatory_workup=[
                "Intact parathyroid hormone (iPTH)",
                "PTH-related peptide (PTHrP) and 25-OH Vitamin D",
                "12-lead ECG for shortened QT interval",
                "Isotonic saline IV volume expansion",
            ],
        ))

    # 21. Normal / Unremarkable Laboratory Profile
    if not inferred:
        abnormal_results = [r for r in results if r.flag not in ("normal", "unknown")]
        if not abnormal_results:
            inferred.append(LabDiseaseInference(
                disease="Normal / Unremarkable Laboratory Profile",
                likelihood="high",
                severity="mild",
                pathophysiology="Normal biochemical and cellular physiological homeostasis with all evaluated parameters falling within established biological reference intervals.",
                clinical_rationale=f"All {len(results)} evaluated laboratory analytes are within normal physiological reference ranges without critical or abnormal flags.",
                supporting_biomarkers=["All tested analytes within normal limits"],
                suggested_confirmatory_workup=[
                    "Routine preventative health maintenance",
                    "Periodic re-evaluation as clinically indicated",
                ],
            ))

    # Sort inferred diseases: critical first, then severe, moderate, mild
    severity_order = {"critical": 0, "severe": 1, "moderate": 2, "mild": 3}
    inferred.sort(key=lambda d: severity_order.get(d.severity, 99))

    return inferred


def generate_lab_clinical_summary(inferred: List[LabDiseaseInference], total_tests: int) -> str:
    """Generate concise, physician-grade laboratory assessment summary."""
    if not inferred:
        return f"Laboratory evaluation of {total_tests} standard analytes demonstrates normal physiological parameters without acute metabolic or cellular dysfunction."
    if len(inferred) == 1 and inferred[0].disease == "Normal / Unremarkable Laboratory Profile":
        return f"Laboratory evaluation of {total_tests} standard analytes demonstrates normal physiological parameters without acute metabolic or cellular dysfunction."

    top_d = inferred[0]
    critical_alerts = [d.disease for d in inferred if d.severity == "critical"]
    crit_clause = f" Critical alert: urgent clinical intervention required for {', '.join(critical_alerts)}." if critical_alerts else ""
    other_d = [d.disease for d in inferred[1:3] if d.disease != top_d.disease]
    other_clause = f" Additional laboratory findings indicate {', '.join(other_d)}." if other_d else ""
    conf_str = f" ({top_d.confidence_score * 100.0:.1f}% neural confidence)" if getattr(top_d, "confidence_score", None) is not None else ""

    return f"Laboratory analysis indicates findings characteristic of {top_d.disease}{conf_str}.{crit_clause}{other_clause} Targeted confirmatory workup and guided clinical monitoring are recommended."


def run_lab_agent(case_input: CaseInput) -> LabReport:
    """Execute complete laboratory evaluation pipeline using trained ClinicalLabNet neural model."""
    has_text = bool(case_input.lab_text and case_input.lab_text.strip())
    has_files = bool(case_input.lab_paths and len(case_input.lab_paths) > 0)

    if not has_text and not has_files:
        return LabReport(
            status="skipped",
            results=[],
            patterns=[],
            inferred_diseases=[],
            clinical_summary="",
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

    # Neural multi-label disease inference using trained ClinicalLabNet PyTorch deep learning model
    inferred_diseases: List[LabDiseaseInference] = []
    try:
        from medagent.models.lab_net import infer_diseases_with_lab_net
        inferred_diseases = infer_diseases_with_lab_net(parsed_results, case_input.patient_sex)
    except Exception as e:
        logger.warning("Neural lab model inference failed, falling back to clinical logic: %s", e)
        inferred_diseases = []

    if not inferred_diseases:
        inferred_diseases = infer_laboratory_diseases(parsed_results, case_input.patient_sex)

    clinical_summary = generate_lab_clinical_summary(inferred_diseases, len(parsed_results))

    status_val = "ok" if parsed_results else ("error" if not unparsed else "ok")
    msg = f"Processed {len(parsed_results)} laboratory tests, synthesized {len(patterns)} clinical patterns, inferred {len(inferred_diseases)} disease diagnoses."
    if notes:
        msg += " " + " ".join(notes)

    return LabReport(
        status=status_val,
        results=parsed_results,
        patterns=patterns,
        inferred_diseases=inferred_diseases,
        clinical_summary=clinical_summary,
        unparsed_lines=unparsed,
        message=msg,
    )
