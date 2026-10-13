"""Safety Agent: deterministic grounding, consistency, contraindication, language, and urgency guardrails."""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from medagent.config import settings
from medagent.schemas import (
    CaseInput,
    ImagingReport,
    LabReport,
    RagReport,
    ClinicalDraft,
    SafetyReport,
    SafetyIssue,
)

logger = logging.getLogger("medagent.safety")

# Red flag symptoms and negation regexes
RED_FLAG_SYMPTOMS: List[Tuple[str, str]] = [
    (r"\b(?:crushing|severe|acute|substernal|tearing)?\s*chest pain\b", "Chest pain"),
    (r"\b(?:sudden|severe|acute|worsening)\s*(?:breathlessness|dyspnea|shortness of breath)\b", "Severe / sudden breathlessness"),
    (r"\b(?:hemoptysis|haemoptysis|coughing up blood)\b", "Haemoptysis"),
    (r"\b(?:syncope|loss of consciousness|fainted|blackout)\b", "Syncope"),
    (r"\b(?:acute confusion|altered mental status|delirium)\b", "Confusion / altered mental status"),
    (r"\b(?:seizure|convulsion|status epilepticus)\b", "Seizure"),
    (r"\b(?:stroke|facial droop|hemiparesis|slurred speech)\b", "Stroke signs / neurological deficit"),
    (r"\b(?:self-harm|suicid(?:al|e)?)\b", "Self-harm / suicide risk"),
    (r"\b(?:hypoxia|cyanosis|spo2\s*<\s*88)\b", "Hypoxia / severe desaturation"),
    (r"\b(?:gi bleed(?:ing)?|melena|hematemesis|rectal bleeding)\b", "Gastrointestinal bleeding"),
    (r"\btearing\s*(?:chest|back|chest\s*and\s*back|back\s*and\s*chest)\s*pain\b", "Tearing chest/back pain (aortic dissection risk)"),
]

NEGATION_PATTERNS: List[str] = [
    r"\bno\b",
    r"\bdenies\b",
    r"\bwithout\b",
    r"\bfree of\b",
    r"\babsence of\b",
    r"\bnegative for\b",
    r"\bnot\b",
    r"\brules? out\b",
]

REASSURING_PHRASES: List[str] = [
    r"\bbenign\b",
    r"\breassuring\b",
    r"\bmild\b",
    r"\bno acute (?:concerns|distress|issues)\b",
    r"\broutine follow-?up\b",
    r"\bdischarged? home\b",
    r"\bunremarkable\b",
    r"\blow risk\b",
    r"\bnon-urgent\b",
]


def check_negation(text: str, match_start: int) -> bool:
    """Check if a symptom match in text is preceded by a negation phrase within 40 characters."""
    window_start = max(0, match_start - 45)
    preceding_window = text[window_start:match_start].lower()
    for neg_pat in NEGATION_PATTERNS:
        if re.search(neg_pat, preceding_window):
            return True
    return False


def sanitize_language(text: str) -> Tuple[str, List[SafetyIssue]]:
    """
    Remove drug doses/schedules and soften over-certain diagnostic assertions.
    """
    issues: List[SafetyIssue] = []
    sanitized = text

    # 1. Remove drug doses and schedules
    dose_pattern = re.compile(
        r"([^.?!;]*?\b(?:\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units|meq)(?:/(?:kg|day|hr|m2))?|"
        r"(?:bid|tid|qid|daily|q\d+h|twice a day|three times daily))\b[^.?!;]*?[.?!;])",
        re.IGNORECASE,
    )

    matches = dose_pattern.findall(sanitized)
    if matches:
        for m in matches:
            issues.append(
                SafetyIssue(
                    severity="warning",
                    code="LANGUAGE_DOSING_REMOVED",
                    message=f"Removed statement containing specific drug dose or schedule: '{m.strip()}'",
                    where="summary/differentials",
                )
            )
            sanitized = sanitized.replace(m, "")

    # 2. Soften over-certain phrasing
    replacements = [
        (r"\bdefinit(?:ely|ively)\b", "likely"),
        (r"\bconfirms\b", "suggests"),
        (r"\bdiagnosed with\b", "consistent with"),
        (r"\bruled out\b", "considered less likely"),
        (r"\bproves?\b", "strongly supports"),
        (r"\bcertainly\b", "likely"),
    ]

    for pat, repl in replacements:
        if re.search(pat, sanitized, re.IGNORECASE):
            issues.append(
                SafetyIssue(
                    severity="info",
                    code="LANGUAGE_CERTAINTY_SOFTENED",
                    message=f"Softened over-certain clinical language matching '{pat}' to '{repl}'.",
                    where="clinical_draft",
                )
            )
            sanitized = re.sub(pat, repl, sanitized, flags=re.IGNORECASE)

    return sanitized.strip(), issues


def check_contraindications(
    case_input: CaseInput,
    draft: ClinicalDraft,
    lab_report: Optional[LabReport],
) -> List[SafetyIssue]:
    """Evaluate dangerous drug-disease, drug-drug, and radiation contraindications."""
    issues: List[SafetyIssue] = []
    text_corpus = (case_input.clinical_text + " " + draft.summary + " " + " ".join(draft.suggested_workup)).lower()

    # Extract lab values
    lab_vals: Dict[str, float] = {}
    if lab_report and lab_report.status == "ok":
        for r in lab_report.results:
            lab_vals[r.test] = r.value

    creat = lab_vals.get("Serum Creatinine", 0.0)
    egfr = lab_vals.get("eGFR", 100.0)
    k_val = lab_vals.get("Potassium", 4.0)
    plt_val = lab_vals.get("Platelet Count", 250.0)
    inr_val = lab_vals.get("INR", 1.0)

    has_renal_impairment = (creat > 1.5) or (egfr < 60.0)

    # 1. Iodinated Contrast with renal impairment
    if "contrast" in text_corpus:
        if has_renal_impairment:
            issues.append(
                SafetyIssue(
                    severity="blocker",
                    code="CONTRAINDICATION_CONTRAST_RENAL",
                    message=f"Iodinated contrast contraindicated with renal impairment (Creatinine: {creat} mg/dL, eGFR: {egfr} mL/min). Risk of Contrast-Induced Nephropathy.",
                    where="suggested_workup/notes",
                )
            )

    # 2. NSAIDs with renal impairment or hyperkalemia
    nsaid_match = re.search(r"\b(?:nsaid|ibuprofen|naproxen|ketorolac|diclofenac|indomethacin|meloxicam)\b", text_corpus)
    if nsaid_match:
        if has_renal_impairment or k_val > 5.0:
            issues.append(
                SafetyIssue(
                    severity="blocker",
                    code="CONTRAINDICATION_NSAIDS_RENAL_HYPERKALEMIA",
                    message=f"NSAIDs contraindicated in renal impairment or hyperkalemia (K: {k_val} mEq/L, Creat: {creat}). Risk of acute renal failure and worsening hyperkalemia.",
                    where="notes/medications",
                )
            )

    # 3. ACE inhibitors / ARBs / Spironolactone with hyperkalemia
    raas_match = re.search(r"\b(?:lisinopril|enalapril|ramipril|losartan|valsartan|spironolactone|eplerenone)\b", text_corpus)
    if raas_match and k_val > 5.0:
        issues.append(
            SafetyIssue(
                severity="blocker",
                code="CONTRAINDICATION_RAAS_HYPERKALEMIA",
                message=f"Potassium-sparing agent or ACEi/ARB detected with hyperkalemia (Potassium: {k_val} mEq/L). Immediate discontinuation required to prevent fatal arrhythmias.",
                where="notes/medications",
            )
        )

    # 4. Antithrombotics / Anticoagulants with low platelets or high INR
    anticoag_match = re.search(r"\b(?:warfarin|heparin|enoxaparin|apixaban|rivaroxaban|dabigatran|aspirin|clopidogrel)\b", text_corpus)
    if anticoag_match:
        if plt_val < 50.0 or inr_val > 2.5:
            issues.append(
                SafetyIssue(
                    severity="blocker",
                    code="CONTRAINDICATION_ANTICOAGULATION_BLEEDING_RISK",
                    message=f"Antithrombotic therapy contraindicated with severe coagulopathy / thrombocytopenia (Platelets: {plt_val} x10^3/uL, INR: {inr_val}). Severe bleeding risk.",
                    where="notes/medications",
                )
            )

    # 5. Pregnancy with ionizing imaging or contrast
    if re.search(r"\b(?:pregnant|pregnancy|gestation|first trimester|second trimester|third trimester)\b", text_corpus):
        if case_input.image_paths or "contrast" in text_corpus or "ct" in text_corpus:
            issues.append(
                SafetyIssue(
                    severity="warning",
                    code="CONTRAINDICATION_PREGNANCY_RADIATION",
                    message="Documented pregnancy alongside radiographic imaging or contrast. Requires fetal dose minimization and pelvic shielding.",
                    where="patient_intake/imaging",
                )
            )

    # 6. Metformin with renal impairment
    if "metformin" in text_corpus and (creat > 1.5 or egfr < 45.0):
        issues.append(
            SafetyIssue(
                severity="blocker",
                code="CONTRAINDICATION_METFORMIN_LACTIC_ACIDOSIS",
                message=f"Metformin contraindicated in renal impairment (eGFR: {egfr} mL/min). Risk of fatal metformin-associated lactic acidosis (MALA).",
                where="medications",
            )
        )

    # 7. Documented allergy to contrast or antibiotics
    has_contrast_allergy = bool(re.search(r"\b(?:contrast|iodine)\s*allerg(?:y|ic)\b|\ballerg(?:y|ic).*?\b(?:contrast|iodine)\b", text_corpus))
    if has_contrast_allergy and "contrast" in text_corpus:
        issues.append(
            SafetyIssue(
                severity="blocker",
                code="CONTRAINDICATION_ALLERGY_CONTRAST",
                message="Documented allergy to iodinated contrast with contrast procedure mentioned. Severe anaphylaxis risk.",
                where="allergies",
            )
        )

    has_penicillin_allergy = bool(re.search(r"\b(?:penicillin|beta-lactam|amoxicillin|ampicillin)\s*allerg(?:y|ic)\b|\ballerg(?:y|ic).*?\b(?:penicillin|amoxicillin|ampicillin)\b", text_corpus))
    if has_penicillin_allergy and any(ab in text_corpus for ab in ("penicillin", "amoxicillin", "ampicillin")):
        issues.append(
            SafetyIssue(
                severity="blocker",
                code="CONTRAINDICATION_ALLERGY_ANTIBIOTIC",
                message="Documented penicillin allergy with beta-lactam mentioned. Risk of severe allergic reaction.",
                where="allergies",
            )
        )

    return issues


def check_clinical_consistency(
    case_input: CaseInput,
    draft: ClinicalDraft,
    imaging_report: Optional[ImagingReport],
    lab_report: Optional[LabReport],
) -> List[SafetyIssue]:
    """Detect inconsistencies between proposed diagnoses and multimodal findings."""
    issues: List[SafetyIssue] = []
    text_corpus = case_input.clinical_text.lower()
    conditions = [d.condition.lower() for d in draft.differentials]

    # Pre-parse labs
    wbc_val = None
    crp_val = None
    esr_val = None
    ferritin_val = None
    if lab_report and lab_report.status == "ok":
        for r in lab_report.results:
            if r.test == "WBC Count":
                wbc_val = r.value
            elif r.test == "C-Reactive Protein":
                crp_val = r.value
            elif r.test == "ESR":
                esr_val = r.value
            elif r.test == "Ferritin":
                ferritin_val = r.value

    # 1. Infective diagnosis with completely normal inflammatory markers
    has_infection_diag = any("pneumonia" in c or "sepsis" in c or "infection" in c for c in conditions)
    if has_infection_diag and lab_report and lab_report.status == "ok":
        if (wbc_val is not None and wbc_val <= 10.0) and (crp_val is not None and crp_val <= 5.0) and (esr_val is not None and esr_val <= 15.0):
            issues.append(
                SafetyIssue(
                    severity="warning",
                    code="CONSISTENCY_INFECTION_NORMAL_MARKERS",
                    message="Infective diagnosis proposed despite completely normal leukocyte count, CRP, and ESR. Consider non-infectious inflammatory or reactive etiologies.",
                    where="differentials",
                )
            )

    # 2. Pneumonia without radiographic infiltrate or fever
    has_pneumonia = any("pneumonia" in c for c in conditions)
    if has_pneumonia:
        has_fever = bool(re.search(r"\bfever\b|\bchills\b|\btemp(?:erature)?\s*>\s*38", text_corpus))
        has_infiltrate = False
        if imaging_report and imaging_report.status == "ok":
            has_infiltrate = any(
                f.finding in ("Consolidation", "Infiltration", "Pneumonia", "Effusion")
                for f in imaging_report.findings
            )
        if not has_infiltrate and not has_fever:
            issues.append(
                SafetyIssue(
                    severity="warning",
                    code="CONSISTENCY_PNEUMONIA_NO_INFILTRATE_OR_FEVER",
                    message="Diagnosis of pneumonia proposed without radiographic consolidation/infiltrate and without documented fever.",
                    where="differentials",
                )
            )

    # 3. Iron deficiency with normal or high ferritin
    has_ida = any("iron deficiency" in c for c in conditions)
    if has_ida and ferritin_val is not None and ferritin_val > 100.0:
        issues.append(
            SafetyIssue(
                severity="warning",
                code="CONSISTENCY_IRON_DEFICIENCY_NORMAL_FERRITIN",
                message=f"Iron deficiency anemia proposed despite normal/elevated ferritin ({ferritin_val} ng/mL). Evaluate for anemia of chronic disease or hemoglobinopathy.",
                where="differentials",
            )
        )

    # 4. Low-confidence imaging or quality notes
    if imaging_report and imaging_report.status == "ok":
        low_conf = [f.finding for f in imaging_report.findings if f.confidence == "low"]
        if low_conf:
            issues.append(
                SafetyIssue(
                    severity="info",
                    code="IMAGING_LOW_CONFIDENCE",
                    message=f"Low-confidence radiographic findings reported: {', '.join(low_conf)}.",
                    where="imaging_report",
                )
            )
        if imaging_report.image_quality_notes:
            issues.append(
                SafetyIssue(
                    severity="info",
                    code="IMAGING_QUALITY_NOTES",
                    message="; ".join(imaging_report.image_quality_notes),
                    where="imaging_report",
                )
            )
    elif imaging_report and imaging_report.status in ("unavailable", "error"):
        issues.append(
            SafetyIssue(
                severity="info",
                code="IMAGING_UNAVAILABLE",
                message=f"Chest radiograph analysis was unavailable: {imaging_report.message}",
                where="imaging_report",
            )
        )

    # 5. Unparsed lab lines
    if lab_report and lab_report.unparsed_lines:
        issues.append(
            SafetyIssue(
                severity="info",
                code="LAB_UNPARSED_LINES",
                message=f"Unparsed lines in lab report: {len(lab_report.unparsed_lines)} lines could not be matched.",
                where="lab_report",
            )
        )

    return issues


def check_grounding_and_citations(
    draft: ClinicalDraft,
    rag_report: Optional[RagReport],
) -> Tuple[List[SafetyIssue], Optional[str]]:
    """Validate citations against RAG store and purge hallucinated citation IDs."""
    issues: List[SafetyIssue] = []
    valid_ids: Set[str] = set()
    if rag_report and rag_report.status == "ok":
        valid_ids = {e.id for e in rag_report.evidence}

    purged_any = False
    for diff in draft.differentials:
        # Check citations
        filtered_ids = []
        for cid in diff.evidence_ids:
            if cid in valid_ids:
                filtered_ids.append(cid)
            else:
                purged_any = True
                issues.append(
                    SafetyIssue(
                        severity="warning",
                        code="GROUNDING_HALLUCINATED_CITATION",
                        message=f"Purged citation ID '{cid}' on '{diff.condition}' which does not exist in retrieved evidence.",
                        where=f"differentials/{diff.condition}",
                    )
                )
        diff.evidence_ids = filtered_ids

        # Flag differentials with no supporting findings
        if not diff.supporting_findings:
            issues.append(
                SafetyIssue(
                    severity="warning",
                    code="GROUNDING_NO_SUPPORTING_FINDINGS",
                    message=f"Differential diagnosis '{diff.condition}' has no listed supporting clinical or diagnostic findings.",
                    where=f"differentials/{diff.condition}",
                )
            )

    revised_note = "Citations reconciled with knowledge store." if purged_any else None
    return issues, revised_note


def check_urgency(
    case_input: CaseInput,
    imaging_report: Optional[ImagingReport],
    lab_report: Optional[LabReport],
) -> Tuple[bool, List[str]]:
    """Detect acute red flags with strict negation handling."""
    urgent_reasons: List[str] = []
    notes_clean = case_input.clinical_text

    # 1. Non-negated red flag symptoms
    for pattern, symptom_name in RED_FLAG_SYMPTOMS:
        for match in re.finditer(pattern, notes_clean, re.IGNORECASE):
            if not check_negation(notes_clean, match.start()):
                urgent_reasons.append(f"Non-negated acute symptom: {symptom_name}")
                break

    # 2. Urgent lab patterns
    if lab_report and lab_report.status == "ok":
        for pat in lab_report.patterns:
            if pat.urgency == "urgent":
                urgent_reasons.append(f"Urgent laboratory pattern: {pat.name} ({pat.description})")

        # Critical lab limits
        for res in lab_report.results:
            if res.flag in ("critical high", "critical low"):
                urgent_reasons.append(f"Critical panic laboratory value: {res.test} = {res.value} {res.unit} ({res.flag})")

    # 3. Possible pneumothorax on imaging
    if imaging_report and imaging_report.status == "ok":
        for f in imaging_report.findings:
            if "pneumothorax" in f.finding.lower() and f.probability >= 0.40:
                urgent_reasons.append(f"Radiographic pneumothorax identified (p={f.probability:.1%})")

    return bool(urgent_reasons), urgent_reasons


def run_safety_agent(
    case_input: CaseInput,
    clinical_draft: ClinicalDraft,
    imaging_report: Optional[ImagingReport] = None,
    lab_report: Optional[LabReport] = None,
    rag_report: Optional[RagReport] = None,
) -> SafetyReport:
    """Execute complete deterministic safety validation suite."""
    all_issues: List[SafetyIssue] = []

    # 1. Grounding check
    grounding_issues, ground_revised = check_grounding_and_citations(clinical_draft, rag_report)
    all_issues.extend(grounding_issues)

    # 2. Language sanitization
    sanitized_summary, lang_issues = sanitize_language(clinical_draft.summary)
    clinical_draft.summary = sanitized_summary
    all_issues.extend(lang_issues)

    # 3. Consistency checks
    consistency_issues = check_clinical_consistency(case_input, clinical_draft, imaging_report, lab_report)
    all_issues.extend(consistency_issues)

    # 4. Contraindication checks
    contra_issues = check_contraindications(case_input, clinical_draft, lab_report)
    all_issues.extend(contra_issues)

    # 5. Urgency checks
    is_urgent, urgent_reasons = check_urgency(case_input, imaging_report, lab_report)

    # 6. Blocker Rule: Reassuring language when urgent findings exist
    summary_lower = clinical_draft.summary.lower()
    if is_urgent:
        for re_pat in REASSURING_PHRASES:
            if re.search(re_pat, summary_lower):
                all_issues.append(
                    SafetyIssue(
                        severity="blocker",
                        code="BLOCKER_REASSURING_DURING_EMERGENCY",
                        message=f"Draft uses reassuring wording ('{re_pat}') despite acute urgent findings: {'; '.join(urgent_reasons[:2])}. Immediate clinical escalation required.",
                        where="summary",
                    )
                )
                break

    # Determine final verdict
    has_blockers = any(i.severity == "blocker" for i in all_issues)
    has_warnings = any(i.severity == "warning" for i in all_issues)

    if has_blockers:
        verdict = "blocked"
    elif has_warnings:
        verdict = "pass_with_warnings"
    else:
        verdict = "pass"

    revisions = []
    if ground_revised:
        revisions.append(ground_revised)
    if any(i.code == "LANGUAGE_DOSING_REMOVED" for i in lang_issues):
        revisions.append("Removed drug dosing statements.")
    if any(i.code == "LANGUAGE_CERTAINTY_SOFTENED" for i in lang_issues):
        revisions.append("Softened over-certain assertions.")

    return SafetyReport(
        verdict=verdict,
        issues=all_issues,
        urgent=is_urgent,
        urgent_reasons=urgent_reasons,
        revised="; ".join(revisions) if revisions else None,
    )
