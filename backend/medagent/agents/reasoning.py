"""Clinical Reasoning Agent: multimodal clinical synthesis using LLM engine with weighted rule-engine fallback."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from medagent.config import settings
from medagent.llm import call_llm_json
from medagent.schemas import (
    CaseInput,
    ImagingReport,
    LabReport,
    RagReport,
    ClinicalDraft,
    Differential,
    Evidence,
)

logger = logging.getLogger("medagent.reasoning")

# Catalog of >= 25 conditions evaluated by rule engine
CONDITIONS_CATALOG: Dict[str, Dict[str, Any]] = {
    "Community-Acquired Pneumonia": {
        "symptoms": [r"\bfever\b", r"\bcough\b", r"\bsputum\b", r"\bcrackles\b", r"\brales\b", r"\bpleuritic\b", r"\bdyspnea\b"],
        "imaging": {"Consolidation": 3.0, "Infiltration": 2.5, "Pneumonia": 3.0, "Effusion": 1.5},
        "labs": {"Leukocytosis with Neutrophilia": 2.5, "Leukocytosis": 1.8, "Raised Inflammatory Markers": 2.0},
        "against_imaging": ["Normal"],
        "workup": ["Sputum gram stain and culture", "Blood cultures prior to antibiotics", "Follow-up chest radiograph in 6 weeks"],
    },
    "Pleural Effusion": {
        "symptoms": [r"\bpleuritic\b", r"\bdullness\b", r"\bdecreased breath sounds\b", r"\bdyspnea\b", r"\borthopnea\b"],
        "imaging": {"Effusion": 4.0, "Atelectasis": 1.5},
        "labs": {"Hypoalbuminaemia": 1.5, "Raised Inflammatory Markers": 1.0},
        "against_imaging": ["Pneumothorax"],
        "workup": ["Diagnostic thoracentesis with Light's criteria", "Pleural fluid cell count and culture", "Chest ultrasound"],
    },
    "Pneumothorax": {
        "symptoms": [r"\bsudden.*chest pain\b", r"\bsharp.*pain\b", r"\bpleuritic\b", r"\bhyperresonance\b", r"\bacute.*dyspnea\b"],
        "imaging": {"Pneumothorax": 4.5},
        "labs": {},
        "against_imaging": ["Effusion"],
        "workup": ["Urgent upright expiratory chest radiograph", "Continuous pulse oximetry", "Thoracic surgery consultation"],
    },
    "Congestive Heart Failure / Pulmonary Edema": {
        "symptoms": [r"\borthopnea\b", r"\bpnd\b", r"\bparoxysmal nocturnal\b", r"\bedema\b", r"\bjvd\b", r"\bcrackles\b"],
        "imaging": {"Cardiomegaly": 3.0, "Edema": 3.5, "Effusion": 2.0},
        "labs": {"Renal Impairment": 1.5, "Hyponatraemia": 1.5},
        "against_imaging": [],
        "workup": ["Serum NT-proBNP or BNP", "Transthoracic echocardiogram", "Daily weight and fluid restriction monitoring"],
    },
    "Cardiomegaly": {
        "symptoms": [r"\bhypertension\b", r"\bfatigue\b", r"\bdyspnea on exertion\b"],
        "imaging": {"Cardiomegaly": 4.0},
        "labs": {},
        "against_imaging": [],
        "workup": ["12-lead electrocardiogram", "Transthoracic echocardiogram to assess chamber size", "Blood pressure log"],
    },
    "Acute Kidney Injury": {
        "symptoms": [r"\boliguria\b", r"\banuria\b", r"\blethargy\b", r"\bnausea\b", r"\bdecreased urine\b", r"\bfluid overload\b"],
        "imaging": {},
        "labs": {"Renal Impairment": 4.0, "Hyperkalaemia": 2.5, "Metabolic Acidosis": 2.0},
        "against_imaging": [],
        "workup": ["Renal ultrasound to exclude hydronephrosis", "Urine electrolytes and fractional excretion of sodium", "Discontinue nephrotoxic agents"],
    },
    "Chronic Kidney Disease": {
        "symptoms": [r"\bchronic kidney\b", r"\bckd\b", r"\bhypertension\b", r"\bdiabetes\b", r"\bnocturia\b", r"\bpruritus\b"],
        "imaging": {},
        "labs": {"Renal Impairment": 3.0, "Normocytic Anaemia": 1.5, "Hyperkalaemia": 1.5},
        "against_imaging": [],
        "workup": ["Spot urine albumin-to-creatinine ratio", "Serial serum creatinine tracking", "Nephrology referral"],
    },
    "Severe Hyperkalemia": {
        "symptoms": [r"\bweakness\b", r"\bpalpitations\b", r"\bparesthesias\b", r"\bflaccid\b"],
        "imaging": {},
        "labs": {"Hyperkalaemia": 4.5, "Renal Impairment": 2.0, "Metabolic Acidosis": 1.5},
        "against_imaging": [],
        "workup": ["Immediate 12-lead ECG for peaked T waves/conduction blocks", "Telemetry cardiac monitoring", "Discontinue ACE inhibitors, spironolactone, and NSAIDs"],
    },
    "Hypokalemia": {
        "symptoms": [r"\bcramps\b", r"\bmuscle weakness\b", r"\bvomiting\b", r"\bdiarrhea\b", r"\bpolyuria\b"],
        "imaging": {},
        "labs": {"Hypokalaemia": 4.0},
        "against_imaging": [],
        "workup": ["Serum magnesium level", "12-lead ECG for U waves and ST changes", "Review diuretic regimen"],
    },
    "Hyponatremia": {
        "symptoms": [r"\bconfusion\b", r"\blethargy\b", r"\bheadache\b", r"\bnausea\b", r"\bseizure\b", r"\bunsteady\b"],
        "imaging": {},
        "labs": {"Hyponatraemia": 4.0},
        "against_imaging": [],
        "workup": ["Serum and urine osmolality", "Urine sodium concentration", "Assessment of clinical volume status"],
    },
    "Microcytic Iron Deficiency Anemia": {
        "symptoms": [r"\bfatigue\b", r"\bpallor\b", r"\bcold intolerance\b", r"\bbrittle nails\b", r"\bpica\b", r"\bmenorrhagia\b"],
        "imaging": {},
        "labs": {"Microcytic Anaemia": 4.0, "Thrombocytosis": 1.2},
        "against_imaging": [],
        "workup": ["Serum ferritin and iron saturation", "Stool occult blood test / colonoscopy if age > 50", "Oral iron replacement therapy"],
    },
    "Macrocytic / Megaloblastic Anemia": {
        "symptoms": [r"\bneuropathy\b", r"\bparesthesias\b", r"\bataxia\b", r"\bglossitis\b", r"\balcohol\b"],
        "imaging": {},
        "labs": {"Macrocytic Anaemia": 4.0, "Vitamin Deficiencies": 2.5},
        "against_imaging": [],
        "workup": ["Serum vitamin B12 and folate levels", "Serum methylmalonic acid and homocysteine", "Peripheral blood smear"],
    },
    "Normocytic Anemia of Chronic Disease": {
        "symptoms": [r"\bchronic\b", r"\brheumatoid\b", r"\bmalignancy\b", r"\bautoimmune\b", r"\bweight loss\b"],
        "imaging": {},
        "labs": {"Normocytic Anaemia": 3.0, "Raised Inflammatory Markers": 2.0},
        "against_imaging": [],
        "workup": ["Reticulocyte count", "Ferritin and transferrin saturation", "Underlying chronic disease evaluation"],
    },
    "Type 2 Diabetes Mellitus": {
        "symptoms": [r"\bpolyuria\b", r"\bpolydipsia\b", r"\bpolyphagia\b", r"\bblurred vision\b", r"\bweight loss\b"],
        "imaging": {},
        "labs": {"Diabetic Range Glycemia": 4.0, "Pre-diabetic Range": 2.5, "Dyslipidaemia": 1.0},
        "against_imaging": [],
        "workup": ["Fasting lipid panel", "Spot urine microalbumin", "Diabetic dilated eye examination"],
    },
    "Diabetic Ketoacidosis": {
        "symptoms": [r"\bkussmaul\b", r"\bfruity breath\b", r"\bvomi.*abdominal\b", r"\bdehydration\b", r"\baltered mental\b"],
        "imaging": {},
        "labs": {"Hyperglycaemia with Low Bicarbonate (DKA Pattern)": 5.0, "Metabolic Acidosis": 3.0, "Diabetic Range Glycemia": 2.0},
        "against_imaging": [],
        "workup": ["Arterial/venous blood gas (pH, lactate)", "Serum beta-hydroxybutyrate and ketones", "Hourly glucose and electrolyte monitoring"],
    },
    "Hypoglycemia": {
        "symptoms": [r"\bsweating\b", r"\bdiaphoresis\b", r"\btremor\b", r"\bconfusion\b", r"\btachycardia\b", r"\bhunger\b"],
        "imaging": {},
        "labs": {"Hypoglycaemia": 4.5},
        "against_imaging": [],
        "workup": ["Immediate bedside capillary blood glucose", "Review insulin and secretagogue doses", "Endocrine workup if non-diabetic"],
    },
    "Acute Hepatocellular Injury": {
        "symptoms": [r"\bjaundice\b", r"\bhepatic\b", r"\bacetaminophen\b", r"\bhepatomegaly\b", r"\bruq pain\b", r"\bitching\b"],
        "imaging": {},
        "labs": {"Hepatocellular Pattern": 4.5, "Hyperbilirubinaemia": 2.0, "Prolonged INR": 1.5},
        "against_imaging": [],
        "workup": ["Viral hepatitis serologies (HAV, HBV, HCV)", "Serum acetaminophen level and toxicology screen", "Coagulation profile (PT/INR)"],
    },
    "Cholestasis / Obstructive Biliary Disease": {
        "symptoms": [r"\bdark urine\b", r"\bpale stool\b", r"\bicterus\b", r"\bpruritus\b", r"\bruq tenderness\b", r"\bgallstones\b"],
        "imaging": {},
        "labs": {"Cholestatic Pattern": 4.0, "Hyperbilirubinaemia": 2.5},
        "against_imaging": [],
        "workup": ["Right upper quadrant abdominal ultrasound", "Magnetic resonance cholangiopancreatography (MRCP)", "Gastroenterology consultation"],
    },
    "Acute Pancreatitis": {
        "symptoms": [r"\bepigastric pain\b", r"\bradiat.*back\b", r"\bnausea\b", r"\bvomi.*pain\b", r"\bgallstones\b", r"\balcohol\b"],
        "imaging": {"Atelectasis": 1.0, "Effusion": 1.0},
        "labs": {"Raised Inflammatory Markers": 1.5, "Metabolic Acidosis": 1.0},
        "against_imaging": [],
        "workup": ["Serum lipase and amylase", "Contrast-enhanced CT abdomen if unresolving after 48-72h", "Abdominal ultrasound for gallstones"],
    },
    "Acute Myocardial Infarction / ACS": {
        "symptoms": [r"\bcrushing.*chest\b", r"\bpressure.*chest\b", r"\bradiat.*arm\b", r"\bradiat.*jaw\b", r"\bdiaphoresis\b", r"\bangina\b"],
        "imaging": {"Cardiomegaly": 1.5, "Edema": 2.0},
        "labs": {"Raised Troponin": 4.5},
        "against_imaging": [],
        "workup": ["Serial 12-lead ECGs (every 15-30 mins)", "Serial high-sensitivity troponins at 0h and 1-2h", "Emergent cardiology consultation for catheterization"],
    },
    "Pulmonary Embolism": {
        "symptoms": [r"\bsudden.*dyspnea\b", r"\bpleuritic.*chest\b", r"\btachypnea\b", r"\bhemoptysis\b", r"\bdvt\b", r"\bleg swelling\b"],
        "imaging": {"Atelectasis": 1.5, "Effusion": 1.5},
        "labs": {"Raised Troponin": 1.5},
        "against_imaging": [],
        "workup": ["CT Pulmonary Angiography (CTPA)", "D-dimer assay if low-intermediate probability", "Lower extremity venous duplex ultrasound"],
    },
    "Sepsis / Severe Systemic Infection": {
        "symptoms": [r"\bseptic\b", r"\bhypotension\b", r"\btachycardia\b", r"\bfever\b", r"\bhypothermia\b", r"\bshivering\b", r"\baltered\b"],
        "imaging": {"Consolidation": 2.0},
        "labs": {"Raised Lactate": 4.0, "Leukocytosis with Neutrophilia": 2.5, "Leukopenia": 2.5, "Raised Inflammatory Markers": 2.5, "Renal Impairment": 2.0},
        "against_imaging": [],
        "workup": ["Blood cultures x 2 sets", "Serial serum lactate every 2-4 hours", "Source-directed imaging and prompt broad-spectrum antibiotics"],
    },
    "COPD Acute Exacerbation": {
        "symptoms": [r"\bcopd\b", r"\bwheezing\b", r"\bbarrel chest\b", r"\bsmoking\b", r"\bincreased sputum\b", r"\bdyspnea\b"],
        "imaging": {},
        "labs": {"Leukocytosis": 1.5},
        "against_imaging": ["Consolidation"],
        "workup": ["Arterial blood gas to check for CO2 retention", "Inhaled bronchodilators with spacer/nebulizer", "Oral corticosteroid burst"],
    },
    "Hypercalcemia": {
        "symptoms": [r"\bconstipation\b", r"\bbone pain\b", r"\bkidney stones\b", r"\bpolyuria\b", r"\bdepression\b"],
        "imaging": {},
        "labs": {"Hypercalcaemia": 4.5},
        "against_imaging": [],
        "workup": ["Intact parathyroid hormone (iPTH)", "PTH-related peptide (PTHrP) and 25-OH Vitamin D", "Aggressive isotonic IV saline hydration"],
    },
    "Metabolic Acidosis": {
        "symptoms": [r"\btachypnea\b", r"\bkussmaul\b", r"\blethargy\b", r"\bvomiting\b", r"\bconfusion\b"],
        "imaging": {},
        "labs": {"Metabolic Acidosis": 4.0, "Raised Lactate": 2.5, "Renal Impairment": 2.0},
        "against_imaging": [],
        "workup": ["Calculate serum anion gap [Na - (Cl + HCO3)]", "Arterial blood gas for pH and pCO2", "Treat underlying etiology (sepsis, DKA, uremia, toxin)"],
    },
};


def run_rule_engine(
    case_input: CaseInput,
    imaging_report: Optional[ImagingReport] = None,
    lab_report: Optional[LabReport] = None,
    rag_report: Optional[RagReport] = None,
) -> ClinicalDraft:
    """Execute weighted scoring rule engine evaluating >=25 clinical conditions."""
    text_lower = case_input.clinical_text.lower()

    # Pre-parse lab pattern names
    lab_pattern_names = set()
    if lab_report and lab_report.status == "ok":
        lab_pattern_names = {p.name for p in lab_report.patterns}

    # Pre-parse imaging findings
    imaging_findings_dict: Dict[str, float] = {}
    if imaging_report and imaging_report.status == "ok":
        for f in imaging_report.findings:
            imaging_findings_dict[f.finding] = f.probability

    scored_conditions: List[Dict[str, Any]] = []

    for cond_name, rules in CONDITIONS_CATALOG.items():
        score = 0.0
        supporting_items: List[str] = []
        data_sources: Set[str] = set()

        # 1. Symptom matches in text
        for pat in rules.get("symptoms", []):
            if re.search(pat, text_lower):
                score += 1.0
                data_sources.add("notes")
                matched_word = re.findall(pat, text_lower)[0]
                if matched_word not in supporting_items:
                    supporting_items.append(matched_word.strip())

        # 2. Imaging matches
        for img_finding, weight in rules.get("imaging", {}).items():
            for detected_finding, prob in imaging_findings_dict.items():
                if img_finding.lower() in detected_finding.lower():
                    pts = weight * prob
                    score += pts
                    data_sources.add("imaging")
                    supporting_items.append(f"{detected_finding} (p={prob:.1%})")

        # 3. Lab pattern matches
        for lab_pat, weight in rules.get("labs", {}).items():
            matched_pat = next((p for p in lab_pattern_names if lab_pat.lower() in p.lower() or p.lower() in lab_pat.lower()), None)
            if matched_pat:
                score += weight
                data_sources.add("labs")
                supporting_items.append(matched_pat)

        # Check discordant / against findings
        against_notes = []
        for ag_img in rules.get("against_imaging", []):
            if ag_img in imaging_findings_dict:
                against_notes.append(f"Presence of {ag_img}")

        if score > 0.8:
            scored_conditions.append({
                "condition": cond_name,
                "score": score,
                "sources_count": len(data_sources),
                "supporting": supporting_items[:5],
                "against": "; ".join(against_notes) if against_notes else "",
                "workup": rules.get("workup", []),
            })

    # Sort descending by score
    scored_conditions.sort(key=lambda x: x["score"], reverse=True)

    # Must provide 2 to 5 differentials
    top_items = scored_conditions[:5]
    if len(top_items) < 2:
        # Add fallback default if sparse
        if not any(item["condition"] == "Acute Viral Respiratory Infection" for item in top_items):
            top_items.append({
                "condition": "Acute Viral Respiratory Infection",
                "score": 1.0,
                "sources_count": 1,
                "supporting": ["Clinical presentation"],
                "against": "Focal bacterial consolidation absent",
                "workup": ["Supportive care", "Symptomatic treatment"],
            })

    # Likelihood assignment:
    # High only when score >= 3.5, ranked at top, and supported by >= 2 data sources
    differentials: List[Differential] = []
    top_score = top_items[0]["score"] if top_items else 0.0

    # Build evidence map from RAG report
    available_citations: List[Evidence] = rag_report.evidence if rag_report else []

    for rank, item in enumerate(top_items):
        is_top = (rank == 0 or (item["score"] >= top_score * 0.85))
        if item["score"] >= 3.5 and is_top and item["sources_count"] >= 2:
            likelihood = "high"
        elif item["score"] >= 2.0:
            likelihood = "moderate"
        else:
            likelihood = "low"

        # Match evidence citations
        cited_ids: List[str] = []
        cond_words = set(re.findall(r"\w+", item["condition"].lower()))
        for ev in available_citations:
            ev_words = set(re.findall(r"\w+", (ev.source + " " + ev.section + " " + ev.passage).lower()))
            if len(cond_words.intersection(ev_words)) >= 1:
                cited_ids.append(ev.id)
                if len(cited_ids) >= 2:
                    break

        rationale = f"Presentation supported by {', '.join(item['supporting']) if item['supporting'] else 'clinical features'}."
        if item["against"]:
            rationale += f" Discordant factors: {item['against']}."

        differentials.append(
            Differential(
                condition=item["condition"],
                likelihood=likelihood,
                rationale=rationale,
                supporting_findings=item["supporting"],
                evidence_ids=cited_ids,
                against=item["against"],
            )
        )

    # Synthesis summary (<= 4 sentences)
    top_cond = differentials[0].condition
    summary_sentences = [
        f"Patient evaluation demonstrates clinical features consistent with {top_cond}.",
        f"Diagnostic synthesis is supported by {len(differentials[0].supporting_findings)} key multimodal findings.",
        f"Differential considerations include {', '.join([d.condition for d in differentials[1:3]])}." if len(differentials) > 1 else "",
        "Diagnostic workup and clinical monitoring are indicated to confirm etiology and guide definitive management.",
    ]
    summary = " ".join([s for s in summary_sentences if s])

    # Workup aggregate
    combined_workup = []
    for item in top_items:
        for w in item.get("workup", []):
            if w not in combined_workup:
                combined_workup.append(w)
            if len(combined_workup) >= 4:
                break

    # Red flags
    red_flags = []
    if "Severe Hyperkalemia" in [d.condition for d in differentials]:
        red_flags.append("Critical hyperkalemia with lethal arrhythmia risk. Urgent ECG and stabilization required.")
    if "Diabetic Ketoacidosis" in [d.condition for d in differentials]:
        red_flags.append("Potential diabetic ketoacidosis with metabolic acidosis requiring immediate resuscitation.")
    if "Pneumothorax" in [d.condition for d in differentials]:
        red_flags.append("Possible pneumothorax; evaluate immediately for tension signs and acute decompensation.")
    if any("troponin" in sf.lower() for d in differentials for sf in d.supporting_findings):
        red_flags.append("Myocardial necrosis marker elevation; evaluate for acute coronary syndrome.")

    return ClinicalDraft(
        summary=summary,
        differentials=differentials,
        suggested_workup=combined_workup,
        red_flags=red_flags,
        uncertainties=["Awaiting confirmatory microbiology or follow-up laboratory testing."],
        mode="rule_engine",
    )


def run_reasoning_agent(
    case_input: CaseInput,
    imaging_report: Optional[ImagingReport] = None,
    lab_report: Optional[LabReport] = None,
    rag_report: Optional[RagReport] = None,
    feedback: Optional[str] = None,
) -> ClinicalDraft:
    """Execute Clinical Reasoning Agent using LLM if configured, otherwise rule engine fallback."""
    # Check if remote LLM provider is active
    if settings.LLM_PROVIDER.lower() != "offline":
        prompt = _build_llm_prompt(case_input, imaging_report, lab_report, rag_report, feedback)
        draft = call_llm_json(prompt, ClinicalDraft, max_retries=2)
        if draft is not None:
            draft.mode = "llm"
            return draft
        logger.warning("Remote LLM reasoning failed or unavailable; invoking deterministic rule engine.")

    # Rule-engine fallback
    return run_rule_engine(case_input, imaging_report, lab_report, rag_report)


def _build_llm_prompt(
    case_input: CaseInput,
    imaging_report: Optional[ImagingReport],
    lab_report: Optional[LabReport],
    rag_report: Optional[RagReport],
    feedback: Optional[str],
) -> str:
    """Construct strict JSON schema prompt for clinical reasoning LLM."""
    evidence_text = "\n".join([f"[{e.id}] {e.source} - {e.section}: {e.passage}" for e in (rag_report.evidence if rag_report else [])])
    img_text = "\n".join([f"- {f.finding}: {f.probability:.1%}, Location: {f.location}" for f in (imaging_report.findings if imaging_report else [])])
    lab_text = "\n".join([f"- {r.test}: {r.value} {r.unit} ({r.flag})" for r in (lab_report.results if lab_report else [])])

    return f"""
You are MedAgent's Clinical Reasoning specialist. Synthesize the provided multimodal clinical data into a structured ClinicalDraft JSON.
Rules:
1. Use ONLY the provided data.
2. Cite ONLY existing evidence IDs from the Evidence list ({[e.id for e in (rag_report.evidence if rag_report else [])]}).
3. Use cautious phrasing (e.g. 'suggests', 'consistent with'). NEVER state a diagnosis as certain.
4. DO NOT provide drug doses or dosing schedules.
5. Return between 2 and 5 ranked differential diagnoses.
6. Summary must be 4 sentences or fewer.
7. List discordant findings as uncertainties.
8. Red flags must highlight acute life threats.
{f'Feedback from Safety Review: {feedback}' if feedback else ''}

Patient: Age={case_input.patient_age}, Sex={case_input.patient_sex}
Clinical Notes:
{case_input.clinical_text}

Imaging Findings:
{img_text or 'None'}

Laboratory Results:
{lab_text or 'None'}

Retrieved Literature Evidence:
{evidence_text or 'None'}
"""
