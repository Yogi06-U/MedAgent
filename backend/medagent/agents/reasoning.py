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

# Clinical disease profiles detailing underlying pathophysiology and mechanism
CONDITION_CLINICAL_PROFILES: Dict[str, Dict[str, str]] = {
    "Community-Acquired Pneumonia": {
        "mechanism": "Acute microbial infection triggering inflammatory exudation and leukocyte accumulation in pulmonary alveolar spaces, causing parenchymal consolidation and impaired oxygen diffusion.",
    },
    "Pleural Effusion": {
        "mechanism": "Pathological accumulation of fluid in the pleural cavity due to elevated hydrostatic pressure, decreased oncotic pressure, or increased capillary permeability from pleural inflammation or malignancy.",
    },
    "Pneumothorax": {
        "mechanism": "Collection of free air in the pleural space disrupting negative intrapleural pressure, resulting in partial or complete ipsilateral lung collapse and ventilation-perfusion mismatch.",
    },
    "Congestive Heart Failure / Pulmonary Edema": {
        "mechanism": "Left ventricular systolic or diastolic failure elevating left atrial and pulmonary venous pressures, driving transudation of fluid into pulmonary interstitial and alveolar compartments.",
    },
    "Cardiomegaly": {
        "mechanism": "Pathological enlargement of the cardiac silhouette resulting from compensatory myocardial hypertrophy or ventricular cavity dilation secondary to chronic pressure or volume overload.",
    },
    "Acute Kidney Injury": {
        "mechanism": "Abrupt decline in glomerular filtration rate leading to retention of nitrogenous wastes (azotemia), dysregulation of fluid balance, and impaired electrolyte and acid-base excretion.",
    },
    "Chronic Kidney Disease": {
        "mechanism": "Progressive, irreversible structural or functional renal parenchyma damage lasting >3 months characterized by nephron loss, glomerulosclerosis, tubulointerstitial fibrosis, and loss of endocrine kidney function.",
    },
    "Severe Hyperkalemia": {
        "mechanism": "Critical extracellular potassium elevation shifting cardiac resting membrane potential toward threshold, causing delayed cardiac conduction, peaked T waves, and high risk of ventricular fibrillation or asystole.",
    },
    "Hypokalemia": {
        "mechanism": "Depletion of serum potassium causing hyperpolarization of excitable cell membranes, impaired neuromuscular transmission, cardiac repolarization abnormalities (prominent U waves), and increased arrhythmia susceptibility.",
    },
    "Hyponatremia": {
        "mechanism": "Hypotonic imbalance of body water relative to sodium content producing an osmotic gradient that drives water into intracellular compartments, predisposing to cerebral astrocyte edema and neurological dysfunction.",
    },
    "Microcytic Iron Deficiency Anemia": {
        "mechanism": "Inadequate iron stores restricting erythroblast heme and hemoglobin synthesis, resulting in microcytic, hypochromic erythrocytes with reduced systemic oxygen delivery.",
    },
    "Macrocytic / Megaloblastic Anemia": {
        "mechanism": "Deficiency of vitamin B12 or folate disrupting purine/thymidylate synthesis and erythroblast nuclear maturation, leading to nuclear-cytoplasmic dyssynchrony and abnormally large erythrocytes.",
    },
    "Normocytic Anemia of Chronic Disease": {
        "mechanism": "Chronic inflammation-induced hepcidin elevation sequestering iron inside macrophages, blunting renal erythropoietin response, and shortening red blood cell survival.",
    },
    "Type 2 Diabetes Mellitus": {
        "mechanism": "Chronic metabolic disorder characterized by peripheral insulin resistance combined with progressive pancreatic beta-cell secretory failure, causing sustained hyperglycemia and micro-/macrovascular risk.",
    },
    "Diabetic Ketoacidosis": {
        "mechanism": "Profound insulin deficiency with excessive counter-regulatory hormone surge driving unchecked lipolysis, hepatic free fatty acid oxidation into ketoacids, high anion gap metabolic acidosis, and severe osmotic dehydration.",
    },
    "Hypoglycemia": {
        "mechanism": "Critically low blood glucose (<70 mg/dL) depriving brain tissue of obligatory energy substrate, initiating autonomic counter-regulatory catecholamine discharge and neuroglycopenic symptoms.",
    },
    "Acute Hepatocellular Injury": {
        "mechanism": "Direct toxic, ischemic, or immunologic destruction of hepatocytes causing leakage of intracellular aminotransferases (ALT, AST) into systemic circulation and acute hepatic metabolic decompensation.",
    },
    "Cholestasis / Obstructive Biliary Disease": {
        "mechanism": "Mechanical obstruction or biochemical failure of biliary secretion resulting in retrograde retention of conjugated bilirubin, bile acids, and alkaline phosphatase in hepatic parenchyma and systemic circulation.",
    },
    "Acute Pancreatitis": {
        "mechanism": "Premature intrapancreatic trypsinogen activation within acinar cells causing enzymatic autodigestion, local tissue necrosis, vascular damage, and acute systemic inflammatory mediator release.",
    },
    "Acute Myocardial Infarction / ACS": {
        "mechanism": "Coronary atherosclerotic plaque rupture with occlusive thrombosis causing transmural or subendocardial ischemia, cardiomyocyte necrosis, and release of cardiac troponin into circulation.",
    },
    "Pulmonary Embolism": {
        "mechanism": "Mechanical occlusion of pulmonary arterial vasculature by dislodged venous thromboemboli causing alveolar dead space expansion, right ventricular afterload spike, and hypoxemia.",
    },
    "Sepsis / Severe Systemic Infection": {
        "mechanism": "Dysregulated, life-threatening host inflammatory and endothelial response to infection causing diffuse microvascular leakage, cellular metabolic failure, tissue hypoperfusion, and multiorgan dysfunction.",
    },
    "COPD Acute Exacerbation": {
        "mechanism": "Acute intensification of lower airway inflammation and bronchial hyperresponsiveness causing dynamic hyperinflation, increased air trapping, mucus hypersecretion, and acute gas exchange deterioration.",
    },
    "Hypercalcemia": {
        "mechanism": "Elevation of extracellular calcium concentration secondary to increased osteoclastic bone resorption or intestinal absorption, lowering cell membrane excitability and impairing renal concentrating ability.",
    },
    "Metabolic Acidosis": {
        "mechanism": "Primary fall in serum bicarbonate concentration due to excessive accumulation of endogenous/exogenous fixed acids or loss of bicarbonate, prompting respiratory compensation with hyperventilation.",
    },
    "No Acute Cardiopulmonary Process Identified": {
        "mechanism": "Normal cardiopulmonary anatomical and physiological status without evidence of parenchymal infiltration, pleural fluid, pulmonary vascular congestion, or chamber dilation.",
    },
    "Preventative Health Maintenance / Baseline Normal": {
        "mechanism": "Physiological baseline within expected limits without evidence of active systemic illness, organ dysfunction, or acute metabolic decompensation.",
    },
    "Preventative Health Maintenance": {
        "mechanism": "Physiological baseline within expected limits without evidence of active systemic illness, organ dysfunction, or acute metabolic decompensation.",
    },
    "Acute Viral Bronchitis / Atypical Infection": {
        "mechanism": "Transient inflammatory response limited to the tracheobronchial tree without dense alveolar consolidation or parenchymal destruction.",
    },
    "Hypertensive Heart Disease / Left Ventricular Hypertrophy": {
        "mechanism": "Concentric remodeling and hypertrophy of left ventricular cardiomyocytes in response to sustained systemic vascular afterload.",
    },
    "Prerenal Azotemia / Dehydration": {
        "mechanism": "Decreased effective circulating renal blood volume reducing glomerular perfusion pressure without structural parenchymal or tubular epithelial injury.",
    },
    "Unstable Angina / Acute Coronary Syndrome": {
        "mechanism": "Transient coronary artery hypoperfusion causing reversible myocardial ischemia without permanent cardiomyocyte necrosis or biomarker elevation.",
    },
    "Choledocholithiasis / Biliary Stricture": {
        "mechanism": "Partial or complete physical obstruction of the extrahepatic common bile duct by calculi or stricture causing proximal biliary hypertension and cholestasis.",
    },
    "Toxic / Drug-Induced Liver Injury (DILI)": {
        "mechanism": "Xenobiotic or drug-metabolite induced direct hepatocyte necrosis or idiosyncratic immunoallergic injury.",
    },
    "Acute Cholecystitis / Biliary Colic": {
        "mechanism": "Cystic duct obstruction by gallstones leading to gallbladder distension, mucosal ischemia, chemical inflammation, and secondary bacterial infection.",
    },
    "Pseudohyperkalemia / Hemolyzed Sample Artifact": {
        "mechanism": "In vitro mechanical rupture of erythrocytes during blood drawing or processing releasing intracellular potassium into plasma, producing artifactual elevation.",
    },
    "Gastrointestinal or Renal Potassium Wasting": {
        "mechanism": "Loss of total body potassium via urinary excretion (diuretics, hyperaldosteronism) or gastrointestinal tract (diarrhea, vomiting).",
    },
    "Syndrome of Inappropriate ADH (SIADH) vs Hypovolemia": {
        "mechanism": "Unregulated non-osmotic secretion of antidiuretic hormone leading to excessive water retention and dilutional hyponatremia.",
    },
    "Hyperosmolar Hyperglycemic State (HHS)": {
        "mechanism": "Severe relative insulin deficiency and profound hyperosmolality causing severe osmotic diuresis and dehydration without ketoacidosis.",
    },
    "Impaired Fasting Glucose / Metabolic Syndrome": {
        "mechanism": "Early insulin resistance with elevated hepatic glucose output and borderline beta-cell compensation.",
    },
    "Primary Hyperparathyroidism vs Malignancy": {
        "mechanism": "Excess autonomous parathyroid hormone secretion or tumor production of PTH-related protein stimulating bone osteoclast resorption and renal calcium reabsorption.",
    },
    "High Anion Gap vs Non-Anion Gap Metabolic Acidosis": {
        "mechanism": "Acid-base disturbance driven either by unmeasured organic anions (lactate, ketones) or hyperchloremic bicarbonate wasting.",
    },
    "Thalassemia Minor / Anemia of Chronic Disease": {
        "mechanism": "Genetic reduction in globin chain synthesis or inflammatory iron restriction producing disproportionate microcytosis.",
    },
    "Vitamin B12 or Folate Deficiency Anemia": {
        "mechanism": "Impaired DNA synthesis in hematopoietic precursors causing megaloblastic maturation arrest and macrocytosis.",
    },
    "Anemia of Chronic Kidney Disease or Inflammation": {
        "mechanism": "Blunted renal erythropoietin secretion compounded by inflammatory hepcidin suppression of iron availability.",
    },
    "Systemic Inflammatory Response Syndrome (SIRS) of Non-Infectious Origin": {
        "mechanism": "Widespread non-microbial systemic cytokine release (e.g. pancreatitis, trauma, tissue ischemia) causing autonomic and endothelial activation.",
    },
}

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
        "symptoms": [r"\bcardiomegaly\b", r"\benlarged heart\b", r"\bcardiothoracic ratio\b", r"\bdisplaced apex\b", r"\bdisplaced apical impulse\b", r"\bdyspnea on exertion\b"],
        "imaging": {"Cardiomegaly": 4.0},
        "labs": {},
        "against_imaging": [],
        "workup": ["12-lead electrocardiogram", "Transthoracic echocardiogram to assess chamber size", "Blood pressure log"],
    },
    "Acute Kidney Injury": {
        "symptoms": [r"\boliguria\b", r"\banuria\b", r"\bdecreased urine\b", r"\bacute kidney\b", r"\baki\b", r"\banuric\b", r"\boliguric\b", r"\bfluid overload\b"],
        "imaging": {},
        "labs": {"Renal Impairment": 4.0, "Hyperkalaemia": 2.5, "Metabolic Acidosis": 2.0},
        "against_imaging": [],
        "workup": ["Renal ultrasound to exclude hydronephrosis", "Urine electrolytes and fractional excretion of sodium", "Discontinue nephrotoxic agents"],
    },
    "Chronic Kidney Disease": {
        "symptoms": [r"\bchronic kidney\b", r"\bckd\b", r"\bnocturia\b", r"\bpruritus\b", r"\begfr\b"],
        "imaging": {},
        "labs": {"Renal Impairment": 3.0, "Normocytic Anaemia": 1.5, "Hyperkalaemia": 1.5},
        "against_imaging": [],
        "workup": ["Spot urine albumin-to-creatinine ratio", "Serial serum creatinine tracking", "Nephrology referral"],
    },
    "Severe Hyperkalemia": {
        "symptoms": [r"\bflaccid weakness\b", r"\bmuscle weakness\b", r"\bpeaked t[- ]waves?\b", r"\bhyperkalemia\b", r"\bhyperkalaemia\b", r"\bmissed.*dialysis\b", r"\bpalpitations\b", r"\bparesthesias\b"],
        "imaging": {},
        "labs": {"Hyperkalaemia": 4.5, "Renal Impairment": 2.0, "Metabolic Acidosis": 1.5},
        "against_imaging": [],
        "workup": ["Immediate 12-lead ECG for peaked T waves/conduction blocks", "Telemetry cardiac monitoring", "Discontinue ACE inhibitors, spironolactone, and NSAIDs"],
    },
    "Hypokalemia": {
        "symptoms": [r"\bhypokalemia\b", r"\bhypokalaemia\b", r"\bmuscle cramps\b", r"\bmuscle weakness\b", r"\bu[- ]waves?\b", r"\bfurosemide\b", r"\bloop diuretic\b"],
        "imaging": {},
        "labs": {"Hypokalaemia": 4.0},
        "against_imaging": [],
        "workup": ["Serum magnesium level", "12-lead ECG for U waves and ST changes", "Review diuretic regimen"],
    },
    "Hyponatremia": {
        "symptoms": [r"\bhyponatremia\b", r"\bhyponatraemia\b", r"\bconfusion\b", r"\blethargy\b", r"\bheadache\b", r"\bseizure\b", r"\bunsteady\b"],
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
        "symptoms": [r"\banemia of chronic\b", r"\banemia of inflammation\b", r"\brheumatoid\b", r"\bautoimmune flare\b", r"\bmalignancy\b", r"\bchronic inflammation\b", r"\bweight loss\b"],
        "imaging": {},
        "labs": {"Normocytic Anaemia": 3.0, "Raised Inflammatory Markers": 2.0},
        "against_imaging": [],
        "workup": ["Reticulocyte count", "Ferritin and transferrin saturation", "Underlying chronic disease evaluation"],
    },
    "Type 2 Diabetes Mellitus": {
        "symptoms": [r"\bpolyuria\b", r"\bpolydipsia\b", r"\bpolyphagia\b", r"\bblurred vision\b", r"\bweight loss\b", r"\bhyperglycemia\b"],
        "imaging": {},
        "labs": {"Diabetic Range Glycemia": 4.0, "Pre-diabetic Range": 2.5, "Dyslipidaemia": 1.0},
        "against_imaging": [],
        "workup": ["Fasting lipid panel", "Spot urine microalbumin", "Diabetic dilated eye examination"],
    },
    "Diabetic Ketoacidosis": {
        "symptoms": [r"\bkussmaul\b", r"\bfruity breath\b", r"\bvomi.*abdominal\b", r"\bdehydration\b", r"\baltered mental\b", r"\btype 1 diabetes\b"],
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
        "symptoms": [r"\bseptic\b", r"\bsepsis\b", r"\bsevere hypotension\b", r"\bhypotension\b", r"\bseptic shock\b", r"\brigors?\b", r"\bshivering\b", r"\bhypothermia\b", r"\btachycardia\b"],
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
        "symptoms": [r"\bhypercalcemia\b", r"\bhypercalcaemia\b", r"\bbone pain\b", r"\bkidney stones\b", r"\bnephrolithiasis\b", r"\bshortened qt\b"],
        "imaging": {},
        "labs": {"Hypercalcaemia": 4.5},
        "against_imaging": [],
        "workup": ["Intact parathyroid hormone (iPTH)", "PTH-related peptide (PTHrP) and 25-OH Vitamin D", "Aggressive isotonic IV saline hydration"],
    },
    "Metabolic Acidosis": {
        "symptoms": [r"\bkussmaul\b", r"\bacidosis\b", r"\bacidotic\b", r"\bhigh anion gap\b", r"\blow bicarbonate\b", r"\btachypnea\b"],
        "imaging": {},
        "labs": {"Metabolic Acidosis": 4.0, "Raised Lactate": 2.5, "Renal Impairment": 2.0},
        "against_imaging": [],
        "workup": ["Calculate serum anion gap [Na - (Cl + HCO3)]", "Arterial blood gas for pH and pCO2", "Treat underlying etiology (sepsis, DKA, uremia, toxin)"],
    },
};

# Medically coherent secondary differentials by condition
CONTEXTUAL_SECONDARY_DIFFERENTIALS: Dict[str, Dict[str, Any]] = {
    # Respiratory
    "Community-Acquired Pneumonia": {
        "condition": "Acute Viral Bronchitis / Atypical Infection",
        "supporting": ["Respiratory presentation", "Cough and dyspnea"],
        "against": "Lacks dense focal lobar consolidation",
        "workup": ["Respiratory viral multiplex PCR", "Repeat chest radiograph in 6 weeks", "Pulse oximetry"],
    },
    "Pleural Effusion": {
        "condition": "Congestive Heart Failure / Transudative Effusion",
        "supporting": ["Decreased breath sounds", "Pleuritic discomfort"],
        "against": "Peripheral dependent edema absent",
        "workup": ["Diagnostic thoracentesis with Light's criteria", "Transthoracic echocardiogram", "Serum NT-proBNP"],
    },
    "Pneumothorax": {
        "condition": "Acute Pleuritis / Musculoskeletal Chest Pain",
        "supporting": ["Acute pleuritic chest discomfort", "Dyspnea"],
        "against": "Mediastinal deviation absent",
        "workup": ["Repeat upright chest radiograph", "Continuous pulse oximetry", "Electrocardiogram"],
    },
    "COPD Acute Exacerbation": {
        "condition": "Acute Bronchitis / Tracheobronchitis",
        "supporting": ["Wheezing and dyspnea", "Cough"],
        "against": "Focal bacterial consolidation absent",
        "workup": ["Spirometry post-stabilization", "Peak expiratory flow measurement", "Chest radiograph"],
    },
    # Cardiac
    "Acute Myocardial Infarction / ACS": {
        "condition": "Unstable Angina / Acute Coronary Syndrome",
        "supporting": ["Anginal chest discomfort", "Diaphoresis and radiation"],
        "against": "Persistent ST elevation absent",
        "workup": ["Serial high-sensitivity troponins at 0h and 2h", "Serial 12-lead ECGs", "Urgent cardiology evaluation"],
    },
    "Congestive Heart Failure / Pulmonary Edema": {
        "condition": "Acute Exacerbation of Diastolic Dysfunction",
        "supporting": ["Fluid overload features", "Orthopnea and dyspnea"],
        "against": "Acute valvular regurgitation murmur absent",
        "workup": ["Serum NT-proBNP", "Transthoracic echocardiogram", "Renal function panel"],
    },
    "Cardiomegaly": {
        "condition": "Hypertensive Heart Disease / Left Ventricular Hypertrophy",
        "supporting": ["Cardiomegaly on imaging", "Dyspnea on exertion"],
        "against": "Pericardial effusion absent",
        "workup": ["12-lead ECG", "Transthoracic echocardiogram", "Ambulatory blood pressure monitoring"],
    },
    # Renal & Electrolytes
    "Acute Kidney Injury": {
        "condition": "Prerenal Azotemia / Dehydration",
        "supporting": ["Decreased urine output", "Elevated renal markers"],
        "against": "Active urinary sediment casts absent",
        "workup": ["Fractional excretion of sodium (FeNa)", "Renal ultrasound", "Fluid challenge responsiveness"],
    },
    "Chronic Kidney Disease": {
        "condition": "Hypertensive Nephrosclerosis",
        "supporting": ["Chronically elevated creatinine", "Hypertension history"],
        "against": "Active glomerulonephritis serologies absent",
        "workup": ["Urine albumin-to-creatinine ratio", "Renal duplex ultrasound", "Nephrology consultation"],
    },
    "Severe Hyperkalemia": {
        "condition": "Pseudohyperkalemia / Hemolyzed Sample Artifact",
        "supporting": ["Elevated serum potassium", "Neuromuscular symptoms"],
        "against": "Peaked T waves on baseline ECG absent",
        "workup": ["Immediate unhemolyzed repeat stat potassium", "Stat 12-lead ECG", "Discontinue potassium supplements"],
    },
    "Hypokalemia": {
        "condition": "Gastrointestinal or Renal Potassium Wasting",
        "supporting": ["Hypokalemia", "Muscle weakness or cramps"],
        "against": "Hyperaldosteronism features absent",
        "workup": ["Urine potassium-to-creatinine ratio", "Serum magnesium", "Review diuretic regimen"],
    },
    "Hyponatremia": {
        "condition": "Syndrome of Inappropriate ADH (SIADH) vs Hypovolemia",
        "supporting": ["Hyponatremia", "Neurological symptoms"],
        "against": "Severe volume depletion orthostasis absent",
        "workup": ["Serum and urine osmolality", "Urine sodium concentration", "Thyroid and adrenal function tests"],
    },
    # Endocrine & Metabolic
    "Diabetic Ketoacidosis": {
        "condition": "Hyperosmolar Hyperglycemic State (HHS)",
        "supporting": ["Marked hyperglycemia", "Dehydration and altered mentation"],
        "against": "Profound hyperosmolality without acidosis absent",
        "workup": ["Serum beta-hydroxybutyrate", "Arterial/venous blood gas", "Serum osmolality"],
    },
    "Type 2 Diabetes Mellitus": {
        "condition": "Impaired Fasting Glucose / Metabolic Syndrome",
        "supporting": ["Hyperglycemia", "Polyuria and polydipsia"],
        "against": "Diabetic ketoacidosis features absent",
        "workup": ["Fasting plasma glucose", "HbA1c confirmation", "Lipid panel"],
    },
    "Hypoglycemia": {
        "condition": "Drug-Induced Hypoglycemia / Autonomic Dysregulation",
        "supporting": ["Autonomic and neuroglycopenic symptoms", "Low blood glucose"],
        "against": "Insulinoma triad documentation absent",
        "workup": ["Immediate capillary glucose test", "Medication reconciliation", "Serum C-peptide and insulin level"],
    },
    "Hypercalcemia": {
        "condition": "Primary Hyperparathyroidism vs Malignancy",
        "supporting": ["Elevated serum calcium", "Bone pain or nephrolithiasis"],
        "against": "Acute hypervitaminosis D absent",
        "workup": ["Intact parathyroid hormone (iPTH)", "PTH-related peptide (PTHrP)", "25-OH Vitamin D"],
    },
    "Metabolic Acidosis": {
        "condition": "High Anion Gap vs Non-Anion Gap Metabolic Acidosis",
        "supporting": ["Low bicarbonate", "Tachypnea and compensatory hyperventilation"],
        "against": "Severe respiratory alkalosis absent",
        "workup": ["Serum electrolytes for anion gap calculation", "Arterial blood gas", "Serum lactate and ketones"],
    },
    # GI & Hepatic
    "Acute Pancreatitis": {
        "condition": "Acute Cholecystitis / Biliary Colic",
        "supporting": ["Severe abdominal pain", "Nausea and vomiting"],
        "against": "Murphy's sign and gallbladder wall thickening absent",
        "workup": ["Serum lipase and amylase", "Abdominal ultrasound", "Liver function panel"],
    },
    "Acute Hepatocellular Injury": {
        "condition": "Toxic / Drug-Induced Liver Injury (DILI)",
        "supporting": ["Transaminitis", "Jaundice and hepatomegaly"],
        "against": "Biliary duct dilatation absent",
        "workup": ["Viral hepatitis panel (A, B, C, E)", "Toxicology screen including acetaminophen", "Abdominal ultrasound"],
    },
    "Cholestasis / Obstructive Biliary Disease": {
        "condition": "Choledocholithiasis / Biliary Stricture",
        "supporting": ["Elevated alkaline phosphatase and bilirubin", "Dark urine and pruritus"],
        "against": "Ascending cholangitis triad (fever) absent",
        "workup": ["Right upper quadrant ultrasound", "MRCP", "Serum lipase"],
    },
    # Hematology
    "Microcytic Iron Deficiency Anemia": {
        "condition": "Thalassemia Minor / Anemia of Chronic Disease",
        "supporting": ["Microcytic anemia", "Pallor and fatigue"],
        "against": "Target cells or basophilic stippling predominant absent",
        "workup": ["Serum ferritin and iron saturation", "Hemoglobin electrophoresis", "Fecal occult blood test"],
    },
    "Macrocytic / Megaloblastic Anemia": {
        "condition": "Vitamin B12 or Folate Deficiency Anemia",
        "supporting": ["Macrocytic anemia", "Neuropathy and ataxia"],
        "against": "Myelodysplastic dysplastic blasts absent",
        "workup": ["Serum B12 and red cell folate", "Methylmalonic acid", "Peripheral blood smear"],
    },
    "Normocytic Anemia of Chronic Disease": {
        "condition": "Anemia of Chronic Kidney Disease or Inflammation",
        "supporting": ["Normocytic anemia", "Chronic inflammatory condition"],
        "against": "Acute hemolytic anemia features absent",
        "workup": ["Reticulocyte production index", "Ferritin and transferrin saturation", "Serum erythropoietin"],
    },
    # Systemic & Vascular
    "Pulmonary Embolism": {
        "condition": "Acute Pleurisy / Musculoskeletal Chest Pain",
        "supporting": ["Pleuritic chest pain", "Tachypnea and dyspnea"],
        "against": "Deep vein thrombosis on venous ultrasound absent",
        "workup": ["CT Pulmonary Angiogram (CTPA)", "Quantitative D-dimer", "12-lead ECG"],
    },
    "Sepsis / Severe Systemic Infection": {
        "condition": "Systemic Inflammatory Response Syndrome (SIRS) of Non-Infectious Origin",
        "supporting": ["Tachycardia and hypotension", "Elevated inflammatory markers"],
        "against": "Identified positive microbial source culture absent",
        "workup": ["Blood cultures x2", "Serial serum lactate", "Broad-spectrum empiric antimicrobial coverage"],
    },
}

# Discriminating clinical concept keywords for grounded RAG evidence matching
CONDITION_CONCEPT_KEYWORDS: Dict[str, List[str]] = {
    "Community-Acquired Pneumonia": ["pneumonia", "consolidation", "infiltrate", "sputum", "bronchial"],
    "Pleural Effusion": ["pleural effusion", "effusion", "thoracentesis", "light's criteria", "pleural fluid"],
    "Pneumothorax": ["pneumothorax", "pleural air", "chest tube", "tension pneumothorax"],
    "Congestive Heart Failure / Pulmonary Edema": ["heart failure", "pulmonary edema", "chf", "bnp", "cardiomegaly", "orthopnea"],
    "Cardiomegaly": ["cardiomegaly", "cardiothoracic ratio", "enlarged heart", "ventricular hypertrophy"],
    "Acute Kidney Injury": ["acute kidney injury", "aki", "creatinine", "oliguria", "prerenal"],
    "Chronic Kidney Disease": ["chronic kidney disease", "ckd", "egfr", "nephropathy", "uremia"],
    "Severe Hyperkalemia": ["hyperkalemia", "hyperkalaemia", "potassium", "peaked t", "calcium gluconate"],
    "Hypokalemia": ["hypokalemia", "hypokalaemia", "potassium", "u waves"],
    "Hyponatremia": ["hyponatremia", "hyponatraemia", "sodium", "siadh", "osmolality"],
    "Microcytic Iron Deficiency Anemia": ["iron deficiency", "ferritin", "microcytic", "transferrin", "iron"],
    "Macrocytic / Megaloblastic Anemia": ["megaloblastic", "macrocytic", "vitamin b12", "folate", "homocysteine"],
    "Normocytic Anemia of Chronic Disease": ["anemia of chronic", "anemia of inflammation", "normocytic", "hepcidin"],
    "Type 2 Diabetes Mellitus": ["diabetes mellitus", "hyperglycemia", "glycemia", "hba1c", "metformin", "diabetic"],
    "Diabetic Ketoacidosis": ["diabetic ketoacidosis", "dka", "ketones", "ketoacidosis", "beta-hydroxybutyrate"],
    "Hypoglycemia": ["hypoglycemia", "hypoglycaemia", "blood glucose", "dextrose", "hypoglycemic"],
    "Acute Hepatocellular Injury": ["hepatocellular", "transaminitis", "hepatitis", "alt", "ast", "liver injury"],
    "Cholestasis / Obstructive Biliary Disease": ["cholestasis", "cholestatic", "biliary", "alkaline phosphatase", "bilirubin", "gallstones"],
    "Acute Pancreatitis": ["pancreatitis", "lipase", "amylase", "epigastric", "pancreatic"],
    "Acute Myocardial Infarction / ACS": ["myocardial infarction", "troponin", "coronary syndrome", "stemi", "nstemi", "angina"],
    "Pulmonary Embolism": ["pulmonary embolism", "d-dimer", "ctpa", "thromboembolism", "deep vein thrombosis"],
    "Sepsis / Severe Systemic Infection": ["sepsis", "septic", "lactate", "bacteremia", "qsofa"],
    "COPD Acute Exacerbation": ["copd", "chronic obstructive", "bronchodilator", "emphysema", "bronchitis"],
    "Hypercalcemia": ["hypercalcemia", "hypercalcaemia", "calcium", "parathyroid", "pth"],
    "Metabolic Acidosis": ["metabolic acidosis", "anion gap", "bicarbonate", "acidemia"],
}

# Essential cardinal diagnostic requirements for each condition to prevent spurious differential inclusion
CARDINAL_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "Severe Hyperkalemia": {
        "lab_patterns": ["Hyperkalaemia", "Severe Hyperkalaemia"],
        "symptoms": [r"\bhyperkalemia\b", r"\bhyperkalaemia\b", r"\bflaccid\b", r"\bpeaked t[- ]waves?\b"],
    },
    "Hypokalemia": {
        "lab_patterns": ["Hypokalaemia"],
        "symptoms": [r"\bhypokalemia\b", r"\bhypokalaemia\b", r"\bu[- ]waves?\b", r"\bfurosemide\b"],
    },
    "Hyponatremia": {
        "lab_patterns": ["Hyponatraemia"],
        "symptoms": [r"\bhyponatremia\b", r"\bhyponatraemia\b"],
    },
    "Hypercalcemia": {
        "lab_patterns": ["Hypercalcaemia"],
        "symptoms": [r"\bhypercalcemia\b", r"\bhypercalcaemia\b", r"\bbone pain\b", r"\bkidney stones\b"],
    },
    "Hypoglycemia": {
        "lab_patterns": ["Hypoglycaemia"],
        "symptoms": [r"\bhypoglycemia\b", r"\bhypoglycaemia\b"],
    },
    "Diabetic Ketoacidosis": {
        "lab_patterns": ["Hyperglycaemia with Low Bicarbonate (DKA Pattern)", "Metabolic Acidosis"],
        "symptoms": [r"\bkussmaul\b", r"\bfruity breath\b", r"\bdka\b", r"\bdiabetic ketoacidosis\b"],
    },
    "Type 2 Diabetes Mellitus": {
        "lab_patterns": ["Diabetic Range Glycemia", "Pre-diabetic Range"],
        "symptoms": [r"\bdiabetes\b", r"\bdiabetic\b", r"\bhyperglycemia\b", r"\bpolyuria\b"],
    },
    "Normocytic Anemia of Chronic Disease": {
        "lab_patterns": ["Normocytic Anaemia"],
        "symptoms": [r"\banemia of chronic\b", r"\banemia of inflammation\b", r"\brheumatoid\b", r"\bautoimmune\b"],
    },
    "Microcytic Iron Deficiency Anemia": {
        "lab_patterns": ["Microcytic Anaemia"],
        "symptoms": [r"\biron deficiency\b", r"\bferritin\b", r"\bpica\b", r"\bmenorrhagia\b"],
    },
    "Macrocytic / Megaloblastic Anemia": {
        "lab_patterns": ["Macrocytic Anaemia", "Vitamin Deficiencies"],
        "symptoms": [r"\bmacrocytic\b", r"\bmegaloblastic\b", r"\bglossitis\b"],
    },
    "Acute Kidney Injury": {
        "lab_patterns": ["Renal Impairment"],
        "symptoms": [r"\boliguria\b", r"\banuria\b", r"\bdecreased urine\b", r"\baki\b", r"\bacute kidney\b"],
    },
    "Chronic Kidney Disease": {
        "lab_patterns": ["Renal Impairment"],
        "symptoms": [r"\bchronic kidney\b", r"\bckd\b", r"\begfr\b"],
    },
    "Acute Hepatocellular Injury": {
        "lab_patterns": ["Hepatocellular Pattern"],
        "symptoms": [r"\bjaundice\b", r"\bhepatomegaly\b", r"\bhepatitis\b"],
    },
    "Cholestasis / Obstructive Biliary Disease": {
        "lab_patterns": ["Cholestatic Pattern"],
        "symptoms": [r"\bdark urine\b", r"\bpale stool\b", r"\bicterus\b", r"\bgallstones\b"],
    },
    "Acute Pancreatitis": {
        "symptoms": [r"\bepigastric pain\b", r"\bradiat.*back\b", r"\bpancreatitis\b", r"\blipase\b"],
    },
    "Acute Myocardial Infarction / ACS": {
        "lab_patterns": ["Raised Troponin"],
        "symptoms": [r"\bcrushing.*chest\b", r"\bpressure.*chest\b", r"\bradiat.*arm\b", r"\bradiat.*jaw\b", r"\bangina\b", r"\bmyocardial infarction\b"],
    },
    "Cardiomegaly": {
        "imaging": ["Cardiomegaly"],
        "symptoms": [r"\bcardiomegaly\b", r"\benlarged heart\b", r"\bcardiothoracic ratio\b"],
    },
    "Pneumothorax": {
        "imaging": ["Pneumothorax"],
        "symptoms": [r"\bsudden.*chest pain\b", r"\bhyperresonance\b", r"\bpneumothorax\b"],
    },
    "Pleural Effusion": {
        "imaging": ["Effusion"],
        "symptoms": [r"\bdullness\b", r"\bdecreased breath sounds\b", r"\bpleural effusion\b"],
    },
    "COPD Acute Exacerbation": {
        "symptoms": [r"\bcopd\b", r"\bwheezing\b", r"\bbarrel chest\b", r"\bsmoking\b", r"\bchronic bronchitis\b", r"\bemphysema\b"],
    },
    "Community-Acquired Pneumonia": {
        "imaging": ["Consolidation", "Infiltration", "Pneumonia"],
        "symptoms": [r"\bcough\b", r"\bsputum\b", r"\bcrackles\b", r"\brales\b", r"\bpneumonia\b"],
    },
    "Pulmonary Embolism": {
        "symptoms": [r"\bpleuritic.*chest\b", r"\bhemoptysis\b", r"\bdvt\b", r"\bleg swelling\b", r"\bpulmonary embolism\b"],
    },
    "Congestive Heart Failure / Pulmonary Edema": {
        "imaging": ["Cardiomegaly", "Edema"],
        "symptoms": [r"\borthopnea\b", r"\bpnd\b", r"\bparoxysmal nocturnal\b", r"\bjvd\b", r"\bheart failure\b"],
    },
    "Sepsis / Severe Systemic Infection": {
        "lab_patterns": ["Raised Lactate"],
        "symptoms": [r"\bseptic\b", r"\bsepsis\b", r"\bsevere hypotension\b", r"\bseptic shock\b", r"\brigors?\b"],
    },
    "Hypercalcemia": {
        "lab_patterns": ["Hypercalcaemia"],
        "symptoms": [r"\bhypercalcemia\b", r"\bhypercalcaemia\b", r"\bbone pain\b", r"\bkidney stones\b"],
    },
    "Metabolic Acidosis": {
        "lab_patterns": ["Metabolic Acidosis"],
        "symptoms": [r"\bkussmaul\b", r"\bacidosis\b", r"\bacidemia\b"],
    },
}


NEGATION_PATTERNS: List[str] = [
    r"\bno\b",
    r"\bdenies\b",
    r"\bwithout\b",
    r"\bfree of\b",
    r"\babsence of\b",
    r"\bnegative for\b",
    r"\bnot\b",
    r"\bunlikely\b",
    r"\brules? out\b",
    r"\bno signs of\b",
    r"\bno evidence of\b",
]


def is_symptom_negated(text: str, match_start: int) -> bool:
    """Check if a symptom in text is preceded by a negation phrase within 50 characters."""
    window_start = max(0, match_start - 50)
    preceding_window = text[window_start:match_start].lower()
    last_punct = max(preceding_window.rfind("."), preceding_window.rfind(";"), preceding_window.rfind("\n"))
    if last_punct != -1:
        preceding_window = preceding_window[last_punct:]
    for neg_pat in NEGATION_PATTERNS:
        if re.search(neg_pat, preceding_window):
            return True
    return False


def _check_cardinal_requirement(
    cond_name: str,
    text_lower: str,
    lab_pattern_names: Set[str],
    imaging_findings_dict: Dict[str, float],
) -> bool:
    """Ensure a proposed condition is anchored in essential clinical criteria."""
    # Guard against false-positive acute pneumonia / sepsis on routine or asymptomatic examinations
    is_explicitly_asymptomatic = any(
        phrase in text_lower for phrase in [
            "asymptomatic", "no complaints", "routine checkup", "routine physical",
            "feeling well", "denies all symptoms", "no acute symptoms", "baseline normal",
            "well-appearing", "unremarkable exam", "routine screening", "annual exam"
        ]
    )
    if cond_name in ("Community-Acquired Pneumonia", "Sepsis / Severe Systemic Infection"):
        if is_explicitly_asymptomatic and not any("leukocytosis" in p.lower() or "inflammatory" in p.lower() for p in lab_pattern_names):
            return False

    reqs = CARDINAL_REQUIREMENTS.get(cond_name)
    if not reqs:
        return True

    # 1. Check cardinal lab pattern
    for pat in reqs.get("lab_patterns", []):
        if any(pat.lower() in p.lower() or p.lower() in pat.lower() for p in lab_pattern_names):
            return True

    # 2. Check cardinal imaging finding (requires clinically elevated probability >= 0.60)
    for img_term in reqs.get("imaging", []):
        if any(img_term.lower() in f.lower() and prob >= 0.60 for f, prob in imaging_findings_dict.items()):
            return True

    # 3. Check cardinal symptom pattern (must not be negated)
    for sym_pat in reqs.get("symptoms", []):
        for m in re.finditer(sym_pat, text_lower):
            if not is_symptom_negated(text_lower, m.start()):
                return True

    return False


def run_rule_engine(
    case_input: CaseInput,
    imaging_report: Optional[ImagingReport] = None,
    lab_report: Optional[LabReport] = None,
    rag_report: Optional[RagReport] = None,
) -> ClinicalDraft:
    """Execute weighted scoring rule engine evaluating >=25 clinical conditions."""
    text_lower = case_input.clinical_text.lower()

    # Pre-parse lab pattern names and inferred lab diseases
    lab_pattern_names = set()
    if lab_report and lab_report.status == "ok":
        lab_pattern_names = {p.name for p in lab_report.patterns}
        lab_pattern_names.update({d.disease for d in lab_report.inferred_diseases})

    # Pre-parse imaging findings
    imaging_findings_dict: Dict[str, float] = {}
    if imaging_report and imaging_report.status == "ok":
        for f in imaging_report.findings:
            imaging_findings_dict[f.finding] = f.probability

    scored_conditions: List[Dict[str, Any]] = []

    for cond_name, rules in CONDITIONS_CATALOG.items():
        # Cardinal diagnostic gate: ensure anchor finding is present
        if not _check_cardinal_requirement(cond_name, text_lower, lab_pattern_names, imaging_findings_dict):
            continue

        score = 0.0
        supporting_items: List[str] = []
        data_sources: Set[str] = set()

        # 1. Symptom matches in text (must not be negated)
        for pat in rules.get("symptoms", []):
            for m in re.finditer(pat, text_lower):
                if not is_symptom_negated(text_lower, m.start()):
                    matched_word = m.group(0).strip()
                    if matched_word not in supporting_items:
                        score += 1.0
                        data_sources.add("notes")
                        supporting_items.append(matched_word)
                    break

        # 2. Imaging matches (requires clinically elevated probability >= 0.60)
        for img_finding, weight in rules.get("imaging", {}).items():
            for detected_finding, prob in imaging_findings_dict.items():
                if img_finding.lower() in detected_finding.lower() and prob >= 0.60:
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
            if any(ag_img.lower() in k.lower() and prob >= 0.50 for k, prob in imaging_findings_dict.items()):
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
    if not top_items:
        # Case is clinically unremarkable / normal evaluation
        top_items = [
            {
                "condition": "No Acute Cardiopulmonary Process Identified",
                "score": 3.5,
                "sources_count": 2,
                "supporting": ["Clinical presentation unremarkable", "Absence of acute abnormal findings"],
                "against": "",
                "workup": ["Routine preventative health maintenance", "Return precautions as clinically indicated"],
            },
            {
                "condition": "Preventative Health Maintenance / Baseline Normal",
                "score": 2.0,
                "sources_count": 1,
                "supporting": ["Baseline vital signs and clinical examination"],
                "against": "",
                "workup": ["Age-appropriate routine screening", "Lifestyle counseling"],
            },
        ]
    elif len(top_items) < 2:
        top_name = top_items[0]["condition"]
        fallback_entry = CONTEXTUAL_SECONDARY_DIFFERENTIALS.get(top_name)
        if fallback_entry and not any(item["condition"] == fallback_entry["condition"] for item in top_items):
            top_items.append({
                "condition": fallback_entry["condition"],
                "score": 1.0,
                "sources_count": 1,
                "supporting": fallback_entry.get("supporting", ["Clinical presentation"]),
                "against": fallback_entry.get("against", ""),
                "workup": fallback_entry.get("workup", ["Clinical observation", "Follow-up monitoring"]),
            })
        elif not any(item["condition"] == "Preventative Health Maintenance" for item in top_items):
            top_items.append({
                "condition": "Preventative Health Maintenance",
                "score": 1.0,
                "sources_count": 1,
                "supporting": ["Clinical observation"],
                "against": "",
                "workup": ["Supportive care", "Routine monitoring"],
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

        # Match evidence citations strictly using clinical concept keywords
        cited_ids: List[str] = []
        concept_terms = CONDITION_CONCEPT_KEYWORDS.get(item["condition"], [])
        if not concept_terms:
            stop_words = {"acute", "chronic", "severe", "disease", "disorder", "syndrome", "type", "with", "without", "and", "or", "of", "in", "the", "a", "an", "pattern", "injury", "failure", "edema", "infection"}
            concept_terms = [w for w in re.findall(r"\w+", item["condition"].lower()) if w not in stop_words and len(w) > 3]

        for ev in available_citations:
            ev_corpus = f"{ev.source} {ev.section} {ev.passage}".lower()
            if any(kw in ev_corpus for kw in concept_terms):
                cited_ids.append(ev.id)
                if len(cited_ids) >= 2:
                    break

        profile = CONDITION_CLINICAL_PROFILES.get(item["condition"])
        mechanism_text = f" Disease mechanism: {profile['mechanism']}" if profile else ""
        rationale = f"Presentation is clinically indicative of {item['condition']}, supported by {', '.join(item['supporting']) if item['supporting'] else 'clinical examination'}."
        if item["against"]:
            rationale += f" Discriminating factors: {item['against']}."
        if mechanism_text:
            rationale += mechanism_text

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
    top_findings = differentials[0].supporting_findings[:3]
    top_profile = CONDITION_CLINICAL_PROFILES.get(top_cond)

    if top_cond in ("No Acute Cardiopulmonary Process Identified", "Preventative Health Maintenance / Baseline Normal", "Preventative Health Maintenance"):
        summary_sentences = [
            "Clinical evaluation and imaging review demonstrate no acute cardiopulmonary pathology or acute organ dysfunction.",
            "Cardiothoracic structures, pulmonary parenchyma, and baseline diagnostic parameters remain within normal limits.",
            "Routine preventative health maintenance and age-appropriate monitoring are recommended.",
        ]
    else:
        mechanism_desc = f", characterized pathophysiologically by {top_profile['mechanism'].rstrip('.').lower()}." if top_profile else "."
        s1 = f"Clinical evaluation indicates features consistent with {top_cond}{mechanism_desc}"
        
        # Explicitly mention significant imaging findings if present
        img_mention = ""
        if imaging_findings_dict:
            sig_findings = [f"{k} ({prob:.1%})" for k, prob in imaging_findings_dict.items() if prob >= 0.50]
            if sig_findings:
                img_mention = f" Concurrently, chest radiography identifies {', '.join(sig_findings)}."

        diff_names = [d.condition for d in differentials[1:3]]
        diff_clause = f", while secondary differential considerations include {', '.join(diff_names)}" if diff_names else ""
        findings_clause = f"Key supporting clinical indicators include {', '.join(top_findings)}" if top_findings else "Clinical assessment is guided by presenting signs"
        s2 = f"{findings_clause}{diff_clause}.{img_mention}"
        
        s3 = "Targeted diagnostic workup and guided serial clinical monitoring are recommended to establish definitive management."
        summary_sentences = [s1, s2, s3]
    summary = " ".join(summary_sentences)

    # Workup aggregate
    combined_workup = []
    for item in top_items:
        for w in item.get("workup", []):
            if w not in combined_workup:
                combined_workup.append(w)
            if len(combined_workup) >= 4:
                break

    # Red flags: require high/moderate likelihood or verified objective lab/imaging alert
    high_mod_conditions = {d.condition for d in differentials if d.likelihood in ("high", "moderate")}
    red_flags = []
    if "Severe Hyperkalemia" in high_mod_conditions or any("hyperkalaemia" in p.lower() or "hyperkalemia" in p.lower() for p in lab_pattern_names):
        red_flags.append("Critical hyperkalemia with lethal arrhythmia risk. Urgent ECG and stabilization required.")
    if "Diabetic Ketoacidosis" in high_mod_conditions or any("dka" in p.lower() for p in lab_pattern_names):
        red_flags.append("Potential diabetic ketoacidosis with metabolic acidosis requiring immediate resuscitation.")
    if "Pneumothorax" in high_mod_conditions or any("pneumothorax" in k.lower() and prob >= 0.70 for k, prob in imaging_findings_dict.items()):
        red_flags.append("Possible pneumothorax; evaluate immediately for tension signs and acute decompensation.")
    if "Acute Myocardial Infarction / ACS" in high_mod_conditions or any("troponin" in sf.lower() for d in differentials for sf in d.supporting_findings):
        red_flags.append("Myocardial necrosis marker elevation; evaluate for acute coronary syndrome.")
    if "Pulmonary Embolism" in high_mod_conditions:
        red_flags.append("High-risk pulmonary embolism suspected; evaluate hemodynamic stability and consider urgent CTPA.")

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
You are MedAgent's Clinical Reasoning specialist. Synthesize the provided clinical data into a structured ClinicalDraft JSON.
Rules:
1. Use ONLY the provided patient data. Explain the underlying disease mechanism and clinical reasons for the primary diagnosis.
2. DO NOT use robotic meta-language such as 'from the multimodal derivation', 'multimodal data', or 'derivation'. Speak as an objective, expert physician.
3. Cite ONLY existing evidence IDs from the Evidence list ({[e.id for e in (rag_report.evidence if rag_report else [])]}).
4. Use cautious phrasing (e.g. 'suggests', 'consistent with'). NEVER state a diagnosis as certain.
5. DO NOT provide drug doses or dosing schedules.
6. Return between 2 and 5 ranked differential diagnoses with detailed pathophysiology explanations in their rationale.
7. Summary must be 4 sentences or fewer and describe the disease mechanism and key clinical reasons.
8. List discordant findings as uncertainties.
9. Red flags must highlight acute life threats.
10. When both Imaging Findings and Laboratory Results are present, you MUST explicitly synthesize both modalities into the summary, differential list, and diagnostic workup.
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
