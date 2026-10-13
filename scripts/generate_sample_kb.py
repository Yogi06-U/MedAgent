"""Generates ~30 sample clinical knowledge base documents labelled 'sample'."""

import os
from pathlib import Path

KNOWLEDGE_DOCS = [
    ("pneumonia.md", "Pneumonia", "Infectious Diseases / Pulmonology",
     """# Pneumonia (Sample: Gale Encyclopedia of Medicine)
## Overview and Pathophysiology
Pneumonia is an acute inflammatory infection of the lung parenchyma, most commonly caused by Streptococcus pneumoniae, atypical bacteria, or respiratory viruses. Infiltrates accumulate in the alveoli impairing oxygenation.
## Clinical Presentation
Symptoms include productive cough, purulent sputum, fever, chills, dyspnea, and pleuritic chest pain. Auscultation reveals localized bronchial breathing, crackles, and dullness to percussion.
## Diagnostic Evaluation
Diagnosis is supported by chest radiography demonstrating focal consolidation, air bronchograms, or patchy infiltrates. Laboratory analysis typically shows leukocytosis with neutrophilia and elevated inflammatory markers (CRP and procalcitonin).
## Management Principles
Empiric antibiotics based on community vs hospital acquisition. Pulse oximetry monitoring, hydration, and supplemental oxygen as clinically indicated."""),

    ("pleural_effusion.md", "Pleural Effusion", "Pulmonology",
     """# Pleural Effusion (Sample: Gale Encyclopedia of Medicine)
## Overview and Pathophysiology
Pleural effusion is the pathological accumulation of fluid in the pleural space, classified as either transudate or exudate using Light's criteria.
## Clinical Presentation
Pleuritic chest pain, progressive dyspnea on exertion, cough, and decreased breath sounds with stony dullness to percussion on the affected side.
## Radiographic Findings
Chest radiography demonstrates blunting of the costophrenic angles, meniscus sign, or homogeneous opacification of the lower hemithorax with contralateral mediastinal shift in massive effusions.
## Diagnostic Evaluation
Diagnostic thoracentesis for protein, LDH, glucose, pH, cell count, and microbiology."""),

    ("pneumothorax.md", "Pneumothorax", "Emergency Medicine / Pulmonology",
     """# Pneumothorax (Sample: Gale Encyclopedia of Medicine)
## Overview and Pathophysiology
Pneumothorax is the presence of air in the pleural cavity causing partial or total lung collapse. Tension pneumothorax is an immediate life-threatening emergency.
## Clinical Presentation
Sudden onset sharp unilateral pleuritic chest pain and acute shortness of breath. Physical exam reveals hyperresonance, decreased breath sounds, and tracheal deviation away from the affected side in tension cases.
## Radiographic Findings
Upright chest X-ray shows visceral pleural line with absent peripheral lung markings.
## Urgent Management
Immediate needle decompression for tension pneumothorax followed by chest tube thoracostomy. Never delay decompression in an unstable patient for imaging."""),

    ("cardiomegaly.md", "Cardiomegaly", "Cardiology",
     """# Cardiomegaly (Sample: Gale Encyclopedia of Medicine)
## Overview
Cardiomegaly refers to an enlarged cardiac silhouette, reflecting ventricular hypertrophy, dilation, or pericardial effusion.
## Radiographic Criteria
Cardiothoracic ratio (CTR) greater than 0.50 on a standard PA chest radiograph. Lateral view may reveal displacement of the retrocardiac space.
## Clinical Correlation
Associated with systemic hypertension, ischemic cardiomyopathy, valvular disease, and heart failure. Symptoms include orthopnea, paroxysmal nocturnal dyspnea, and peripheral edema.
## Diagnostic Workup
Echocardiogram is the diagnostic gold standard to evaluate ejection fraction, chamber dimensions, and wall motion abnormalities."""),

    ("pulmonary_edema.md", "Pulmonary Edema", "Cardiology / Critical Care",
     """# Pulmonary Edema (Sample: Gale Encyclopedia of Medicine)
## Overview and Pathophysiology
Accumulation of fluid in pulmonary interstitium and alveoli due to elevated capillary hydrostatic pressure (cardiogenic) or altered microvascular permeability.
## Clinical Presentation
Severe acute dyspnea, orthopnea, pink frothy sputum, diaphoresis, and diffuse bibasilar crackles.
## Radiographic Findings
Prominent pulmonary vascular congestion, cephalization of vessels, Kerley B lines, peribronchial cuffing, and bat-wing / butterfly alveolar infiltrates.
## Diagnostic and Therapeutic Actions
High-dose loop diuretics, nitrates for preload reduction, non-invasive positive pressure ventilation (BiPAP), and cardiac monitoring."""),

    ("atelectasis.md", "Atelectasis", "Pulmonology",
     """# Atelectasis (Sample: Gale Encyclopedia of Medicine)
## Overview
Atelectasis is the collapse or incomplete expansion of lung parenchyma, frequently occurring postoperatively or due to bronchial obstruction.
## Radiographic Features
Linear or platelike opacities, volume loss with elevation of the ipsilateral hemidiaphragm, and mediastinal shift toward the affected side.
## Clinical Presentation
Mild tachypnea, decreased breath sounds, hypoxia, and low-grade fever postoperatively.
## Management
Incentive spirometry, early mobilization, chest physiotherapy, and clearing of airway secretions."""),

    ("consolidation.md", "Lung Consolidation", "Pulmonology / Radiology",
     """# Lung Consolidation (Sample: Gale Encyclopedia of Medicine)
## Definition
Replacement of normal alveolar air by inflammatory exudate, pus, blood, edema, or cells.
## Imaging Characteristics
Dense confluent radiographic opacity that obscures pulmonary vessels, often with visible air bronchograms. Location may correspond to specific lobes or segments.
## Differential Diagnosis
Bacterial pneumonia, acute aspiration, pulmonary hemorrhage, and localized alveolar proteinosis.
## Clinical Correlation
Correlate with fever, sputum purulence, and inflammatory markers."""),

    ("acute_kidney_injury.md", "Acute Kidney Injury", "Nephrology",
     """# Acute Kidney Injury (Sample: Gale Encyclopedia of Medicine)
## Diagnostic Criteria
Abrupt decline in renal function defined by an increase in serum creatinine by ≥0.3 mg/dL within 48 hours or ≥1.5-fold from baseline, or oliguria (<0.5 mL/kg/h for 6 hours).
## Etiologies
Prerenal (hypovolemia, shock, heart failure), intrinsic (acute tubular necrosis, interstitial nephritis, glomerulonephritis), and postrenal obstructive uropathy.
## Critical Warnings and Contraindications
Avoid iodinated contrast and nephrotoxic agents (NSAIDs, aminoglycosides). Adjust or discontinue metformin, ACE inhibitors, and spironolactone.
## Laboratory Findings
Elevated BUN and creatinine, hyperkalemia, metabolic acidosis, and hyperphosphatemia."""),

    ("chronic_kidney_disease.md", "Chronic Kidney Disease", "Nephrology",
     """# Chronic Kidney Disease (Sample: Gale Encyclopedia of Medicine)
## Definition
Kidney damage or estimated GFR (eGFR) < 60 mL/min/1.73m2 persisting for greater than 3 months.
## Staging
Stage 1 (eGFR ≥90 with proteinuria), Stage 2 (60-89), Stage 3 (30-59), Stage 4 (15-29), Stage 5 / End-stage (<15).
## Clinical Implications
Anemia of chronic disease, secondary hyperparathyroidism, fluid retention, and hyperkalemia. Strict dosing adjustments required for renally cleared medications."""),

    ("hyperkalemia.md", "Hyperkalemia", "Nephrology / Emergency Medicine",
     """# Hyperkalemia (Sample: Gale Encyclopedia of Medicine)
## Definition and Urgency
Serum potassium concentration exceeding 5.0 mEq/L; levels ≥6.0 mEq/L represent a medical emergency due to fatal cardiac arrhythmogenic potential.
## Electrocardiographic Changes
Peaked T waves, flattened P waves, prolonged PR interval, QRS widening, and sine wave pattern preceding ventricular fibrillation.
## Medication Review & Contraindications
Immediately discontinue potassium-sparing diuretics (spironolactone), ACE inhibitors, ARBs, and NSAIDs.
## Acute Stabilization
Intravenous calcium gluconate for cardiac membrane stabilization, followed by insulin + dextrose, beta-2 agonists, and potassium binders or emergent hemodialysis."""),

    ("hypokalemia.md", "Hypokalemia", "Endocrinology / Nephrology",
     """# Hypokalemia (Sample: Gale Encyclopedia of Medicine)
## Definition
Serum potassium < 3.5 mEq/L. Severe hypokalemia is defined as potassium < 2.5 mEq/L.
## Etiologies
Loop and thiazide diuretics, gastrointestinal losses (vomiting, diarrhea), hyperaldosteronism, and intracellular shifting.
## Complications
Muscle weakness, hyporeflexia, cramps, paralytic ileus, flattened T waves, ST depression, prominent U waves, and cardiac arrhythmias.
## Management
Oral or intravenous potassium chloride replacement, with concurrent correction of hypomagnesemia."""),

    ("hyponatremia.md", "Hyponatremia", "Endocrinology / Nephrology",
     """# Hyponatremia (Sample: Gale Encyclopedia of Medicine)
## Overview
Serum sodium concentration < 135 mEq/L. Severe hyponatremia is < 120 mEq/L.
## Classification
Hypovolemic, euvolemic (SIADH, psychogenic polydipsia), and hypervolemic (CHF, cirrhosis, nephrotic syndrome).
## Symptoms
Headache, nausea, confusion, lethargy, seizures, and cerebral edema if acute.
## Treatment Precautions
Correction must not exceed 8-10 mEq/L in 24 hours to avoid osmotic demyelination syndrome (central pontine myelinolysis)."""),

    ("iron_deficiency_anemia.md", "Iron Deficiency Anemia", "Hematology",
     """# Iron Deficiency Anemia (Sample: Gale Encyclopedia of Medicine)
## Pathophysiology
Microcytic, hypochromic anemia resulting from inadequate iron stores required for hemoglobin synthesis.
## Laboratory Features
Low hemoglobin, low MCV (<80 fL), low MCH, low serum iron, high TIBC, and characteristically low serum ferritin (<30 ng/mL). Note: ferritin is an acute phase reactant.
## Clinical Features
Fatigue, pallor, pica, brittle nails, koilonychia, and cold intolerance.
## Investigation
Evaluate for occult gastrointestinal bleeding, menorrhagia, or malabsorption."""),

    ("macrocytic_anemia.md", "Macrocytic Anemia", "Hematology",
     """# Macrocytic Anemia (Sample: Gale Encyclopedia of Medicine)
## Overview
Anemia with elevated mean corpuscular volume (MCV > 100 fL).
## Etiologies
Megaloblastic: Vitamin B12 deficiency (pernicious anemia, malabsorption) and folate deficiency. Non-megaloblastic: alcoholism, liver disease, hypothyroidism, myelodysplasia.
## Clinical Distinctions
Vitamin B12 deficiency causes subacute combined degeneration of the spinal cord (neuropathy, ataxia, loss of vibration sense), which is absent in folate deficiency.
## Workup
Serum B12, serum and RBC folate, methylmalonic acid, and homocysteine levels."""),

    ("leukocytosis.md", "Leukocytosis", "Hematology / Infectious Diseases",
     """# Leukocytosis (Sample: Gale Encyclopedia of Medicine)
## Definition
Elevated total white blood cell count > 11.0 x10^3/uL.
## Etiologies
Bacterial infection (neutrophilic leukocytosis with left shift / bands), viral infection (relative lymphocytosis), severe physical stress, inflammation, and hematologic malignancies.
## Clinical Correlation
Evaluate accompanied symptoms: fever, localized tenderness, purulent discharge, and systemic inflammatory response criteria."""),

    ("leukopenia.md", "Leukopenia & Neutropenia", "Hematology / Oncology",
     """# Leukopenia and Neutropenia (Sample: Gale Encyclopedia of Medicine)
## Definition
Total WBC count < 4.0 x10^3/uL. Neutropenia is defined as absolute neutrophil count (ANC) < 1.5 x10^3/uL. Severe neutropenia is ANC < 0.5 x10^3/uL.
## Risks
Severe risk of opportunistic bacterial and fungal infections. Febrile neutropenia requires immediate empiric broad-spectrum antibiotic therapy within 1 hour.
## Etiologies
Bone marrow suppression, chemotherapy, viral infections (HIV, hepatitis), autoimmune destruction, and severe sepsis."""),

    ("thrombocytopenia.md", "Thrombocytopenia", "Hematology",
     """# Thrombocytopenia (Sample: Gale Encyclopedia of Medicine)
## Definition
Platelet count < 150 x10^3/uL. Severe thrombocytopenia is < 50 x10^3/uL (increased bleeding risk) and < 20 x10^3/uL (spontaneous bleeding risk).
## Etiologies
Immune thrombocytopenia (ITP), drug-induced (heparin-induced thrombocytopenia HIT), thrombotic microangiopathy (TTP/HUS), DIC, and cirrhosis / hypersplenism.
## Safety Warning
Anticoagulants and antiplatelet drugs are strictly contraindicated in severe thrombocytopenia due to risk of fatal hemorrhage."""),

    ("type_2_diabetes.md", "Type 2 Diabetes Mellitus", "Endocrinology",
     """# Type 2 Diabetes Mellitus (Sample: Gale Encyclopedia of Medicine)
## Diagnostic Criteria
Fasting plasma glucose ≥ 126 mg/dL, 2-hour postprandial glucose ≥ 200 mg/dL during OGTT, HbA1c ≥ 6.5 %, or random glucose ≥ 200 mg/dL with classic symptoms of hyperglycemia.
## Clinical Features
Polyuria, polydipsia, polyphagia, unintentional weight loss, and blurred vision.
## Complications
Diabetic nephropathy, retinopathy, peripheral neuropathy, and accelerated cardiovascular atherosclerosis."""),

    ("diabetic_ketoacidosis.md", "Diabetic Ketoacidosis", "Endocrinology / Emergency Medicine",
     """# Diabetic Ketoacidosis (Sample: Gale Encyclopedia of Medicine)
## Triad
Hyperglycemia (>250 mg/dL), high anion gap metabolic acidosis (bicarbonate < 18 mEq/L, pH < 7.30), and ketonemia / ketonuria.
## Clinical Manifestations
Kussmaul breathing, fruity breath odor, nausea, vomiting, abdominal pain, dehydration, and altered mental status.
## Emergency Management
Aggressive IV fluid resuscitation, intravenous regular insulin infusion, and vigilant monitoring of serum potassium prior to and during insulin administration."""),

    ("hypoglycemia.md", "Hypoglycemia", "Endocrinology / Emergency Medicine",
     """# Hypoglycemia (Sample: Gale Encyclopedia of Medicine)
## Definition
Blood glucose concentration < 70 mg/dL. Severe hypoglycemia is < 54 mg/dL.
## Whipple Triad
Symptoms consistent with hypoglycemia, documented low plasma glucose, and symptom resolution upon glucose administration.
## Symptoms
Autonomic: sweating, tremors, tachycardia, hunger. Neuroglycopenic: confusion, drowsiness, seizures, and coma.
## Acute Treatment
Fast-acting carbohydrates (15-20g) if conscious; IV 50% dextrose (D50W) or intramuscular glucagon if unconscious."""),

    ("hepatocellular_injury.md", "Hepatocellular Injury", "Gastroenterology / Hepatology",
     """# Acute Hepatocellular Injury (Sample: Gale Encyclopedia of Medicine)
## Laboratory Hallmark
Marked elevation of alanine aminotransferase (ALT) and aspartate aminotransferase (AST), often > 5-10 times the upper limit of normal.
## Etiologies
Viral hepatitis (A, B, C, E), acetaminophen toxicity, ischemic hepatitis (shock liver), autoimmune hepatitis, and drug-induced liver injury.
## Assessment of Hepatic Function
Bilirubin, prothrombin time / INR (measures synthetic capacity), and serum albumin."""),

    ("cholestasis.md", "Cholestasis & Jaundice", "Gastroenterology",
     """# Cholestasis and Obstructive Jaundice (Sample: Gale Encyclopedia of Medicine)
## Laboratory Pattern
Predominant elevation of alkaline phosphatase (ALP) and gamma-glutamyl transferase (GGT), accompanied by conjugated hyperbilirubinemia.
## Etiologies
Extrahepatic biliary obstruction (choledocholithiasis, cholangiocarcinoma, pancreatic head mass) and intrahepatic cholestasis (primary biliary cholangitis, drug-induced).
## Clinical Symptoms
Pruritus, dark tea-colored urine, pale acholic stools, and scleral icterus. Right upper quadrant ultrasound is first-line imaging."""),

    ("myocardial_infarction.md", "Acute Myocardial Infarction", "Cardiology / Emergency Medicine",
     """# Acute Myocardial Infarction (Sample: Gale Encyclopedia of Medicine)
## Diagnostic Definition
Rise and/or fall of cardiac troponin with at least one value above the 99th percentile URL, accompanied by ischemic symptoms or ECG changes (ST elevation or new Q waves).
## Symptoms
Crushing retrosternal chest pain radiating to left arm, neck, or jaw, diaphoresis, dyspnea, and nausea.
## Red Flag Alert
Acute chest pain with elevated troponin requires immediate 12-lead ECG and cardiology evaluation for coronary revascularization."""),

    ("congestive_heart_failure.md", "Congestive Heart Failure", "Cardiology",
     """# Congestive Heart Failure (Sample: Gale Encyclopedia of Medicine)
## Clinical Syndrome
Inability of the heart to pump sufficient blood to meet metabolic demands, characterized by elevated intracardiac pressures.
## Framingham Criteria
Major criteria: paroxysmal nocturnal dyspnea, neck vein distention, rales, cardiomegaly, pulmonary edema, S3 gallop. Minor criteria: bilateral ankle edema, night cough, dyspnea on ordinary exertion, hepatomegaly.
## Diagnostic Workup
Serum BNP or NT-proBNP, echocardiogram, and chest radiograph demonstrating cardiomegaly and vascular redistribution."""),

    ("pulmonary_embolism.md", "Pulmonary Embolism", "Emergency Medicine / Pulmonology",
     """# Pulmonary Embolism (Sample: Gale Encyclopedia of Medicine)
## Pathophysiology
Occlusion of the pulmonary arterial bed by a thrombus, usually dislodged from deep venous thrombosis of the lower extremities.
## Clinical Triad
Acute dyspnea, pleuritic chest pain, and tachypnea. Hemoptysis and syncope indicate massive embolism.
## Diagnostic Stratification
Wells criteria scoring, D-dimer assay (high negative predictive value), and CT Pulmonary Angiography (CTPA) as definitive imaging standard.
## Contraindications
Contrast allergy or acute kidney injury require alternative evaluation (ventilation-perfusion V/Q scan)."""),

    ("copd_exacerbation.md", "COPD Acute Exacerbation", "Pulmonology",
     """# COPD Acute Exacerbation (Sample: Gale Encyclopedia of Medicine)
## Definition
Acute sustained worsening of respiratory symptoms (dyspnea, cough, sputum volume and/or purulence) beyond normal day-to-day variations.
## Radiographic Features
Hyperinflated lung fields, flattened diaphragms, increased retrosternal airspace, and exclusion of concurrent pneumonia or pneumothorax.
## Medical Management
Inhaled short-acting bronchodilators, systemic corticosteroids, supplemental controlled oxygen (target SpO2 88-92% to avoid hypercapnia), and antibiotics if purulence is present."""),

    ("sepsis.md", "Sepsis & Septic Shock", "Critical Care / Infectious Diseases",
     """# Sepsis and Septic Shock (Sample: Gale Encyclopedia of Medicine)
## Sepsis-3 Definition
Life-threatening organ dysfunction caused by a dysregulated host response to infection (SOFA score increase ≥2).
## Clinical Red Flags
Hypotension (SBP < 90 mmHg or MAP < 65 mmHg), hyperlactatemia (> 2.0 mmol/L), tachypnea (> 22/min), altered mental status, and oliguria.
## Hour-1 Bundle
Measure blood lactate, obtain blood cultures prior to antibiotics, administer broad-spectrum antimicrobials, initiate 30 mL/kg crystalloid fluid resuscitation for hypotension or lactate ≥ 4.0 mmol/L."""),

    ("hypercalcemia.md", "Hypercalcemia", "Endocrinology",
     """# Hypercalcemia (Sample: Gale Encyclopedia of Medicine)
## Definition
Total serum calcium > 10.2 mg/dL. Severe hypercalcemia is > 14.0 mg/dL (hypercalcemic crisis).
## Common Etiologies
Primary hyperparathyroidism (elevated or inappropriately normal PTH) and malignancy (PTHrP mediated or osteolytic metastases).
## Clinical Manifestations
Stones (nephrolithiasis), bones (bone pain, fractures), groans (abdominal cramping, pancreatitis, constipation), and psychic overtones (depression, confusion).
## Emergency Treatment
Intravenous isotonic saline hydration, loop diuretics after hydration, calcitonin, and IV bisphosphonates."""),

    ("metabolic_acidosis.md", "Metabolic Acidosis", "Nephrology / Critical Care",
     """# Metabolic Acidosis (Sample: Gale Encyclopedia of Medicine)
## Definition
Low serum bicarbonate (<22 mEq/L) and low arterial pH (<7.35).
## High Anion Gap Acidosis (MUDPILES / GOLDMARK)
Methanol, Uremia, Diabetic ketoacidosis, Propylene glycol, Infection/Iron/Isoniazid, Lactic acidosis, Ethylene glycol, Salicylates. Anion gap = Na - (Cl + HCO3). Normal is 8-12 mEq/L.
## Normal Anion Gap (Hyperchloremic)
Renal tubular acidosis, diarrhea, acetazolamide."""),

    ("pancreatitis.md", "Acute Pancreatitis", "Gastroenterology / Emergency Medicine",
     """# Acute Pancreatitis (Sample: Gale Encyclopedia of Medicine)
## Atlanta Criteria (2 of 3 Required)
1. Acute severe epigastric pain radiating to the back.
2. Serum lipase or amylase elevated ≥ 3 times upper limit of normal.
3. Characteristic cross-sectional imaging findings (pancreatic edema, peripancreatic fluid collections).
## Common Causes
Gallstones (cholelithiasis), alcohol abuse, severe hypertriglyceridemia (>1000 mg/dL), and ERCP.
## Management
Aggressive intravenous hydration, multimodal analgesia, early enteral nutrition, and monitoring for systemic inflammatory complications."""),
]

def main():
    target_dir = Path(__file__).resolve().parent.parent / "backend" / "medagent" / "knowledge"
    target_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for filename, title, category, content in KNOWLEDGE_DOCS:
        file_path = target_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        count += 1
    print(f"Generated {count} sample clinical knowledge documents in {target_dir}")

if __name__ == "__main__":
    main()
