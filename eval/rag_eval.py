"""RAG Agent evaluation harness: evaluates Recall@1, Recall@3, and Recall@5 across 50 question-document pairs."""

from typing import Any, Dict, List, Tuple
from medagent.agents.rag import query_index

# 50 clinical questions and their ground truth target documents
RAG_EVAL_PAIRS: List[Tuple[str, str]] = [
    ("What are the diagnostic criteria and typical infiltrates in community-acquired pneumonia?", "pneumonia"),
    ("How is pleural effusion classified using Light criteria?", "pleural_effusion"),
    ("What is the acute management and needle decompression for tension pneumothorax?", "pneumothorax"),
    ("What cardiothoracic ratio defines cardiomegaly on a PA chest radiograph?", "cardiomegaly"),
    ("What are the radiographic features of pulmonary edema including Kerley B lines?", "pulmonary_edema"),
    ("How does postoperative atelectasis present and what volume loss signs appear?", "atelectasis"),
    ("What is the definition and air bronchogram appearance of lung consolidation?", "consolidation"),
    ("What is the KDIGO definition of acute kidney injury and creatinine rise?", "acute_kidney_injury"),
    ("What are the eGFR thresholds and staging criteria for chronic kidney disease?", "chronic_kidney_disease"),
    ("What ECG changes indicate urgent hyperkalemia and peaked T waves?", "hyperkalemia"),
    ("How does hypokalemia cause U waves, muscle weakness, and cardiac arrhythmias?", "hypokalemia"),
    ("What are the dangers of rapid correction of hyponatremia and osmotic demyelination?", "hyponatremia"),
    ("Why is serum ferritin low in microcytic hypochromic iron deficiency anemia?", "iron_deficiency_anemia"),
    ("How do Vitamin B12 and folate deficiencies cause macrocytic megaloblastic anemia?", "macrocytic_anemia"),
    ("When does leukocytosis with neutrophilic left shift indicate acute bacterial infection?", "leukocytosis"),
    ("What defines severe neutropenia and when is febrile neutropenia an emergency?", "leukopenia"),
    ("What are the bleeding risks and platelet thresholds in severe thrombocytopenia?", "thrombocytopenia"),
    ("What fasting plasma glucose and HbA1c levels establish the diagnosis of diabetes mellitus?", "type_2_diabetes"),
    ("What is the diagnostic triad of diabetic ketoacidosis with high anion gap acidosis?", "diabetic_ketoacidosis"),
    ("What is Whipple's triad and the definition of hypoglycemia under 70 mg/dL?", "hypoglycemia"),
    ("What transaminase elevations characterize acute toxic or viral hepatocellular injury?", "hepatocellular_injury"),
    ("How do alkaline phosphatase and GGT elevations identify cholestatic jaundice?", "cholestasis"),
    ("What criteria define acute myocardial infarction with high-sensitivity troponin elevation?", "myocardial_infarction"),
    ("What are the Framingham criteria and BNP levels in congestive heart failure?", "congestive_heart_failure"),
    ("How do Wells score and CT pulmonary angiography diagnose acute pulmonary embolism?", "pulmonary_embolism"),
    ("What radiographic signs of hyperinflation appear in acute COPD exacerbations?", "copd_exacerbation"),
    ("What is the Sepsis-3 definition and the Hour-1 bundle fluid and antibiotic resuscitation?", "sepsis"),
    ("What causes hypercalcemia and what are the symptoms of stones, bones, and abdominal groans?", "hypercalcemia"),
    ("How do you calculate the serum anion gap to evaluate metabolic acidosis?", "metabolic_acidosis"),
    ("What are the Atlanta criteria and serum lipase thresholds for acute pancreatitis?", "pancreatitis"),
    ("What pathogens commonly cause lobar consolidation in bacterial pneumonia?", "pneumonia"),
    ("When is thoracentesis indicated for blunted costophrenic angles in pleural fluid?", "pleural_effusion"),
    ("What are the physical exam findings of hyperresonance in pneumothorax?", "pneumothorax"),
    ("What imaging view best differentiates pericardial effusion from cardiomegaly?", "cardiomegaly"),
    ("What is the bat-wing butterfly infiltrate on radiograph indicative of?", "pulmonary_edema"),
    ("How does incentive spirometry prevent postoperative lung collapse and atelectasis?", "atelectasis"),
    ("How do air bronchograms help distinguish alveolar consolidation from atelectasis?", "consolidation"),
    ("Why should iodinated contrast be avoided in acute kidney injury?", "acute_kidney_injury"),
    ("What are the complications of secondary hyperparathyroidism in CKD?", "chronic_kidney_disease"),
    ("When is intravenous calcium gluconate indicated for high potassium?", "hyperkalemia"),
    ("What is the relationship between hypomagnesemia and refractory hypokalemia?", "hypokalemia"),
    ("What is the maximum recommended daily correction rate for chronic hyponatremia?", "hyponatremia"),
    ("Why does total iron binding capacity increase in iron deficiency anemia?", "iron_deficiency_anemia"),
    ("What neurological symptoms distinguish B12 deficiency from folate deficiency?", "macrocytic_anemia"),
    ("What is the difference between leukocytosis and a leukemoid reaction?", "leukocytosis"),
    ("What broad-spectrum antibiotics are given within one hour for febrile neutropenia?", "leukopenia"),
    ("Why are antiplatelet agents contraindicated when platelets drop below 50,000?", "thrombocytopenia"),
    ("What oral glucose tolerance test value confirms diabetes mellitus?", "type_2_diabetes"),
    ("Why is potassium monitoring essential before starting insulin in DKA?", "diabetic_ketoacidosis"),
    ("How is neuroglycopenia manifested during acute insulin-induced hypoglycemia?", "hypoglycemia"),
]


def evaluate_rag_retrieval() -> Dict[str, Any]:
    """Calculate Recall@1, Recall@3, and Recall@5 over the 50 evaluation pairs."""
    r1_hits = 0
    r3_hits = 0
    r5_hits = 0

    for query, target_doc in RAG_EVAL_PAIRS:
        hits = query_index(query, top_k=5)
        hit_docs = [h[0]["source"].lower() for h in hits]

        # Check if target_doc is in the source filename
        if len(hit_docs) >= 1 and target_doc in hit_docs[0]:
            r1_hits += 1
        if any(target_doc in doc for doc in hit_docs[:3]):
            r3_hits += 1
        if any(target_doc in doc for doc in hit_docs[:5]):
            r5_hits += 1

    total = len(RAG_EVAL_PAIRS)
    return {
        "total_queries_evaluated": total,
        "recall_at_1": round(r1_hits / total, 4),
        "recall_at_3": round(r3_hits / total, 4),
        "recall_at_5": round(r5_hits / total, 4),
        "r1_count": r1_hits,
        "r3_count": r3_hits,
        "r5_count": r5_hits,
    }


if __name__ == "__main__":
    res = evaluate_rag_retrieval()
    print("=" * 60)
    print("RAG AGENT RETRIEVAL BENCHMARK (50 QA Pairs)")
    print("=" * 60)
    print(f"Total Question-Document Pairs : {res['total_queries_evaluated']}")
    print(f"Recall@1                     : {res['recall_at_1']:.1%} ({res['r1_count']}/{res['total_queries_evaluated']})")
    print(f"Recall@3                     : {res['recall_at_3']:.1%} ({res['r3_count']}/{res['total_queries_evaluated']})")
    print(f"Recall@5                     : {res['recall_at_5']:.1%} ({res['r5_count']}/{res['total_queries_evaluated']})")
