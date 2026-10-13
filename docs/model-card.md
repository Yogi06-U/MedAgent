# MedAgent Clinical Model Card

## 1. Model Details

### 1.1 Chest Radiography Model (TorchXRayVision DenseNet-121)
- **Model Architecture:** DenseNet-121 with single-channel 224x224 input and 18 pathology logit outputs.
- **Weights Tag:** `densenet121-res224-all` (TorchXRayVision library; Cohen et al., MIDL 2020 / arXiv:2011.08199).
- **Pretraining Datasets:** NIH ChestX-ray14, Stanford CheXpert, PhysioNet MIMIC-CXR, PadChest (BIMCV), Google Health OpenI, RSNA Pneumonia / Kaggle.
- **Model Parameters:** 6,966,034 parameters.
- **Inference Mode:** CPU-only, deterministic evaluation mode (`model.eval()`). Weights cached locally in `data/weights/`.

### 1.2 Knowledge Embedding Model (MiniLM-L6-v2)
- **Model Architecture:** SentenceTransformers `all-MiniLM-L6-v2` (6-layer MiniLM, 384-dimensional dense vectors).
- **Knowledge Base:** Gale Encyclopedia of Medicine clinical reference texts.
- **Retrieval Metric:** Cosine similarity with L2-normalized embeddings.

### 1.3 Clinical Reasoning Engine
- **Engine Modes:** Dual architecture:
  1. Provider-Agnostic LLM Schema Caller (OpenAI, Anthropic, Gemini) with JSON schema enforcement.
  2. Offline Weighted Rule Engine evaluating $\ge 25$ conditions across clinical notes, imaging findings ($\times$ probability), and lab patterns.

---

## 2. Intended Clinical Use & User Persona

- **Primary Persona:** Licensed physicians, radiologists, and clinicians evaluating adult patients in emergency, inpatient, or ambulatory settings.
- **Clinical Task:** Differential diagnosis generation, multimodal data fusion (notes + labs + radiographs), contraindication checking, and clinical evidence retrieval.
- **Human-in-the-Loop Requirement:** This system is an adjunct **clinical decision support** tool. It is NOT an autonomous diagnostic agent. A licensed physician must review, edit, approve, or reject every case before any clinical report is released.

---

## 3. Out-of-Scope Uses & Explicit Limitations

- **Pediatric Patients:** Model pretraining was conducted predominantly on adult patient cohorts. Pediatric chest radiographs exhibit different anatomical proportions and thymus shadows.
- **Autonomous Dosing / Treatment:** MedAgent strictly prohibits providing medication dosages or autonomous prescription generation.
- **Non-CXR Imaging:** The imaging pipeline is strictly validated on frontal chest radiographs (CR/DX). CT, MRI, ultrasound, or cross-sectional scans are rejected.
- **Definitive Diagnostic Assertions:** The system never asserts a diagnosis as absolute certainty. All outputs must reflect probabilistic likelihoods and differential possibilities.

---

## 4. Failure Modes & Mitigations

| Failure Mode | Clinical Risk | Technical Mitigation |
| :--- | :--- | :--- |
| **Pneumothorax false negative** | Missed life-threatening tension pneumothorax | Explicit safety check flags clinical emergency and sets `urgent=True`. |
| **High contrast or inverted image** | Degradation of DenseNet pathology detection | Load-time quality checks detect MONOCHROME1 inversion, low contrast, and invalid aspect ratios. |
| **Missing laboratory units** | Erroneous order-of-magnitude lab interpretation | Unit normalization converts to SI standard; unparsed lines are explicitly flagged in `unparsed_lines`. |
| **Hallucinated citations** | Clinician relies on non-existent medical evidence | Deterministic Safety Agent cross-references all evidence IDs against retrieved passages and removes invalid citations. |
| **Contraindicated medication recommendation** | Prescription of NSAIDs/ACEi in hyperkalemia or contrast in severe renal failure | Deterministic safety rule evaluates creatinine/eGFR and potassium before allowing approval. |

---

## 5. Performance Metrics & Benchmarks

Measured on the MedAgent evaluation suite (`eval/results.md`):

- **Imaging Macro AUROC:** 0.798 across 14 benchmark pathologies (Cohen et al. multi-dataset benchmark).
- **Grad-CAM Localization Alignment:** 75.0% peak alignment inside thoracic lung fields across 20 radiographs.
- **Lab Agent Accuracy:** 100.0% field extraction accuracy and 100.0% pattern F1 score across 55 panels.
- **Safety Fault Catch Rate:** 100.0% detection rate across 32 seeded clinical and language faults.
- **Reasoning Concordance:** 100.0% Top-1 diagnostic concordance across 25 standardized vignettes.
