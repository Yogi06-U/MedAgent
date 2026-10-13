# MedAgent System Architecture

## 1. High-Level Overview

MedAgent is a multimodal, multi-agent clinical decision-support system designed to assist healthcare professionals in diagnostic synthesis while maintaining strict physician oversight and deterministic safety bounds.

```
Notes · X-ray · Labs · Voice
  → Orchestrator (LangGraph): intake → plan
      ├─ Imaging Agent ┐ (parallel; run only if radiograph exists)
      └─ Lab Agent     ┘ (parallel; run only if lab results exist)
  → RAG Agent (evidence retrieval from clinical knowledge base)
  → Clinical Reasoning Agent (multimodal fusion into differential)
      └─ Rationale-Guided 2nd Retrieval (fills missing evidence)
  → Safety Agent (deterministic checks + contraindications)
      └─ [If blocked & retries < 1] → Back to Reasoning with feedback
  → Physician Gate (LangGraph interrupt; review / edit / approve / reject)
  → Finalize (WeasyPrint / ReportLab PDF + SQLite audit log)
```

---

## 2. Agent Specifications & Boundaries

Every agent boundary is strictly typed using **Pydantic v2** models defined in `medagent.schemas`. No raw untyped dictionaries cross module boundaries.

### 2.1 Intake & Orchestrator Agent (`backend/medagent/graph.py`)
- Ingests clinical notes, patient demographics, image uploads, lab files/text, and voice recordings.
- Formulates a conditional execution plan:
  - If no radiograph is attached, skips Imaging Agent.
  - If no lab data is attached, skips Lab Agent.
  - If both are missing, routes directly to RAG using clinical notes.
- If any specialist agent encounters an unrecoverable runtime exception, it catches the error and returns a typed report with `status="error"` and a descriptive message; the graph continues execution with the remaining agents.

### 2.2 Imaging Agent (`backend/medagent/agents/imaging.py`)
- **Modality:** Frontal Chest Radiographs (CR/DX). Rejects non-CXR modalities (CT/MRI/Ultrasound) with explicit status `"unavailable"`.
- **Pretrained Model:** TorchXRayVision DenseNet-121 (`densenet121-res224-all`) evaluated in inference mode on CPU.
- **Grad-CAM Localization:** Computes exact gradients of the target pathology logit with respect to the last DenseNet convolutional feature map (before the final ReLU and pooling). Upsamples to $224 \times 224$ and normalizes to $[0, 1]$.
- **Heatmap Overlay:** Saves a high-resolution $512 \times 512$ PNG overlay (Jet colormap blended over grayscale radiograph).
- **Anatomical Mapping:** Calculates laterality (patient right vs. left vs. bilateral) and vertical zone (upper, mid, lower) from the centroid of activations exceeding $0.6$. Special handling routes cardiomegaly to the cardiac silhouette.

### 2.3 Lab Agent (`backend/medagent/agents/lab.py`)
- **Parser:** Deterministic regex parser with an alias catalog of $\ge 60$ common tests. Extracts test name, numerical float value, units, and printed reference intervals.
- **Unit Normalization:** Converts non-standard units (e.g., $\mu\text{mol/L}$, $\text{mmol/L}$, $\text{lakhs/cumm}$, $\text{ng/L}$) to standardized clinical SI units.
- **Reference Database:** Evaluates sex- and age-specific intervals against `data/reference_ranges.json`. Printed report intervals strictly override default intervals.
- **Rule Engine:** Evaluates 20+ clinical patterns (anemias, leukocytosis, acute renal impairment, electrolyte emergencies, DKA, hepatocellular and cholestatic patterns).

### 2.4 RAG Agent (`backend/medagent/agents/rag.py`)
- **Knowledge Base:** Gale Encyclopedia of Medicine reference corpus.
- **Chunking:** Semantic heading splits followed by $\approx 900$ character chunks with $120$ character overlap on sentence boundaries.
- **Embedding Store:** Local dense cosine index powered by `sentence-transformers/all-MiniLM-L6-v2` ($384\text{-d}$ normalized vectors).
- **Query Formulation:** Generates focused clinical queries from imaging findings, urgent lab patterns, and primary notes.
- **Two-Pass Retrieval:** Following differential generation, any differential lacking citation evidence triggers an automated rationale-guided second pass.

### 2.5 Clinical Reasoning Agent (`backend/medagent/agents/reasoning.py`)
- **Dual Engine Architecture:**
  - **Primary:** Provider-agnostic LLM schema caller (OpenAI, Anthropic, Gemini) returning structured `ClinicalDraft` JSON.
  - **Fallback:** Deterministic weighted scoring rule engine evaluating $\ge 25$ medical conditions against symptom regexes, imaging probabilities, and lab patterns.
- Guarantees $2$ to $5$ differentials, explicit non-supporting/against findings, suggested diagnostic workups, and red flags.

### 2.6 Safety Agent (`backend/medagent/agents/safety.py`)
- **Grounding Validation:** Strips non-existent evidence IDs; flags differentials lacking observed clinical findings.
- **Contraindication Checks:** Metformin in renal impairment; contrast with eGFR $<30$; NSAIDs/ACEi in hyperkalemia; anticoagulation in severe thrombocytopenia.
- **Language Sanitation:** Automatically redacts drug dosing instructions and softens absolute certainty language ("definitely", "confirmed diagnosis").
- **Urgency Detection:** Flags acute emergencies (tension pneumothorax, severe hyperkalemia, acute MI, DKA) while correctly ignoring negated findings ("denies chest pain").
- **Gating Feedback Loop:** If verdict is `blocked` and `retries < 1`, passes structured clinical feedback back to the Reasoning Agent for one automated remediation cycle.

### 2.7 Physician Gate & Persistence (`backend/medagent/graph.py`)
- LangGraph `interrupt` halts execution before any final report can be generated.
- State is persisted to a local **SQLite checkpointer** (`medagent.db`).
- A licensed physician must explicitly `Approve`, `Edit`, or `Reject` the case draft in the UI.

---

## 3. Communication & Streaming Protocol

- **Server-Sent Events (SSE):** Streaming progress updates are published in real-time via `/api/cases/{case_id}/stream`.
- **Typed Event Schema:** Each event carries `node`, `status`, `summary`, and timestamp.
- **Audit Logging:** Every transition is recorded in an append-only SQLite audit table (`audit.py`). No Patient Health Information (PHI) is logged at `INFO` level.
