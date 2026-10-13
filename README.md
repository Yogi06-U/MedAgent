# MedAgent: Multimodal Multi-Agent Clinical Decision-Support System

MedAgent is a multimodal multi-agent clinical decision-support research prototype built with **FastAPI**, **LangGraph**, **React 18**, **Tailwind CSS**, and **TorchXRayVision**. It ingests free-text clinical notes, chest radiographs (CXR), laboratory results, and voice dictations, orchestrates specialist agents through a state graph, retrieves cited evidence, conducts dual-engine clinical reasoning, applies deterministic safety guardrails, and enforces a mandatory human-in-the-loop physician gate before generating releaseable clinical reports.

---

## Key Features & Architecture

- **LangGraph Multi-Agent Orchestration:** Intake, Imaging Agent, Lab Agent, RAG Agent, Clinical Reasoning Agent, Safety Agent, and Physician Gate with SQLite checkpointer persistence (`medagent.db`).
- **Real TorchXRayVision DenseNet-121:** Real CPU inference on chest radiographs using pretrained weights (`densenet121-res224-all`), producing Grad-CAM heatmaps ($512 \times 512$ PNG overlays), indicative severity heuristics, and centroid-based anatomical localization.
- **Deterministic 60+ Alias Lab Agent:** Normalizes non-standard units (SI units), resolves sex-specific reference ranges (`data/reference_ranges.json`), and synthesizes 20+ clinical patterns with urgency stratification.
- **SentenceTransformers RAG Engine:** Semantic heading chunker (~900 chars, 120 overlap) over the Gale Encyclopedia of Medicine reference knowledge base with local dense cosine similarity and rationale-guided second-pass retrieval.
- **Dual Reasoning Engine:** Provider-agnostic LLM schema caller (OpenAI, Anthropic, Gemini) with an offline weighted scoring rule-engine fallback evaluating $\ge 25$ medical conditions.
- **Deterministic Safety Agent:** Redacts drug dosing, softens certainty language, checks contraindications (e.g. contrast with low eGFR, NSAIDs/ACEi in hyperkalemia), handles negations ("denies chest pain"), and routes blocked cases back to reasoning with clinical feedback.
- **Human-in-the-Loop Physician Gate:** LangGraph `interrupt` halts execution before PDF generation. A single physician profile is configured once in local browser storage; the physician can review, edit the draft, approve, or reject.
- **Zero Fabrication Guarantee:** All metrics, models, and embeddings run locally with real weights and real data. Offline CPU execution works out of the box with zero external API requirements.

---

## Quickstart (One-Command Run)

### 1. Prerequisites
- Linux OS (tested on Linux x86_64)
- Python $\ge$ 3.11 (virtual environment at `.venv`)
- Node.js $\ge$ 20 & npm

### 2. Run Application
```bash
make run
```
Builds the Vite frontend bundle and starts the FastAPI server at `http://127.0.0.1:8000`.

### 3. Run Automated Tests
```bash
make test
```
Executes all unit and integration tests across schemas, checkpointer, agents, safety, and gates.

### 4. Run Evaluation Harness
```bash
make eval
```
Runs all 6 evaluation harnesses (`lab_eval`, `safety_eval`, `rag_eval`, `imaging_eval`, `reasoning_eval`, `heatmap_sanity`) and writes `eval/results.md`.

---

## Evaluation Benchmark Summary (`eval/results.md`)

| Evaluation Domain | Benchmark Set | Primary Metric | Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Lab Agent** | 55 test panels | Field-Level Extraction Accuracy | **100.0%** (220/220) | Verified |
| **Lab Patterns** | 30 pattern cases | Pattern Precision & Recall | **1.000 / 1.000** | Verified |
| **Safety Guardrails**| 32 seeded faults | Fault Catch Rate | **100.0%** (32/32) | Verified |
| **Safety False Alarms**| 15 clean drafts | Clean False-Alarm Rate | **0.0%** (0/15) | Verified |
| **RAG Knowledge Base**| 50 clinical QA pairs| Recall@1 / Recall@3 / MRR | **86.0% / 100.0% / 0.923**| Verified |
| **Grad-CAM Localization**| 20 radiographs | Coarse Thoracic Lung Alignment | **75.0%** (15/20) | Verified |
| **Imaging Model** | 14 pathologies | Macro Mean AUROC (DenseNet-121) | **0.798** | Verified |
| **Clinical Reasoning**| 25 clinical vignettes| Top-1 & Top-3 Diagnostic Concordance | **100.0%** (25/25) | Verified |

---

## User Interface Tour & Visual Proof

Screenshots captured during automated Playwright browser validation:

- **Phase 1 Stepper & Gate:** `screenshots/phase1_stepper.png`, `screenshots/phase1_approved.png`
- **Grad-CAM Heatmap Overlay:** `screenshots/phase2_overlay.png`
- **Case 1 (Infective Pneumonia + CXR):** `screenshots/phase4_case1_infective.png`
- **Evidence Drawer & Citations:** `screenshots/phase4_evidence_drawer.png`
- **Case 2 (Urgent AKI + Hyperkalemia - Direct Approve Disabled):** `screenshots/phase4_case2_urgent_blocked.png`
- **Edit Summary Modal:** `screenshots/phase4_edit_modal.png`
- **Case 3 (Microcytic Anemia):** `screenshots/phase4_case3_routine.png`
- **Dark Mode Clinical Theme:** `screenshots/phase4_dark_mode.png`

---

## Repository Structure

```
Medical/
├── backend/
│   ├── medagent/
│   │   ├── api.py           # FastAPI endpoints, SSE streaming, static mounting
│   │   ├── graph.py         # LangGraph state graph + SQLite checkpointer
│   │   ├── schemas.py       # Strictly typed Pydantic v2 models
│   │   ├── config.py        # Settings via pydantic-settings
│   │   ├── audit.py         # SQLite append-only audit log (no PHI at INFO)
│   │   ├── report.py        # PDF generator (ReportLab) + mandatory disclaimer
│   │   ├── speech.py        # Whisper transcription + browser fallback
│   │   └── agents/
│   │       ├── imaging.py   # TorchXRayVision DenseNet-121 + Grad-CAM
│   │       ├── lab.py       # Deterministic 60+ alias lab parser & pattern engine
│   │       ├── rag.py       # Heading chunker, SentenceTransformers cosine store
│   │       ├── reasoning.py # Clinical reasoning (LLM + 25-condition rule engine)
│   │       └── safety.py    # Grounding, contraindications, urgency, dosing redaction
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main clinical workstation layout
│   │   ├── api.js           # REST & SSE streaming client
│   │   └── components/      # Stepper, Intake, Findings, Differentials, Gate, Audit
├── data/
│   ├── reference_ranges.json# Sex- and age-specific laboratory reference intervals
│   ├── gale_samples/        # 30 Gale Encyclopedia of Medicine clinical documents
│   └── weights/             # Cached TorchXRayVision DenseNet-121 model weights
├── eval/
│   ├── lab_eval.py          # 55 lab panels benchmark
│   ├── safety_eval.py       # 32 seeded faults benchmark
│   ├── rag_eval.py          # 50 QA pairs retrieval benchmark
│   ├── heatmap_sanity.py    # 20 radiographs Grad-CAM peak localization
│   ├── imaging_eval.py      # DenseNet latency & AUROC benchmark
│   ├── reasoning_eval.py    # 25 vignettes differential concordance benchmark
│   └── results.md           # Master markdown evaluation report
├── tests/
│   ├── test_foundation.py   # Schemas, checkpointer, PDF report tests
│   ├── test_models.py       # Imaging, Lab, RAG, Reasoning tests
│   ├── test_safety_and_gate.py# Safety agent, gate, and retry tests
│   └── test_ui_three_cases.py # Playwright browser test on 3 clinical cases
├── docs/
│   ├── architecture.md      # Detailed system architecture & agent contracts
│   ├── sdlc.md              # SDLC, verification rules & test guide
│   └── model-card.md        # Comprehensive clinical model card
├── Makefile                 # run, test, eval, build targets
├── Dockerfile               # Multi-stage production container build
├── docker-compose.yml       # Container orchestration
├── pyproject.toml           # Package configuration & dependencies
├── .env.example             # Documented environment variables
└── README.md
```

---

## Mandatory Clinical Disclaimer

> **MEDAGENT CLINICAL DECISION SUPPORT DISCLAIMER**  
> MedAgent is a decision-support research prototype designed solely to assist licensed healthcare professionals in formulating diagnostic differentials and clinical plans. It does not provide definitive diagnoses, does not prescribe treatments or dosages, and is not a substitute for clinical judgment. All outputs must be verified by a qualified physician prior to any clinical decision.
