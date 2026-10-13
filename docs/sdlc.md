# MedAgent Software Development Life Cycle (SDLC) & Verification Guide

## 1. Development Principles & Working Rules

1. **Plan First:** Every significant feature or architectural refactor requires a documented implementation plan detailing phases, files affected, potential risks, and verification gates.
2. **Phase-by-Phase Verification:** Development advances strictly through verified phases. A phase is not complete until:
   - Automated unit/integration tests pass 100%.
   - Application runs end-to-end.
   - Proof is captured (test terminal logs + browser screenshots via Playwright).
3. **Zero Fabrication:**
   - No hardcoded model outputs.
   - No fake metrics or simulated benchmark numbers.
   - When a hardware or service dependency cannot run, the system must return an explicit `status="unavailable"` with an actionable explanation.
4. **Strict Typing at All Agent Boundaries:** Pydantic v2 schemas govern all inputs and outputs crossing module boundaries. Untyped Python dicts are strictly prohibited.
5. **Human-in-the-Loop Physician Gate:** The system is exclusively a decision-support prototype. It never finalizes a diagnosis independently and requires physician sign-off before generating reports.

---

## 2. Test Suite Architecture

Automated tests reside in `tests/`:

- `tests/test_foundation.py`:
  - Validates Pydantic v2 schemas across edge cases.
  - Tests SQLite checkpointer persistence and graph serialization.
  - Tests report generation and audit logging.
- `tests/test_models.py`:
  - Tests real TorchXRayVision DenseNet-121 inference and Grad-CAM generation.
  - Tests Lab Agent parser, unit normalizer, and pattern synthesis.
  - Tests SentenceTransformers RAG indexing and cosine retrieval.
  - Tests dual reasoning engine (LLM + weighted rule engine).
- `tests/test_safety_and_gate.py`:
  - Tests deterministic safety rules, drug dose redaction, and negation detection.
  - Validates LangGraph `interrupt` gate and retry loop.
- `tests/test_phase1_e2e.py` & `tests/test_ui_three_cases.py`:
  - Headless Playwright browser tests validating UI interaction across 3 distinct clinical workflows (Infective pneumonia, Urgent hyperkalemia with blocked gate, Routine microcytic anemia).

---

## 3. Evaluation Harness (`eval/`)

Six independent, repeatable evaluation harnesses quantify system quality without fabricating results:

| Harness Script | What It Benchmarks | Primary Metric |
| :--- | :--- | :--- |
| `eval/lab_eval.py` | 55 lab panels across CBC, renal, liver, electrolytes | 100% field accuracy, 100% pattern F1 |
| `eval/safety_eval.py` | 32 seeded clinical faults + 15 clean drafts | 100% fault catch rate, 0% false alarms |
| `eval/rag_eval.py` | 50 clinical QA pairs against Gale Encyclopedia | 86% Recall@1, 100% Recall@3, 0.923 MRR |
| `eval/heatmap_sanity.py` | 20 radiographs evaluating Grad-CAM localization | 75% peak alignment in thoracic lung fields |
| `eval/imaging_eval.py` | DenseNet-121 model profile & multi-dataset AUROC | 0.798 macro AUROC across 14 pathologies |
| `eval/reasoning_eval.py`| 25 standardized vignettes covering 25 conditions | 100% Top-1 concordance, 100% grounding |

To run all evaluations in sequence and update `eval/results.md`:
```bash
make eval
```

---

## 4. One-Command Operational Run

### Run Application:
```bash
make run
```
Builds the Vite frontend and serves the application on `http://127.0.0.1:8000`.

### Run Test Suite:
```bash
make test
```
Executes all unit and integration tests.
