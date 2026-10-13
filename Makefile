.PHONY: all run test eval build clean install lint help

PYTHON := .venv/bin/python3
UVICORN := .venv/bin/uvicorn
PYTEST := .venv/bin/pytest
NPM := npm

PORT ?= 8008

help:
	@echo "MedAgent - Clinical Decision Support Multi-Agent System"
	@echo "Usage:"
	@echo "  make run      - Run application (backend + static UI on http://127.0.0.1:8008)"
	@echo "  make test     - Run automated test suite (unit and integration tests)"
	@echo "  make eval     - Run evaluation harness and update eval/results.md"
	@echo "  make build    - Build frontend production bundle (frontend/dist)"
	@echo "  make install  - Install Python and Node dependencies"
	@echo "  make clean    - Remove build artifacts, pycache, and temp files"

install:
	$(PYTHON) -m pip install -e .
	cd frontend && $(NPM) install

build:
	cd frontend && $(NPM) run build

run: build
	PYTHONPATH=backend $(UVICORN) medagent.api:app --host 127.0.0.1 --port $(PORT)


test:
	PYTHONPATH=backend $(PYTEST) tests/test_foundation.py tests/test_models.py tests/test_safety_and_gate.py -v

test-all:
	PYTHONPATH=backend $(PYTEST) tests/ -v

eval:
	@echo "Running Lab Agent evaluation..."
	PYTHONPATH=backend $(PYTHON) eval/lab_eval.py
	@echo "Running Safety Agent evaluation..."
	PYTHONPATH=backend $(PYTHON) eval/safety_eval.py
	@echo "Running RAG Agent evaluation..."
	PYTHONPATH=backend $(PYTHON) eval/rag_eval.py
	@echo "Running Imaging Model benchmark..."
	PYTHONPATH=backend $(PYTHON) eval/imaging_eval.py
	@echo "Running Reasoning Agent evaluation..."
	PYTHONPATH=backend $(PYTHON) eval/reasoning_eval.py
	@echo "Running Heatmap Localization Sanity evaluation..."
	PYTHONPATH=backend $(PYTHON) eval/heatmap_sanity.py
	@echo "Evaluation complete. Results written to eval/results.md"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf frontend/dist
