# Multi-stage Dockerfile for MedAgent
# Stage 1: Build Frontend Assets
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend Runtime
FROM python:3.12-slim AS runner

# System dependencies for PyTorch CPU, imaging, and PDF generation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy application code and assets
COPY backend/ ./backend/
COPY data/ ./data/
COPY samples/ ./samples/
COPY Makefile ./

# Copy compiled frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Environment configuration
ENV PYTHONPATH=/app/backend \
    HOST=0.0.0.0 \
    PORT=8000 \
    LLM_PROVIDER=offline \
    VECTOR_BACKEND=numpy \
    HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "medagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
