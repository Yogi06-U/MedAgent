"""Configuration and environment settings for MedAgent."""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable override support."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    KNOWLEDGE_DIR: Path = (BASE_DIR / "data" / "gale_samples") if (BASE_DIR / "data" / "gale_samples").exists() else (BASE_DIR / "backend" / "medagent" / "knowledge")
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    DB_PATH: Path = BASE_DIR / "medagent.db"

    # Reference data
    REFERENCE_RANGES_FILE: Path = (BASE_DIR / "data" / "reference_ranges.json") if (BASE_DIR / "data" / "reference_ranges.json").exists() else (BASE_DIR / "backend" / "medagent" / "data" / "reference_ranges.json")

    # LLM Settings
    LLM_PROVIDER: str = "offline"  # 'offline', 'gemini', 'openai', 'anthropic'
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = ""

    # Vector / RAG Settings
    VECTOR_BACKEND: str = "numpy"  # 'numpy', 'pinecone'
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Imaging & Laboratory Model Settings
    TORCHXRAYVISION_WEIGHTS: str = "densenet121-res224-all"
    CLINICAL_LAB_WEIGHTS: Path = (BASE_DIR / "data" / "weights" / "clinical_lab_model.pt")
    ENABLE_MEDGEMMA: bool = False
    MEDGEMMA_MODEL_ID: str = "google/medgemma-4b-it"

    # Speech / Audio
    WHISPER_MODEL: str = "tiny"

    # Server & Security
    HOST: str = "127.0.0.1"
    PORT: int = 8008
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        self.REFERENCE_RANGES_FILE.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
