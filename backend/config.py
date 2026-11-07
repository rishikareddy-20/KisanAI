"""Backend configuration. Values come from env vars (see .env.example) with defaults."""
from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Directory containing backend/, models/, etc.
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load .env so non-prefixed keys read via os.environ are available too.
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KISANAI_",
        extra="ignore",
    )

    # Vision
    tflite_model: str = "models/kisanai_efficientnet.tflite"
    labels: str = "models/labels.json"
    confidence_threshold: float = 0.55

    # RAG
    kb_path: str = "knowledge_base/diseases.json"
    rag_index: str = "backend/rag/index"

    # TTS
    tts_provider: str = "none"  # none | gtts

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173"

    def path(self, value: str) -> Path:
        """Resolve a relative config path against the project root."""
        p = Path(value)
        return p if p.is_absolute() else (ROOT_DIR / p)

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
