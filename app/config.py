from functools import lru_cache
from pathlib import Path
import os


class Settings:
    app_name: str = "Trellis Document Classifier"
    app_version: str = "1.0.0"
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/document_classifier.joblib"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
    max_document_chars: int = int(os.getenv("MAX_DOCUMENT_CHARS", "100000"))
    require_model_artifact: bool = os.getenv("REQUIRE_MODEL_ARTIFACT", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()
