from functools import lru_cache
from pathlib import Path
import os


class Settings:
    app_name: str = "Document Classifier"
    app_version: str = "1.0.0"
    model_path: Path = Path(
        os.getenv("MODEL_PATH", "models/linear_svc_tfidf_calibrated.joblib")
    )
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
    confidence_threshold_overridden: bool = "CONFIDENCE_THRESHOLD" in os.environ
    max_document_chars: int = int(os.getenv("MAX_DOCUMENT_CHARS", "100000"))
    require_model_artifact: bool = os.getenv("REQUIRE_MODEL_ARTIFACT", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()
