from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from app.config import Settings

logger = logging.getLogger(__name__)


KNOWN_LABELS = [
    "business",
    "entertainment",
    "food",
    "graphics",
    "historical",
    "medical",
    "politics",
    "space",
    "sport",
    "technologie",
]

ALL_LABELS = [*KNOWN_LABELS, "other"]


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    raw_label: str
    is_other: bool


class PredictiveModel(Protocol):
    def predict(self, texts: list[str]) -> list[str]:
        ...


class ProbabilisticModel(PredictiveModel, Protocol):
    classes_: np.ndarray

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        ...


class ModelUnavailableError(RuntimeError):
    pass


class MockClassifier:
    """Deterministic lightweight stand-in for the final TF-IDF model artifact."""

    labels = KNOWN_LABELS
    keyword_map = {
        "business": {"market", "company", "revenue", "investment", "stock", "customer"},
        "entertainment": {"movie", "music", "actor", "series", "concert", "festival"},
        "food": {"recipe", "restaurant", "chef", "meal", "flavor", "kitchen"},
        "graphics": {"image", "render", "pixel", "design", "animation", "3d"},
        "historical": {"history", "century", "ancient", "war", "empire", "museum"},
        "medical": {"patient", "doctor", "disease", "treatment", "clinical", "hospital"},
        "politics": {"election", "government", "policy", "senate", "minister", "vote"},
        "space": {"planet", "nasa", "orbit", "telescope", "galaxy", "mission"},
        "sport": {"team", "match", "season", "coach", "score", "championship"},
        "technologie": {"software", "computer", "ai", "data", "device", "network"},
    }

    def predict_with_confidence(self, text: str) -> Prediction:
        normalized = text.lower()
        scores = {
            label: sum(1 for keyword in keywords if keyword in normalized)
            for label, keywords in self.keyword_map.items()
        }
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        if best_score > 0:
            confidence = min(0.65 + (best_score * 0.08), 0.95)
            return Prediction(
                label=best_label,
                confidence=round(confidence, 4),
                raw_label=best_label,
                is_other=False,
            )

        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(self.labels)
        raw_label = self.labels[index]
        return Prediction(label="other", confidence=0.35, raw_label=raw_label, is_other=True)


class DocumentClassifierService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: ProbabilisticModel | None = None
        self.mock_model = MockClassifier()
        self.model_type = "mock-keyword-classifier"
        self.artifact_loaded = False

    def load(self) -> None:
        model_path = Path(self.settings.model_path)

        if not model_path.exists():
            if self.settings.require_model_artifact:
                raise ModelUnavailableError(f"Model artifact not found at {model_path}")

            logger.warning("Model artifact not found; using mock classifier")
            return

        loaded_model = joblib.load(model_path)

        if not hasattr(loaded_model, "predict") or not hasattr(loaded_model, "predict_proba"):
            raise ModelUnavailableError(
                "Loaded model must expose predict and predict_proba methods"
            )

        self.model = loaded_model
        self.model_type = "tfidf-calibrated-linearsvc"
        self.artifact_loaded = True
        logger.info("Loaded model artifact", extra={"path": str(model_path)})

    def predict(self, text: str) -> Prediction:
        if self.model is None:
            if self.settings.require_model_artifact:
                raise ModelUnavailableError("Model artifact is required but not loaded")
            return self.mock_model.predict_with_confidence(text)

        raw_label = str(self.model.predict([text])[0])
        probas = self.model.predict_proba([text])
        confidence = float(np.max(probas[0]))
        label = raw_label if confidence >= self.settings.confidence_threshold else "other"

        return Prediction(
            label=label,
            confidence=round(confidence, 4),
            raw_label=raw_label,
            is_other=label == "other",
        )

    def metadata(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "artifact_path": self.settings.model_path.as_posix(),
            "artifact_loaded": self.artifact_loaded,
            "artifact_required": self.settings.require_model_artifact,
            "confidence_threshold": self.settings.confidence_threshold,
            "labels": ALL_LABELS,
        }
