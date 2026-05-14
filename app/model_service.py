from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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
        self.artifact_sklearn_version: str | None = None
        self.other_label = "other"
        self.labels = ALL_LABELS

    def load(self) -> None:
        model_path = Path(self.settings.model_path)

        if not model_path.exists():
            if self.settings.require_model_artifact:
                raise ModelUnavailableError(f"Model artifact not found at {model_path}")

            logger.warning("Model artifact not found; using mock classifier")
            return

        loaded_artifact = joblib.load(model_path)
        loaded_model = self._extract_model(loaded_artifact)

        if not hasattr(loaded_model, "predict") or not hasattr(loaded_model, "predict_proba"):
            raise ModelUnavailableError(
                "Loaded model must expose predict and predict_proba methods"
            )

        self.model = loaded_model
        self.model_type = "tfidf-calibrated-linearsvc"
        self.artifact_loaded = True
        self.labels = self._labels_from_model(loaded_model)
        logger.info("Loaded model artifact", extra={"path": str(model_path)})

    def _extract_model(self, artifact: Any) -> ProbabilisticModel:
        if not isinstance(artifact, dict):
            return artifact

        model = artifact.get("model")
        if model is None:
            raise ModelUnavailableError("Model artifact dictionary is missing 'model'")

        artifact_threshold = artifact.get("threshold")
        if artifact_threshold is not None and not self.settings.confidence_threshold_overridden:
            self.settings.confidence_threshold = float(artifact_threshold)

        other_label = artifact.get("other_label")
        if other_label:
            self.other_label = str(other_label)

        sklearn_version = artifact.get("sklearn_version")
        if sklearn_version:
            self.artifact_sklearn_version = str(sklearn_version)

        return model

    def _labels_from_model(self, model: ProbabilisticModel) -> list[str]:
        model_classes = [str(label) for label in getattr(model, "classes_", [])]
        if not model_classes:
            model_classes = KNOWN_LABELS

        if self.other_label not in model_classes:
            model_classes.append(self.other_label)

        return model_classes

    def predict(self, text: str) -> Prediction:
        if self.model is None:
            if self.settings.require_model_artifact:
                raise ModelUnavailableError("Model artifact is required but not loaded")
            return self.mock_model.predict_with_confidence(text)

        raw_label = str(self.model.predict([text])[0])
        probas = self.model.predict_proba([text])
        confidence = float(np.max(probas[0]))
        label = (
            raw_label
            if confidence >= self.settings.confidence_threshold
            else self.other_label
        )

        return Prediction(
            label=label,
            confidence=round(confidence, 4),
            raw_label=raw_label,
            is_other=label == self.other_label,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "artifact_path": self.settings.model_path.as_posix(),
            "artifact_loaded": self.artifact_loaded,
            "artifact_required": self.settings.require_model_artifact,
            "confidence_threshold": self.settings.confidence_threshold,
            "labels": self.labels,
            "artifact_sklearn_version": self.artifact_sklearn_version,
        }
