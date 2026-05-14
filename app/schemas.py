from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.config import get_settings


class ClassificationRequest(BaseModel):
    document_text: str = Field(
        ...,
        description="Text document to classify.",
        examples=["The team won the championship after a dramatic final match."],
    )

    @field_validator("document_text")
    @classmethod
    def validate_document_text(cls, value: str) -> str:
        settings = get_settings()
        text = value.strip()

        if not text:
            raise ValueError("document_text must not be empty")

        if len(text) > settings.max_document_chars:
            raise ValueError(
                f"document_text must be at most {settings.max_document_chars} characters"
            )

        return text


class ClassificationResponse(BaseModel):
    message: str = "Classification successful"
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_label: str
    is_other: bool


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str


class ModelMetadataResponse(BaseModel):
    model_type: str
    artifact_path: str
    artifact_loaded: bool
    artifact_required: bool
    confidence_threshold: float
    labels: list[str]
    artifact_sklearn_version: str | None = None


class ErrorResponse(BaseModel):
    message: str
    detail: Any | None = None
