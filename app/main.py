from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.logging_config import configure_logging
from app.model_service import DocumentClassifierService, ModelUnavailableError
from app.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    ErrorResponse,
    HealthResponse,
    ModelMetadataResponse,
)

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

model_service = DocumentClassifierService(settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="REST API for classifying text documents into assessment categories.",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception(
            "Unhandled request failure",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
            },
        )
        raise

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "message": "Invalid request body",
            "detail": jsonable_encoder(exc.errors()),
        },
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
    )


@app.get("/model/metadata", response_model=ModelMetadataResponse)
def model_metadata() -> ModelMetadataResponse:
    return ModelMetadataResponse(**model_service.metadata())


@app.post(
    "/classify_document",
    response_model=ClassificationResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Invalid request body"},
        503: {"model": ErrorResponse, "description": "Model unavailable"},
        500: {"model": ErrorResponse, "description": "Unexpected inference failure"},
    },
)
def classify_document(request: ClassificationRequest) -> ClassificationResponse:
    try:
        prediction = model_service.predict(request.document_text)
    except ModelUnavailableError as exc:
        logger.error("Model unavailable", extra={"error_type": type(exc).__name__})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "message": "Model unavailable",
                "detail": "The classification model is not ready to serve predictions.",
            },
        )
    except Exception as exc:
        logger.exception("Inference failed", extra={"error_type": type(exc).__name__})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "message": "Inference failed",
                "detail": "An unexpected error occurred while classifying the document.",
            },
        )

    logger.info(
        "Document classified",
        extra={
            "label": prediction.label,
            "confidence": prediction.confidence,
            "raw_label": prediction.raw_label,
            "is_other": prediction.is_other,
        },
    )

    return ClassificationResponse(
        label=prediction.label,
        confidence=prediction.confidence,
        raw_label=prediction.raw_label,
        is_other=prediction.is_other,
    )
