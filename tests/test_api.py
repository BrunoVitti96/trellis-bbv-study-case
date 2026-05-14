import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_service import ALL_LABELS, ModelUnavailableError


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_classify_document_success(client):
    response = client.post(
        "/classify_document",
        json={
            "document_text": "The team won the championship after the coach changed tactics."
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["message"] == "Classification successful"
    assert payload["label"] in ALL_LABELS
    assert 0 <= payload["confidence"] <= 1
    assert payload["raw_label"] in ALL_LABELS
    assert isinstance(payload["is_other"], bool)


def test_classify_document_missing_document_text(client):
    response = client.post("/classify_document", json={})

    assert response.status_code == 422
    assert response.json()["message"] == "Invalid request body"


def test_classify_document_empty_document_text(client):
    response = client.post("/classify_document", json={"document_text": "   "})

    assert response.status_code == 422
    assert response.json()["message"] == "Invalid request body"


def test_classify_document_too_long(client):
    response = client.post("/classify_document", json={"document_text": "a" * 100001})

    assert response.status_code == 422
    assert response.json()["message"] == "Invalid request body"


def test_model_unavailable_returns_503(client, monkeypatch):
    from app import main

    def raise_unavailable(_text):
        raise ModelUnavailableError("not loaded")

    monkeypatch.setattr(main.model_service, "predict", raise_unavailable)

    response = client.post(
        "/classify_document",
        json={"document_text": "This text should trigger a service-level model failure."},
    )

    assert response.status_code == 503
    assert response.json()["message"] == "Model unavailable"


def test_unexpected_inference_error_returns_500(client, monkeypatch):
    from app import main

    def raise_unexpected(_text):
        raise RuntimeError("boom")

    monkeypatch.setattr(main.model_service, "predict", raise_unexpected)

    response = client.post(
        "/classify_document",
        json={"document_text": "This text should trigger an unexpected inference failure."},
    )

    assert response.status_code == 500
    assert response.json()["message"] == "Inference failed"


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_metadata(client):
    response = client.get("/model/metadata")

    assert response.status_code == 200
    payload = response.json()
    assert payload["confidence_threshold"] == 0.45
    assert payload["artifact_path"] == "models/linear_svc_tfidf_calibrated.joblib"
    assert payload["labels"] == ALL_LABELS
