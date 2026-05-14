# Trellis Document Classification API

FastAPI service for the Trellis document classification case study. The current implementation uses a deterministic mocked classifier behind the same interface expected by the final TF-IDF + calibrated LinearSVC `joblib` model.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

## Production-Style Run

```bash
gunicorn app.main:app --workers 2 --worker-class uvicorn_worker.UvicornWorker --bind 0.0.0.0:8000
```

## Docker

```bash
docker build -t trellis-document-classifier .
docker run -p 8000:8000 trellis-document-classifier
```

## Endpoints

### `POST /classify_document`

Request:

```json
{
  "document_text": "The team won the championship after a dramatic final match."
}
```

Success response, `200 OK`:

```json
{
  "message": "Classification successful",
  "label": "sport",
  "confidence": 0.81,
  "raw_label": "sport",
  "is_other": false
}
```

Error responses:

- `422 Unprocessable Entity`: invalid request body, missing `document_text`, empty text, or text longer than `MAX_DOCUMENT_CHARS`.
- `503 Service Unavailable`: model artifact is required but unavailable.
- `500 Internal Server Error`: unexpected inference failure.

### `GET /health`

Returns API liveness.

### `GET /model/metadata`

Returns model type, artifact path, artifact status, confidence threshold, and labels.

## Model Artifact

The final TF-IDF model should be saved at `models/document_classifier.joblib` or configured with `MODEL_PATH`.

The intended production model mirrors the notebook decision:

- Train on all labels except `other`.
- Use TF-IDF with calibrated LinearSVC so `predict_proba` is available.
- Return the raw predicted label when confidence is at least `0.45`.
- Return `other` when confidence is below `0.45`.

Set `REQUIRE_MODEL_ARTIFACT=true` to fail startup/inference if the artifact is missing.

## Tests

```bash
pytest
```
