# Document Classification API

FastAPI service for the document classification case study. The API loads the saved TF-IDF + calibrated LinearSVC `joblib` artifact from the `models` folder and exposes it through a production-style FastAPI interface.

## Approach

The model exploration is documented in `training_notebook/FindingBestModel.ipynb`. The dataset contains 11 folders, but the `other` folder is intentionally not used as a supervised training class. The trained classifier learns the 10 known categories (`business`, `entertainment`, `food`, `graphics`, `historical`, `medical`, `politics`, `space`, `sport`, and `technologie`), then uses a confidence threshold to return `other` for low-confidence, likely out-of-distribution documents.

Two main approaches were evaluated:

- **ModernBERT**: a transformer encoder was adapted with a classification head for the 10 known classes. Training was done in two stages: first only the new head was trained, then the full model was fine-tuned. This gradual unfreezing avoids a large optimization shock to the pretrained weights when the randomly initialized head starts learning. ModernBERT produced the strongest validation performance in the notebook and was trained on Google Colab GPU.
- **TF-IDF + calibrated LinearSVC**: this classical text-classification baseline reached comparable performance while being much smaller, simpler to deploy, and faster for CPU inference. Because the case study asks to consider computational cost and scaling to large volumes, this is the model selected for the API.

The final API artifact is therefore a `TfidfVectorizer` + `LinearSVC` pipeline wrapped with calibration so `predict_proba` is available. The calibrated confidence is used both for reporting confidence and for deciding when to return `other`.

## Serving Design

The FastAPI application loads the model once during application startup through the lifespan hook in `app.main`. The loaded estimator is kept in the process-level `DocumentClassifierService`, so each request only performs validation, vectorization, prediction, and response serialization. This avoids reloading the `joblib` artifact on every request, which would add unnecessary latency and disk I/O.

For production-style serving, the recommended command uses Gunicorn with Uvicorn workers:

```bash
gunicorn app.main:app --workers 2 --worker-class uvicorn_worker.UvicornWorker --bind 0.0.0.0:8000
```

Uvicorn provides the ASGI server implementation for FastAPI, while Gunicorn manages multiple worker processes, restarts unhealthy workers, and lets the service use more CPU cores. This matters more than making the inference endpoint `async`: model inference is CPU-bound work, so an `async def` endpoint would not make the classification computation non-blocking. Scaling this service is better handled by multiple worker processes and, if needed, multiple container replicas.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation.

The `scripts/classify_with_api.ipynb` notebook contains example classification requests against the running API, so reviewers can quickly test the endpoint with sample document text.

## Production-Style Run

```bash
gunicorn app.main:app --workers 2 --worker-class uvicorn_worker.UvicornWorker --bind 0.0.0.0:8000
```

This command is intended for Linux-based production or the provided Docker image. On Windows, use the local Uvicorn command above for development.

## Docker

```bash
docker build -t document-classifier .
docker run -p 8000:8000 document-classifier
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
  "confidence": 0.69,
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

The final TF-IDF model is saved at `models/linear_svc_tfidf_calibrated.joblib` by default. You can override the path with `MODEL_PATH`.

The intended production model mirrors the notebook decision:

- Train on all labels except `other`.
- Use TF-IDF with calibrated LinearSVC so `predict_proba` is available.
- Return the raw predicted label when confidence is at least the artifact threshold, currently `0.45`.
- Return `other` when confidence is below that threshold.

`REQUIRE_MODEL_ARTIFACT` defaults to `true` so the service fails fast when the model artifact is missing. Set it to `false` only for local fallback testing.

## Tests

```bash
pytest
```
