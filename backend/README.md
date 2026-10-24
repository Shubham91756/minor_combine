# Parkinsons Models Backend

This small FastAPI app exposes three endpoints that wrap the trained models for handwriting, motor, and gait predictions.

Prerequisites
- Python 3.10+ (project uses a virtual environment `.venv`)

Install

1. cd backend
2. python -m venv .venv; .\.venv\Scripts\Activate.ps1
3. pip install -r requirements.txt

Run

Start the server:

APP_ENV=development uvicorn backend.main:app --host 127.0.0.1 --port 8001

For a deployment, configure an API key and do not use `--reload`:

```bash
export APP_ENV=production
export PARKINSONS_API_KEY="$(openssl rand -hex 32)"
export FRONTEND_ORIGINS="https://app.example.com"
uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

Authenticated requests must include:

```text
X-API-Key: <value of PARKINSONS_API_KEY>
```

When `APP_ENV=development` and no API key is configured, only loopback
requests are accepted. The `/healthz` endpoint remains available for a
reverse proxy liveness check. Do not expose the development configuration to
the public internet.

Endpoints

- POST /predict/handwriting
  - multipart file: image (jpg/png)
  - returns JSON with label and probability

- POST /predict/motor
  - JSON body: {"features": {"c1_mean": 0.1, "c1_std": 0.2, ...}}
  - returns JSON with prediction and probability

- POST /predict/motor_csv
  - multipart file: CSV with feature columns matching trained motor features
  - returns list of predictions for each row

- POST /predict/gait
  - multipart file: gait timeseries .txt (tab/space separated columns as in dataset)
  - returns JSON with prediction and probability

Examples (curl)

1) Handwriting:
curl -X POST "http://127.0.0.1:8000/predict/handwriting" -F "file=@/path/to/image.jpg"

2) Motor (JSON):
curl -X POST "http://127.0.0.1:8000/predict/motor" -H "Content-Type: application/json" -d '{"features": {"c1_mean": 0.1}}'

3) Motor CSV:
curl -X POST "http://127.0.0.1:8000/predict/motor_csv" -F "file=@/path/to/motor_features.csv"

4) Gait:
curl -X POST "http://127.0.0.1:8000/predict/gait" -F "file=@/path/to/GaCo01_01.txt"

Authenticated example:

```bash
curl -H "X-API-Key: $PARKINSONS_API_KEY" \
  -X POST "http://127.0.0.1:8001/predict/motor_csv" \
  -F "file=@/path/to/motor_features.csv"
```

The frontend reads `REACT_APP_API_KEY` when direct API authentication is
required. Browser-embedded API keys are visible to users, so for production
prefer an authenticated reverse proxy or a per-user authentication system.
Prediction history is not persisted in the browser unless
`REACT_APP_PERSIST_HISTORY=1` is explicitly set.
