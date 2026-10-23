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

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

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
