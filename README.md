# NEUROFUSION – PD , Early Detection and Classification of Parkinson’s Disease


This repository contains code, trained models and example data used to build
three clinical-signal classifiers: a gait signals classifier, a motor-tasks
classifier, and a handwriting image classifier. The project includes data
preprocessing utilities, training scripts, evaluation tooling and a FastAPI
backend that exposes model inference endpoints for each task.

This README documents the full pipeline, features, model architectures,
training recipes and how to use the models and backend. It is intended to be
used as the basis for a methods section in a research paper and to produce
slides for a presentation.

## Contents

- Project overview
- Data and feature engineering (3 canonical feature groups)
- Models (gait, motor, handwriting) — architectures & training
- Evaluation summaries and sample metrics
- Deployment and inference pipeline (FastAPI endpoints)
- Files and artifacts (paths to trained models)
- Reproducibility notes and suggestions for further work

---

## Results summary (from model result files)

### Gait model

- **Confusion matrix:**
  - True Negatives (class 0): 83
  - False Positives: 9
  - False Negatives: 0
  - True Positives (class 1): 214
  - **Accuracy:** 97.1%
  - **Precision (class 1):** 96%
  - **Recall (class 1):** 100%
- **ROC AUC:** 0.852 (good to excellent discrimination)
- **Average Precision (AP):** 0.935 (excellent)

Interpretation: The gait model is extremely good at finding class '1' (abnormal gait), with perfect recall and high precision. The ROC and PR curves confirm strong performance.

### Motor model

- **Confusion matrix (3-class):**
  - Class 0: 68 correct, 10 as class 1, 1 as class 2
  - Class 1: 2 as class 0, 269 correct, 5 as class 2
  - Class 2: 5 as class 0, 21 as class 1, 88 correct
- **ROC AUC:**
  - Class 0: 0.951 (excellent)
  - Class 1: 0.901 (very good)
  - Micro-average: 0.887
- **Average Precision (AP):**
  - Class 1: 0.897 (excellent)
  - Class 0: 0.755 (decent)
  - Micro-average: 0.769

Interpretation: The motor model is excellent at identifying class 1, strong for class 0, but weaker for class 2 (often confused with class 1). ROC and PR curves confirm high discriminatory power, but class 2 is a weakness.

### Handwriting model

- **Confusion matrix:**
  - True Negatives (healthy): 82
  - False Positives: 9
  - False Negatives: 6
  - True Positives (patient): 50
- **Validation loss (30 epochs):**
  - Starts ~0.147, ends <0.09 (steady improvement)
- **Accuracy (50 epochs):**
  - Highly variable: from ~76% to ~91.5%
  - Final epoch: ~91.5%

Interpretation: The handwriting model shows learning (validation loss decreases), but accuracy is unstable across epochs, suggesting the need for more stable training (possibly lower learning rate). Final accuracy can be high, but reliability is questionable.

---

## Methods (detailed)

This section collects precise methodological details to support a reproducible Methods section in a paper. Copy the relevant subsections into your manuscript and cite the repository and commit.

### Data preprocessing

- Gait and motor timeseries preprocessing steps (implemented in `models/utils/*` and trainer scripts):
  1. Resampling: signals are resampled to a fixed sampling rate when required (median of `dt` is used to detect original sampling). If downsampling is needed, a decimation with anti-aliasing filter is applied.
  2. Detrending: linear detrend applied per channel to remove sensor drift.
  3. Bandpass filtering (optional, used for enhanced gait features): 0.5–20 Hz Butterworth bandpass (4th order) applied to accelerometer channels to emphasize gait-related frequencies.
  4. Magnitude computation: vector magnitude is computed as sqrt(x^2 + y^2 + z^2) and used to derive global statistics (`acc_mag_mean`, `acc_mag_std`).
  5. Windowing (motor): features are computed either on the whole recording (per-subject) or using non-overlapping windows of configurable length (default 2s) and aggregated using mean/median.

Exact code references: `models/eda/normalize_and_compare.py`, `models/trainers/train_gait.py`, and `tools/further_retrain_motor.py` contain the implementations and parameters used in the reported runs.

### Feature formulas (selected)

- Mean: \(\mu = \frac{1}{N} \sum_{i=1}^N x_i\)
- Standard deviation: \(\sigma = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i-\mu)^2}\)
- RMS (root mean square): \(\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2}\)
- Energy: \(\text{Energy} = \sum_{i=1}^N x_i^2\)
- Peak-to-peak: \(\text{PTP} = \max(x) - \min(x)\)
- Spectral energy (band): compute FFT and sum squared magnitudes in the frequency band of interest (e.g., 0.5–3.0 Hz for step cadence).

These formulas are implemented in `models/utils/feature_utils.py` (search for `rms`, `energy`, `ptp`). If you need the exact implementation lines, I can extract and paste them verbatim.

### Training hyperparameters and selection

- Gait (RandomForest): n_estimators=100, random_state=42, class_weight=None. Stratified 80/20 train/test split.
- Motor (RandomForest retrain used in repo): n_estimators=300, class_weight='balanced', StratifiedKFold(n_splits=5) with selection by recall on class 0 (healthy). Synthetic healthy augmentation: 5000 synthesized samples drawn from multivariate Gaussian using observed healthy mean/covariance.
- Handwriting (EfficientNetB0 fine-tune): backbone=EfficientNetB0 imagenet weights, optimizer=Adam, initial_lr=1e-4 (head), fine-tune_lr=1e-5, batch_size=32 (configurable), epochs=head_train_epochs + finetune_epochs (configurable). EarlyStopping with patience=5 on validation loss; ModelCheckpoint saves the best validation model.

Seeds
- All scripts set numpy and sklearn random_state where applicable (42). Keras/TensorFlow seed is set via `tf.random.set_seed(42)` in the trainer.

### Cross-validation and model selection

- Motor: StratifiedKFold (n_splits=5). For each fold synthetic healthy samples were added to the training fold only (not validation) to avoid leakage. The selected fold was the one with the highest recall on the healthy class in validation (to prioritize sensitivity for healthy detection as required in this project).
- Gait: hold-out stratified split reported in `gait_classification_report.txt` and per-file ROC/AP computed from `gait_per_file_predictions.csv`.

### Metrics definitions

- Accuracy: (TP + TN) / total
- Precision (class i): TP_i / (TP_i + FP_i)
- Recall (sensitivity, class i): TP_i / (TP_i + FN_i)
- F1-score: harmonic mean of precision and recall
- ROC AUC: area under ROC curve (for multiclass computed in one-vs-rest by class)
- Average Precision (AP): area under the precision-recall curve (class-wise and micro-avg)
- Brier score: mean squared error between predicted probability and true label (used in calibration analysis)

### Reproducibility & environment

- Python & packages: a sample of installed packages (truncated) was written in the Appendix; please use the `.venv` from this repo or recreate using `pip install -r backend/requirements.txt` and `pip install -r models/requirements.txt` where provided.
- Hardware: handwriting fine-tuning benefits from GPU (NVIDIA CUDA); RandomForest training runs fine on CPU.
- Re-run commands (examples):

```powershell
.venv\Scripts\python.exe tools/further_retrain_motor.py    # retrain motor model
.venv\Scripts\python.exe models/trainers/train_gait.py   # retrain gait model
.venv\Scripts\python.exe models/trainers/train_handwriting_efficient.py --mode end2end
```

### Limitations and ethical considerations

- Data provenance: training data combines control and patient cohorts collected at different centers; care must be taken when interpreting model generalization across populations.
- Class imbalance: motor dataset contains a dominant class; synthetic augmentation was used to mitigate imbalance but may introduce distributional artifacts. Evaluate on truly held-out clinical cohorts before deployment.
- Clinical use: models are research prototypes and not validated for clinical diagnosis. Do not use for patient management without thorough prospective evaluation and regulatory review.

---

If you'd like, I can now:
- extract the exact lines of code that implement each feature (e.g., RMS, energy) and insert them into this README; or
- run the motor probability extraction and compute motor ROC/AP and permutation importance, then append the resulting numeric outputs and plots into the README and `visualizations/`.

Tell me which of those you'd like next and I will proceed.

This project implements three classifiers that operate on different input
modalities but share a common design principle: produce a small set of
interpretable features from raw signals (or embeddings for images), train a
robust classifier, evaluate both per-sample and aggregated performance, and
expose inference via a simple REST API.

- Gait classifier: RandomForest trained on per-file gait summary features.
- Motor classifier: RandomForest trained on per-subject aggregated features
  extracted from motor task timeseries; synthetic healthy augmentation used
  during training to improve sensitivity for the healthy class.
- Handwriting classifier: EfficientNet-based Keras model (pretrained
  EfficientNetB0 backbone with a small head and optional fine-tuning).

Models and example artifacts are stored under `models/trained_models/`.

## Key features and data schema (3 canonical feature groups)

Across the gait and motor pipelines we compute a common set of statistical and
signal-derived features. For the README / paper, we highlight three types of
features used throughout the project and why they are useful:

1. Time-domain summary statistics (per channel)
   - Examples: mean, std (standard deviation), median, min, max, peak-to-peak
     (ptp), root-mean-square (rms), energy (sum of squares).
   - Why: capture baseline offsets, variability and gross amplitude — useful
     for discriminating healthy vs. pathologic movement where amplitude and
     variance patterns differ.

2. Derived kinematic metrics (peak features, cadence / step metrics)
   - Examples (gait-specific enhanced features): center-of-pressure (COP)
     related features, step-time and step-length proxies, spectral energy in
     frequency bands.
   - Why: these capture rhythm and periodic structure which are often altered
     by disease (irregular cadence, asymmetric steps, altered power spectra).

3. Global/summary features (aggregated magnitude statistics)
   - Examples: vector magnitude mean/std (acc_mag_mean, acc_mag_std), total
     energy across channels, recording duration and sampling stats (nrows,
     median_dt).
   - Why: provide cross-signal normalization and context (short/long
     recordings, sensor magnitude differences).

These three feature groups are combined into a final per-file or per-subject
feature vector that is used to train the tabular classifiers. The motor
pipeline additionally uses per-subject aggregation (mean/median across time
windows) and performs synthetic augmentation for the healthy class during
training to balance representation.

## Models, training recipes and implementation details

Below I provide the training recipe, architecture and implementation notes
for each of the three models. Exact code references (scripts) are listed so
you can reproduce the experiments.

### Gait classifier

- Artifact: `models/trained_models/gait_signals/model_gait_enhanced_calibrated.joblib`
- Training script: `models/trainers/train_gait.py`

Training recipe and architecture
- Input: raw gait timeseries files (.txt) read by `models.utils.dataset.load_gait_file`.
- Feature extraction: `extract_gait_features` (basic) or
  `extract_gait_features_enhanced` (COP, step metrics, spectral energy).
- Labels: inferred heuristically from filenames (Pt → patient, Co → control).
- Model: scikit-learn RandomForestClassifier (n_estimators=100, random_state=42).
- Train/test split: stratified 80/20 split; random_state=42.
- Persistence: model saved with joblib to the artifact path.

Relevant code snippet: `models/trainers/train_gait.py` — builds the feature
matrix, trains RandomForest and prints + saves classification report.

Evaluation
- Source classification report (saved under `models/trained_models/gait_signals/gait_classification_report.txt`):

```
precision / recall / f1 / support

0: precision 1.00, recall 0.90, f1 0.95 (support 92)
1: precision 0.96, recall 1.00, f1 0.98 (support 214)

accuracy: 0.97 (306 samples)
```

Additional artifacts produced during analysis:
- `models/trained_models/gait_signals/gait_per_file_predictions.csv` — per-file
  probabilities and predicted labels used to compute ROC / PR curves (AUC ≈ 0.852, AP ≈ 0.935).

Notes
- The gait model uses tabular, interpretable features. If you need feature
  importances, load the joblib artifact and inspect `.feature_importances_`.
  If the artifact is saved as a dict with 'features' and 'model', the
  `features` list provides column ordering used at training time.

### Motor classifier

- Artifact: `models/trained_models/motor_baseline_full/motor_rf_further_retrained.joblib`
- Training script: `tools/further_retrain_motor.py` (this script was used to
  retrain the motor model inside the current environment to avoid sklearn
  version compatibility issues). The training logic builds on files saved by
  earlier preprocessing and evaluation scripts under `models/trained_models/motor_baseline_full/`.

Training recipe and architecture
- Input: per-subject aggregated features in
  `models/trained_models/motor_baseline_full/motor_features_per_subject_recomputed.csv`.
- Labels: `motor_per_subject_predictions.csv` and related prediction files
  provide truth labels; the retrain script filters to classes 0 and 1 (binary)
  and uses `label_true` as target.
- Synthetic augmentation: the retrain script synthesizes multivariate samples
  for the healthy class (class 0) using the covariance of observed healthy
  samples (function `multivariate_synth`) to generate 5000 synthetic
  healthy samples; these are appended to training folds to improve recall for healthy.
- Cross-validation & model selection: 5-fold StratifiedKFold CV used to pick
  the best fold optimizing recall for class 0 (healthy). RandomForest was
  trained with n_estimators=300 and class_weight='balanced'.
- Persistence: the best model and the feature list are saved as an artifact
  dict: {'model': clf, 'features': list(X.columns)} and dumped with joblib.

Relevant code snippets: `tools/further_retrain_motor.py` — loads features,
builds synthetic healthy samples, runs CV, and writes the best model and a
report file (if available).

Evaluation
- The retraining procedure prints fold-wise recall scores and writes a
  short report (if present) including the classification report and confusion
  matrix for the best fold. This artifact can be found next to the model
  (search for `motor_further_retrain_report.txt` in the motor model folder).

Notes and gotchas
- The motor model was originally serialized with a scikit-learn version that
  caused unpickle errors in this environment. To maintain reproducibility
  for deployment, the model was retrained inside the current `.venv` and
  saved to `motor_rf_further_retrained.joblib`.
- The `motor_service.py` loader is defensive: it captures load tracebacks and
  exposes `get_load_error()` to the FastAPI `/info` endpoint. This helps
  diagnose version incompatibilities without crashing the server.

### Handwriting classifier (EfficientNet)

- Artifact: `models/trained_models/handwriting_efficient_best_long/best_model_finetune.h5`
- Training script: `models/trainers/train_handwriting_efficient.py`

Training recipe and architecture
- Input: a validated list of handwriting image paths (text file with one
  image per line). The training script builds a tf.data pipeline and uses
  Keras preprocessing layers for augmentation.
- Backbone: EfficientNetB0 pretrained on ImageNet used as the feature
  extractor. Two modes supported:
  - embed: extract embeddings with frozen backbone and train a tabular
    classifier (LightGBM or RandomForest) on embeddings.
  - end2end: build a small head on top of the frozen backbone, train the
    head, then optionally unfreeze and fine-tune the last N layers.
- Augmentation: RandomFlip, RandomRotation, RandomZoom, RandomContrast,
  RandomTranslation. Optional mixup and oversampling are supported.
- Optimization: Adam optimizer with configurable learning rates; optional
  cosine-decay schedule; early stopping and model checkpoint callbacks.
- Output: saved Keras models (SavedModel + checkpoints) and training
  curves if training history is available.

Evaluation
- When run in end2end mode the script computes classification report and
  confusion matrix on the validation set and saves training curves as PNGs.

Notes
- The saved artifact `best_model_finetune.h5` is compatible with the TF
  runtime used when training; ensure the deployed TF version supports the
  same Keras layers if you plan to load it elsewhere.

## Inference & deployment (pipeline)

This project exposes a FastAPI backend (`backend/main.py`) that provides
endpoints for inference and basic diagnostics. The backend loads the model
artifacts at startup (or lazily) and exposes endpoints (examples):

- `/predict/gait` — accepts a gait text file upload, extracts features, and
  returns predicted class, label and probabilities.
- `/predict/motor` — accepts a JSON or CSV with features (or an uploaded
  CSV) and returns predicted class, label and probability info.
- `/predict/handwriting` — accepts an image file and returns the handwriting
  class prediction (relies on the Keras model or an embedding + tabular
  classifier depending on how the model was saved).
- `/info` — returns safe model status and feature-count metadata. Internal
  model paths and load tracebacks are never returned to clients.

Key implementation notes
- The backend requires `PARKINSONS_API_KEY` in production and expects it in
  the `X-API-Key` header. In development, requests without a key are allowed
  only from loopback addresses. `/healthz` is the only unauthenticated
  liveness endpoint.
- Uploads are type-checked, capped at 10 MB, stored under generated temporary
  names, and removed after prediction. Original filenames are not exposed in
  motor sessions.
- Internal tracebacks are logged with an error ID and are not returned to
  clients.
- CSV handling: the motor loader coerces columns to numeric and fills
  non-convertible values with 0.0 to avoid type conversion errors during
  prediction. The loader respects the `features` ordering saved in the
  artifact (if present) and will add missing columns with zero values so the
  trained model receives the same feature vector shape it was trained on.

Example usage (curl-like, replace with your client):

1) Gait file upload (example)

```bash
curl -X POST "http://localhost:8000/predict/gait" -F "file=@sample_data/gait_signals/GaCo01_01.txt"
```

2) Motor JSON prediction (example)

```bash
curl -X POST "http://localhost:8000/predict/motor" -H "Content-Type: application/json" -d @sample_data/motor_tasks/example_patient_confirmed.json
```

3) Handwriting image prediction (example)

```bash
curl -X POST "http://localhost:8000/predict/handwriting" -F "file=@path/to/handwriting.jpg"
```

The backend server can be started during development with:

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

For deployment, set `APP_ENV=production`, `PARKINSONS_API_KEY`, and
`FRONTEND_ORIGINS`, then run without `--reload`. Put the service behind HTTPS
and a reverse proxy, and use a separate authentication system for
multi-user access instead of embedding a shared browser API key.

## Evaluation highlights and plots

- Gait classifier:
  - Classification report (per-file) shows accuracy ≈ 0.97 with high
    precision/recall on both classes (report in
    `models/trained_models/gait_signals/gait_classification_report.txt`).
  - ROC AUC ≈ 0.852 and Average Precision (AP) ≈ 0.935 computed from
    `gait_per_file_predictions.csv` and saved plots under `visualizations/`.

- Motor classifier:
  - Retrained inside the current environment to fix joblib/scikit-learn
    incompatibilities. Training used 5-fold stratified CV with synthetic
    healthy (class 0) augmentation and RandomForest (n_estimators=300,
    class_weight='balanced'). The final artifact is
    `motor_rf_further_retrained.joblib` and includes the features ordering.
  - ROC/PR plots for motor were produced and saved under `visualizations/` during analysis.

- Handwriting classifier:
  - EfficientNetB0 backbone with a light head, optional fine-tuning. The
    training script supports embedding mode (extract embeddings and train a
    LightGBM/RF classifier) which is useful for small handwriting datasets.

## Files & artifacts (important paths)

- Gait model: `models/trained_models/gait_signals/model_gait_enhanced_calibrated.joblib`
- Gait per-file predictions: `models/trained_models/gait_signals/gait_per_file_predictions.csv`
- Gait classification report: `models/trained_models/gait_signals/gait_classification_report.txt`

- Motor model: `models/trained_models/motor_baseline_full/motor_rf_further_retrained.joblib`
- Motor features (per-subject): `models/trained_models/motor_baseline_full/motor_features_per_subject_recomputed.csv`
- Motor per-subject predictions: `models/trained_models/motor_baseline_full/predictions_improved.csv` (contains true labels)

- Handwriting model: `models/trained_models/handwriting_efficient_best_long/best_model_finetune.h5`
- Handwriting training list: `models/trained_models/handwriting_all_valid_images.txt` (used for training)

## Reproducibility notes and dependencies

- Python environment: project uses a virtual environment `.venv` with
  standard scientific Python packages: numpy, pandas, scikit-learn, joblib,
  tensorflow (for handwriting), matplotlib, and FastAPI (uvicorn) for the
  backend. See `backend/requirements.txt` and `models/requirements.txt` for
  more details.
- Random seeds: training scripts set deterministic random_state/seed values
  (42, 1337) to make runs reproducible where possible.

Practical tips
- If you need to run the motor model artifact produced elsewhere, ensure
  your scikit-learn version matches the one used to save the model. If you
  observe unpickle errors, retrain the model in the current environment or
  re-export with a compatible scikit-learn version.
- For handwriting, training with GPU and mixed precision speeds up
  experiments. The training script has flags to enable mixed precision and
  distributed training.

## How to use this README for a paper & slides

- Methods section: copy the "Data and feature engineering" and the three
  "Models" subsections. Include the training hyperparameters and the
  synthetic augmentation method used for the motor classifier (multivariate
  Gaussian sampling from observed healthy covariance matrix).
- Results: include the classification reports and ROC/AP metrics. Use the
  saved plots under `visualizations/` (ROC, PR, confusion matrices) as
  figures for your paper and slides.
- Pipeline diagram (for slides): show raw input → feature extraction
  (time-domain, derived kinematics, aggregated magnitude) → model (RF or
  EfficientNet) → outputs (per-file/probabilities, per-subject aggregated
  decision). Mention the API endpoints used for deployment.

## Next steps and suggested analyses

- Calibration: compute reliability diagrams and Brier scores for each
  classifier to quantify probability calibration.
- Feature importance: compute permutation importances (or SHAP values) for
  the RandomForest models for interpretable explanations.
- Cross-dataset validation: evaluate models on held-out datasets or
  synthetic perturbations (noise, downsample) to test robustness.
- Ensemble: try embedding + tabular classifier (for handwriting) or a
  small ensemble of RF + logistic-regression for motor/gait for more stable
  probabilities.

---

## Frontend & FastAPI (development and deployment)

This project includes a React frontend (in `frontend/`) and a FastAPI backend (`backend/main.py`). The frontend is configured to proxy API calls to `http://localhost:8000` during development; production builds should call the backend at the deployed URL (or use a reverse proxy).

1) Frontend (development)

- Navigate to the frontend folder and start the development server (it uses `react-scripts`):

```powershell
cd frontend
npm install    # first time only
npm start
```

- The `package.json` contains a `proxy` entry set to `http://localhost:8000`, so requests like `fetch('/predict/gait')` will be forwarded to the FastAPI backend during development. The frontend also implements a fallback that directly contacts `http://127.0.0.1:8000` if the proxied request fails.

2) Backend (development)

- Start the FastAPI server from the project root (in the `.venv`):

```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- The `/info` endpoint is used by the frontend at startup to display model status and paths. If the motor model failed to load, the `/info` route will include the loader traceback to help debug.

3) Example payloads and endpoints

- Handwriting (image): POST `/predict/handwriting` with multipart form file field `file` (image/*). The frontend uses `postFile('/predict/handwriting', file)`.
- Motor JSON: POST `/predict/motor` with JSON body `{ "features": {"c1_mean": 0.12, ...}}`. The frontend tolerates either the inner `features` object or the top-level features object.
- Motor CSV: POST `/predict/motor_csv` with a multipart file field `file` (CSV). The frontend sends files to `/predict/motor_csv` and expects JSON with `predictions` array.
- Gait timeseries: POST `/predict/gait` with multipart file field `file` (plain text timeseries). The frontend sends selected `.txt` files to `/predict/gait`.

4) CORS and production notes

- In production, configure CORS properly on the FastAPI side (via `fastapi.middleware.cors.CORSMiddleware`) or serve the frontend from the same origin as the API to avoid CORS complexity.
- The frontend expects certain shapes in backend responses (for example, handwriting returns `label`, `pred`, `prob_pos`; motor may return `pred`, `probs`, `prob_pos`, `label`). Keep these conventions if modifying endpoints.

5) Deployment recommendations

- Option A (simple): build the frontend and serve static files from a web server (NGINX) with the FastAPI backend behind a Gunicorn/Uvicorn process manager. Use HTTPS and a reverse proxy.
- Option B (single-host): configure the webserver to serve the built frontend and proxy API routes to the FastAPI application.
- Option C (containerized): create Docker images for backend and frontend and orchestrate with docker-compose or Kubernetes.

6) Troubleshooting

- If the frontend shows an error like "Unexpected response (status 404)" during a file post, check that (a) the backend is running, (b) the endpoint path is `'/predict/...'`, and (c) the proxy in `frontend/package.json` is set correctly.
- Use the backend `/info` endpoint to inspect loaded model paths and any loader errors.


If you'd like, I can now:
- generate a compact Methods section in markdown suitable for a paper;
- produce a 6–8 slide PowerPoint draft (bullet slides + figures) from these
  results; or
- add calibration, permutation importance and SHAP plots and include them
  in an HTML report under `visualizations/report.html`.

Tell me which of the above you'd like next and I will continue.


---

<!-- appended enrichment -->
````markdown
## Appendix: dataset & artifact statistics (auto-collected)

- Sample files (sample_data): gait=11, motor=6, handwriting=0
- Gait predictions: 306 rows, class balance: {1: 214, 0: 92}
- Motor features: 469 rows, 54 columns (including id/label). Missing values: 0
- Motor predictions: 469 rows, class balance: {1: 276, 2: 114, 0: 79}
- Handwriting training list not found; count sample images in sample_data/handwriting if available
- Gait model artifact present: True (D:\Bunny\ML_mini_project\models\trained_models\gait_signals\model_gait_enhanced_calibrated.joblib)
- Motor model artifact present: True (D:\Bunny\ML_mini_project\models\trained_models\motor_baseline_full\motor_rf_further_retrained.joblib)
- Handwriting model artifact present: True (D:\Bunny\ML_mini_project\models\trained_models\handwriting_efficient_best_long\best_model_finetune.h5)

### Installed packages (sample)
```
absl-py==2.3.1
anyio==4.11.0
astunparse==1.6.3
cachetools==6.2.1
certifi==2025.10.5
charset-normalizer==3.4.4
click==8.3.0
cloudpickle==3.1.1
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
fastapi==0.95.2
flatbuffers==25.9.23
fonttools==4.60.1
gast==0.6.0
google-auth-oauthlib==1.0.0
google-auth==2.41.1
google-pasta==0.2.0
grpcio==1.75.1
h11==0.16.0
h5py==3.15.1
httptools==0.7.1
idna==3.11
imbalanced-learn==0.11.0
joblib==1.2.0
keras==2.14.0
kiwisolver==1.4.9
libclang==18.1.1
llvmlite==0.45.1
lxml==6.0.2
markdown==3.9
markupsafe==3.0.3
matplotlib==3.10.7
ml-dtypes==0.2.0
numba==0.62.1
numpy==1.26.2
oauthlib==3.3.1
opt-einsum==3.4.0
packaging==25.0
pandas==2.1.2
... (truncated)
```
