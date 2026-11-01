# Multi-Dataset Machine Learning Framework for Early Prediction of Liver Cirrhosis

A production-grade, end-to-end Machine Learning and MLOps framework for early detection and risk stratification of liver cirrhosis and liver disease across heterogeneous clinical datasets.

The system features robust preprocessing, hyperparameter optimization, probability calibration, threshold optimization, REST API serving via FastAPI, automated test suites with safe synthetic datasets, containerization, GitHub Actions CI/CD, and Prometheus/Grafana monitoring.

---

## Architecture Overview

```
                      +---------------------------------------+
                      | Raw Multi-Datasets                    |
                      | - Hepatitis C (UCI)                   |
                      | - Indian Liver Patient Dataset (ILPD) |
                      | - Liver Patient Dataset (LPD Train)   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | src/dataset.py: Data Harmonization    |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | src/preprocessing.py: Imputer + Scaler|
                      | (Median Imputation + Strict Ordering) |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | src/model.py: CatBoost + Threshold    |
                      | F1 Optimization + Brier Calibration   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | artifacts/model_pipeline.joblib       |
                      | (Atomic Bundle: Preprocessor + Model) |
                      +-------------------+-------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
        +-------------------------------+   +-------------------------------+
        | api/main.py: FastAPI REST API |   | Prometheus & Grafana Telemetry|
        | - GET  /health                |   | - /metrics                    |
        | - GET  /ready                 |   | - Latency Percentiles (p95)   |
        | - POST /predict               |   | - Request Counts / Error Rates|
        | - POST /predict/batch         |   | - Risk Level Distribution     |
        | - GET  /model-info            |   +-------------------------------+
        +-------------------------------+
```

---

## Directory Structure

```
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml                # GitHub Actions: Lint, Test, Docker Build & Push to GHCR
├── api/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application factory, lifespan, CORS, error handling
│   ├── routes.py                   # REST endpoints (/health, /ready, /predict, /model-info)
│   └── schemas.py                  # Pydantic v2 schemas for request validation & documentation
├── artifacts/
│   └── model_pipeline.joblib       # Serialized bundle: preprocessor, CatBoost classifier, threshold
├── deploy/
│   ├── deploy.sh                   # Automated deployment script for Linux/macOS
│   ├── k8s/
│   │   └── deployment.yaml         # Kubernetes Deployment & Service manifests
│   ├── nginx/
│   │   └── cirrhosis-api.conf      # Nginx reverse proxy with rate limiting
│   └── systemd/
│       └── cirrhosis-api.service   # Linux systemd service unit
├── monitoring/
│   ├── dashboards/
│   │   └── liver_cirrhosis_dashboard.json # Pre-built Grafana monitoring dashboard
│   ├── grafana/
│   │   └── provisioning/
│   │       ├── dashboards/dashboards.yml
│   │       └── datasources/prometheus_datasource.yml
│   └── prometheus.yml              # Prometheus scrape configuration
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration with pydantic-settings
│   ├── dataset.py                  # Multi-dataset loader, column cleaning, harmonization
│   ├── model.py                    # CirrhosisPredictionPipeline encapsulation & persistence
│   ├── predict.py                  # Inference service with caching & singleton pattern
│   ├── preprocessing.py            # ClinicalFeaturePreprocessor (median imputer + scaler)
│   └── train.py                    # Training script with Optuna tuning & artifact export
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures and safe synthetic clinical data
│   ├── test_api.py                 # Integration tests for FastAPI endpoints
│   ├── test_model.py               # Model attributes, risk levels, threshold logic tests
│   └── test_preprocessing.py       # Preprocessor consistency & missing value tests
├── Dockerfile                      # Non-root, multi-stage production container
├── docker-compose.yml              # Unified orchestration (API + Prometheus + Grafana)
├── final_year_project.ipynb        # Experimentation notebook
├── requirements.txt                # Production dependencies
├── .env.example                    # Environment variable configuration template
└── README.md
```

---

## Phase 1 — Python Code Refactoring

The preprocessing, training, and inference logic have been moved from the exploratory notebook into modular Python packages:

1. **`src/dataset.py`**:
   - Cleans and standardizes column names.
   - Harmonizes heterogeneous labels into a unified binary target:
     - Hepatitis C: `Target != "0=Blood Donor"`
     - LPD / ILPD: `Result == 1` or `Dataset == 1`
   - Harmonizes laboratory feature naming (e.g. `Total_Proteins` to `Total_Protiens`).

2. **`src/preprocessing.py`**:
   - `ClinicalFeaturePreprocessor` enforces a deterministic feature ordering:
     `["Age", "Total_Bilirubin", "Albumin", "ALT", "AST", "ALP", "Total_Protiens"]`
   - Imputes missing clinical values using training set medians (`SimpleImputer`).
   - Normalizes feature scales using `StandardScaler`.
   - Resolves case variations and aliases (`total_bilirubin`, `Total_Bilirubin`).

3. **`src/model.py` & `src/train.py`**:
   - `CirrhosisPredictionPipeline` atomically packages the preprocessor, classifier, decision threshold, version, and training metadata into a single `.joblib` artifact.
   - Run training:
     ```bash
     python3 src/train.py
     # Or with Optuna hyperparameter tuning:
     python3 src/train.py --tune --trials 20
     ```

---

## Phase 2 — REST API (FastAPI)

Run the server locally:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI is accessible at: `http://localhost:8000/docs`

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check confirming service is running |
| `GET` | `/ready` | Readiness check verifying model pipeline is loaded |
| `POST` | `/predict` | Clinical prediction with Pydantic validation & risk stratification |
| `POST` | `/predict/batch`| Batch prediction for up to 500 patient records |
| `GET` | `/model-info` | Model metadata, feature list, version, and training metrics |
| `GET` | `/metrics` | Prometheus telemetry exposition |

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "total_bilirubin": 2.4,
    "albumin": 3.1,
    "alt": 48.0,
    "ast": 52.0,
    "alp": 210.0,
    "total_proteins": 6.7
  }'
```

### Example Response

```json
{
  "prediction": 1,
  "prediction_label": "High Risk",
  "cirrhosis_probability": 0.8842,
  "decision_threshold": 0.52,
  "risk_level": "High",
  "model_version": "1.0.0",
  "disclaimer": "RESEARCH NOTICE: Model outputs are statistical predictions generated for investigative and educational purposes only. They do not constitute medical advice, formal diagnosis, or treatment recommendations."
}
```

---

## Phase 3 — Automated Testing & Synthetic Data

All tests use safe, synthetic clinical profiles without exposing real patient records.

Run the test suite:
```bash
pytest -v
```

Lint the codebase:
```bash
flake8 src api tests --max-line-length=120 --exclude=.venv,__pycache__
```

### Test Coverage Highlights
- **Input Validation**: Tests boundary conditions (negative age, out-of-range enzymes, forbidden fields).
- **Missing Value Handling**: Verifies median imputation when clinical fields are omitted (`None`).
- **Feature Ordering**: Confirms deterministic feature alignment across shuffled dictionary and DataFrame columns.
- **Model Invariants**: Tests probability range $[0.0, 1.0]$, threshold logic, and risk categorization.
- **API Endpoints**: Validates status codes (200, 422, 503) and JSON schemas.

---

## Phase 4 — Dockerization

### Build and Run with Docker

Build the Docker image:
```bash
docker build -t liver-cirrhosis-api:latest .
```

Run as a non-root container:
```bash
docker run -d \
  -p 8000:8000 \
  --name cirrhosis-api \
  --env-file .env.example \
  liver-cirrhosis-api:latest
```

### Run the Full Stack with Docker Compose

Start the API, Prometheus, and Grafana in a single command:
```bash
docker compose up -d --build
```

Access the services:
- **API & Docs**: `http://localhost:8000/docs`
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (User: `admin`, Password: `admin`)

---

## Phase 5 — CI/CD Pipeline (GitHub Actions)

The workflow defined in `.github/workflows/ci-cd.yml` automates quality checks and deployment:

1. **Pull Requests (`main`, `master`)**:
   - Checks out the repository.
   - Sets up Python 3.11 with pip caching.
   - Installs dependencies.
   - Runs `flake8` linting.
   - Executes `pytest` test suite.

2. **Push to `main`**:
   - Executes linting and test stages.
   - Logs in to GitHub Container Registry (`ghcr.io`) using `GITHUB_TOKEN`.
   - Builds the Docker container with GitHub Actions layer caching.
   - Publishes the container image tagged with the Git commit SHA and `latest`.

---

## Phase 6 — Monitoring & Production Deployment

### Prometheus Metrics
The service exports metrics on `/metrics`:
- `http_requests_total`: Request volume partitioned by status code and method.
- `http_request_duration_seconds`: Request latency histogram.
- `cirrhosis_predictions_total`: Prediction count partitioned by status and risk category.
- `cirrhosis_prediction_duration_seconds`: Pipeline inference latency histogram.
- `cirrhosis_model_loaded`: Model availability gauge (1 = healthy, 0 = unavailable).

### Grafana Dashboard
A pre-configured dashboard (`monitoring/dashboards/liver_cirrhosis_dashboard.json`) is automatically provisioned with:
- Model Readiness Status Indicator
- Prediction Volume Stratified by Clinical Risk (Low, Moderate, High, Critical)
- Inference Latency Percentiles (p50, p90, p99)
- HTTP Status Code Rates & Error Tracking (4xx / 5xx)

### Linux Production Deployment (systemd & Nginx)

1. Deploy the systemd service:
   ```bash
   sudo cp deploy/systemd/cirrhosis-api.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now cirrhosis-api
   ```

2. Configure Nginx reverse proxy:
   ```bash
   sudo cp deploy/nginx/cirrhosis-api.conf /etc/nginx/sites-available/
   sudo ln -s /etc/nginx/sites-available/cirrhosis-api.conf /etc/nginx/sites-enabled/
   sudo nginx -t && sudo systemctl reload nginx
   ```

3. Automated script:
   ```bash
   ./deploy/deploy.sh
   ```

---

## Disclaimer

**RESEARCH USE ONLY**: This software is intended strictly for research and educational purposes. The predictions, risk categories, and probabilities generated by this framework do not constitute medical diagnoses, clinical guidance, or treatment plans.
