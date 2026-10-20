# Heart Disease Risk — MLOps Assignment 1

End-to-end MLOps pipeline for the UCI Cleveland Heart Disease dataset: data ingestion and
cleaning, feature engineering, multi-model training with **MLflow** experiment tracking and a
**Model Registry**, a **FastAPI** inference service, a multi-stage **Docker** image, **Kubernetes**
manifests (+ autoscaling), **Prometheus/Grafana** monitoring, and a 4-stage **GitHub Actions** CI
pipeline (lint → test → train → docker build/smoke-test).

> Built for AIMLCZG523 (MLOps) Assignment 1. See [reports/report.md](reports/report.md) /
> [reports/MLOps_Assignment1_Report.docx](reports/MLOps_Assignment1_Report.docx) for the full
> write-up with results, screenshots and the architecture diagram.

### Submission links

- Project video: [GitHub video](https://github.com/is-zen-genius/mlops_assignment_1/blob/main/recordings/mlops_2025ae05040.mov)
- Project video: [Google Drive folder](https://drive.google.com/drive/folders/1UO6gMZc_yPuTMv8h-T_kK07NyMYBEhN2)

```
UCI dataset -> ingest -> clean -> features -> train (MLflow) -> package -> FastAPI -> Docker -> Kubernetes
                                                                              |              |
                                                                       /metrics + JSON logs -> Prometheus -> Grafana
```

## Contents

1. [Project layout](#project-layout)
2. [Prerequisites](#prerequisites)
3. [Quickstart — three ways to run this](#quickstart--three-ways-to-run-this)
4. [Option A — run locally with Python](#option-a--run-locally-with-python)
5. [Option B — run with Docker](#option-b--run-with-docker)
6. [Option C — run with Kubernetes](#option-c--run-with-kubernetes)
7. [CI/CD](#cicd)
8. [API reference](#api-reference)
9. [Results](#results)
10. [Troubleshooting](#troubleshooting)

## Project layout

```
.
├── .github/workflows/ci.yml     # lint -> test -> train -> docker build + smoke test
├── data/
│   ├── raw/, processed/             # downloaded + cleaned CSVs
│   └── sample_patient.json          # example request body for /predict
├── deployment/
│   ├── k8s/                         # namespace, configmap, deployment, services, HPA
│   └── monitoring/                  # Prometheus, Grafana, dashboards/
├── models/                      # model.joblib, model_metadata.json, mlflow_model/
├── notebooks/01_exploratory_data_analysis.ipynb
├── reports/                     # report.md, generated .docx, figures/, model_comparison.csv
├── screenshots/, recordings/    # evidence for submission (see each folder's README)
├── scripts/                     # simulate_traffic.py, make_architecture_diagram.py,
│                                 # export_eda_figures.py, generate_report.py
├── src/                         # config, ingest, clean, features, train, package_model,
│                                 # inference, schemas, metrics, logging_utils, main (FastAPI)
├── tests/                       # 25 pytest cases
├── Dockerfile
└── requirements.txt              # single file for everything - see "Prerequisites" below
```

## Prerequisites

| Tool                | Needed for       | Notes                                             |
| ------------------- | ---------------- | ------------------------------------------------- |
| Python 3.12         | Option A (local) | `python3.12 -m venv .venv`                      |
| Docker              | Options B and C  | Docker Desktop or any Docker engine               |
| kubectl + a cluster | Option C         | Docker Desktop, Rancher Desktop, minikube or kind |

There is a **single `requirements.txt`** for the whole project (data/training, the FastAPI
service, pytest/ruff, notebook execution, and the DOCX report generator). Local development,
the Docker image, and CI all install from this one file — nothing else to track or keep in sync.

## Quickstart — three ways to run this

|                         | Command                                                                        | What you get                                                                               |
| ----------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| **A. Local**      | `pip install -r requirements.txt` then run the `src/*.py` scripts directly | Fastest inner loop; best for development, training, and the notebook                       |
| **B. Docker**     | `docker build` + `docker run`                                              | The exact artifact CI builds; isolated, reproducible, no local Python needed to *run* it |
| **C. Kubernetes** | `kubectl apply -f deployment/k8s/`                                           | Production-like: 2 replicas, rolling updates, autoscaling, Prometheus/Grafana              |

Each option is self-contained below — pick whichever matches what you're trying to do.

## Option A — run locally with Python

Everything here uses the **one** `requirements.txt` (training deps, the FastAPI service,
pytest/ruff, notebook execution, and the DOCX report generator all in one file).

### A.1 Install

```bash
git clone <this-repo-url>
cd mlops_assignment_1

python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verified clean during development: a brand-new venv, `pip install -r requirements.txt`, then
`pytest` (25 passed) and `ruff check src tests scripts` (0 errors) with nothing else installed.

### A.2 Data pipeline

```bash
python src/ingest.py     # download UCI Cleveland data -> data/raw/heart_disease_raw.csv
python src/clean.py      # impute ca/thal, binarise target -> data/processed/heart_disease_clean.csv
```

Scaling/encoding live in the shared sklearn pipeline (`src/features.py`), so training and serving
use identical transforms. Full EDA with executed plots:
[notebooks/01_exploratory_data_analysis.ipynb](notebooks/01_exploratory_data_analysis.ipynb)
(or regenerate the headline figures standalone with `python scripts/export_eda_figures.py`).

### A.3 Train + track with MLflow

`src/train.py` grid-searches **Logistic Regression**, **Random Forest** and **Gradient Boosting**
with stratified 5-fold CV, logging everything to MLflow (SQLite-backed, so the Model Registry
works):

```bash
python src/train.py
```

```bash
mlflow ui --backend-store-uri sqlite:///tracking.db --port 5000
# http://localhost:5000 -> experiment "heart-disease-risk"
```

Each model family is one parent run (params, best-candidate metrics, confusion-matrix/ROC/PR
plots, the fitted pipeline) with its top-5 grid-search candidates as nested child runs, and the
full comparison table at [reports/model_comparison.csv](reports/model_comparison.csv).

### A.4 Package the model

```bash
# models/model.joblib, models/model_metadata.json, models/mlflow_model/ + registry + round-trip check
python src/package_model.py

python src/inference.py                       # CLI predict on data/sample_patient.json
python src/inference.py path/to/other.json    # or any other patient record
```

Registers the winning run as `heart-risk-classifier` and points the **`champion`** alias at it, so
it can be resolved as `models:/heart-risk-classifier@champion`.

### A.5 Tests & linting

```bash
pytest                                   # 25 unit/API tests (pytest.ini sets pythonpath=src)
pytest --cov=src --cov-report=term-missing

ruff check src tests scripts             # lint (config in pyproject.toml, line length 100)
```

Both commands exit non-zero and print exactly what's wrong (file, line, assertion/rule) if
something's broken — see [reports/report.md](reports/report.md#7-cicd-pipeline--automated-testing)
for a real captured example of each failure mode.

### A.6 Run the API directly (no Docker)

```bash
PYTHONPATH=src uvicorn main:app --app-dir src --host 0.0.0.0 --port 8001 --reload
```

- Swagger UI: [http://localhost:8001/docs](http://localhost:8001/docs)
- Health: [http://localhost:8001/health](http://localhost:8001/health) · Metrics: [http://localhost:8001/metrics](http://localhost:8001/metrics)

```bash
curl http://localhost:8001/health
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d @data/sample_patient.json
```

### A.7 Reproduce the whole pipeline in one line

```bash
ruff check src tests scripts \
  && pytest \
  && python src/ingest.py \
  && python src/clean.py \
  && python src/train.py \
  && python src/package_model.py \
  && python src/inference.py
```

## Option B — run with Docker

Multi-stage build (`python:3.12-slim`): a builder stage resolves the project's single
`requirements.txt` into a venv, the runtime stage copies just that venv + the 7 modules the API
needs + `model.joblib`, runs as a **non-root user**, and exposes port **8001** with a
`HEALTHCHECK`. The image is self-contained — no local Python environment is needed to *run* it,
only Docker.

### B.1 Build

```bash
# models/model.joblib must exist first (Option A.3 + A.4), or re-use the one already committed
python src/train.py && python src/package_model.py   # only if you want to retrain first

docker build -t heart-risk-api:1.0 .
```

> Note: because the image installs the single project-wide `requirements.txt` (training +
> serving + dev deps), it's ~1.1 GB rather than a hand-trimmed ~550 MB — a deliberate trade-off
> for having only one requirements file to maintain. The non-root user, multi-stage build and
> minimal `COPY` list are unaffected.

### B.2 Run and verify

```bash
docker run -d --name heart-risk-api -p 8001:8001 heart-risk-api:1.0

docker ps
docker inspect --format='{{.State.Health.Status}}' heart-risk-api   # "healthy"

curl http://localhost:8001/health
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d @data/sample_patient.json
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"age": 9999}'   # deliberately invalid -> 422

docker logs heart-risk-api | grep '"event"'    # structured JSON request/prediction logs
```

### B.3 Stop and clean up

```bash
docker stop heart-risk-api && docker rm heart-risk-api
```

### B.4 Docker troubleshooting

| Symptom                                                    | Fix                                                                                                                                                 |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError: models/model.joblib` during build    | Run `python src/train.py && python src/package_model.py` first (Option A.3/A.4)                                                                   |
| Container exits immediately                                | `docker logs heart-risk-api` — almost always a missing/corrupt `models/model.joblib`                                                           |
| `docker inspect ... Health.Status` stuck on `starting` | Wait for the 10s `--start-period`; if it never becomes `healthy`, check `docker logs`                                                         |
| Port already in use                                        | Another process (or a previous container) is on 8001 — `docker ps`, then `docker rm -f <name>`, or map a different host port: `-p 8002:8001` |

## Option C — run with Kubernetes

Manifests in [deployment/k8s/](deployment/k8s/) (apply in filename order). This was tested
end-to-end on a local Rancher Desktop (k3s) cluster during development — see
[reports/report.md](reports/report.md#9-production-deployment-kubernetes) for the full real
command-by-command walkthrough with output.

| File                         | Resource                                                                    |
| ---------------------------- | --------------------------------------------------------------------------- |
| `00-namespace.yaml`        | Namespace `heart-risk`                                                    |
| `10-configmap.yaml`        | `LOG_LEVEL` env var for the API                                           |
| `20-deployment.yaml`       | Deployment, 2 replicas, rolling update, probes, non-root, resource limits   |
| `30-service.yaml`          | `LoadBalancer` Service, port **8080** → container port 8001        |
| `35-service-headless.yaml` | Headless Service so Prometheus can discover every pod                       |
| `40-hpa.yaml`              | HorizontalPodAutoscaler, 2-5 replicas at 70% CPU (needs `metrics-server`) |

### C.1 Before you start

```bash
kubectl config current-context    # make sure this is the cluster you think it is!
```

Make the image available to the cluster:

```bash
# Docker Desktop / Rancher Desktop share the local Docker daemon - no extra step needed,
# `docker build -t heart-risk-api:1.0 .` (Option B.1) is already enough.

# minikube:
minikube image load heart-risk-api:1.0
# kind:
kind load docker-image heart-risk-api:1.0
```

### C.2 Deploy the API

```bash
kubectl apply -f deployment/k8s/
kubectl rollout status deployment/heart-risk-api -n heart-risk
```

### C.3 Verify

```bash
kubectl get all -n heart-risk -o wide
kubectl get endpoints -n heart-risk
kubectl get hpa -n heart-risk

# Docker Desktop / Rancher Desktop: LoadBalancer is exposed on localhost or a local VM IP
# (check the EXTERNAL-IP column from `kubectl get all`, usually localhost or 192.168.x.x)
curl http://localhost:8080/health
# otherwise, port-forward:
kubectl port-forward svc/heart-risk-api 8080:8080 -n heart-risk

curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
  -d @data/sample_patient.json
```

Swagger UI is also reachable at `http://<service-ip>:8080/docs` for interactive testing.

### C.4 Logs, scaling, updates, rollback

```bash
kubectl logs -f deployment/heart-risk-api -n heart-risk
kubectl logs -l app=heart-risk-api -n heart-risk --prefix     # both pods, labelled

kubectl scale deployment/heart-risk-api --replicas=3 -n heart-risk
kubectl rollout restart deployment/heart-risk-api -n heart-risk

# after building a new image tag:
kubectl set image deployment/heart-risk-api api=heart-risk-api:1.1 -n heart-risk
kubectl rollout status deployment/heart-risk-api -n heart-risk
kubectl rollout undo deployment/heart-risk-api -n heart-risk   # roll back if needed
```

### C.5 Deploy monitoring (Prometheus + Grafana)

```bash
kubectl apply -f deployment/monitoring/00-namespace.yaml

# Grafana admin credentials come from a Secret, never a hard-coded value in the manifest:
kubectl create secret generic grafana-admin-credentials \
  --from-literal=admin-user=admin \
  --from-literal=admin-password='<choose-your-own-password>' \
  -n monitoring

kubectl create configmap grafana-dashboards \
  --from-file=deployment/monitoring/dashboards/heart-risk-dashboard.json \
  -n monitoring

kubectl apply -f deployment/monitoring/10-prometheus.yaml -f deployment/monitoring/20-grafana.yaml
kubectl rollout status deployment/prometheus -n monitoring
kubectl rollout status deployment/grafana -n monitoring
```

### C.6 Access the dashboards

```bash
kubectl port-forward svc/prometheus 9090:9090 -n monitoring   # http://localhost:9090/targets
kubectl port-forward svc/grafana 3000:3000 -n monitoring      # http://localhost:3000
```

Log into Grafana with the username/password you chose in the Secret above. The
"Heart Risk API - Overview" dashboard is auto-provisioned (request rate by status, p95 latency,
error rate, predictions/min by label, median confidence, total requests, live pod count).

### C.7 Generate traffic so the dashboards have data

```bash
kubectl port-forward svc/heart-risk-api 8080:8080 -n heart-risk &
python scripts/simulate_traffic.py --url http://localhost:8080 --requests 300 --delay 0.2
```

### C.8 Tear down

```bash
kubectl delete -f deployment/monitoring/20-grafana.yaml -f deployment/monitoring/10-prometheus.yaml
kubectl delete secret grafana-admin-credentials -n monitoring
kubectl delete configmap grafana-dashboards -n monitoring
kubectl delete namespace monitoring

kubectl delete -f deployment/k8s/

kubectl get all --all-namespaces | grep -iE 'heart-risk|grafana|prometheus'   # should be empty
```

### C.9 Kubernetes troubleshooting

| Symptom                                    | Fix                                                                                                                                                                                 |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ErrImagePull` / `ImagePullBackOff`    | The image isn't on the cluster node — load it (`minikube image load ...`, `kind load docker-image ...`) or push to a registry and update `deployment/k8s/20-deployment.yaml` |
| Pods `Running` but not `Ready`         | `kubectl describe pod -l app=heart-risk-api -n heart-risk` — readiness probe hits `/health` on port 8001                                                                       |
| `localhost:8080` not reachable           | Use `kubectl port-forward svc/heart-risk-api 8080:8080 -n heart-risk`, or check the real `EXTERNAL-IP` from `kubectl get all -n heart-risk`                                   |
| HPA shows `<unknown>/70%`                | `metrics-server` isn't installed, or hasn't completed its first collection cycle yet (wait ~1 min)                                                                                |
| Grafana pod stuck in `ContainerCreating` | The `grafana-dashboards` ConfigMap or `grafana-admin-credentials` Secret is missing — see C.5                                                                                  |
| Prometheus target down/empty               | Deploy the API first; check `kubectl get endpoints heart-risk-api-headless -n heart-risk`                                                                                         |

## CI/CD

[.github/workflows/ci.yml](.github/workflows/ci.yml) — 4 sequential jobs, each gating the next
(`needs:`), so a lint/test failure stops the pipeline before training or building an image:

1. **lint** — `ruff check src tests scripts`, uploads `ruff-report.txt`
2. **test** — `pytest --junitxml=... --cov=src`, uploads the JUnit + coverage reports
3. **train** — ingest → clean → train (MLflow) → package → CLI smoke test, uploads
   `models/`, `reports/`, `mlruns/`
4. **docker** — builds the image from the `train` job's artifacts, runs it, polls until Docker
   reports the container `healthy`, then calls `/health` and `/predict`

A real failure of both the `lint` and `test` steps (a deliberate lint violation and a deliberate
failing test, each added and then removed again) was captured and is in
[reports/report.md](reports/report.md#7-cicd-pipeline--automated-testing) — both produced a clear,
specific error and a non-zero exit code, which is what makes the corresponding CI job fail red.

Reproduce the whole thing locally, from lint through a Docker build:

```bash
ruff check src tests scripts \
  && pytest \
  && python src/ingest.py \
  && python src/clean.py \
  && python src/train.py \
  && python src/package_model.py \
  && python src/inference.py \
  && docker build -t heart-risk-api:1.0 .
```

## API reference

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`

All 13 fields required, validated with Pydantic (`src/schemas.py`); invalid input → `422`.

| Field        | Type  | Range     | Meaning                                       |
| ------------ | ----- | --------- | --------------------------------------------- |
| `age`      | int   | 1–120    | Age in years                                  |
| `sex`      | int   | 0–1      | 1 = male, 0 = female                          |
| `cp`       | int   | 1–4      | Chest pain type                               |
| `trestbps` | int   | 50–260   | Resting blood pressure (mm Hg)                |
| `chol`     | int   | 80–700   | Serum cholesterol (mg/dl)                     |
| `fbs`      | int   | 0–1      | Fasting blood sugar > 120 mg/dl               |
| `restecg`  | int   | 0–2      | Resting ECG result                            |
| `thalach`  | int   | 50–250   | Max heart rate achieved                       |
| `exang`    | int   | 0–1      | Exercise-induced angina                       |
| `oldpeak`  | float | 0–10     | ST depression                                 |
| `slope`    | int   | 1–3      | Slope of peak exercise ST segment             |
| `ca`       | int   | 0–4      | Major vessels coloured                        |
| `thal`     | int   | 3, 6 or 7 | 3 normal, 6 fixed defect, 7 reversible defect |

```json
{"prediction": 0, "label": "no disease", "confidence": 0.7999, "probability_disease": 0.2001}
```

### `GET /metrics`

Prometheus text exposition format (`heart_api_*` counters/histograms).

## Results

Held-out test set (20%, stratified, `random_state=42`), ranked by cross-validated ROC-AUC — see
[reports/model_comparison.csv](reports/model_comparison.csv) and
[reports/report.md](reports/report.md#12-results-summary) for the full table and plots:

| Model                                    | CV ROC-AUC       | Test accuracy | Test ROC-AUC |
| ---------------------------------------- | ---------------- | ------------- | ------------ |
| **Logistic Regression** (selected) | **0.9082** | 0.8689        | 0.9578       |
| Random Forest                            | 0.9010           | 0.8852        | 0.9524       |
| Gradient Boosting                        | 0.8840           | 0.8852        | 0.9448       |

## Troubleshooting

General/local issues — see [B.4](#b4-docker-troubleshooting) for Docker-specific and
[C.9](#c9-kubernetes-troubleshooting) for Kubernetes-specific troubleshooting.

| Symptom                                                                 | Fix                                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError` in tests                                        | Run `pytest` from the project root so `pytest.ini`'s `pythonpath = src` applies                                                                                                                                                       |
| `ImportError: FallbackAsyncAdaptedQueuePool` from `sqlalchemy.pool` | SQLAlchemy 2.1+ isn't compatible with this MLflow version — already pinned to `SQLAlchemy==2.0.55` in `requirements.txt`; if you see this, re-install from the committed `requirements.txt` rather than installing mlflow separately |
| MLflow UI shows no runs                                                 | Start it from the project root with `--backend-store-uri sqlite:///tracking.db`                                                                                                                                                           |
| `FileNotFoundError: models/model.joblib`                              | Run `python src/train.py && python src/package_model.py` first (Option A.3/A.4) — needed before building the Docker image or deploying to Kubernetes                                                                                     |
