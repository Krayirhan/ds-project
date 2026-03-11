<div align="center">

# 🏨 Hotel Booking Cancellation Prediction

### Production-Grade MLOps Platform for Revenue Protection

*Cost-sensitive ML decisioning · Real-time API serving · Full observability · Kubernetes-native*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose%20v2-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5?style=flat-square&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-945DD6?style=flat-square&logo=dvc&logoColor=white)](https://dvc.org)
[![Coverage](https://img.shields.io/badge/Coverage-%E2%89%A580%25-brightgreen?style=flat-square)](pyproject.toml)
[![Version](https://img.shields.io/badge/Version-1.2.0-0075FF?style=flat-square)](CHANGELOG.md)

---

**This is not a notebook experiment.** It is a complete, deployable MLOps platform — built to predict hotel booking cancellations and translate those predictions into measurable revenue impact through policy-driven, business-aware decisions.

---

✦ **Business-aware ML** — decisions optimised against a real cost matrix, not just accuracy  
✦ **Production API** — versioned FastAPI endpoints with rate limiting, auth, and hot-reload  
✦ **Full observability** — Prometheus · Grafana · Jaeger tracing · automated drift alerts  
✦ **Safe deployments** — blue/green slots, canary ingress, automated policy rollback  
✦ **70+ test modules** · 119+ tracked experiment runs · 5-stage DVC pipeline

</div>

---

## Table of Contents

1. [Quick Summary](#quick-summary)
2. [Business Problem](#business-problem)
3. [Project Scope](#project-scope)
4. [Architecture](#architecture)
5. [Core Capabilities](#core-capabilities)
6. [Data & Modeling](#data--modeling)
7. [Results](#results)
8. [API & System Interfaces](#api--system-interfaces)
9. [Deployment & Runtime Topology](#deployment--runtime-topology)
10. [Monitoring, Drift & Reliability](#monitoring-drift--reliability)
11. [Repository Structure](#repository-structure)
12. [Quick Start](#quick-start)
13. [Testing & CI](#testing--ci)
14. [Documentation Map](#documentation-map)
15. [Roadmap](#roadmap)

---

## Quick Summary

| | |
|---|---|
| **Problem domain** | Hotel reservation management — cancellation risk |
| **System type** | End-to-end MLOps platform (training + serving + monitoring) |
| **ML approach** | Supervised binary classification with cost-sensitive threshold optimisation |
| **Serving layer** | Versioned REST API (`/v1`, `/v2`) · real-time probability + policy decisions |
| **Observability** | Prometheus · Grafana · Jaeger · PSI/KS drift monitoring · automated rollback |
| **Deployment** | Docker Compose (dev) · Kubernetes + Helm (prod) · blue/green · canary |
| **Test suite** | 70+ test modules · ≥ 80% branch coverage enforced in CI |
| **Experiment history** | 119+ tracked runs · DVC pipeline · per-run lineage artefacts |
| **Extended features** | React dashboard · RAG chat assistant · guest CRM · pgvector knowledge store |

---

## Business Problem

### Why cancellations matter

Hotel booking cancellations are one of the largest sources of revenue loss in hospitality. A cancelled room that goes unsold — or is resold below market rate at the last minute — directly erodes margin. At scale, unpredicted cancellation spikes cause:

- **Revenue leakage** from last-minute discounted re-bookings or unsold inventory
- **Operational inefficiency** in staffing, housekeeping, and supply planning
- **Customer experience degradation** when overbooking strategies misfire

### What this system changes

A naive model that predicts "cancel / don't cancel" doesn't solve the business problem — it just moves it. What matters is *what action to take* and *at what cost*.

This platform takes a different approach:

1. **Predict** the probability that each booking will result in a cancellation
2. **Decide** — using a configurable business cost matrix — whether the expected cost of intervention outweighs the cost of inaction
3. **Monitor** those decisions over time, checking for model degradation and data drift
4. **Rollback automatically** when real-world performance deviates from expectations

The result: a system where the ML model is a component in a *revenue-protection workflow*, not a standalone experiment.

> **Core insight:** A missed cancellation (FN) costs ~10× more than a false alarm (FP). Standard accuracy or F1 metrics are blind to this asymmetry. This system is not.

---

## Project Scope

The project is intentionally broad. To keep the narrative clear, capabilities are separated into two layers.

### Core Platform

The primary focus — production ML from data to monitored deployment:

| Component | Description |
|---|---|
| **ML Pipeline** | 5-stage DVC pipeline: preprocess → split → train → evaluate → predict |
| **Cost-Sensitive Decisioning** | Threshold optimisation driven by a configurable TP/FP/FN/TN cost matrix |
| **Two-Stage Calibration** | Sigmoid then isotonic regression calibration for reliable probability outputs |
| **Inference API** | Versioned FastAPI service (`/v1`, `/v2`) with auth, rate limiting, hot-reload |
| **Drift Monitoring** | PSI + KS-based feature and prediction drift with webhook alerting |
| **Policy Lifecycle** | CLI-driven promote/rollback with blue/green slot support |
| **Safe Deployment** | Docker Compose, Kubernetes manifests, Helm chart, canary ingress |
| **Observability** | Prometheus metrics, Grafana dashboards, Jaeger distributed tracing |

### Extended Features

Valuable additions built on top of the core platform:

| Component | Description |
|---|---|
| **React Dashboard** | Full-stack UI (Vite + React 18) with model analytics, run history, system health |
| **RAG Chat Assistant** | SSE-streaming chat powered by Ollama (`qwen2.5:7b`) + pgvector HNSW retrieval |
| **Guest CRM** | CRUD API for hotel guest records (PostgreSQL) |
| **Knowledge Store** | pgvector-backed document store with Prometheus instrumentation |

---

## Architecture

### High-Level Overview

```
  ┌─────────────────────────────────────────────────────────┐
  │                      Data Layer                         │
  │   Raw CSV  ──►  DVC Pipeline  ──►  Parquet Datasets    │
  └──────────────────────────┬──────────────────────────────┘
                             │ train / calibrate / evaluate
  ┌──────────────────────────▼──────────────────────────────┐
  │                     Model Registry                      │
  │          models/<run_id>/   ·   models/latest.json      │
  └──────────────────────────┬──────────────────────────────┘
                             │ load on startup / hot-reload
  ┌──────────────────────────▼──────────────────────────────┐
  │                    Inference API                        │
  │       FastAPI  ─  /v1  ·  /v2  ─  predict + decide     │
  └──────────┬──────────────────────────────────┬───────────┘
             │ serve                            │ scrape
  ┌──────────▼───────────┐         ┌────────────▼──────────┐
  │   Client / Frontend  │         │   Prometheus · Grafana │
  │   React 18 SPA       │         │   Jaeger Tracing       │
  └──────────────────────┘         └───────────────────────┘
             │
  ┌──────────▼──────────────────────────────────────────────┐
  │                  Drift Monitor (CI / cron)              │
  │   PSI · KS · AUC · Brier · webhook alerts · rollback   │
  └─────────────────────────────────────────────────────────┘
```

### Detailed Technical Flow

```
┌─────────────────────── React 18 SPA (Vite) ──────────────────────────┐
│  useAuth · useRuns · useChat · useTheme · useSystem · Chart.js        │
│  Pages: Overview · Model Analysis · Run History · System Status       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ HTTP / SSE
┌───────────────────────────────▼──────────────────────────────────────┐
│                      FastAPI Application                             │
│                  (src/api.py + api_lifespan.py)                      │
│                                                                      │
│  ┌─ /v1 · /v2 ──────────────────────────────────────────────────┐   │
│  │  predict_proba  →  model.predict_proba()                      │   │
│  │  decide         →  policy threshold  →  binary decision       │   │
│  └───────────────────────────────────────────────────────────────┘   │
│  ┌─ /dashboard ──────────────────────────────────────────────────┐   │
│  │  DashboardStore (SQLAlchemy · PostgreSQL)                     │   │
│  │  Redis cache layer (TTL 45 s)                                 │   │
│  └───────────────────────────────────────────────────────────────┘   │
│  ┌─ /chat ────────────────────────────────────────────────────── ┐   │
│  │  SSE stream  →  Ollama (qwen2.5:7b)  →  pgvector HNSW        │   │
│  └───────────────────────────────────────────────────────────────┘   │
│  ┌─ /health · /ready · /metrics · /reload ───────────────────────┐   │
│  │  Liveness · Readiness · Prometheus scrape · Hot-reload        │   │
│  └───────────────────────────────────────────────────────────────┘   │
└──────────┬──────────────┬────────────────────┬───────────────────────┘
           │              │                    │
     PostgreSQL         Redis              Ollama (qwen2.5:7b)
     + pgvector      (sessions             Local LLM + embeddings
     (data + runs)    + cache)
           │
     Prometheus ──► Grafana (dashboards)
     Jaeger      ──► Distributed trace viewer
```

**Architecture Decision Records** — 13 documented design decisions in [`docs/adr/`](docs/adr/)

<details>
<summary>View all ADRs</summary>

| ADR | Decision |
|---|---|
| [ADR-001](docs/adr/001-ml-pipeline-architecture.md) | ML pipeline architecture |
| [ADR-002](docs/adr/002-cost-sensitive-decisioning.md) | Cost-sensitive decisioning |
| [ADR-003](docs/adr/003-api-serving-design.md) | API serving design |
| [ADR-004](docs/adr/004-helm-gitops-deployment.md) | Helm + GitOps deployment |
| [ADR-005](docs/adr/005-distributed-tracing.md) | Distributed tracing |
| [ADR-006](docs/adr/006-data-validation-framework.md) | Data validation framework |
| [ADR-007](docs/adr/007-api-versioning.md) | API versioning strategy |
| [ADR-008](docs/adr/008-lua-rate-limiting.md) | Lua-based rate limiting |
| [ADR-009](docs/adr/009-model-health-metrics.md) | Model health metrics |
| [ADR-010](docs/adr/010-versioned-api-v1-v2.md) | Versioned API (v1 / v2) |
| [ADR-011](docs/adr/011-ollama-self-hosted.md) | Self-hosted LLM (Ollama) |
| [ADR-012](docs/adr/012-postgres-plus-redis.md) | Storage: PostgreSQL + Redis |
| [ADR-013](docs/adr/013-calibration-sigmoid-isotonic.md) | Two-stage calibration |

</details>

---

## Core Capabilities

### Training Pipeline
5-stage DVC pipeline (`preprocess → split → train → evaluate → predict`) with full data lineage tracking. Each run produces a timestamped artefact set and a lineage JSON. Reproducible with `dvc repro`.

### Cost-Sensitive Decisioning
Decisions are not made at a fixed 0.5 threshold. The system sweeps action-rate constraints (5 %, 10 %, 15 %, 20 %, 30 %) and selects the threshold that maximises realised profit under the configured cost matrix. A standard classifier optimised for accuracy would ignore this entirely.

### Two-Stage Model Calibration
Raw classifier probabilities are often poorly calibrated. The platform applies sigmoid calibration followed by isotonic regression, producing reliable, well-ordered probability outputs. This matters because the decision logic depends on probability magnitudes, not just rankings.

### Versioned Inference API
FastAPI serves two independent model versions (`/v1`, `/v2`) simultaneously. Both support `predict_proba` (probability) and `decide` (binary policy decision). Zero-downtime hot-reload is available via `POST /reload`.

### PSI / KS Drift Monitoring
After deployment, the system continuously checks for feature drift (PSI) and prediction drift (PSI + KS test). Alert flags trigger webhook notifications. If `any_alert=true` is detected in CI, a policy rollback is automatically initiated.

### Policy Lifecycle Management
Every promoted model run is versioned. Operators can promote specific run IDs to blue or green slots, inspect active policies via CLI, and roll back to the previous policy in a single command — no re-deployment required.

### Safe Deployment Strategy
The platform supports blue/green slot promotion, NGINX-weighted canary ingress (default: 10 % traffic), Kubernetes HPA, PDB, and network policies. Canary weight is controlled via annotations — no application code changes needed.

---

## Data & Modeling

### Dataset

| Property | Value |
|---|---|
| Source | [Hotel Booking Demand — Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |
| Records | ~119,000 bookings |
| Cancellation rate | ~37 % |
| Features | Booking metadata, lead time, special requests, distribution channel, customer type |

### Pipeline Stages

```
Raw CSV
  │
  ├─ 1. Preprocess
  │     Feature engineering · categorical encoding · Pandera schema validation
  │     Anomaly detection · reference statistics saved for drift baseline
  │
  ├─ 2. Split   (test_size=0.20 · cal_size=0.20 · seed=42)
  │     train.parquet  ·  cal.parquet  ·  test.parquet
  │
  ├─ 3. Train   (cv_folds=5)
  │     Cross-validated scikit-learn classifier
  │     Hyperparameter optimisation (src/hpo.py)
  │
  ├─ 4. Calibrate
  │     Stage 1: sigmoid calibration  (CalibratedClassifierCV)
  │     Stage 2: isotonic calibration (on held-out cal set)
  │     Both variants saved as separate artefacts
  │
  └─ 5. Evaluate
        AUC · F1 · Brier score · precision · recall
        Cost-matrix profit sweep across action-rate constraints
        Feature importance · threshold sweep report
```

### Decision Logic

Model selection is not based on AUC alone. The evaluation compares models on **realised profit** under the business cost matrix:

| | Predicted: Cancel | Predicted: No Cancel |
|---|---|---|
| **Actual: Cancel** | ✅ TP = **+$180** | ❌ FN = **−$200** |
| **Actual: No Cancel** | ⚠️ FP = **−$20** | TN = $0 |

A missed cancellation (FN) costs 10× more than a false alarm (FP). The policy threshold is therefore set to maximise expected profit, not F1 or accuracy.

---

## Results

### Model Comparison (test set, n = 23,878)

| Model | Threshold | ROC AUC | F1 | Precision | Recall | Brier Score |
|---|---|---|---|---|---|---|
| Baseline | 0.50 (default) | **0.896** | 0.731 | 0.812 | 0.665 | — |
| Baseline + sigmoid calibration | — | 0.896 | — | — | — | 0.122 |
| **Baseline + isotonic calibration** | — | 0.896 | — | — | — | **0.121** |
| **Decision policy** | **0.35** (profit-optimal) | 0.896 | **0.754** | 0.714 | **0.798** | 0.121 |

### Key observations

- **Calibration matters:** Isotonic calibration reduces Brier score by ~0.8 % relative and produces better-ordered probabilities for threshold sweeping.
- **Threshold matters more:** Shifting from the default 0.50 to the profit-optimal 0.35 increases recall by **+13.3 pp** — capturing 1,178 additional true cancellations at the cost of only 1,035 additional false alarms.
- **Business outcome:** At the 30 % action-rate constraint, the profit-optimal policy achieves a positive realised profit (+$428,180 on the test set) compared to a net loss under the 5 % constraint — demonstrating that the deployment policy, not just the model, determines business value.

> **Selected model:** Isotonic-calibrated classifier with decision threshold 0.35, tuned to maximise realised profit under the configured cost matrix.

---

## API & System Interfaces

Full API documentation is available at `http://localhost:8000/docs` (Swagger UI) when the server is running.

### Authentication

All inference and admin endpoints require:
```
x-api-key: <DS_API_KEY>
```

### Endpoint Summary

**Core inference**

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/predict_proba` | Cancellation probability — model v1 |
| `POST` | `/v2/predict_proba` | Cancellation probability — model v2 |
| `POST` | `/v1/decide` | Policy-based binary decision — model v1 |
| `POST` | `/v2/decide` | Policy-based binary decision — model v2 |
| `POST` | `/reload` | Hot-reload model + policy (zero downtime) |

**Operational**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness — model + policy loaded |
| `GET` | `/metrics` | Prometheus scrape target |

**Dashboard & auth** (see [docs/architecture.md](docs/architecture.md) for full detail)

| Method | Path | Description |
|---|---|---|
| `GET` | `/dashboard/api/overview` | Metrics summary |
| `GET` | `/dashboard/api/runs` | Experiment run history |
| `POST` | `/auth/login` · `/auth/logout` | Session management |

**Rate limiting options:** `RATE_LIMIT_BACKEND=memory` (single-pod / dev) · `redis` (distributed / prod)

---

## Deployment & Runtime Topology

### Local development

```bash
# Install dependencies
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Run API only
python main.py serve-api --host 0.0.0.0 --port 8000

# Run frontend
cd apps/frontend && npm install && npm run dev
```

### Dockerised full stack

```bash
docker compose -f docker-compose.dev.yml up --build
```

| Service | URL | Notes |
|---|---|---|
| API | http://localhost:8000 | FastAPI + ML |
| Frontend | http://localhost:5173 | React SPA |
| Grafana | http://localhost:3000 | Dashboards |
| Prometheus | http://localhost:9090 | Metrics |
| Jaeger | http://localhost:16686 | Tracing |

### Kubernetes

Manifests in [`deploy/k8s/`](deploy/k8s/) — namespace, Deployment, Service, HPA, PDB, NetworkPolicy, Ingress, canary Deployment + canary Ingress (NGINX, default 10 % weight).

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/hpa.yaml
kubectl apply -f deploy/k8s/canary-deployment.yaml
kubectl apply -f deploy/k8s/canary-ingress.yaml
```

Helm chart: [`deploy/helm/`](deploy/helm/) · GitOps config: [`deploy/gitops/`](deploy/gitops/)

### Policy management (CLI)

```bash
# Promote a specific run
python main.py promote-policy --run-id 20260311_015136

# Blue/green slot assignment
python main.py promote-policy --run-id 20260311_015136 --slot blue

# Rollback
python main.py rollback-policy
python main.py rollback-policy --slot blue
```

---

## Monitoring, Drift & Reliability

```bash
python main.py monitor
```

### What is checked

| Signal | Method | Action |
|---|---|---|
| Feature drift | PSI (Population Stability Index) | Alert flag if threshold exceeded |
| Prediction drift | PSI + KS test | Alert flag + webhook |
| Model performance | AUC · Brier score · realised profit | Alert flag if degraded |
| Service health | `/health` · `/ready` endpoints | Kubernetes liveness/readiness probes |

### Alert delivery

Set `ALERT_WEBHOOK_URL` to receive alert payloads. Failed deliveries are queued in a dead-letter queue and retried:

```bash
python main.py retry-webhook-dlq --url https://your-webhook.example.com
```

### Automated rollback in CI

The `monitor.yml` GitHub Actions workflow runs on schedule and on push. If `any_alert=true` is found in `reports/monitoring/latest_monitoring_report.json`, **policy rollback is triggered automatically** before it causes further impact.

### Observability stack

| Tool | Role |
|---|---|
| Prometheus | Metrics collection (request count, inference latency, retrieval latency) |
| Grafana | Pre-built dashboards in [`deploy/monitoring/`](deploy/monitoring/) |
| Jaeger | Distributed tracing via OTLP (configured in `api_lifespan.py`) |

---

## Repository Structure

```
hotel-booking-cancellation-prediction/
│
├── apps/
│   ├── backend/           Production entrypoint (python -m apps.backend.main)
│   └── frontend/          React 18 SPA (Vite · Chart.js · react-router-dom)
│
├── data/
│   ├── raw/               hotel_bookings.csv — download from Kaggle, not in Git
│   ├── interim/           Intermediate processing artefacts
│   └── processed/         DVC-tracked Parquet splits (train / cal / test)
│
├── deploy/
│   ├── k8s/               Kubernetes manifests (Deployment · HPA · PDB · canary)
│   ├── helm/              Helm chart
│   ├── gitops/            GitOps sync configuration
│   ├── dev/               Local dev overrides
│   └── monitoring/        Grafana dashboard JSON · Prometheus alert rules
│
├── docs/
│   ├── architecture.md    Full system architecture reference
│   ├── runbook.md         On-call runbook
│   ├── slo.md             Service Level Objectives
│   └── adr/               13 Architecture Decision Records
│
├── models/                Trained artefacts — not in Git, produced by pipeline
│
├── notebooks/             Exploratory analysis
│
├── perf/
│   ├── locustfile.py      Locust load test
│   └── k6_smoke.js        k6 SLO smoke test
│
├── reports/
│   ├── metrics/           Per-run evaluation JSON files · profit sweeps · lineage
│   ├── monitoring/        Drift monitoring reports
│   └── figures/           Generated plots
│
├── scripts/
│   └── check_setup.py     10-point environment validation script
│
├── src/                   Core ML + API source package
│   ├── api.py             FastAPI app factory
│   ├── api_lifespan.py    Startup / shutdown / tracing
│   ├── api_v1.py          v1 prediction endpoints
│   ├── api_v2.py          v2 prediction endpoints
│   ├── calibration.py     Two-stage calibration
│   ├── cost_matrix.py     Business cost matrix + threshold sweep
│   ├── data_validation.py Pandera schema · drift · anomaly
│   ├── dashboard.py       Dashboard router + Redis cache
│   ├── monitoring.py      PSI / KS drift analysis
│   ├── policy.py          Policy promote / rollback / slot management
│   ├── metrics.py         Prometheus definitions
│   └── chat/              RAG pipeline · Ollama client · pgvector store
│
├── tests/                 70+ pytest modules · unit · integration · smoke
│
├── dvc.yaml               DVC pipeline definition
├── params.yaml            Hyperparameters + cost matrix
├── docker-compose.dev.yml Full dev stack
├── docker-compose.prod.yml Production compose
├── Dockerfile             Multi-stage production image
└── pyproject.toml         Build config · ruff · mypy · coverage settings
```

---

## Quick Start

### Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12 | ML pipeline + API |
| Docker Desktop | 24+ | Full stack via Compose |
| Docker Compose v2 | built-in | Service orchestration |
| Git | any | Clone repo |
| Ollama | latest | Chat assistant (optional) |
| Node.js | ≥ 18 | Frontend dev (optional) |

### 1 — Clone

```bash
git clone https://github.com/<YOUR_USERNAME>/hotel-booking-cancellation-prediction.git
cd hotel-booking-cancellation-prediction
```

### 2 — Validate environment (10 automated checks)

```bash
python scripts/check_setup.py
```

### 3 — Configure

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux / macOS
```

Edit the four required keys in `.env`:

```dotenv
DS_API_KEY=<strong-random-string>
DASHBOARD_ADMIN_PASSWORD_ADMIN=<bcrypt-hash>
POSTGRES_PASSWORD=<db-password>
GF_ADMIN_PASSWORD=<grafana-password>
```

> ⚠️ Never commit `.env` — it is in `.gitignore`.

### 4 — Train the model

> Download `hotel_bookings.csv` from [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) and place it in `data/raw/`.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows — use source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python main.py preprocess
python main.py train
python main.py evaluate
```

Or reproduce the full DVC pipeline:

```bash
dvc repro
```

### 5 — Start the full stack

```bash
docker compose -f docker-compose.dev.yml up --build
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Frontend | http://localhost:5173 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Jaeger | http://localhost:16686 |

```bash
# Stop (data preserved)
docker compose -f docker-compose.dev.yml down

# Stop + delete volumes
docker compose -f docker-compose.dev.yml down -v
```

---

## Testing & CI

```bash
pytest
```

Branch coverage ≥ 80 % enforced (configured in `pyproject.toml`, checked in CI via `--cov-fail-under=80`).

| Test category | Modules | Examples |
|---|---|---|
| Unit — ML | `test_train.py` · `test_calibration.py` · `test_features.py` | Training logic, calibration variants, feature engineering |
| Unit — policy | `test_policy.py` · `test_cost_matrix.py` | Threshold selection, slot management, rollback |
| Unit — validation | `test_data_validation.py` · `test_schema_validation.py` | Pandera schema, drift detectors, anomaly flags |
| Integration — API | `test_api.py` · `test_api_versions.py` · `test_dashboard.py` | Endpoint contracts, auth, rate limiting |
| Integration — stores | `test_dashboard_store.py` · `test_guest_store.py` | DB read/write, Redis cache |
| End-to-end | `test_pipeline_smoke.py` · `test_predict_full.py` | Full pipeline pass-through |

**CI/CD workflows (GitHub Actions):**

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | Push / PR | Lint · type-check · test · coverage gate |
| `deploy.yml` | Push to `main` | Build image · push registry · deploy |
| `monitor.yml` | Schedule + push | Drift check · auto-rollback |
| `security.yml` | Push / PR | Dependency vulnerability scan |

**SLO targets (k6 smoke test):**

| Metric | Threshold |
|---|---|
| p95 response time | < 300 ms |
| p99 response time | < 800 ms |
| Error rate | < 1 % |

---

## Documentation Map

| Topic | Location |
|---|---|
| Full architecture | [docs/architecture.md](docs/architecture.md) |
| On-call runbook | [docs/runbook.md](docs/runbook.md) |
| Service Level Objectives | [docs/slo.md](docs/slo.md) |
| Architecture decisions | [docs/adr/](docs/adr/) — ADR-001 through ADR-013 |
| API reference (live) | `http://localhost:8000/docs` (Swagger UI) |
| Environment variables | [.env.example](.env.example) — fully annotated |
| Pipeline parameters | [params.yaml](params.yaml) |
| Deployment manifests | [deploy/k8s/](deploy/k8s/) · [deploy/helm/](deploy/helm/) |
| Monitoring dashboards | [deploy/monitoring/](deploy/monitoring/) |
| Experiment history | `reports/metrics/<run_id>/` |
| Version history | [CHANGELOG.md](CHANGELOG.md) |

---

## Roadmap

### Model & Evaluation
- [ ] Hyperparameter optimisation with Optuna (foundation in `src/hpo.py` — needs full integration)
- [ ] Online learning / incremental retraining on production traffic
- [ ] Feature store integration for consistent train/serve feature pipelines
- [ ] SHAP-based explainability endpoint (`/explain`)

### Platform
- [ ] Multi-tenant API key management
- [ ] S3 / GCS remote storage for DVC artefacts (currently local)
- [ ] Model registry UI with artefact comparison across runs
- [ ] Shadow mode scoring (run new model in parallel without affecting decisions)

### Operations
- [ ] Alert routing to PagerDuty / Slack natively (currently webhook-generic)
- [ ] Automated retraining pipeline triggered by drift thresholds
- [ ] Chaos engineering test suite for resilience validation
- [ ] GPU inference support for larger embedding models in the RAG pipeline

---

<div align="center">

**Core:** FastAPI · scikit-learn · DVC · PostgreSQL · Redis · Prometheus · Grafana · Jaeger · Kubernetes

**Extended:** React 18 · Ollama · pgvector · Helm · GitHub Actions · Locust · k6

</div>
