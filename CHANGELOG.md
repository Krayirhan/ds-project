# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-02-18

### Added
- **Helm Chart**: Parameterized Kubernetes deployment via Helm chart (`deploy/helm/ds-project/`)
  - Environment-specific values: `values-staging.yaml`, `values-production.yaml`
  - Templates: Deployment, Service, Ingress, HPA, PDB, NetworkPolicy, Canary, PrometheusRule
- **GitOps (ArgoCD + Flux)**: Declarative deployment sync (`deploy/gitops/`)
  - ArgoCD: Project + staging (auto-sync) + production (manual approve)
  - Flux: GitRepository + HelmRelease for both environments
- **Distributed Tracing**: OpenTelemetry instrumentation (`src/tracing.py`)
  - OTLP gRPC export to Jaeger/Tempo
  - Automatic FastAPI span creation
  - Custom ML inference spans with `ml.*` attributes
  - Configurable via `OTEL_ENABLED`, `OTEL_EXPORTER_OTLP_ENDPOINT`
- **docker-compose dev stack**: Single-command local environment (`docker-compose.dev.yml`)
  - API + Redis + Prometheus + Grafana + Jaeger
  - Pre-configured datasources and dashboard provisioning
- **Staging deploy workflow**: CI/CD pipeline with approval gate (`.github/workflows/deploy.yml`)
  - Build → Deploy staging → Smoke tests → Manual approve → Deploy production
- **Data validation framework**: Pandera-based schema + distribution assertions (`src/data_validation.py`)
  - Raw data schema (hotel bookings contract)
  - Processed data schema (post-preprocessing)
  - Inference payload validation
  - Distribution drift checks with configurable tolerance
  - Reference stats generation for monitoring
- **API versioning**: `/v1` and `/v2` prefix routing (`src/api_v1.py`, `src/api_v2.py`)
  - V1: Backward-compatible original endpoints
  - V2: Enhanced responses with `meta` block (api_version, model_used, latency_ms, request_id)
  - Root endpoints preserved for backward compatibility
- **Architecture Decision Records**: ADR documentation (`docs/adr/`)
- **CHANGELOG**: Structured change tracking (this file)

### Changed
- API app description updated to reference versioned endpoints
- Tracing integrated into predict_proba and decide endpoints

## [1.1.0] - 2026-02-17

### Added
- Blue/Green deployment slots with promote/rollback CLI commands
- Canary deployment support (K8s manifests)
- Rate limiting with Redis backend support
- API key authentication middleware
- Prometheus metrics (request count, latency histogram, inference counters)
- Grafana dashboard JSON
- PrometheusRule alerts (p95 latency, 5xx rate, inference errors)
- Alertmanager config (Slack + PagerDuty routing)
- Network policy for pod traffic isolation
- PodDisruptionBudget for safe rollouts
- HPA (CPU-based autoscaling)
- Webhook DLQ retry mechanism
- HPO with Optuna
- SHAP + permutation importance explainability
- MLflow experiment tracking integration
- Cost-sensitive decision framework
- Pipeline smoke tests in CI

## [1.0.0] - 2026-02-16

### Added
- Initial ML pipeline: preprocess → train → evaluate → predict → monitor
- Baseline LogisticRegression + challenger models (XGBoost, LightGBM, CatBoost, HistGB)
- Calibration (Isotonic + Sigmoid)
- FastAPI serving endpoint (`/predict_proba`, `/decide`, `/reload`)
- Feature spec contract (JSON schema)
- Decision policy engine with profit-optimal threshold selection
- DVC pipeline definition
- Docker multi-stage build
- CI pipeline (lint, type check, security, tests, coverage gate)
- Basic data validation (schema checks, target label validation)

[Unreleased]: https://github.com/your-org/ds-project/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/your-org/ds-project/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/your-org/ds-project/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/your-org/ds-project/releases/tag/v1.0.0
