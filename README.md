# ds_project — Production DS Pipeline

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py preprocess
python main.py train
python main.py evaluate
python main.py predict
```

## API Serving

Set API key first:

```bash
set DS_API_KEY=your-secret-key
set RATE_LIMIT_BACKEND=redis
set REDIS_URL=redis://localhost:6379/0
python main.py serve-api --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` (liveness)
- `GET /ready` (readiness with model+policy loaded)
- `GET /metrics` (Prometheus)
- `POST /predict_proba`
- `POST /decide`
- `POST /reload` (reload model+policy without restart)

Send header:
- `x-api-key: <DS_API_KEY>`

Rate limit backends:
- `memory` (single pod/dev)
- `redis` (distributed, multi-replica production)

## Monitoring

```bash
python main.py monitor
```

Outputs:
- `reports/monitoring/<run_id>/monitoring_report.json`
- `reports/monitoring/latest_monitoring_report.json`

Supports:
- data drift (PSI)
- prediction drift (PSI + KS)
- outcome monitoring (AUC, Brier, realized profit)
- alert flags

Webhook alerts:
- set `ALERT_WEBHOOK_URL`

Dead-letter queue retry:

```bash
python main.py retry-webhook-dlq --url https://example.com/webhook
```

## Rollout / Rollback

Promote a run policy:

```bash
python main.py promote-policy --run-id 20260217_220731
```

Blue/green slot promotion:

```bash
python main.py promote-policy --run-id 20260217_220731 --slot blue
python main.py promote-policy --run-id 20260217_220731 --slot green
```

Rollback to previous policy:

```bash
python main.py rollback-policy
```

Slot rollback:

```bash
python main.py rollback-policy --slot blue
```

## Tests

```bash
pytest
```

Includes:
- policy unit tests
- schema validation tests
- end-to-end smoke test

## Container

```bash
docker build -t ds-project:latest .
docker run -e DS_API_KEY=your-secret-key -p 8000:8000 ds-project:latest
```

## Kubernetes

Manifests under [deploy/k8s](deploy/k8s):
- namespace, deployment, service
- HPA
- network policy
- PDB
- secret example
- canary deployment and canary ingress

Apply:

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/secrets.example.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/hpa.yaml
kubectl apply -f deploy/k8s/network-policy.yaml
kubectl apply -f deploy/k8s/pdb.yaml
kubectl apply -f deploy/k8s/ingress.yaml
kubectl apply -f deploy/k8s/canary-deployment.yaml
kubectl apply -f deploy/k8s/canary-ingress.yaml
```

Canary traffic split:
- `deploy/k8s/canary-ingress.yaml` uses NGINX canary annotations
- default weight is 10%

## CI/CD

GitHub Actions workflows:
- `.github/workflows/ci.yml`
- `.github/workflows/monitor.yml`
- `.github/workflows/security.yml`

`monitor.yml` includes automatic policy rollback if monitoring reports `any_alert=true`.

## Load / Performance Tests

Locust:

```bash
locust -f perf/locustfile.py --host http://127.0.0.1:8000
```

k6:

```bash
k6 run perf/k6_smoke.js
```

SLO checks are encoded in k6 thresholds (`p95 < 300ms`, `p99 < 800ms`).

## Data Lineage / Versioning

- DVC pipeline definition: [dvc.yaml](dvc.yaml)
- Experiment parameters: [params.yaml](params.yaml)
- Preprocess lineage artifact: `reports/metrics/data_lineage_preprocess.json`
- Train lineage artifact: `reports/metrics/<run_id>/data_lineage.json`

Use:

```bash
dvc repro
```

## Runtime Dashboard & Alert Routing

- Prometheus rules: `deploy/monitoring/prometheus-rule.yaml`
- Alertmanager routing (Slack + PagerDuty): `deploy/monitoring/alertmanager-config.yaml`
- Grafana dashboard JSON: `deploy/monitoring/grafana-dashboard.json`

## Contracts and Integrity

- Policy includes `policy_version` and `feature_schema_version`
- Serving verifies model SHA256 checksum against policy
- Predict/API fail-fast on contract mismatch

## Runbook

See [docs/runbook.md](docs/runbook.md).

## SLO

See [docs/slo.md](docs/slo.md).

## Environment Variables

See [.env.example](.env.example).
