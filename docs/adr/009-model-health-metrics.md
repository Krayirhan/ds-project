# ADR-009: Prometheus Gauges for Model Health Monitoring

## Status
Accepted

## Date
2026-02-19

## Context

The API exposes a `/metrics` endpoint (Prometheus format) for operational observability. Initially it tracked request-level metrics (latency, error rates, inference counts) but lacked **model health** signals that are critical for detecting silent model degradation:

- **PSI (Population Stability Index)**: detects feature drift between training and serving distributions
- **AUC**: tracks model discriminative ability on labeled data (when ground truth is available)
- **Action rate**: % of requests resulting in a positive decision — anomaly may indicate model drift
- **Label drift**: change in observed positive rate in recent predictions

Without these metrics, model degradation goes undetected until business KPIs are affected.

## Decision

Add four new Prometheus `Gauge` metrics to `src/metrics.py`:

| Metric Name | Labels | Description |
|-------------|--------|-------------|
| `ds_model_roc_auc` | `model`, `run_id` | ROC AUC on labeled evaluation data |
| `ds_feature_psi` | `feature`, `run_id` | PSI score per feature |
| `ds_model_action_rate` | `window`, `run_id` | Fraction of requests with positive decision |
| `ds_label_drift_rate` | `window`, `run_id` | Observed label positive rate in recent window |

`Gauge` (not `Counter` or `Histogram`) because these are **current state** measurements, not cumulative counts.

Additionally, three `PrometheusRule` alerts in `deploy/monitoring/prometheus-rule.yaml`:

| Alert | Condition | Severity | Window |
|-------|-----------|----------|--------|
| `DSProjectHighPSI` | `ds_feature_psi > 0.2` | warning | 5m |
| `DSProjectLowAUC` | `ds_model_roc_auc < 0.65` | critical | 10m |
| `DSProjectActionRateAnomaly` | `abs(rate - 7d_avg) > 0.15` | warning | 15m |

**PSI thresholds** follow industry convention: < 0.1 (stable), 0.1–0.2 (minor shift), > 0.2 (major shift — action required).

## Consequences

### Positive
- ✅ Model health is observable without waiting for business KPI degradation
- ✅ PSI alert provides early warning of data distribution changes
- ✅ AUC alert triggers when model accuracy falls below acceptable threshold
- ✅ Action rate anomaly catches edge cases (e.g., threshold misconfiguration)
- ✅ Labels (`model`, `run_id`, `feature`) enable per-model and per-feature Grafana breakdowns

### Negative
- ❌ PSI and AUC gauges must be set by the monitoring CLI (`python main.py monitor`) — not real-time
- ❌ Stale metrics if monitoring job fails silently (mitigated by `up` metric + alerting)

### Neutral
- Gauge cardinality is bounded: O(n_features) for PSI, O(1) for others per run_id

## Alternatives Considered

| Alternative | Rejected Because |
|-------------|-----------------|
| Histogram for PSI | PSI is a single scalar per feature; histogram adds no value |
| Push to external monitoring DB (InfluxDB, etc.) | Adds infrastructure; breaks existing Prometheus/Grafana stack |
| Business-layer-only monitoring | Delay between model degradation and business impact may be weeks |

## References
- [PSI: Population Stability Index](https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html)
- [Prometheus Gauge](https://prometheus.io/docs/concepts/metric_types/#gauge)
- `src/metrics.py`: gauge definitions
- `deploy/monitoring/prometheus-rule.yaml`: alert rules
