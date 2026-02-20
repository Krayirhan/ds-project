from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter(
    "ds_api_requests_total",
    "Total API requests",
    ["path", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "ds_api_request_latency_seconds",
    "API request latency in seconds",
    ["path", "method"],
)

INFERENCE_ROWS = Counter(
    "ds_api_inference_rows_total",
    "Total number of records processed by inference endpoints",
    ["endpoint", "model"],
)

INFERENCE_ERRORS = Counter(
    "ds_api_inference_errors_total",
    "Total number of inference errors",
    ["endpoint", "model"],
)

# ── Drift & quality gauges (set by monitoring CLI / scheduled job) ──────────

MODEL_AUC = Gauge(
    "ds_model_roc_auc",
    "Latest model ROC-AUC score from evaluation run",
    ["model", "run_id"],
)

PSI_SCORE = Gauge(
    "ds_feature_psi",
    "Population Stability Index for each feature vs reference distribution",
    ["feature"],
)

ACTION_RATE = Gauge(
    "ds_model_action_rate",
    "Fraction of records where the model recommends action=1",
    ["model", "run_id"],
)

LABEL_DRIFT = Gauge(
    "ds_label_drift_rate",
    "Observed positive label rate delta vs reference (abs value)",
    ["run_id"],
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
