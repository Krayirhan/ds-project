from __future__ import annotations

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

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


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
