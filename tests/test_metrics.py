"""Tests for src.metrics — Prometheus metric definitions."""

from __future__ import annotations

from src.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    INFERENCE_ROWS,
    INFERENCE_ERRORS,
    GUEST_RISK_FALLBACK_TOTAL,
    MODEL_AUC,
    PSI_SCORE,
    ACTION_RATE,
    LABEL_DRIFT,
    KNOWLEDGE_RETRIEVAL_TOTAL,
    KNOWLEDGE_RETRIEVAL_EMPTY,
    KNOWLEDGE_RETRIEVAL_HIT_COUNT,
    KNOWLEDGE_SIMILARITY_SCORE,
    render_metrics,
)


class TestMetricDefinitions:
    """All Prometheus collectors are importable and have correct type."""

    def test_request_count_is_counter(self) -> None:
        from prometheus_client import Counter

        assert isinstance(REQUEST_COUNT, Counter)

    def test_request_latency_is_histogram(self) -> None:
        from prometheus_client import Histogram

        assert isinstance(REQUEST_LATENCY, Histogram)

    def test_inference_rows_is_counter(self) -> None:
        from prometheus_client import Counter

        assert isinstance(INFERENCE_ROWS, Counter)

    def test_inference_errors_is_counter(self) -> None:
        from prometheus_client import Counter

        assert isinstance(INFERENCE_ERRORS, Counter)

    def test_guest_risk_fallback_is_counter(self) -> None:
        from prometheus_client import Counter

        assert isinstance(GUEST_RISK_FALLBACK_TOTAL, Counter)

    def test_model_auc_is_gauge(self) -> None:
        from prometheus_client import Gauge

        assert isinstance(MODEL_AUC, Gauge)

    def test_psi_score_is_gauge(self) -> None:
        from prometheus_client import Gauge

        assert isinstance(PSI_SCORE, Gauge)

    def test_action_rate_is_gauge(self) -> None:
        from prometheus_client import Gauge

        assert isinstance(ACTION_RATE, Gauge)

    def test_label_drift_is_gauge(self) -> None:
        from prometheus_client import Gauge

        assert isinstance(LABEL_DRIFT, Gauge)


class TestKnowledgeMetrics:
    """Knowledge retrieval metrics are defined properly."""

    def test_retrieval_total(self) -> None:
        from prometheus_client import Counter

        assert isinstance(KNOWLEDGE_RETRIEVAL_TOTAL, Counter)

    def test_retrieval_empty(self) -> None:
        from prometheus_client import Counter

        assert isinstance(KNOWLEDGE_RETRIEVAL_EMPTY, Counter)

    def test_hit_count_histogram(self) -> None:
        from prometheus_client import Histogram

        assert isinstance(KNOWLEDGE_RETRIEVAL_HIT_COUNT, Histogram)

    def test_similarity_score_histogram(self) -> None:
        from prometheus_client import Histogram

        assert isinstance(KNOWLEDGE_SIMILARITY_SCORE, Histogram)


class TestRenderMetrics:
    """render_metrics returns Prometheus text format."""

    def test_returns_bytes_and_content_type(self) -> None:
        payload, content_type = render_metrics()
        assert isinstance(payload, bytes)
        assert "text/plain" in content_type or "text/plain" in str(content_type)

    def test_contains_metric_names(self) -> None:
        payload, _ = render_metrics()
        text = payload.decode("utf-8")
        assert "ds_api_requests_total" in text or "ds_api_request_latency" in text
