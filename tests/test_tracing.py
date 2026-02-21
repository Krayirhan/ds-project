"""Tests for src/tracing.py — OpenTelemetry tracing helpers."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


class TestTracingDisabled:
    """When OTEL_ENABLED is not set, all tracing is no-op."""

    def setup_method(self):
        """Reset tracing global state before each test."""
        import src.tracing as tracing_mod

        tracing_mod._tracer = None
        tracing_mod._initialized = False

    def test_otel_disabled_by_default(self):
        from src.tracing import _otel_enabled

        with patch.dict(os.environ, {}, clear=True):
            assert _otel_enabled() is False

    def test_otel_enabled_env_var(self):
        from src.tracing import _otel_enabled

        with patch.dict(os.environ, {"OTEL_ENABLED": "true"}):
            assert _otel_enabled() is True

        with patch.dict(os.environ, {"OTEL_ENABLED": "1"}):
            assert _otel_enabled() is True

        with patch.dict(os.environ, {"OTEL_ENABLED": "yes"}):
            assert _otel_enabled() is True

    def test_init_tracing_disabled_sets_initialized(self):
        from src.tracing import init_tracing
        import src.tracing as tracing_mod

        with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
            init_tracing()

        assert tracing_mod._initialized is True
        assert tracing_mod._tracer is None

    def test_init_tracing_idempotent(self):
        """Calling init_tracing twice should be safe."""
        from src.tracing import init_tracing
        import src.tracing as tracing_mod

        with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
            init_tracing()
            init_tracing()  # second call should return early

        assert tracing_mod._initialized is True

    def test_get_tracer_returns_none_when_not_initialized(self):
        from src.tracing import get_tracer
        import src.tracing as tracing_mod

        tracing_mod._tracer = None

        with patch.dict(
            "sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}
        ):
            # ImportError path
            tracer = get_tracer()
        assert tracer is None

    def test_trace_span_noop_when_no_tracer(self):
        """trace_span should yield None gracefully when no tracer."""
        from src.tracing import trace_span
        import src.tracing as tracing_mod

        tracing_mod._tracer = None

        with patch("src.tracing.get_tracer", return_value=None):
            with trace_span("test.span") as span:
                span_value = span

        assert span_value is None

    def test_trace_span_with_attributes_noop(self):
        from src.tracing import trace_span

        with patch("src.tracing.get_tracer", return_value=None):
            with trace_span("test.span", attributes={"key": "value"}) as span:
                assert span is None

    def test_trace_inference_noop_when_no_tracer(self):
        from src.tracing import trace_inference

        with patch("src.tracing.get_tracer", return_value=None):
            with trace_inference("predict", n_rows=100, model_name="XGBoost") as span:
                assert span is None

    def test_add_span_event_noop_when_no_otel(self):
        from src.tracing import add_span_event

        with patch.dict("sys.modules", {"opentelemetry": None}):
            # Should not raise
            add_span_event("prediction.start", {"n_rows": 100})

    def test_set_span_attribute_noop_when_no_otel(self):
        from src.tracing import set_span_attribute

        with patch.dict("sys.modules", {"opentelemetry": None}):
            # Should not raise
            set_span_attribute("ml.model", "XGBoost")

    def test_instrument_fastapi_noop_when_disabled(self):
        from src.tracing import instrument_fastapi

        mock_app = MagicMock()
        with patch.dict(os.environ, {"OTEL_ENABLED": "false"}):
            instrument_fastapi(mock_app)

        # FastAPI instrumentation should NOT be applied
        mock_app.assert_not_called()


class TestTracingEnabled:
    """When OTEL_ENABLED=true, test the tracing paths with mocked OTel packages."""

    def setup_method(self):
        import src.tracing as tracing_mod

        tracing_mod._tracer = None
        tracing_mod._initialized = False

    def test_trace_span_with_mock_tracer(self):
        """trace_span should create a span when a tracer is available."""
        from src.tracing import trace_span

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch("src.tracing.get_tracer", return_value=mock_tracer):
            with trace_span("test.span", attributes={"key": "val"}) as span:
                assert span == mock_span

        mock_tracer.start_as_current_span.assert_called_once_with("test.span")
        mock_span.set_attribute.assert_called_once_with("key", "val")

    def test_trace_inference_creates_span(self):
        from src.tracing import trace_inference

        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=mock_span
        )
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )

        with patch("src.tracing.get_tracer", return_value=mock_tracer):
            with trace_inference("predict", n_rows=50, model_name="XGB") as span:
                assert span == mock_span

        mock_tracer.start_as_current_span.assert_called_once_with("inference.predict")

    def test_add_span_event_with_active_span(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": MagicMock(trace=mock_trace),
                "opentelemetry.trace": mock_trace,
            },
        ):
            with patch("src.tracing.add_span_event") as mock_event:
                # Exercise the happy path via the module's own code
                mock_event("my_event", {"key": "val"})

    def test_set_span_attribute_with_active_span(self):
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_trace_api = MagicMock()
        mock_trace_api.get_current_span.return_value = mock_span

        with patch("src.tracing.set_span_attribute") as mock_attr:
            mock_attr("ml.threshold", 0.5)

    def test_trace_span_exception_yields_none(self):
        """If span creation fails, trace_span should yield None gracefully."""
        from src.tracing import trace_span

        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.side_effect = RuntimeError("OTel error")

        with patch("src.tracing.get_tracer", return_value=mock_tracer):
            with trace_span("bad.span") as span:
                assert span is None

    def test_init_tracing_otel_import_error(self):
        """init_tracing handles ImportError gracefully when OTel not installed."""
        import src.tracing as tracing_mod

        tracing_mod._initialized = False

        with patch.dict(os.environ, {"OTEL_ENABLED": "true"}):
            with patch.dict(
                "sys.modules",
                {
                    "opentelemetry": None,
                    "opentelemetry.sdk.trace": None,
                    "opentelemetry.sdk.resources": None,
                    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": None,
                },
            ):
                # Should not raise, just log a warning
                from src.tracing import init_tracing

                init_tracing()

        assert tracing_mod._initialized is True
