from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock

import src.tracing as tracing


class _TracerProvider:
    def __init__(self, resource=None):
        self.resource = resource
        self.processors = []

    def add_span_processor(self, processor):
        self.processors.append(processor)


class _BatchSpanProcessor:
    def __init__(self, exporter):
        self.exporter = exporter


class _Resource:
    @staticmethod
    def create(attrs):
        return {"attrs": attrs}


class _OTLPSpanExporter:
    def __init__(self, endpoint, insecure):
        self.endpoint = endpoint
        self.insecure = insecure


def _fake_otel_modules(trace_api):
    root = types.ModuleType("opentelemetry")
    root.trace = trace_api

    sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
    sdk_trace.TracerProvider = _TracerProvider

    sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
    sdk_trace_export.BatchSpanProcessor = _BatchSpanProcessor

    sdk_resources = types.ModuleType("opentelemetry.sdk.resources")
    sdk_resources.Resource = _Resource
    sdk_resources.SERVICE_NAME = "service.name"

    otlp_exporter = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    otlp_exporter.OTLPSpanExporter = _OTLPSpanExporter

    return {
        "opentelemetry": root,
        "opentelemetry.trace": trace_api,
        "opentelemetry.sdk.trace": sdk_trace,
        "opentelemetry.sdk.trace.export": sdk_trace_export,
        "opentelemetry.sdk.resources": sdk_resources,
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter": otlp_exporter,
    }


def setup_function():
    tracing._tracer = None
    tracing._initialized = False


def test_init_tracing_success(monkeypatch):
    trace_api = types.SimpleNamespace()
    trace_api.set_tracer_provider = MagicMock()
    trace_api.get_tracer = MagicMock(return_value="TRACER")

    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    modules = _fake_otel_modules(trace_api)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    tracing.init_tracing(service_name="svc-a")

    assert tracing._initialized is True
    assert tracing._tracer == "TRACER"
    assert trace_api.set_tracer_provider.called
    assert trace_api.get_tracer.called


def test_instrument_fastapi_success(monkeypatch):
    called = {"args": None, "kwargs": None}

    class _FastAPIInstrumentor:
        @staticmethod
        def instrument_app(*args, **kwargs):
            called["args"] = args
            called["kwargs"] = kwargs

    fastapi_inst = types.ModuleType("opentelemetry.instrumentation.fastapi")
    fastapi_inst.FastAPIInstrumentor = _FastAPIInstrumentor

    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation.fastapi", fastapi_inst)

    app = object()
    tracing.instrument_fastapi(app)
    assert called["args"][0] is app
    assert called["kwargs"]["excluded_urls"] == "health,ready,metrics"


def test_get_tracer_import_success(monkeypatch):
    tracing._tracer = None
    trace_api = types.SimpleNamespace(get_tracer=lambda _name: "TRACE-OBJ")
    root = types.ModuleType("opentelemetry")
    root.trace = trace_api
    monkeypatch.setitem(sys.modules, "opentelemetry", root)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_api)

    out = tracing.get_tracer()
    assert out == "TRACE-OBJ"


def test_trace_span_and_trace_inference_with_attributes(monkeypatch):
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=span)
    tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr(tracing, "get_tracer", lambda: tracer)

    with tracing.trace_span("unit.span", attributes={"a": 1}) as out:
        assert out is span
    span.set_attribute.assert_called_once_with("a", 1)

    with tracing.trace_inference("predict", n_rows=3, model_name="xgb") as out2:
        assert out2 is span
    tracer.start_as_current_span.assert_any_call("inference.predict")


def test_add_event_and_set_attribute_recording_and_not_recording(monkeypatch):
    recording_span = MagicMock()
    recording_span.is_recording.return_value = True

    trace_api = types.SimpleNamespace(get_current_span=lambda: recording_span)
    root = types.ModuleType("opentelemetry")
    root.trace = trace_api
    monkeypatch.setitem(sys.modules, "opentelemetry", root)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_api)

    tracing.add_span_event("model.loaded", {"count": 1})
    tracing.set_span_attribute("ml.rows", 10)
    recording_span.add_event.assert_called_once()
    recording_span.set_attribute.assert_called_once_with("ml.rows", 10)

    non_recording = MagicMock()
    non_recording.is_recording.return_value = False
    trace_api2 = types.SimpleNamespace(get_current_span=lambda: non_recording)
    root2 = types.ModuleType("opentelemetry")
    root2.trace = trace_api2
    monkeypatch.setitem(sys.modules, "opentelemetry", root2)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_api2)
    tracing.add_span_event("ignored")
    tracing.set_span_attribute("ignored", "x")
    non_recording.add_event.assert_not_called()
    non_recording.set_attribute.assert_not_called()


def test_trace_helpers_tolerate_internal_errors(monkeypatch):
    class _BadTrace:
        @staticmethod
        def get_current_span():
            raise RuntimeError("broken")

    root = types.ModuleType("opentelemetry")
    root.trace = _BadTrace
    monkeypatch.setitem(sys.modules, "opentelemetry", root)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", _BadTrace)

    tracing.add_span_event("e")
    tracing.set_span_attribute("k", "v")

    # keep OTEL disabled path stable for this test process
    os.environ.pop("OTEL_ENABLED", None)


def test_init_tracing_runtime_exception_branch(monkeypatch):
    trace_api = types.SimpleNamespace()
    trace_api.set_tracer_provider = MagicMock()
    trace_api.get_tracer = MagicMock(return_value="TRACER")

    class _BrokenOTLPSpanExporter:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("exporter boom")

    sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
    sdk_trace.TracerProvider = _TracerProvider

    sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
    sdk_trace_export.BatchSpanProcessor = _BatchSpanProcessor

    sdk_resources = types.ModuleType("opentelemetry.sdk.resources")
    sdk_resources.Resource = _Resource
    sdk_resources.SERVICE_NAME = "service.name"

    otlp_exporter = types.ModuleType("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
    otlp_exporter.OTLPSpanExporter = _BrokenOTLPSpanExporter

    root = types.ModuleType("opentelemetry")
    root.trace = trace_api

    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setitem(sys.modules, "opentelemetry", root)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_api)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", sdk_trace)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace.export", sdk_trace_export)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.resources", sdk_resources)
    monkeypatch.setitem(
        sys.modules,
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        otlp_exporter,
    )

    tracing.init_tracing(service_name="svc-a")
    assert tracing._initialized is True


def test_instrument_fastapi_import_and_runtime_error_branches(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "true")

    # ImportError branch
    monkeypatch.delitem(sys.modules, "opentelemetry.instrumentation.fastapi", raising=False)
    tracing.instrument_fastapi(object())

    # Runtime exception branch
    class _BadFastAPIInstrumentor:
        @staticmethod
        def instrument_app(*_args, **_kwargs):
            raise RuntimeError("instrument boom")

    bad_mod = types.ModuleType("opentelemetry.instrumentation.fastapi")
    bad_mod.FastAPIInstrumentor = _BadFastAPIInstrumentor
    monkeypatch.setitem(sys.modules, "opentelemetry.instrumentation.fastapi", bad_mod)

    tracing.instrument_fastapi(object())
