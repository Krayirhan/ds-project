from __future__ import annotations

from contextlib import asynccontextmanager
import os
import time
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response

from .api_shared import (
    ServingState,
    RecordsPayload,
    HealthResponse,
    ReadyResponse,
    PredictProbaResponse,
    DecideResponse,
    ReloadResponse,
    ErrorResponse,
    load_serving_state,
    error_response,
)
from .config import ExperimentConfig
from .metrics import INFERENCE_ERRORS, INFERENCE_ROWS, REQUEST_COUNT, REQUEST_LATENCY, render_metrics
from .predict import predict_with_policy, validate_and_prepare_features
from .rate_limit import BaseRateLimiter, build_rate_limiter
from .tracing import init_tracing, instrument_fastapi, trace_inference, set_span_attribute
from .utils import get_logger

logger = get_logger("api")


def _api_key_required() -> bool:
    return ExperimentConfig().api.require_api_key


def _expected_api_key() -> str | None:
    cfg = ExperimentConfig().api
    return os.getenv(cfg.api_key_env_var)


def _build_runtime_rate_limiter() -> BaseRateLimiter:
    cfg = ExperimentConfig().api
    env_backend = os.getenv("RATE_LIMIT_BACKEND")
    env_redis_url = os.getenv("REDIS_URL")
    env_key_prefix = os.getenv("RATE_LIMIT_REDIS_KEY_PREFIX")
    return build_rate_limiter(
        backend=env_backend or cfg.rate_limit_backend,
        redis_url=env_redis_url or cfg.redis_url,
        key_prefix=env_key_prefix or cfg.redis_key_prefix,
    )


def _load_serving_state() -> ServingState:
    return load_serving_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_tracing(service_name="ds-project-api")
    instrument_fastapi(app)
    app.state.serving = _load_serving_state()
    app.state.rate_limiter = _build_runtime_rate_limiter()
    app.state.shutting_down = False
    yield
    app.state.shutting_down = True


app = FastAPI(
    title="DS Project Serving API",
    version="1.1.0",
    description=(
        "Production inference API for hotel cancellation decisioning. "
        "Supports health checks, readiness checks, probability scoring, policy-based actioning, "
        "runtime metrics, and safe model/policy reload. "
        "Versioned endpoints available under /v1 and /v2 prefixes."
    ),
    lifespan=lifespan,
)

# ── Register versioned routers ────────────────────────────────────────
from .api_v1 import router_v1, _set_app_ref as _v1_set_app  # noqa: E402
from .api_v2 import router_v2, _set_app_ref as _v2_set_app  # noqa: E402

app.include_router(router_v1)
app.include_router(router_v2)
_v1_set_app(app)
_v2_set_app(app)


def _error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    request_id: str | None,
):
    return error_response(
        status_code=status_code,
        error_code=error_code,
        message=message,
        request_id=request_id,
    )


def _get_serving_state() -> ServingState:
    serving = getattr(app.state, "serving", None)
    if serving is None:
        serving = _load_serving_state()
        app.state.serving = serving
    return serving


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = rid
    started = time.time()

    # Security gate
    if _api_key_required():
        expected = _expected_api_key()
        if not expected:
            return _error_response(
                status_code=503,
                error_code="api_key_not_configured",
                message="API key is not configured",
                request_id=rid,
            )
        got = request.headers.get("x-api-key")
        if got != expected:
            return _error_response(
                status_code=401,
                error_code="unauthorized",
                message="Unauthorized",
                request_id=rid,
            )

    # Rate limit gate
    client = request.client.host if request.client else "unknown"
    limiter = getattr(app.state, "rate_limiter", None)
    if limiter is None:
        limiter = _build_runtime_rate_limiter()
        app.state.rate_limiter = limiter
    if not limiter.allow(client, ExperimentConfig().api.rate_limit_per_minute):
        return _error_response(
            status_code=429,
            error_code="rate_limit_exceeded",
            message="Rate limit exceeded",
            request_id=rid,
        )

    # Request size guard (8MB)
    max_bytes = 8 * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                return _error_response(
                    status_code=413,
                    error_code="payload_too_large",
                    message="Payload too large",
                    request_id=rid,
                )
        except ValueError:
            return _error_response(
                status_code=400,
                error_code="invalid_content_length",
                message="Invalid content-length",
                request_id=rid,
            )

    if bool(getattr(app.state, "shutting_down", False)):
        return _error_response(
            status_code=503,
            error_code="service_shutting_down",
            message="Service is shutting down",
            request_id=rid,
        )

    response = await call_next(request)
    response.headers["x-request-id"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"

    latency_ms = round((time.time() - started) * 1000.0, 2)
    logger.info(
        f"request path={request.url.path} status={response.status_code} latency_ms={latency_ms}",
        extra={"request_id": rid},
    )
    REQUEST_COUNT.labels(path=request.url.path, method=request.method, status=str(response.status_code)).inc()
    REQUEST_LATENCY.labels(path=request.url.path, method=request.method).observe((time.time() - started))
    return response


@app.get("/health")
def health() -> HealthResponse:
    return {
        "status": "ok",
        "service": "alive",
    }


@app.get("/ready")
def ready() -> ReadyResponse:
    serving = _get_serving_state()
    return {
        "status": "ok",
        "service": "ready",
        "model": serving.policy.selected_model,
        "policy_path": str(serving.policy_path),
    }


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


@app.post(
    "/predict_proba",
    response_model=PredictProbaResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def predict_proba(payload: RecordsPayload) -> PredictProbaResponse:
    try:
        serving = _get_serving_state()
        max_rows = ExperimentConfig().api.max_payload_records
        if len(payload.records) > max_rows:
            raise ValueError(f"Payload too large. Max records={max_rows}")
        df = pd.DataFrame(payload.records)
        with trace_inference("predict_proba", n_rows=len(df), model_name=str(getattr(serving.policy, 'selected_model', ''))):
            X, schema_report = validate_and_prepare_features(df, serving.feature_spec, fail_on_missing=True)
            proba = serving.model.predict_proba(X)[:, 1]
            set_span_attribute("ml.result_count", int(len(proba)))
        INFERENCE_ROWS.labels(endpoint="predict_proba").inc(len(proba))
        return PredictProbaResponse(
            n=int(len(proba)),
            proba=[float(x) for x in proba],
            schema_report=schema_report,
        )
    except Exception as e:
        INFERENCE_ERRORS.labels(endpoint="predict_proba").inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/decide",
    response_model=DecideResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def decide(payload: RecordsPayload) -> DecideResponse:
    try:
        serving = _get_serving_state()
        max_rows = ExperimentConfig().api.max_payload_records
        if len(payload.records) > max_rows:
            raise ValueError(f"Payload too large. Max records={max_rows}")
        df = pd.DataFrame(payload.records)
        with trace_inference("decide", n_rows=len(df), model_name=str(serving.policy.selected_model_artifact)):
            actions_df, pred_report = predict_with_policy(
                model=serving.model,
                policy=serving.policy,
                df_input=df,
                feature_spec_payload=serving.feature_spec,
                model_used=serving.policy.selected_model_artifact,
            )
            set_span_attribute("ml.result_count", int(len(actions_df)))
            set_span_attribute("ml.threshold_used", float(pred_report.get("threshold_used", 0) if isinstance(pred_report, dict) else getattr(pred_report, 'threshold_used', 0)))
        INFERENCE_ROWS.labels(endpoint="decide").inc(len(actions_df))
        return DecideResponse(
            n=int(len(actions_df)),
            results=actions_df.to_dict(orient="records"),
            report=pred_report,
        )
    except Exception as e:
        INFERENCE_ERRORS.labels(endpoint="decide").inc()
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/reload", response_model=ReloadResponse, responses={500: {"model": ErrorResponse}})
def reload_serving_state() -> ReloadResponse:
    try:
        serving = _load_serving_state()
        app.state.serving = serving
        return {
            "status": "ok",
            "message": "Serving state reloaded",
            "model": serving.policy.selected_model,
            "policy_path": str(serving.policy_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    rid = getattr(request.state, "request_id", None)
    detail = str(exc.detail)
    return _error_response(
        status_code=exc.status_code,
        error_code="http_error",
        message=detail,
        request_id=rid,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", None)
    logger.exception(f"Unhandled exception: {exc}", extra={"request_id": rid})
    return _error_response(
        status_code=500,
        error_code="internal_error",
        message="Internal server error",
        request_id=rid,
    )
