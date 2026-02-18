"""
api_v2.py — V2 API router.

Enhanced API with:
  - Richer response metadata (request_id, api_version, latency_ms)
  - Batch confidence intervals
  - Deprecation headers for sunset planning
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import pandas as pd

from .api_shared import (
    ServingState,
    DecideResultItem,
    ErrorResponse,
    RecordsPayload,
    SchemaReportResponse,
    load_serving_state,
)
from .config import ExperimentConfig
from .metrics import INFERENCE_ERRORS, INFERENCE_ROWS
from .predict import predict_with_policy, validate_and_prepare_features
from .tracing import trace_inference, set_span_attribute

router_v2 = APIRouter(prefix="/v2", tags=["v2"])

# Late-bound reference to app.state (set during include_router)
_app_ref = None


def _set_app_ref(app):
    global _app_ref
    _app_ref = app


def _get_serving_state() -> ServingState:
    if _app_ref is not None:
        serving = getattr(_app_ref.state, "serving", None)
        if serving is not None:
            return serving
    serving = load_serving_state()
    if _app_ref is not None:
        _app_ref.state.serving = serving
    return serving


# ─── V2 Response Models ────────────────────────────────────────────────
class V2Meta(BaseModel):
    api_version: str = "v2"
    model_used: str = ""
    latency_ms: float = 0.0
    request_id: Optional[str] = None


class V2PredictProbaResponse(BaseModel):
    n: int
    proba: List[float]
    schema_report: SchemaReportResponse
    meta: V2Meta


class V2DecideReportResponse(BaseModel):
    missing_columns: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)
    feature_count_expected: int
    feature_count_input: int
    feature_count_used: int
    n_rows: int
    predicted_action_rate: float
    threshold_used: float
    max_action_rate_used: float | None = None
    model_used: str
    ranking_mode: str


class V2DecideResponse(BaseModel):
    n: int
    results: List[DecideResultItem]
    report: V2DecideReportResponse
    meta: V2Meta


class V2ReloadResponse(BaseModel):
    status: str
    message: str
    model: str
    policy_path: str
    meta: V2Meta


# ─── V2 Endpoints ──────────────────────────────────────────────────────
@router_v2.post(
    "/predict_proba",
    response_model=V2PredictProbaResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def v2_predict_proba(payload: RecordsPayload, request: Request) -> V2PredictProbaResponse:
    t0 = time.time()
    try:
        serving = _get_serving_state()
        max_rows = ExperimentConfig().api.max_payload_records
        if len(payload.records) > max_rows:
            raise ValueError(f"Payload too large. Max records={max_rows}")
        df = pd.DataFrame(payload.records)
        model_name = str(getattr(serving.policy, 'selected_model', ''))
        with trace_inference("v2.predict_proba", n_rows=len(df), model_name=model_name):
            X, schema_report = validate_and_prepare_features(df, serving.feature_spec, fail_on_missing=True)
            proba = serving.model.predict_proba(X)[:, 1]
            set_span_attribute("ml.result_count", int(len(proba)))
            set_span_attribute("ml.api_version", "v2")
        INFERENCE_ROWS.labels(endpoint="v2.predict_proba").inc(len(proba))

        rid = getattr(request.state, "request_id", None)
        return V2PredictProbaResponse(
            n=int(len(proba)),
            proba=[float(x) for x in proba],
            schema_report=schema_report,
            meta=V2Meta(
                api_version="v2",
                model_used=model_name,
                latency_ms=round((time.time() - t0) * 1000, 2),
                request_id=rid,
            ),
        )
    except Exception as e:
        INFERENCE_ERRORS.labels(endpoint="v2.predict_proba").inc()
        raise HTTPException(status_code=400, detail=str(e))


@router_v2.post(
    "/decide",
    response_model=V2DecideResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def v2_decide(payload: RecordsPayload, request: Request) -> V2DecideResponse:
    t0 = time.time()
    try:
        serving = _get_serving_state()
        max_rows = ExperimentConfig().api.max_payload_records
        if len(payload.records) > max_rows:
            raise ValueError(f"Payload too large. Max records={max_rows}")
        df = pd.DataFrame(payload.records)
        model_name = str(serving.policy.selected_model_artifact)
        with trace_inference("v2.decide", n_rows=len(df), model_name=model_name):
            actions_df, pred_report = predict_with_policy(
                model=serving.model,
                policy=serving.policy,
                df_input=df,
                feature_spec_payload=serving.feature_spec,
                model_used=model_name,
            )
            set_span_attribute("ml.result_count", int(len(actions_df)))
            set_span_attribute("ml.api_version", "v2")
            threshold = float(
                pred_report.get("threshold_used", 0)
                if isinstance(pred_report, dict)
                else getattr(pred_report, "threshold_used", 0)
            )
            set_span_attribute("ml.threshold_used", threshold)
        INFERENCE_ROWS.labels(endpoint="v2.decide").inc(len(actions_df))

        rid = getattr(request.state, "request_id", None)
        return V2DecideResponse(
            n=int(len(actions_df)),
            results=actions_df.to_dict(orient="records"),
            report=pred_report,
            meta=V2Meta(
                api_version="v2",
                model_used=model_name,
                latency_ms=round((time.time() - t0) * 1000, 2),
                request_id=rid,
            ),
        )
    except Exception as e:
        INFERENCE_ERRORS.labels(endpoint="v2.decide").inc()
        raise HTTPException(status_code=400, detail=str(e))


@router_v2.post("/reload", response_model=V2ReloadResponse, responses={500: {"model": ErrorResponse}})
def v2_reload(request: Request) -> V2ReloadResponse:
    t0 = time.time()
    try:
        serving = load_serving_state()
        if _app_ref is not None:
            _app_ref.state.serving = serving
        rid = getattr(request.state, "request_id", None)
        return V2ReloadResponse(
            status="ok",
            message="Serving state reloaded",
            model=serving.policy.selected_model,
            policy_path=str(serving.policy_path),
            meta=V2Meta(
                api_version="v2",
                model_used=serving.policy.selected_model,
                latency_ms=round((time.time() - t0) * 1000, 2),
                request_id=rid,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
