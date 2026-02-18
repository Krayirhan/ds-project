"""
api_v1.py — V1 API router.

Original API endpoints mounted under /v1 prefix.
Maintains backward compatibility.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
import pandas as pd

from .api_shared import (
    ServingState,
    DecideResponse,
    ErrorResponse,
    PredictProbaResponse,
    RecordsPayload,
    ReloadResponse,
    load_serving_state,
)
from .config import ExperimentConfig
from .metrics import INFERENCE_ERRORS, INFERENCE_ROWS
from .predict import predict_with_policy, validate_and_prepare_features
from .tracing import trace_inference, set_span_attribute

router_v1 = APIRouter(prefix="/v1", tags=["v1"])

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


@router_v1.post(
    "/predict_proba",
    response_model=PredictProbaResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def v1_predict_proba(payload: RecordsPayload) -> PredictProbaResponse:
    try:
        serving = _get_serving_state()
        max_rows = ExperimentConfig().api.max_payload_records
        if len(payload.records) > max_rows:
            raise ValueError(f"Payload too large. Max records={max_rows}")
        df = pd.DataFrame(payload.records)
        with trace_inference("v1.predict_proba", n_rows=len(df), model_name=str(getattr(serving.policy, 'selected_model', ''))):
            X, schema_report = validate_and_prepare_features(df, serving.feature_spec, fail_on_missing=True)
            proba = serving.model.predict_proba(X)[:, 1]
            set_span_attribute("ml.result_count", int(len(proba)))
        INFERENCE_ROWS.labels(endpoint="v1.predict_proba").inc(len(proba))
        return PredictProbaResponse(
            n=int(len(proba)),
            proba=[float(x) for x in proba],
            schema_report=schema_report,
        )
    except Exception as e:
        INFERENCE_ERRORS.labels(endpoint="v1.predict_proba").inc()
        raise HTTPException(status_code=400, detail=str(e))


@router_v1.post(
    "/decide",
    response_model=DecideResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def v1_decide(payload: RecordsPayload) -> DecideResponse:
    try:
        serving = _get_serving_state()
        max_rows = ExperimentConfig().api.max_payload_records
        if len(payload.records) > max_rows:
            raise ValueError(f"Payload too large. Max records={max_rows}")
        df = pd.DataFrame(payload.records)
        with trace_inference("v1.decide", n_rows=len(df), model_name=str(serving.policy.selected_model_artifact)):
            actions_df, pred_report = predict_with_policy(
                model=serving.model,
                policy=serving.policy,
                df_input=df,
                feature_spec_payload=serving.feature_spec,
                model_used=serving.policy.selected_model_artifact,
            )
            set_span_attribute("ml.result_count", int(len(actions_df)))
        INFERENCE_ROWS.labels(endpoint="v1.decide").inc(len(actions_df))
        return DecideResponse(
            n=int(len(actions_df)),
            results=actions_df.to_dict(orient="records"),
            report=pred_report,
        )
    except Exception as e:
        INFERENCE_ERRORS.labels(endpoint="v1.decide").inc()
        raise HTTPException(status_code=400, detail=str(e))


@router_v1.post("/reload", response_model=ReloadResponse, responses={500: {"model": ErrorResponse}})
def v1_reload() -> ReloadResponse:
    try:
        serving = load_serving_state()
        if _app_ref is not None:
            _app_ref.state.serving = serving
        return {
            "status": "ok",
            "message": "Serving state reloaded",
            "model": serving.policy.selected_model,
            "policy_path": str(serving.policy_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
