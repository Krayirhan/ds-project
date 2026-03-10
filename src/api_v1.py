"""
api_v1.py - V1 API router.

Original API endpoints mounted under /v1 prefix.
Maintains backward compatibility.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.routing import APIRoute

from .api_shared import (
    ServingState,
    DecideResponse,
    ErrorResponse,
    PredictProbaResponse,
    RecordsPayload,
    ReloadResponse,
    exec_decide,
    exec_predict_proba,
    get_shared_app_ref,
    load_serving_state,
    reload_serving_state_for_app,
    require_admin_key,
)
from .metrics import INFERENCE_ERRORS

V1_DEPRECATION_HEADER = "true"
V1_SUNSET_HEADER = "Wed, 31 Dec 2026 23:59:59 GMT"
V1_SUCCESSOR_LINK_HEADER = '</v2>; rel="successor-version"'


class V1DeprecationRoute(APIRoute):
    """Inject RFC-aligned deprecation metadata on all v1 responses."""

    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def route_handler(request: Request):
            response = await original_route_handler(request)
            response.headers["Deprecation"] = V1_DEPRECATION_HEADER
            response.headers["Sunset"] = V1_SUNSET_HEADER
            response.headers["Link"] = V1_SUCCESSOR_LINK_HEADER
            return response

        return route_handler


router_v1 = APIRouter(prefix="/v1", tags=["v1"], route_class=V1DeprecationRoute)

# Backward-compat shim for tests that monkeypatch module-level app refs.
_app_ref = None


def _load_serving_state() -> ServingState:
    return load_serving_state()


def _resolve_app_ref():
    return _app_ref or get_shared_app_ref()


def _get_serving_state() -> ServingState:
    """Retrieve current serving state, loading and caching when needed."""
    app_ref = _resolve_app_ref()
    if app_ref is not None:
        serving = getattr(app_ref.state, "serving", None)
        if serving is not None:
            return serving

    serving = _load_serving_state()
    if app_ref is not None:
        app_ref.state.serving = serving
    return serving


@router_v1.post(
    "/predict_proba",
    response_model=PredictProbaResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def v1_predict_proba(payload: RecordsPayload) -> PredictProbaResponse:
    serving = _get_serving_state()
    try:
        proba, schema_report, _ = exec_predict_proba(
            payload, serving, "v1.predict_proba"
        )
        return PredictProbaResponse(
            n=int(len(proba)),
            proba=proba,
            schema_report=schema_report,
        )
    except ValueError as e:
        INFERENCE_ERRORS.labels(
            endpoint="v1.predict_proba", model=_model_name(serving)
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        INFERENCE_ERRORS.labels(
            endpoint="v1.predict_proba", model=_model_name(serving)
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


@router_v1.post(
    "/decide",
    response_model=DecideResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def v1_decide(payload: RecordsPayload) -> DecideResponse:
    serving = _get_serving_state()
    try:
        actions_df, pred_report, _ = exec_decide(payload, serving, "v1.decide")
        return DecideResponse(
            n=int(len(actions_df)),
            results=actions_df.to_dict(orient="records"),
            report=pred_report,
        )
    except ValueError as e:
        INFERENCE_ERRORS.labels(endpoint="v1.decide", model=_model_name(serving)).inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        INFERENCE_ERRORS.labels(endpoint="v1.decide", model=_model_name(serving)).inc()
        raise HTTPException(status_code=500, detail=str(e))


def _model_name(serving: ServingState | None) -> str:
    """Extract model artifact name from serving state safely."""
    return str(
        getattr(getattr(serving, "policy", None), "selected_model_artifact", "") or ""
    )


@router_v1.post(
    "/reload",
    response_model=ReloadResponse,
    responses={403: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def v1_reload(request: Request) -> ReloadResponse:
    require_admin_key(request)

    app_ref = _resolve_app_ref()
    if app_ref is None:
        raise HTTPException(status_code=500, detail="Application state is unavailable.")

    try:
        serving = await reload_serving_state_for_app(
            app_ref, loader=_load_serving_state
        )
        return {
            "status": "ok",
            "message": "Serving state reloaded",
            "model": serving.policy.selected_model,
            "policy_path": str(serving.policy_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
