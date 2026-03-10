from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api_shared import RecordsPayload
import src.api_v1 as api_v1
import src.api_v2 as api_v2


def _dummy_serving(model_name: str = "xgb") -> SimpleNamespace:
    policy = SimpleNamespace(
        selected_model=model_name,
        selected_model_artifact="models/xgb.joblib",
    )
    return SimpleNamespace(
        policy=policy, policy_path=Path("reports/decision_policy.json")
    )


def _schema_report() -> dict:
    return {
        "missing_columns": [],
        "extra_columns": [],
        "feature_count_expected": 3,
        "feature_count_input": 3,
        "feature_count_used": 3,
    }


def _decide_report() -> dict:
    return {
        "missing_columns": [],
        "extra_columns": [],
        "feature_count_expected": 3,
        "feature_count_input": 3,
        "feature_count_used": 3,
        "n_rows": 1,
        "predicted_action_rate": 1.0,
        "threshold_used": 0.5,
        "max_action_rate_used": 0.8,
        "model_used": "xgb",
        "ranking_mode": "threshold",
    }


def _dummy_request(
    headers: dict | None = None, request_id: str = "rid-1"
) -> SimpleNamespace:
    return SimpleNamespace(
        headers=headers or {},
        state=SimpleNamespace(request_id=request_id),
    )


def _normalized_json(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True))


def _contract_snapshot_path() -> Path:
    return Path(__file__).parent / "snapshots" / "api_version_contract_snapshot.json"


def test_v1_get_serving_state_uses_app_state(monkeypatch):
    serving = _dummy_serving()
    monkeypatch.setattr(
        api_v1, "_app_ref", SimpleNamespace(state=SimpleNamespace(serving=serving))
    )
    assert api_v1._get_serving_state() is serving


def test_v1_get_serving_state_loads_when_missing(monkeypatch):
    state = SimpleNamespace(serving=None)
    app_ref = SimpleNamespace(state=state)
    monkeypatch.setattr(api_v1, "_app_ref", app_ref)
    loaded = _dummy_serving("loaded-model")
    monkeypatch.setattr(api_v1, "load_serving_state", lambda: loaded)
    assert api_v1._get_serving_state() is loaded
    assert app_ref.state.serving is loaded


def test_v1_predict_proba_error_paths(monkeypatch):
    payload = RecordsPayload(records=[{"lead_time": 10}])
    monkeypatch.setattr(api_v1, "_get_serving_state", lambda: _dummy_serving())

    monkeypatch.setattr(
        api_v1,
        "exec_predict_proba",
        lambda *_: (_ for _ in ()).throw(ValueError("bad")),
    )
    with pytest.raises(HTTPException) as ex1:
        api_v1.v1_predict_proba(payload)
    assert ex1.value.status_code == 400

    monkeypatch.setattr(
        api_v1,
        "exec_predict_proba",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(HTTPException) as ex2:
        api_v1.v1_predict_proba(payload)
    assert ex2.value.status_code == 500


def test_v1_decide_error_paths(monkeypatch):
    payload = RecordsPayload(records=[{"lead_time": 10}])
    monkeypatch.setattr(api_v1, "_get_serving_state", lambda: _dummy_serving())

    monkeypatch.setattr(
        api_v1, "exec_decide", lambda *_: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(HTTPException) as ex1:
        api_v1.v1_decide(payload)
    assert ex1.value.status_code == 400

    monkeypatch.setattr(
        api_v1, "exec_decide", lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(HTTPException) as ex2:
        api_v1.v1_decide(payload)
    assert ex2.value.status_code == 500


def test_v1_reload_forbidden(monkeypatch):
    monkeypatch.setenv("DS_ADMIN_KEY", "admin-secret")
    req = _dummy_request(headers={})
    with pytest.raises(HTTPException) as ex:
        asyncio.run(api_v1.v1_reload(req))
    assert ex.value.status_code == 403


def test_v1_reload_success_and_failure(monkeypatch):
    monkeypatch.setenv("DS_ADMIN_KEY", "admin-secret")
    monkeypatch.setattr(
        api_v1,
        "_app_ref",
        SimpleNamespace(
            state=SimpleNamespace(serving=None, _reload_lock=asyncio.Lock())
        ),
    )
    req = _dummy_request(headers={"x-admin-key": "admin-secret"})

    monkeypatch.setattr(api_v1, "load_serving_state", lambda: _dummy_serving("xgb"))
    out = asyncio.run(api_v1.v1_reload(req))
    assert out["status"] == "ok"
    assert out["model"] == "xgb"

    monkeypatch.setattr(
        api_v1,
        "load_serving_state",
        lambda: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    with pytest.raises(HTTPException) as ex:
        asyncio.run(api_v1.v1_reload(req))
    assert ex.value.status_code == 500


def test_v1_model_name_helper():
    assert api_v1._model_name(None) == ""
    assert api_v1._model_name(_dummy_serving()) == "models/xgb.joblib"


def test_v1_routes_emit_rfc_deprecation_headers(monkeypatch):
    monkeypatch.setattr(api_v1, "_get_serving_state", lambda: _dummy_serving())
    monkeypatch.setattr(
        api_v1, "exec_predict_proba", lambda *_: ([0.42], _schema_report(), "xgb")
    )

    app = FastAPI()
    app.include_router(api_v1.router_v1)
    client = TestClient(app)

    response = client.post("/v1/predict_proba", json={"records": [{"lead_time": 10}]})

    assert response.status_code == 200
    assert response.headers["Deprecation"] == api_v1.V1_DEPRECATION_HEADER
    assert response.headers["Sunset"] == api_v1.V1_SUNSET_HEADER
    assert response.headers["Link"] == api_v1.V1_SUCCESSOR_LINK_HEADER


def test_v2_get_serving_state_paths(monkeypatch):
    serving = _dummy_serving()
    monkeypatch.setattr(
        api_v2, "_app_ref", SimpleNamespace(state=SimpleNamespace(serving=serving))
    )
    assert api_v2._get_serving_state() is serving

    state = SimpleNamespace(serving=None)
    app_ref = SimpleNamespace(state=state)
    monkeypatch.setattr(api_v2, "_app_ref", app_ref)
    loaded = _dummy_serving("v2-model")
    monkeypatch.setattr(api_v2, "load_serving_state", lambda: loaded)
    assert api_v2._get_serving_state() is loaded
    assert app_ref.state.serving is loaded


def test_v2_predict_proba_error_paths(monkeypatch):
    payload = RecordsPayload(records=[{"lead_time": 10}])
    req = _dummy_request()
    monkeypatch.setattr(api_v2, "_get_serving_state", lambda: _dummy_serving())
    monkeypatch.setattr(api_v2, "set_span_attribute", lambda *_: None)

    monkeypatch.setattr(
        api_v2,
        "exec_predict_proba",
        lambda *_: (_ for _ in ()).throw(ValueError("bad")),
    )
    with pytest.raises(HTTPException) as ex1:
        api_v2.v2_predict_proba(payload, req)
    assert ex1.value.status_code == 400

    monkeypatch.setattr(
        api_v2,
        "exec_predict_proba",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(HTTPException) as ex2:
        api_v2.v2_predict_proba(payload, req)
    assert ex2.value.status_code == 500


def test_v2_decide_error_paths(monkeypatch):
    payload = RecordsPayload(records=[{"lead_time": 10}])
    req = _dummy_request()
    monkeypatch.setattr(api_v2, "_get_serving_state", lambda: _dummy_serving())
    monkeypatch.setattr(api_v2, "set_span_attribute", lambda *_: None)

    monkeypatch.setattr(
        api_v2, "exec_decide", lambda *_: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(HTTPException) as ex1:
        api_v2.v2_decide(payload, req)
    assert ex1.value.status_code == 400

    monkeypatch.setattr(
        api_v2, "exec_decide", lambda *_: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(HTTPException) as ex2:
        api_v2.v2_decide(payload, req)
    assert ex2.value.status_code == 500


def test_v2_success_paths(monkeypatch):
    payload = RecordsPayload(records=[{"lead_time": 10}])
    req = _dummy_request(request_id="rid-success")
    serving = _dummy_serving("xgb")
    monkeypatch.setattr(api_v2, "_get_serving_state", lambda: serving)
    monkeypatch.setattr(api_v2, "set_span_attribute", lambda *_: None)

    monkeypatch.setattr(
        api_v2, "exec_predict_proba", lambda *_: ([0.12], _schema_report(), "xgb")
    )
    proba_resp = api_v2.v2_predict_proba(payload, req)
    assert proba_resp.n == 1
    assert proba_resp.meta.api_version == "v2"

    actions_df = pd.DataFrame(
        [
            {
                "proba": 0.88,
                "action": 1,
                "threshold_used": 0.5,
                "max_action_rate_used": 0.8,
                "model_used": "xgb",
            }
        ]
    )
    monkeypatch.setattr(
        api_v2, "exec_decide", lambda *_: (actions_df, _decide_report(), "xgb")
    )
    decide_resp = api_v2.v2_decide(payload, req)
    assert decide_resp.n == 1
    assert decide_resp.meta.request_id == "rid-success"


def test_v2_reload_failure(monkeypatch):
    monkeypatch.setenv("DS_ADMIN_KEY", "admin-secret")
    monkeypatch.setattr(
        api_v2,
        "_app_ref",
        SimpleNamespace(
            state=SimpleNamespace(serving=None, _reload_lock=asyncio.Lock())
        ),
    )
    req = _dummy_request(headers={"x-admin-key": "admin-secret"})

    monkeypatch.setattr(
        api_v2,
        "load_serving_state",
        lambda: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    with pytest.raises(HTTPException) as ex:
        asyncio.run(api_v2.v2_reload(req))
    assert ex.value.status_code == 500


def test_v2_model_name_helper():
    assert api_v2._model_name(None) == ""
    assert api_v2._model_name(_dummy_serving()) == "models/xgb.joblib"


def test_v2_explain_success_fallback_and_404(monkeypatch, tmp_path):
    metrics_root = tmp_path / "reports" / "metrics"
    run_id = "run_x"
    run_dir = metrics_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "permutation_importance.json").write_text(
        json.dumps(
            {
                "method": "permutation_importance",
                "scoring": "roc_auc",
                "n_repeats": 5,
                "n_features": 1,
                "ranking": [
                    {
                        "feature": "lead_time",
                        "importance_mean": 0.2,
                        "importance_std": 0.01,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        api_v2,
        "Paths",
        lambda: SimpleNamespace(reports_metrics=metrics_root),
    )
    req = _dummy_request(request_id="rid-explain")
    out = api_v2.v2_explain(run_id, req)
    assert out.run_id == run_id
    assert out.method == "permutation_importance"
    assert out.meta.request_id == "rid-explain"
    assert out.meta.explain_schema_version == api_v2.EXPLAIN_SCHEMA_VERSION

    (run_dir / "permutation_importance.json").unlink()
    (run_dir / "feature_importance.json").write_text(
        json.dumps({"lead_time": 0.2, "adr": 0.1}),
        encoding="utf-8",
    )
    fallback = api_v2.v2_explain(run_id, req)
    assert fallback.method == "feature_importance"
    assert len(fallback.ranking) == 2
    assert fallback.meta.explain_schema_version == api_v2.EXPLAIN_SCHEMA_VERSION

    (run_dir / "feature_importance.json").unlink()
    with pytest.raises(HTTPException) as ex:
        api_v2.v2_explain(run_id, req)
    assert ex.value.status_code == 404


def _build_api_contract_snapshot(monkeypatch, tmp_path: Path) -> dict:
    payload = RecordsPayload(records=[{"lead_time": 10}])
    request_payload = payload.model_dump()
    request = _dummy_request(
        headers={"x-admin-key": "admin-secret"}, request_id="rid-1"
    )
    serving = _dummy_serving("xgb")

    monkeypatch.setattr(
        api_v1,
        "_app_ref",
        SimpleNamespace(
            state=SimpleNamespace(serving=None, _reload_lock=asyncio.Lock())
        ),
    )
    monkeypatch.setattr(api_v1, "_get_serving_state", lambda: serving)
    monkeypatch.setattr(
        api_v1, "exec_predict_proba", lambda *_: ([0.11], _schema_report(), "xgb")
    )
    monkeypatch.setattr(
        api_v1,
        "exec_decide",
        lambda *_: (
            pd.DataFrame(
                [
                    {
                        "proba": 0.91,
                        "action": 1,
                        "threshold_used": 0.5,
                        "max_action_rate_used": 0.8,
                        "model_used": "xgb",
                    }
                ]
            ),
            _decide_report(),
            "xgb",
        ),
    )
    monkeypatch.setattr(api_v1, "_load_serving_state", lambda: serving)
    monkeypatch.setenv("DS_ADMIN_KEY", "admin-secret")

    monkeypatch.setattr(
        api_v2,
        "_app_ref",
        SimpleNamespace(
            state=SimpleNamespace(serving=None, _reload_lock=asyncio.Lock())
        ),
    )
    monkeypatch.setattr(api_v2, "_get_serving_state", lambda: serving)
    monkeypatch.setattr(api_v2, "set_span_attribute", lambda *_: None)
    monkeypatch.setattr(
        api_v2, "exec_predict_proba", lambda *_: ([0.11], _schema_report(), "xgb")
    )
    monkeypatch.setattr(
        api_v2,
        "exec_decide",
        lambda *_: (
            pd.DataFrame(
                [
                    {
                        "proba": 0.91,
                        "action": 1,
                        "threshold_used": 0.5,
                        "max_action_rate_used": 0.8,
                        "model_used": "xgb",
                    }
                ]
            ),
            _decide_report(),
            "xgb",
        ),
    )
    monkeypatch.setattr(api_v2, "_load_serving_state", lambda: serving)

    metrics_root = tmp_path / "reports" / "metrics"
    run_dir = metrics_root / "run_contract"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "permutation_importance.json").write_text(
        json.dumps(
            {
                "method": "permutation_importance",
                "scoring": "roc_auc",
                "n_repeats": 5,
                "n_features": 1,
                "ranking": [
                    {
                        "feature": "lead_time",
                        "importance_mean": 0.2,
                        "importance_std": 0.01,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api_v2,
        "Paths",
        lambda: SimpleNamespace(reports_metrics=metrics_root),
    )

    v1_predict = api_v1.v1_predict_proba(payload).model_dump()
    v1_decide = api_v1.v1_decide(payload).model_dump()
    v1_reload = asyncio.run(api_v1.v1_reload(request))

    v2_predict = api_v2.v2_predict_proba(payload, request).model_dump()
    v2_decide = api_v2.v2_decide(payload, request).model_dump()
    v2_reload = api_v2.V2ReloadResponse.model_validate(
        asyncio.run(api_v2.v2_reload(request))
    ).model_dump()
    v2_explain = api_v2.v2_explain("run_contract", request).model_dump()

    for response_payload in (v2_predict, v2_decide, v2_reload, v2_explain):
        if isinstance(response_payload, dict) and "meta" in response_payload:
            response_payload["meta"]["latency_ms"] = 0.0

    snapshot = {
        "request_models": {
            "records_payload": RecordsPayload.model_json_schema(),
        },
        "response_models": {
            "v1_predict_proba": api_v1.PredictProbaResponse.model_json_schema(),
            "v1_decide": api_v1.DecideResponse.model_json_schema(),
            "v1_reload": api_v1.ReloadResponse.model_json_schema(),
            "v2_predict_proba": api_v2.V2PredictProbaResponse.model_json_schema(),
            "v2_decide": api_v2.V2DecideResponse.model_json_schema(),
            "v2_reload": api_v2.V2ReloadResponse.model_json_schema(),
            "v2_explain": api_v2.V2ExplainResponse.model_json_schema(),
        },
        "response_examples": {
            "request_payload": request_payload,
            "v1_predict_proba": v1_predict,
            "v1_decide": v1_decide,
            "v1_reload": v1_reload,
            "v2_predict_proba": v2_predict,
            "v2_decide": v2_decide,
            "v2_reload": v2_reload,
            "v2_explain": v2_explain,
        },
        "headers": {
            "v1_deprecation": api_v1.V1_DEPRECATION_HEADER,
            "v1_sunset": api_v1.V1_SUNSET_HEADER,
            "v1_successor_link": api_v1.V1_SUCCESSOR_LINK_HEADER,
        },
        "versions": {
            "v2_explain_schema_version": api_v2.EXPLAIN_SCHEMA_VERSION,
        },
    }
    return _normalized_json(snapshot)


@pytest.mark.contract
def test_api_contract_snapshot_guard(monkeypatch, tmp_path):
    snapshot_path = _contract_snapshot_path()
    actual = _build_api_contract_snapshot(monkeypatch, tmp_path)

    if os.getenv("UPDATE_API_CONTRACT_SNAPSHOT") == "1":
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(actual, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    assert snapshot_path.exists(), (
        f"Missing API contract snapshot: {snapshot_path}. "
        "Run with UPDATE_API_CONTRACT_SNAPSHOT=1 to generate it."
    )
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert actual == expected, (
        "API contract snapshot changed (possible breaking change). "
        "Review intentionally and regenerate with UPDATE_API_CONTRACT_SNAPSHOT=1."
    )
