import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
import pandas as pd

from src.api import app


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


def _dummy_serving():
    policy = SimpleNamespace(
        selected_model="xgb",
        selected_model_artifact="models/xgb.joblib",
    )
    return SimpleNamespace(
        policy=policy,
        policy_path=Path("reports/decision_policy.json"),
    )


def test_health_ready_endpoints():
    os.environ.setdefault("DS_API_KEY", "test-key")
    with TestClient(app) as client:
        r1 = client.get("/health", headers={"x-api-key": "test-key"})
        assert r1.status_code == 200
        r2 = client.get("/ready", headers={"x-api-key": "test-key"})
        assert r2.status_code in (200, 503)
        r3 = client.get("/metrics", headers={"x-api-key": "test-key"})
        assert r3.status_code == 200
        assert "ds_api_requests_total" in r3.text


def test_decide_requires_api_key():
    os.environ.setdefault("DS_API_KEY", "test-key")
    with TestClient(app) as client:
        payload = {"records": [{"lead_time": 10}]}
        r = client.post("/decide", json=payload)
        assert r.status_code in (401, 422, 400)
        if r.status_code == 401:
            body = r.json()
            assert "error_code" in body
            assert body["error_code"] == "unauthorized"


def test_wrong_api_key_is_rejected():
    os.environ.setdefault("DS_API_KEY", "test-key")
    with TestClient(app) as client:
        # /ready operasyonel endpoint olduğu için API key muafiyeti vardır.
        # Korumalı bir endpoint test ediyoruz.
        r = client.post(
            "/predict_proba",
            json={"records": []},
            headers={"x-api-key": "wrong"},
        )
        assert r.status_code == 401
        body = r.json()
        assert body["error_code"] == "unauthorized"
        assert "request_id" in body


def test_reload_endpoint():
    os.environ.setdefault("DS_API_KEY", "test-key")
    with TestClient(app) as client:
        r = client.post("/reload", headers={"x-api-key": "test-key"})
        assert r.status_code in (200, 500)


def test_v1_predict_proba_success():
    os.environ["DS_API_KEY"] = "test-key"
    with (
        patch("src.api_v1._get_serving_state", return_value=_dummy_serving()),
        patch(
            "src.api_v1.exec_predict_proba",
            return_value=([0.21], _schema_report(), "xgb"),
        ),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/v1/predict_proba",
                json={"records": [{"lead_time": 10}]},
                headers={"x-api-key": "test-key"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 1
    assert body["proba"] == [0.21]
    assert body["schema_report"]["feature_count_used"] == 3


def test_v2_decide_success_with_meta():
    os.environ["DS_API_KEY"] = "test-key"
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
    with (
        patch("src.api_v2._get_serving_state", return_value=_dummy_serving()),
        patch(
            "src.api_v2.exec_decide",
            return_value=(actions_df, _decide_report(), "xgb"),
        ),
        patch("src.api_v2.set_span_attribute"),
    ):
        with TestClient(app) as client:
            r = client.post(
                "/v2/decide",
                json={"records": [{"lead_time": 10}]},
                headers={"x-api-key": "test-key"},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 1
    assert body["meta"]["api_version"] == "v2"
    assert body["meta"]["model_used"] == "xgb"
    assert body["report"]["ranking_mode"] == "threshold"


def test_v2_reload_requires_admin_key():
    os.environ["DS_API_KEY"] = "test-key"
    os.environ["DS_ADMIN_KEY"] = "admin-secret"
    with TestClient(app) as client:
        r = client.post("/v2/reload", headers={"x-api-key": "test-key"})
    assert r.status_code == 403


def test_v2_reload_success():
    os.environ["DS_API_KEY"] = "test-key"
    os.environ["DS_ADMIN_KEY"] = "admin-secret"
    with patch("src.api_v2.load_serving_state", return_value=_dummy_serving()):
        with TestClient(app) as client:
            r = client.post(
                "/v2/reload",
                headers={
                    "x-api-key": "test-key",
                    "x-admin-key": "admin-secret",
                },
            )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "xgb"
    assert body["meta"]["api_version"] == "v2"
