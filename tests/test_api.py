import os

from fastapi.testclient import TestClient

from src.api import app


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
        r = client.get("/ready", headers={"x-api-key": "wrong"})
        assert r.status_code == 401
        body = r.json()
        assert body["error_code"] == "unauthorized"
        assert "request_id" in body


def test_reload_endpoint():
    os.environ.setdefault("DS_API_KEY", "test-key")
    with TestClient(app) as client:
        r = client.post("/reload", headers={"x-api-key": "test-key"})
        assert r.status_code in (200, 500)
