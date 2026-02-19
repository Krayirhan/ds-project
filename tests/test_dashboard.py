import os

import pytest
from fastapi.testclient import TestClient

from src.api import app

# Test password — set as plaintext env var before any dashboard_auth import
_TEST_PASSWORD = "TestDashboard123!"


@pytest.fixture(autouse=True)
def _set_dashboard_env(monkeypatch):
    """Ensure test credentials are available for every test in this module."""
    monkeypatch.setenv("DS_API_KEY", "test-key")
    monkeypatch.setenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", _TEST_PASSWORD)


def _login_headers(client: TestClient) -> dict:
    payload = {"username": "admin", "password": _TEST_PASSWORD}
    r = client.post("/auth/login", json=payload)
    assert r.status_code == 200, f"Login failed: {r.text}"
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_dashboard_api_requires_login():
    os.environ.setdefault("DS_API_KEY", "test-key")
    with TestClient(app) as client:
        runs = client.get("/dashboard/api/runs")
        assert runs.status_code == 401


def test_dashboard_overview_returns_train_and_test_metrics():
    os.environ.setdefault("DS_API_KEY", "test-key")
    with TestClient(app) as client:
        headers = _login_headers(client)
        response = client.get("/dashboard/api/overview", headers=headers)
        assert response.status_code == 200
        body = response.json()
        assert "run_id" in body
        assert "models" in body
        if body["models"]:
            first = body["models"][0]
            assert "train_cv_roc_auc_mean" in first
            assert "test_roc_auc" in first


def test_dashboard_runs_returns_available_runs():
    os.environ.setdefault("DS_API_KEY", "test-key")
    with TestClient(app) as client:
        headers = _login_headers(client)
        runs = client.get("/dashboard/api/runs", headers=headers)
        assert runs.status_code == 200
        payload = runs.json()
        assert "runs" in payload


def test_dashboard_db_status_returns_connection_info():
    os.environ.setdefault("DS_API_KEY", "test-key")
    with TestClient(app) as client:
        headers = _login_headers(client)
        r = client.get("/dashboard/api/db-status", headers=headers)
        assert r.status_code == 200
        body = r.json()
        assert "database_backend" in body
        assert "connected" in body
