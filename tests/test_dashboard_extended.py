from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from src.config import Paths
import src.dashboard as dashboard


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class _ConnOk:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def exec_driver_sql(self, _sql: str):
        return 1


class _ConnFail:
    def __enter__(self):
        raise RuntimeError("db down")

    def __exit__(self, exc_type, exc, tb):
        return False


class _EngineOk:
    def connect(self):
        return _ConnOk()


class _EngineFail:
    def connect(self):
        return _ConnFail()


class _StoreListFail:
    def __init__(self):
        self.engine = _EngineOk()

    def list_runs(self, limit: int = 20):
        raise RuntimeError("cannot list runs")


class _StoreUpsertFail:
    def __init__(self):
        self.engine = _EngineOk()

    def upsert_snapshot(self, _snapshot):
        raise RuntimeError("cannot persist snapshot")


@pytest.fixture(autouse=True)
def reset_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard, "_store", None)


@pytest.fixture
def project_paths(tmp_path: Path) -> Paths:
    p = Paths(project_root=tmp_path)
    p.reports_metrics.mkdir(parents=True, exist_ok=True)
    p.reports_monitoring.mkdir(parents=True, exist_ok=True)
    p.models.mkdir(parents=True, exist_ok=True)
    return p


def test_read_json_and_run_dirs(project_paths: Paths) -> None:
    assert dashboard._read_json(project_paths.reports / "missing.json", {"d": 1}) == {"d": 1}

    bad = project_paths.reports / "bad.json"
    bad.write_text("{invalid", encoding="utf-8")
    assert dashboard._read_json(bad, {"x": 2}) == {"x": 2}

    (project_paths.reports_metrics / "run_20260101").mkdir(parents=True, exist_ok=True)
    (project_paths.reports_metrics / "run_20260102").mkdir(parents=True, exist_ok=True)
    dirs = dashboard._run_dirs(project_paths.reports_metrics)
    assert dirs == sorted(dirs, reverse=True)


def test_detect_latest_run_id(project_paths: Paths) -> None:
    _write_json(project_paths.reports_metrics / "latest.json", {"run_id": "run_a"})
    assert dashboard._detect_latest_run_id(project_paths.reports_metrics) == "run_a"

    (project_paths.reports_metrics / "run_20260101").mkdir(parents=True, exist_ok=True)
    (project_paths.reports_metrics / "run_20260103").mkdir(parents=True, exist_ok=True)
    (project_paths.reports_metrics / "latest.json").unlink()
    assert dashboard._detect_latest_run_id(project_paths.reports_metrics) == "run_20260103"

    empty_root = project_paths.project_root / "empty_metrics"
    empty_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(HTTPException):
        dashboard._detect_latest_run_id(empty_root)


def test_load_snapshot_success_and_missing_run(project_paths: Paths) -> None:
    run_id = "run_20260220"
    run_dir = project_paths.reports_metrics / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "cv_summary.json", {"model_a": {"roc_auc_mean": 0.81, "roc_auc_std": 0.02, "cv_folds": 5}})
    _write_json(
        run_dir / "decision_policy.json",
        {
            "selected_model": "model_a",
            "threshold": 0.5,
            "expected_net_profit": 123.4,
            "max_action_rate": 0.3,
            "ranking_mode": "threshold",
        },
    )
    _write_json(
        run_dir / "model_a_metrics.json",
        {
            "roc_auc": 0.79,
            "f1": 0.67,
            "precision": 0.70,
            "recall": 0.65,
            "threshold": 0.5,
            "n_test": 120,
            "positive_rate_test": 0.33,
        },
    )
    _write_json(run_dir / "bad_metrics.json", ["not-a-dict"])
    _write_json(run_dir / "calibration_metrics.json", {"roc_auc": 0.1, "f1": 0.1, "precision": 0.1, "recall": 0.1})
    _write_json(project_paths.reports_metrics / "latest.json", {"run_id": run_id})

    snapshot = dashboard._load_snapshot(paths=project_paths)
    assert snapshot["run_id"] == run_id
    assert snapshot["champion"]["selected_model"] == "model_a"
    assert len(snapshot["models"]) == 1
    assert snapshot["models"][0]["test_roc_auc"] == 0.79

    with pytest.raises(HTTPException):
        dashboard._load_snapshot(paths=project_paths, run_id="missing-run")


def test_mask_database_url() -> None:
    masked = dashboard._mask_database_url("postgresql://user:secret@db.example.com:5432/app")
    assert "secret" not in masked
    assert "***" in masked


def test_init_dashboard_store_and_persist_snapshot_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dashboard, "SQLALCHEMY_AVAILABLE", False)
    dashboard.init_dashboard_store()
    assert dashboard._store is None

    monkeypatch.setattr(dashboard, "SQLALCHEMY_AVAILABLE", True)
    with patch("src.dashboard.DashboardStore", side_effect=RuntimeError("init failed")):
        dashboard.init_dashboard_store()
    assert dashboard._store is None

    monkeypatch.setattr(dashboard, "_store", None)
    dashboard._persist_snapshot({"run_id": "r1"})

    monkeypatch.setattr(dashboard, "_store", _StoreUpsertFail())
    dashboard._persist_snapshot({"run_id": "r2"})


def test_dashboard_runs_and_db_status_error_paths(
    monkeypatch: pytest.MonkeyPatch, project_paths: Paths
) -> None:
    (project_paths.reports_metrics / "run_20260220").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dashboard, "Paths", lambda: project_paths)
    monkeypatch.setattr(dashboard, "_store", _StoreListFail())

    runs_payload = dashboard.dashboard_runs(limit=10, _user={"username": "admin"})
    assert runs_payload["db_enabled"] is True
    assert "run_20260220" in runs_payload["runs"]

    class _StoreBadConnection:
        def __init__(self):
            self.engine = _EngineFail()

    monkeypatch.setattr(dashboard, "_store", _StoreBadConnection())
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@localhost:5432/app")
    db_status = dashboard.dashboard_db_status(_user={"username": "admin"})
    assert db_status["connected"] is False
    assert db_status["database_backend"] == "postgresql"
    assert "***" in db_status["database_url"]


def test_dashboard_monitoring_paths(monkeypatch: pytest.MonkeyPatch, project_paths: Paths) -> None:
    monkeypatch.setattr(dashboard, "Paths", lambda: project_paths)

    latest_payload = {"alerts": {"any_alert": False}, "source": "latest"}
    _write_json(project_paths.reports_monitoring / "latest_monitoring_report.json", latest_payload)
    got_latest = dashboard.dashboard_monitoring(_user={"username": "admin"})
    assert got_latest["source"] == "latest"

    (project_paths.reports_monitoring / "latest_monitoring_report.json").unlink()
    _write_json(project_paths.reports_monitoring / "2026-02-20" / "monitoring_report.json", {"source": "dated"})
    got_dated = dashboard.dashboard_monitoring(_user={"username": "admin"})
    assert got_dated["source"] == "dated"

    for path in project_paths.reports_monitoring.rglob("*"):
        if path.is_file():
            path.unlink()
    with pytest.raises(HTTPException):
        dashboard.dashboard_monitoring(_user={"username": "admin"})


def test_dashboard_explain_paths(monkeypatch: pytest.MonkeyPatch, project_paths: Paths) -> None:
    monkeypatch.setattr(dashboard, "Paths", lambda: project_paths)
    run_id = "run_x"
    run_dir = project_paths.reports_metrics / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(project_paths.reports_metrics / "latest.json", {"run_id": run_id})

    _write_json(
        run_dir / "permutation_importance.json",
        {
            "method": "permutation",
            "scoring": "roc_auc",
            "n_repeats": 5,
            "n_features": 2,
            "ranking": [{"feature": "lead_time", "importance_mean": 0.12, "importance_std": 0.01}],
        },
    )
    expl = dashboard.dashboard_explain(run_id=run_id, _user={"username": "admin"})
    assert expl["method"] == "permutation"

    (run_dir / "permutation_importance.json").unlink()
    _write_json(run_dir / "feature_importance.json", {"lead_time": 0.2, "adr": 0.1})
    fallback = dashboard.dashboard_explain(run_id=run_id, _user={"username": "admin"})
    assert fallback["method"] == "feature_importance"
    assert len(fallback["ranking"]) == 2

    (run_dir / "feature_importance.json").unlink()
    with pytest.raises(HTTPException):
        dashboard.dashboard_explain(run_id=run_id, _user={"username": "admin"})


def test_dashboard_system_degraded(monkeypatch: pytest.MonkeyPatch, project_paths: Paths) -> None:
    monkeypatch.setattr(dashboard, "Paths", lambda: project_paths)
    monkeypatch.setattr(dashboard, "_store", None)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")

    artifact = project_paths.project_root / "models" / "artifact.joblib"
    artifact.write_text("dummy", encoding="utf-8")
    registry_path = project_paths.project_root / "models" / "registry.json"
    _write_json(registry_path, {"slot1": {"path": str(artifact), "model_name": "xgb"}})
    _write_json(project_paths.models / "latest.json", {"run_id": "run1", "model_registry": str(registry_path)})

    with patch("urllib.request.urlopen", side_effect=RuntimeError("ollama down")):
        system = dashboard.dashboard_system(_user={"username": "admin"})

    assert system["overall"] in {"degraded", "partial"}
    assert "database" in system["services"]
    assert "redis" in system["services"]
    assert "ollama" in system["services"]
    assert "model" in system["services"]


def test_dashboard_system_all_ok(monkeypatch: pytest.MonkeyPatch, project_paths: Paths) -> None:
    monkeypatch.setattr(dashboard, "Paths", lambda: project_paths)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./reports/dashboard.db")
    monkeypatch.setenv("REDIS_URL", "redis://user:pass@localhost:6379/0")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")

    class _StoreOk:
        def __init__(self):
            self.engine = _EngineOk()

    monkeypatch.setattr(dashboard, "_store", _StoreOk())

    class _RedisClient:
        def ping(self):
            return True

    class _RedisFactory:
        @staticmethod
        def from_url(*args, **kwargs):
            return _RedisClient()

    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=_RedisFactory))

    class _Resp:
        def read(self):
            return json.dumps({"models": [{"name": "llama3.2:3b"}]}).encode("utf-8")

    artifact = project_paths.project_root / "models" / "artifact.joblib"
    artifact.write_text("dummy", encoding="utf-8")
    registry_path = project_paths.project_root / "models" / "registry.json"
    _write_json(registry_path, {"slot1": {"path": str(artifact), "model_name": "xgb"}})
    _write_json(project_paths.models / "latest.json", {"run_id": "run1", "model_registry": str(registry_path)})

    with patch("urllib.request.urlopen", return_value=_Resp()):
        system = dashboard.dashboard_system(_user={"username": "admin"})

    assert system["overall"] == "ok"
