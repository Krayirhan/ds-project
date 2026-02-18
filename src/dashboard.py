from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from .config import Paths
from .dashboard_auth import require_dashboard_user
from .utils import get_logger

logger = get_logger("dashboard")

try:
    from sqlalchemy import (
        Column,
        DateTime,
        Float,
        Integer,
        MetaData,
        String,
        Table,
        create_engine,
        delete,
        insert,
        select,
    )

    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False


router_dashboard = APIRouter(prefix="/dashboard/api", tags=["dashboard"])

_store = None


class DashboardStore:
    def __init__(self, database_url: str):
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError("SQLAlchemy is not available")

        self.database_url = database_url
        self.engine = create_engine(database_url, pool_pre_ping=True, future=True)
        self.metadata = MetaData()

        self.runs = Table(
            "experiment_runs",
            self.metadata,
            Column("run_id", String(64), primary_key=True),
            Column("selected_model", String(256), nullable=True),
            Column("threshold", Float, nullable=True),
            Column("expected_net_profit", Float, nullable=True),
            Column("max_action_rate", Float, nullable=True),
            Column("source_path", String(1024), nullable=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )

        self.model_metrics = Table(
            "model_metrics",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String(64), nullable=False),
            Column("model_name", String(256), nullable=False),
            Column("train_cv_roc_auc_mean", Float, nullable=True),
            Column("train_cv_roc_auc_std", Float, nullable=True),
            Column("test_roc_auc", Float, nullable=True),
            Column("test_f1", Float, nullable=True),
            Column("test_precision", Float, nullable=True),
            Column("test_recall", Float, nullable=True),
            Column("test_threshold", Float, nullable=True),
            Column("n_test", Integer, nullable=True),
            Column("positive_rate_test", Float, nullable=True),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )

    def create_schema(self) -> None:
        self.metadata.create_all(self.engine)

    def upsert_snapshot(self, snapshot: Dict[str, Any]) -> None:
        run_id = snapshot["run_id"]
        champion = snapshot.get("champion") or {}
        now = datetime.now(timezone.utc)

        with self.engine.begin() as conn:
            conn.execute(delete(self.model_metrics).where(self.model_metrics.c.run_id == run_id))
            conn.execute(delete(self.runs).where(self.runs.c.run_id == run_id))

            conn.execute(
                insert(self.runs).values(
                    run_id=run_id,
                    selected_model=champion.get("selected_model"),
                    threshold=champion.get("threshold"),
                    expected_net_profit=champion.get("expected_net_profit"),
                    max_action_rate=champion.get("max_action_rate"),
                    source_path=snapshot.get("source_path"),
                    updated_at=now,
                )
            )

            for row in snapshot.get("models", []):
                conn.execute(
                    insert(self.model_metrics).values(
                        run_id=run_id,
                        model_name=row.get("model_name"),
                        train_cv_roc_auc_mean=row.get("train_cv_roc_auc_mean"),
                        train_cv_roc_auc_std=row.get("train_cv_roc_auc_std"),
                        test_roc_auc=row.get("test_roc_auc"),
                        test_f1=row.get("test_f1"),
                        test_precision=row.get("test_precision"),
                        test_recall=row.get("test_recall"),
                        test_threshold=row.get("test_threshold"),
                        n_test=row.get("n_test"),
                        positive_rate_test=row.get("positive_rate_test"),
                        updated_at=now,
                    )
                )

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.engine.begin() as conn:
            stmt = (
                select(self.runs)
                .order_by(self.runs.c.updated_at.desc())
                .limit(limit)
            )
            rows = conn.execute(stmt).mappings().all()

        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "run_id": row["run_id"],
                    "selected_model": row["selected_model"],
                    "threshold": row["threshold"],
                    "expected_net_profit": row["expected_net_profit"],
                    "max_action_rate": row["max_action_rate"],
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                }
            )
        return out


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read JSON file %s: %s", path, exc)
        return default


def _run_dirs(metrics_root: Path) -> List[str]:
    if not metrics_root.exists():
        return []
    dirs = [p.name for p in metrics_root.iterdir() if p.is_dir()]
    dirs.sort(reverse=True)
    return dirs


def _detect_latest_run_id(metrics_root: Path) -> str:
    latest_json = _read_json(metrics_root / "latest.json", {})
    latest_run = latest_json.get("run_id") if isinstance(latest_json, dict) else None
    if latest_run:
        return str(latest_run)

    dirs = _run_dirs(metrics_root)
    if not dirs:
        raise HTTPException(status_code=404, detail="No experiment run found under reports/metrics")
    return dirs[0]


def _load_snapshot(paths: Paths, run_id: str | None = None) -> Dict[str, Any]:
    metrics_root = paths.reports_metrics
    selected_run = run_id or _detect_latest_run_id(metrics_root)
    run_dir = metrics_root / selected_run
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {selected_run}")

    cv_summary = _read_json(run_dir / "cv_summary.json", {})
    decision_policy = _read_json(run_dir / "decision_policy.json", {})

    test_metrics: Dict[str, Dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*_metrics.json")):
        if path.name == "calibration_metrics.json":
            continue
        payload = _read_json(path, {})
        if not isinstance(payload, dict):
            continue
        required = {"roc_auc", "f1", "precision", "recall"}
        if required.issubset(payload.keys()):
            model_name = path.name.replace("_metrics.json", "")
            test_metrics[model_name] = payload

    model_names = sorted(set(cv_summary.keys()) | set(test_metrics.keys()))
    models: List[Dict[str, Any]] = []

    for name in model_names:
        cv = cv_summary.get(name, {}) if isinstance(cv_summary, dict) else {}
        tm = test_metrics.get(name, {})
        models.append(
            {
                "model_name": name,
                "train_cv_roc_auc_mean": cv.get("roc_auc_mean"),
                "train_cv_roc_auc_std": cv.get("roc_auc_std"),
                "cv_folds": cv.get("cv_folds"),
                "test_roc_auc": tm.get("roc_auc"),
                "test_f1": tm.get("f1"),
                "test_precision": tm.get("precision"),
                "test_recall": tm.get("recall"),
                "test_threshold": tm.get("threshold"),
                "n_test": tm.get("n_test"),
                "positive_rate_test": tm.get("positive_rate_test"),
            }
        )

    models.sort(key=lambda x: (x.get("test_roc_auc") is None, -(x.get("test_roc_auc") or 0.0)))

    return {
        "run_id": selected_run,
        "available_runs": _run_dirs(metrics_root),
        "source_path": str(run_dir),
        "champion": {
            "selected_model": decision_policy.get("selected_model"),
            "threshold": decision_policy.get("threshold"),
            "expected_net_profit": decision_policy.get("expected_net_profit"),
            "max_action_rate": decision_policy.get("max_action_rate"),
            "ranking_mode": decision_policy.get("ranking_mode"),
        },
        "models": models,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def init_dashboard_store() -> None:
    global _store

    if not SQLALCHEMY_AVAILABLE:
        logger.warning("SQLAlchemy is not installed; dashboard persistence disabled")
        _store = None
        return

    database_url = os.getenv("DATABASE_URL", "sqlite:///./reports/dashboard.db")

    try:
        _store = DashboardStore(database_url=database_url)
        _store.create_schema()
        logger.info("Dashboard database initialized: %s", database_url)
    except Exception as exc:
        logger.warning("Could not initialize dashboard database (%s): %s", database_url, exc)
        _store = None


def _persist_snapshot(snapshot: Dict[str, Any]) -> None:
    if _store is None:
        return
    try:
        _store.upsert_snapshot(snapshot)
    except Exception as exc:
        logger.warning("Dashboard snapshot could not be persisted: %s", exc)


@router_dashboard.get("/overview")
def dashboard_overview(
    run_id: str | None = Query(default=None),
    _user: Dict[str, Any] = Depends(require_dashboard_user),
):
    paths = Paths()
    snapshot = _load_snapshot(paths=paths, run_id=run_id)
    _persist_snapshot(snapshot)
    snapshot["db_enabled"] = _store is not None
    return snapshot


@router_dashboard.get("/runs")
def dashboard_runs(
    limit: int = Query(default=20, ge=1, le=100),
    _user: Dict[str, Any] = Depends(require_dashboard_user),
):
    paths = Paths()
    filesystem_runs = _run_dirs(paths.reports_metrics)

    db_runs: List[Dict[str, Any]] = []
    if _store is not None:
        try:
            db_runs = _store.list_runs(limit=limit)
        except Exception as exc:
            logger.warning("Dashboard runs could not be read from database: %s", exc)

    return {
        "runs": filesystem_runs[:limit],
        "db_runs": db_runs,
        "db_enabled": _store is not None,
    }


def _mask_database_url(url: str) -> str:
    return re.sub(r"://([^:/?#]+):([^@]+)@", r"://\1:***@", url)


@router_dashboard.get("/db-status")
def dashboard_db_status(_user: Dict[str, Any] = Depends(require_dashboard_user)):
    database_url = os.getenv("DATABASE_URL", "sqlite:///./reports/dashboard.db")
    masked_url = _mask_database_url(database_url)
    backend = "postgresql" if database_url.startswith("postgresql") else "sqlite"

    connected = False
    reason = "Dashboard store not initialized"
    if _store is not None:
        try:
            with _store.engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            connected = True
            reason = "ok"
        except Exception as exc:
            reason = str(exc)

    return {
        "db_enabled": _store is not None,
        "database_backend": backend,
        "database_url": masked_url,
        "connected": connected,
        "reason": reason,
    }
