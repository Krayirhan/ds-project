"""Tests for src.dashboard_store — DashboardStore persistence layer."""

from __future__ import annotations

import pytest

from src.dashboard_store import DashboardStore


@pytest.fixture()
def store(tmp_path) -> DashboardStore:
    """Create a fresh DashboardStore with an isolated SQLite DB."""
    db_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    s = DashboardStore(database_url=db_url)
    s.create_schema()
    return s


def _sample_snapshot(run_id: str = "run-001") -> dict:
    return {
        "run_id": run_id,
        "champion": {
            "selected_model": "xgboost",
            "threshold": 0.42,
            "expected_net_profit": 12000.0,
            "max_action_rate": 0.15,
        },
        "source_path": "models/20260301_123456",
        "models": [
            {
                "model_name": "xgboost",
                "train_cv_roc_auc_mean": 0.88,
                "train_cv_roc_auc_std": 0.02,
                "test_roc_auc": 0.87,
                "test_f1": 0.79,
                "test_precision": 0.81,
                "test_recall": 0.77,
                "test_threshold": 0.42,
                "n_test": 5000,
                "positive_rate_test": 0.37,
            },
            {
                "model_name": "lightgbm",
                "train_cv_roc_auc_mean": 0.86,
                "train_cv_roc_auc_std": 0.03,
                "test_roc_auc": 0.85,
                "test_f1": 0.76,
                "test_precision": 0.78,
                "test_recall": 0.74,
                "test_threshold": 0.40,
                "n_test": 5000,
                "positive_rate_test": 0.37,
            },
        ],
    }


class TestUpsertSnapshot:
    """DashboardStore.upsert_snapshot writes and replaces data correctly."""

    def test_insert_and_list(self, store: DashboardStore) -> None:
        store.upsert_snapshot(_sample_snapshot("run-001"))
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-001"
        assert runs[0]["selected_model"] == "xgboost"

    def test_upsert_replaces(self, store: DashboardStore) -> None:
        store.upsert_snapshot(_sample_snapshot("run-001"))
        updated = _sample_snapshot("run-001")
        updated["champion"]["selected_model"] = "lightgbm"
        store.upsert_snapshot(updated)

        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0]["selected_model"] == "lightgbm"

    def test_multiple_runs(self, store: DashboardStore) -> None:
        store.upsert_snapshot(_sample_snapshot("run-001"))
        store.upsert_snapshot(_sample_snapshot("run-002"))
        store.upsert_snapshot(_sample_snapshot("run-003"))

        runs = store.list_runs(limit=10)
        assert len(runs) == 3

    def test_list_respects_limit(self, store: DashboardStore) -> None:
        for i in range(5):
            store.upsert_snapshot(_sample_snapshot(f"run-{i:03d}"))
        runs = store.list_runs(limit=3)
        assert len(runs) == 3


class TestListRuns:
    """DashboardStore.list_runs returns correct ordering and structure."""

    def test_empty_store(self, store: DashboardStore) -> None:
        assert store.list_runs() == []

    def test_returns_expected_keys(self, store: DashboardStore) -> None:
        store.upsert_snapshot(_sample_snapshot())
        run = store.list_runs()[0]
        expected_keys = {
            "run_id",
            "selected_model",
            "threshold",
            "expected_net_profit",
            "max_action_rate",
            "updated_at",
        }
        assert set(run.keys()) == expected_keys

    def test_ordered_by_most_recent(self, store: DashboardStore) -> None:
        store.upsert_snapshot(_sample_snapshot("run-old"))
        store.upsert_snapshot(_sample_snapshot("run-new"))
        runs = store.list_runs()
        # Most recent first
        assert runs[0]["run_id"] == "run-new"


class TestCreateSchema:
    """DashboardStore.create_schema creates tables idempotently."""

    def test_idempotent(self, store: DashboardStore) -> None:
        store.create_schema()  # second call should not raise
        store.upsert_snapshot(_sample_snapshot())
        assert len(store.list_runs()) == 1
