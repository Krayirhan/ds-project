"""Tests for src/cli/evaluate.py — cmd_evaluate."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import ExperimentConfig, Paths


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def cfg() -> ExperimentConfig:
    return ExperimentConfig()


@pytest.fixture()
def paths(tmp_path: Path) -> Paths:
    p = Paths(project_root=tmp_path)
    p.data_processed.mkdir(parents=True, exist_ok=True)
    p.models.mkdir(parents=True, exist_ok=True)
    p.reports_metrics.mkdir(parents=True, exist_ok=True)
    return p


def _make_test_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "lead_time": rng.integers(0, 100, 100),
            "adr": rng.uniform(50, 300, 100),
            "arrival_date_month": rng.choice(["January", "March", "July"], 100),
            "is_canceled": rng.integers(0, 2, 100),
        }
    )


def _mock_model():
    m = MagicMock()
    m.predict_proba.return_value = np.column_stack(
        [np.linspace(0.1, 0.9, 100), np.linspace(0.9, 0.1, 100)]
    )
    return m


# ── cmd_evaluate ──────────────────────────────────────────────────────────────


class TestCmdEvaluate:
    def _setup_run(
        self, paths: Paths, run_id: str, model_name: str = "XGBoost"
    ) -> Path:
        """Write model_registry.json and a dummy model artifact for a run."""
        run_dir = paths.reports_metrics / run_id
        run_dir.mkdir(parents=True)

        # Fake model artifact
        model_path = paths.models / f"{model_name}.joblib"
        model_path.write_bytes(b"dummy")

        registry = {model_name: f"models/{model_name}.joblib"}
        (run_dir / "model_registry.json").write_text(json.dumps(registry))
        (run_dir / "model_checksums.json").write_text(
            json.dumps({model_name: "abc123"})
        )

        return run_dir

    def test_raises_if_no_model_registry(self, paths, cfg):
        from src.cli.evaluate import cmd_evaluate

        df = _make_test_df()
        df.to_parquet(paths.data_processed / "test.parquet", index=False)

        # Write latest.json pointing to a run without registry
        run_id = "missing-registry-run"
        (paths.reports_metrics / run_id).mkdir(parents=True)
        (paths.reports_metrics / "latest.json").write_text(
            json.dumps({"run_id": run_id})
        )

        with pytest.raises(FileNotFoundError, match="model_registry"):
            cmd_evaluate(paths, cfg, run_id=run_id)

    def test_raises_if_no_loadable_models(self, paths, cfg):
        from src.cli.evaluate import cmd_evaluate

        df = _make_test_df()
        df.to_parquet(paths.data_processed / "test.parquet", index=False)

        run_id = "empty-models-run"
        run_dir = paths.reports_metrics / run_id
        run_dir.mkdir(parents=True)
        # Registry points to non-existent file
        (run_dir / "model_registry.json").write_text(
            json.dumps({"Ghost": "models/ghost.joblib"})
        )

        with pytest.raises(ValueError, match="No model artifacts"):
            cmd_evaluate(paths, cfg, run_id=run_id)

    def test_cmd_evaluate_success(self, paths, cfg):
        from src.cli.evaluate import cmd_evaluate

        df = _make_test_df()
        df.to_parquet(paths.data_processed / "test.parquet", index=False)

        run_id = "eval-run-001"
        self._setup_run(paths, run_id)

        with (
            patch("src.cli.evaluate.safe_load", return_value=_mock_model()),
            patch("src.cli.evaluate.evaluate_binary_classifier") as mock_eval,
            patch("src.cli.evaluate.sweep_thresholds") as mock_sweep,
            patch("src.cli.evaluate.sweep_thresholds_for_profit") as mock_profit,
            patch("src.cli.evaluate.ExperimentTracker") as mock_tracker_cls,
            patch("src.cli.evaluate.json_write"),
            patch("src.cli.evaluate.copy_to_latest"),
            patch("src.cli.evaluate.mark_latest"),
        ):
            mock_eval.return_value = {"roc_auc": 0.82, "threshold": 0.5}
            mock_sweep.return_value = pd.DataFrame(
                {"threshold": [0.3, 0.5, 0.7], "f1": [0.7, 0.75, 0.65]}
            )
            profit_result = MagicMock()
            profit_result.best_threshold = 0.5
            profit_result.best_profit = 150.0
            profit_result.rows = [{"threshold": 0.5, "profit": 150.0}]
            mock_profit.return_value = profit_result

            mock_tracker = MagicMock()
            mock_tracker_cls.return_value = mock_tracker
            mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

            result_run_id = cmd_evaluate(paths, cfg, run_id=run_id)

        assert result_run_id == run_id

    def test_cmd_evaluate_resolves_latest_run(self, paths, cfg):
        from src.cli.evaluate import cmd_evaluate

        df = _make_test_df()
        df.to_parquet(paths.data_processed / "test.parquet", index=False)

        run_id = "auto-resolved-run"
        self._setup_run(paths, run_id)
        (paths.reports_metrics / "latest.json").write_text(
            json.dumps({"run_id": run_id})
        )

        with (
            patch("src.cli.evaluate.safe_load", return_value=_mock_model()),
            patch(
                "src.cli.evaluate.evaluate_binary_classifier",
                return_value={"roc_auc": 0.80},
            ),
            patch(
                "src.cli.evaluate.sweep_thresholds",
                return_value=pd.DataFrame({"threshold": [0.5], "f1": [0.75]}),
            ),
            patch(
                "src.cli.evaluate.sweep_thresholds_for_profit",
                return_value=MagicMock(best_threshold=0.5, best_profit=100.0, rows=[]),
            ),
            patch("src.cli.evaluate.ExperimentTracker") as mock_tracker_cls,
            patch("src.cli.evaluate.json_write"),
            patch("src.cli.evaluate.copy_to_latest"),
            patch("src.cli.evaluate.mark_latest"),
        ):
            mock_tracker = MagicMock()
            mock_tracker_cls.return_value = mock_tracker
            mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

            result_run_id = cmd_evaluate(paths, cfg)

        assert result_run_id == run_id
