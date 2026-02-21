"""Tests for src/cli/explain.py — cmd_explain."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config import ExperimentConfig, Paths


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
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "lead_time": rng.integers(0, 100, 50),
            "adr": rng.uniform(50, 300, 50),
            "arrival_date_month": rng.choice(["January", "March"], 50),
            "is_canceled": rng.integers(0, 2, 50),
        }
    )


def _write_policy(paths: Paths, run_id: str = "explain-run-001") -> None:
    run_dir = paths.reports_metrics / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    policy = {
        "selected_model": "XGBoost",
        "selected_model_artifact": "models/xgb.joblib",
    }
    (run_dir / "decision_policy.json").write_text(json.dumps(policy))
    (paths.project_root / "models" / "xgb.joblib").write_bytes(b"dummy")


class TestCmdExplain:
    def test_raises_if_no_policy_found(self, paths, cfg):
        from src.cli.explain import cmd_explain

        with pytest.raises(FileNotFoundError, match="Decision policy not found"):
            cmd_explain(paths, cfg, run_id="nonexistent-run")

    def test_raises_if_no_artifact_in_policy(self, paths, cfg):
        from src.cli.explain import cmd_explain

        run_id = "no-artifact-run"
        run_dir = paths.reports_metrics / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "decision_policy.json").write_text(
            json.dumps({"selected_model": "XGBoost"})
        )

        with pytest.raises(ValueError, match="selected_model_artifact"):
            cmd_explain(paths, cfg, run_id=run_id)

    def test_raises_if_model_file_missing(self, paths, cfg):
        from src.cli.explain import cmd_explain

        run_id = "missing-model-run"
        run_dir = paths.reports_metrics / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "decision_policy.json").write_text(
            json.dumps(
                {
                    "selected_model": "XGBoost",
                    "selected_model_artifact": "models/nonexistent.joblib",
                }
            )
        )
        # Don't create the model file

        with (
            patch("src.cli.explain.safe_load", return_value=None),
        ):
            with pytest.raises(FileNotFoundError, match="Model not found"):
                cmd_explain(paths, cfg, run_id=run_id)

    def test_cmd_explain_success_with_permutation(self, paths, cfg):
        from src.cli.explain import cmd_explain

        run_id = "explain-run-001"
        _write_policy(paths, run_id)

        df = _make_test_df()
        df.to_parquet(paths.data_processed / "test.parquet", index=False)

        mock_perm_result = {
            "importances_mean": [0.1, 0.2, 0.05],
            "feature_names": ["lead_time", "adr", "arrival_date_month"],
        }

        with (
            patch("src.cli.explain.safe_load", return_value=MagicMock()),
            patch(
                "src.cli.explain.compute_permutation_importance",
                return_value=mock_perm_result,
            ),
            patch(
                "src.cli.explain.compute_shap_values", return_value={"shap_summary": []}
            ),
            patch("src.cli.explain.save_explainability_report"),
        ):
            result_run_id = cmd_explain(paths, cfg, run_id=run_id, sample_size=10)

        assert result_run_id == run_id

    def test_cmd_explain_resolves_latest_run(self, paths, cfg):
        from src.cli.explain import cmd_explain

        run_id = "latest-explain-run"
        _write_policy(paths, run_id)
        (paths.models / "latest.json").write_text(json.dumps({"run_id": run_id}))

        df = _make_test_df()
        df.to_parquet(paths.data_processed / "test.parquet", index=False)

        with (
            patch("src.cli.explain.safe_load", return_value=MagicMock()),
            patch("src.cli.explain.compute_permutation_importance", return_value={}),
            patch("src.cli.explain.compute_shap_values", return_value={}),
            patch("src.cli.explain.save_explainability_report"),
            patch("src.cli.explain.resolve_latest_run_id", return_value=run_id),
        ):
            result_run_id = cmd_explain(paths, cfg, sample_size=5)

        assert result_run_id == run_id
