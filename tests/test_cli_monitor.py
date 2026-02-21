"""Tests for src/cli/monitor.py — cmd_monitor."""

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


def _make_input_df() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "lead_time": rng.integers(0, 100, 80),
            "adr": rng.uniform(50, 300, 80),
            "arrival_date_month": rng.choice(["January", "March", "July"], 80),
            "is_canceled": rng.integers(0, 2, 80),
        }
    )


def _make_policy_dict(paths: Paths, model_name: str = "xgb") -> dict:
    """Create a minimal policy file and return the policy dict."""
    model_artifact = f"models/{model_name}.joblib"
    (paths.project_root / "models").mkdir(exist_ok=True)
    (paths.project_root / model_artifact).write_bytes(b"dummy")

    policy = {
        "policy_version": ExperimentConfig().contract.policy_version,
        "run_id": "monitor-run-001",
        "selected_model": model_name,
        "selected_model_artifact": model_artifact,
        "threshold": 0.5,
        "action_rate": 0.3,
    }
    policy_file = paths.reports_metrics / "decision_policy.json"
    policy_file.write_text(json.dumps(policy))
    return policy


class TestCmdMonitor:
    def test_raises_on_policy_version_mismatch(self, paths, cfg):
        from src.cli.monitor import cmd_monitor

        policy = _make_policy_dict(paths)
        policy["policy_version"] = "wrong-version-999"
        (paths.reports_metrics / "decision_policy.json").write_text(json.dumps(policy))

        with (
            patch("src.cli.monitor.load_decision_policy") as mock_load,
        ):
            mock_policy = MagicMock()
            mock_policy.raw = policy
            mock_load.return_value = mock_policy

            with pytest.raises(ValueError, match="Policy version mismatch"):
                cmd_monitor(paths, cfg, run_id="monitor-run-001")

    def test_raises_if_no_model_artifact_in_policy(self, paths, cfg):
        from src.cli.monitor import cmd_monitor

        _make_policy_dict(paths)

        with (
            patch("src.cli.monitor.load_decision_policy") as mock_load,
        ):
            mock_policy = MagicMock()
            mock_policy.raw = {
                "policy_version": cfg.contract.policy_version,
                "run_id": "monitor-run-001",
            }
            mock_policy.selected_model_artifact = None
            mock_load.return_value = mock_policy

            with pytest.raises(ValueError, match="selected_model_artifact"):
                cmd_monitor(paths, cfg, run_id="monitor-run-001")

    def test_cmd_monitor_success_no_input(self, paths, cfg):
        from src.cli.monitor import cmd_monitor

        _make_policy_dict(paths)
        df = _make_input_df()
        df.to_parquet(paths.data_processed / "dataset.parquet", index=False)
        df.to_parquet(paths.data_processed / "test.parquet", index=False)

        mock_drift = MagicMock()
        mock_drift.psi_scores = {"lead_time": 0.05}
        mock_drift.js_scores = {"lead_time": 0.02}
        mock_drift.any_alert = False

        mock_build_alerts_result = {
            "psi_alert": False,
            "any_alert": False,
        }

        with (
            patch("src.cli.monitor.load_decision_policy") as mock_load,
            patch("joblib.load", return_value=MagicMock()),
            patch("src.cli.monitor.read_input_dataset", return_value=df),
            patch(
                "src.cli.monitor.predict_with_policy",
                return_value=(df.assign(action=0, proba=0.5, score=0.5), MagicMock()),
            ),
            patch("src.cli.monitor.data_drift_report", return_value=mock_drift),
            patch(
                "src.cli.monitor.prediction_drift_report",
                return_value=MagicMock(any_alert=False),
            ),
            patch(
                "src.cli.monitor.build_alerts", return_value=mock_build_alerts_result
            ),
            patch(
                "src.cli.monitor.validate_distributions",
                return_value=MagicMock(any_alert=False),
            ),
            patch(
                "src.cli.monitor.detect_label_drift",
                return_value=MagicMock(any_alert=False),
            ),
            patch(
                "src.cli.monitor.detect_correlation_drift",
                return_value=MagicMock(any_alert=False),
            ),
            patch(
                "src.cli.monitor.validate_data_volume", return_value=MagicMock(ok=True)
            ),
            patch(
                "src.cli.monitor.detect_feature_importance_drift",
                return_value=MagicMock(any_alert=False),
            ),
            patch("src.cli.monitor.load_feature_spec", return_value=MagicMock()),
            patch("src.cli.monitor.json_write"),
            patch("src.cli.monitor.mark_latest"),
            patch("src.cli.monitor.notify_webhook"),
        ):
            mock_policy = MagicMock()
            mock_policy.raw = {
                "policy_version": cfg.contract.policy_version,
                "run_id": "monitor-run-001",
            }
            mock_policy.selected_model_artifact = "models/xgb.joblib"
            mock_load.return_value = mock_policy

            run_id = cmd_monitor(paths, cfg, run_id="monitor-run-001")

        assert run_id is not None

    def test_cmd_monitor_any_alert_triggers_webhook(self, paths, cfg):
        from src.cli.monitor import cmd_monitor

        _make_policy_dict(paths)
        df = _make_input_df()
        df.to_parquet(paths.data_processed / "dataset.parquet", index=False)
        df.to_parquet(paths.data_processed / "test.parquet", index=False)

        mock_build_alerts_result = {"psi_alert": True, "any_alert": True}

        with (
            patch("src.cli.monitor.load_decision_policy") as mock_load,
            patch("joblib.load", return_value=MagicMock()),
            patch("src.cli.monitor.read_input_dataset", return_value=df),
            patch(
                "src.cli.monitor.predict_with_policy",
                return_value=(df.assign(action=0, proba=0.5, score=0.5), MagicMock()),
            ),
            patch(
                "src.cli.monitor.data_drift_report",
                return_value=MagicMock(psi_scores={}, js_scores={}, any_alert=True),
            ),
            patch(
                "src.cli.monitor.prediction_drift_report",
                return_value=MagicMock(any_alert=True),
            ),
            patch(
                "src.cli.monitor.build_alerts", return_value=mock_build_alerts_result
            ),
            patch(
                "src.cli.monitor.validate_distributions",
                return_value=MagicMock(any_alert=False),
            ),
            patch(
                "src.cli.monitor.detect_label_drift",
                return_value=MagicMock(any_alert=False),
            ),
            patch(
                "src.cli.monitor.detect_correlation_drift",
                return_value=MagicMock(any_alert=False),
            ),
            patch(
                "src.cli.monitor.validate_data_volume", return_value=MagicMock(ok=True)
            ),
            patch(
                "src.cli.monitor.detect_feature_importance_drift",
                return_value=MagicMock(any_alert=False),
            ),
            patch("src.cli.monitor.load_feature_spec", return_value=MagicMock()),
            patch("src.cli.monitor.json_write"),
            patch("src.cli.monitor.mark_latest"),
            patch("src.cli.monitor.notify_webhook") as mock_notify,
        ):
            mock_policy = MagicMock()
            mock_policy.raw = {
                "policy_version": cfg.contract.policy_version,
                "run_id": "monitor-run-001",
            }
            mock_policy.selected_model_artifact = "models/xgb.joblib"
            mock_load.return_value = mock_policy

            cmd_monitor(paths, cfg, run_id="monitor-run-001")

        # Webhook should be called when any_alert is True
        mock_notify.assert_called_once()
