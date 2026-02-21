"""Tests for src/cli/predict.py — cmd_predict."""

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


def _write_policy(paths: Paths, run_id: str = "pred-run-001") -> Path:
    policy = {
        "policy_version": ExperimentConfig().contract.policy_version,
        "run_id": run_id,
        "selected_model": "XGBoost",
        "selected_model_artifact": "models/xgb.joblib",
        "threshold": 0.5,
        "action_rate": 0.3,
        "feature_schema_version": "v1",
    }
    policy_file = paths.reports_metrics / "decision_policy.json"
    policy_file.write_text(json.dumps(policy))
    (paths.project_root / "models" / "xgb.joblib").write_bytes(b"dummy")
    return policy_file


def _make_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "lead_time": rng.integers(0, 100, 50),
            "adr": rng.uniform(50, 300, 50),
            "arrival_date_month": rng.choice(["January", "March"], 50),
            "is_canceled": rng.integers(0, 2, 50),
        }
    )


class TestCmdPredict:
    def test_raises_on_policy_version_mismatch(self, paths, cfg):
        from src.cli.predict import cmd_predict

        policy_file = _write_policy(paths)
        policy_data = json.loads(policy_file.read_text())
        policy_data["policy_version"] = "incompatible-999"
        policy_file.write_text(json.dumps(policy_data))

        with (
            patch("src.cli.predict.load_decision_policy") as mock_load,
        ):
            mock_policy = MagicMock()
            mock_policy.raw = policy_data
            mock_load.return_value = mock_policy

            with pytest.raises(ValueError, match="Policy version mismatch"):
                cmd_predict(paths, cfg, run_id="pred-run-001")

    def test_raises_if_model_artifact_missing(self, paths, cfg):
        from src.cli.predict import cmd_predict

        _write_policy(paths)

        with (
            patch("src.cli.predict.load_decision_policy") as mock_load,
        ):
            mock_policy = MagicMock()
            mock_policy.raw = {
                "policy_version": cfg.contract.policy_version,
                "run_id": "pred-run-001",
                "feature_schema_version": "v1",
            }
            mock_policy.selected_model_artifact = "models/nonexistent.joblib"
            mock_load.return_value = mock_policy

            with pytest.raises(FileNotFoundError):
                cmd_predict(paths, cfg, run_id="pred-run-001")

    def test_raises_on_checksum_mismatch(self, paths, cfg):
        from src.cli.predict import cmd_predict

        _write_policy(paths)

        with (
            patch("src.cli.predict.load_decision_policy") as mock_load,
            patch("src.cli.predict.sha256_file", return_value="wrong-sha"),
            patch("joblib.load", return_value=MagicMock()),
        ):
            mock_policy = MagicMock()
            mock_policy.raw = {
                "policy_version": cfg.contract.policy_version,
                "run_id": "pred-run-001",
                "feature_schema_version": "v1",
                "selected_model_sha256": "expected-sha-abc",
            }
            mock_policy.selected_model_artifact = "models/xgb.joblib"
            mock_load.return_value = mock_policy

            with pytest.raises(ValueError, match="checksum mismatch"):
                cmd_predict(paths, cfg, run_id="pred-run-001")

    def test_cmd_predict_success(self, paths, cfg):
        from src.cli.predict import cmd_predict

        _write_policy(paths)
        df = _make_df()

        result_df = df.copy()
        result_df["score"] = 0.6
        result_df["action"] = 1

        with (
            patch("src.cli.predict.load_decision_policy") as mock_load,
            patch("joblib.load", return_value=MagicMock()),
            patch("src.cli.predict.sha256_file", return_value="abc"),
            patch("src.cli.predict.load_feature_spec") as mock_spec,
            patch(
                "src.cli.predict.read_input_dataset", return_value=(df, "input.parquet")
            ),
            patch(
                "src.cli.predict.predict_with_policy",
                return_value=(result_df, MagicMock()),
            ),
            patch("src.cli.predict.write_parquet"),
            patch("src.cli.predict.json_write"),
            patch("src.cli.predict.mark_latest"),
        ):
            feat_spec = MagicMock()
            feat_spec.get.return_value = "v1"
            mock_spec.return_value = feat_spec
            mock_policy = MagicMock()
            mock_policy.raw = {
                "policy_version": cfg.contract.policy_version,
                "run_id": "pred-run-001",
                "feature_schema_version": "v1",
                "selected_model_sha256": "abc",
            }
            mock_policy.selected_model_artifact = "models/xgb.joblib"
            mock_load.return_value = mock_policy

            run_id = cmd_predict(paths, cfg, run_id="pred-run-001")

        assert run_id == "pred-run-001"

    def test_cmd_predict_no_artifact_in_policy_raises(self, paths, cfg):
        from src.cli.predict import cmd_predict

        _write_policy(paths)

        with (
            patch("src.cli.predict.load_decision_policy") as mock_load,
        ):
            mock_policy = MagicMock()
            mock_policy.raw = {
                "policy_version": cfg.contract.policy_version,
                "run_id": "pred-run-001",
            }
            mock_policy.selected_model_artifact = None
            mock_load.return_value = mock_policy

            with pytest.raises(ValueError, match="selected_model_artifact"):
                cmd_predict(paths, cfg, run_id="pred-run-001")

    def test_cmd_predict_resolves_relative_input_path(self, paths, cfg):
        from src.cli.predict import cmd_predict

        _write_policy(paths)
        df = _make_df()

        captured = {"path": None}

        def _capture_read_input_dataset(path):
            captured["path"] = path
            return df

        with (
            patch("src.cli.predict.load_decision_policy") as mock_load,
            patch("joblib.load", return_value=MagicMock()),
            patch("src.cli.predict.sha256_file", return_value="abc"),
            patch(
                "src.cli.predict.load_feature_spec",
                return_value={"schema_version": "v1"},
            ),
            patch(
                "src.cli.predict.read_input_dataset",
                side_effect=_capture_read_input_dataset,
            ),
            patch(
                "src.cli.predict.predict_with_policy",
                return_value=(df.assign(action=1, score=0.5), {"n_rows": len(df)}),
            ),
            patch("src.cli.predict.write_parquet"),
            patch("src.cli.predict.json_write"),
            patch("src.cli.predict.mark_latest"),
        ):
            mock_policy = MagicMock()
            mock_policy.raw = {
                "policy_version": cfg.contract.policy_version,
                "run_id": "pred-run-001",
                "feature_schema_version": "v1",
                "selected_model_sha256": "abc",
            }
            mock_policy.selected_model_artifact = "models/xgb.joblib"
            mock_load.return_value = mock_policy

            cmd_predict(
                paths,
                cfg,
                input_path="data/processed/inference.parquet",
                run_id="pred-run-001",
            )

        assert (
            captured["path"] == paths.project_root / "data/processed/inference.parquet"
        )

    def test_cmd_predict_feature_schema_mismatch_raises(self, paths, cfg):
        from src.cli.predict import cmd_predict

        _write_policy(paths)

        with (
            patch("src.cli.predict.load_decision_policy") as mock_load,
            patch("joblib.load", return_value=MagicMock()),
            patch("src.cli.predict.read_input_dataset", return_value=_make_df()),
            patch(
                "src.cli.predict.load_feature_spec",
                return_value={"schema_version": "v2"},
            ),
        ):
            mock_policy = MagicMock()
            mock_policy.raw = {
                "policy_version": cfg.contract.policy_version,
                "run_id": "pred-run-001",
                "feature_schema_version": "v1",
            }
            mock_policy.selected_model_artifact = "models/xgb.joblib"
            mock_load.return_value = mock_policy

            with pytest.raises(ValueError, match="Feature schema version mismatch"):
                cmd_predict(paths, cfg, run_id="pred-run-001")
