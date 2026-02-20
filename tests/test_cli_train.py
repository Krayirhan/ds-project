"""Tests for src/cli/train.py — cmd_train and _load_splits."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import ExperimentConfig, Paths


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def cfg() -> ExperimentConfig:
    return ExperimentConfig()


@pytest.fixture()
def paths(tmp_path: Path) -> Paths:
    # Paths auto-derives all subdirs from project_root
    p = Paths(project_root=tmp_path)
    p.data_processed.mkdir(parents=True, exist_ok=True)
    p.models.mkdir(parents=True, exist_ok=True)
    p.reports_metrics.mkdir(parents=True, exist_ok=True)
    return p


def _make_dummy_df() -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "lead_time": rng.integers(0, 100, 200),
            "adr": rng.uniform(50, 300, 200),
            "arrival_date_month": rng.choice(
                ["January", "March", "July"], 200
            ),
            "is_canceled": rng.integers(0, 2, 200),
        }
    )


# ── _load_splits ──────────────────────────────────────────────────────────────

class TestLoadSplits:
    def test_loads_persisted_splits_when_all_exist(self, paths, cfg):
        from src.cli.train import _load_splits

        df = _make_dummy_df()
        for name in ("train", "cal", "test"):
            p = paths.data_processed / f"{name}.parquet"
            df.to_parquet(p, index=False)

        train, cal, test = _load_splits(paths, cfg)
        assert len(train) > 0
        assert len(cal) > 0
        assert len(test) > 0

    def test_falls_back_to_in_memory_split_if_missing(self, paths, cfg):
        from src.cli.train import _load_splits

        df = _make_dummy_df()
        dataset_path = paths.data_processed / "dataset.parquet"
        df.to_parquet(dataset_path, index=False)

        train, cal, test = _load_splits(paths, cfg)
        # Fallback should also persist the splits
        assert (paths.data_processed / "train.parquet").exists()
        assert (paths.data_processed / "cal.parquet").exists()
        assert (paths.data_processed / "test.parquet").exists()

    def test_fallback_raises_if_no_dataset_either(self, paths, cfg):
        from src.cli.train import _load_splits

        with pytest.raises(Exception):
            _load_splits(paths, cfg)


# ── cmd_train ─────────────────────────────────────────────────────────────────

class TestCmdTrain:
    @patch("src.cli.train.ExperimentTracker")
    @patch("src.cli.train.train_candidate_models")
    @patch("src.cli.train.calibrate_frozen_classifier")
    @patch("src.cli.train.generate_reference_stats", return_value={})
    @patch("src.cli.train.generate_reference_categories", return_value={})
    @patch("src.cli.train.generate_reference_correlations", return_value={})
    @patch("src.cli.train.run_validation_profile", return_value=MagicMock(summary={}, passed=True, hard_failures=[]))
    @patch("src.cli.train.validate_processed_data", return_value=MagicMock(passed=True, errors=[]))
    @patch("src.cli.train.validate_row_counts", return_value={"ok": True, "passed": True, "difference": 0, "train": 200, "cal": 200, "test": 200, "dataset_rows": 600, "split_total": 600})
    def test_cmd_train_returns_run_id(
        self,
        mock_row_counts,
        mock_validate,
        mock_profile,
        mock_corr,
        mock_cats,
        mock_ref,
        mock_calib,
        mock_train,
        mock_tracker_cls,
        paths,
        cfg,
    ):
        import numpy as np
        from src.cli.train import cmd_train

        df = _make_dummy_df()
        for name in ("train", "cal", "test", "dataset"):
            df.to_parquet(paths.data_processed / f"{name}.parquet", index=False)

        # Mock train result with proper numpy array for cv_scores
        train_result = MagicMock()
        train_result.cv_scores = np.array([0.85])
        train_result.feature_spec.to_dict.return_value = {}
        train_result.feature_spec.numeric = []
        train_result.feature_spec.categorical = []
        train_result.feature_spec.all_features = []
        train_result.feature_dtypes = {}
        mock_train.return_value = {"XGBoost": train_result}

        # Mock calibration result with serialisable metrics
        mock_calib.return_value = MagicMock(calibrated_model=MagicMock(), metrics={})

        # Mock tracker as context manager
        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker
        mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("joblib.dump"),
            patch("src.cli.train.json_write"),
            patch("src.cli.train.sha256_file", return_value="sha-abc"),
            patch("src.cli.train.copy_to_latest"),
            patch("src.cli.train.mark_latest"),
        ):
            run_id = cmd_train(paths, cfg, run_id="test-run-001")

        assert run_id == "test-run-001"

    def test_cmd_train_generates_run_id_if_none(self, paths, cfg):
        import numpy as np
        from src.cli.train import cmd_train

        df = _make_dummy_df()
        for name in ("train", "cal", "test", "dataset"):
            df.to_parquet(paths.data_processed / f"{name}.parquet", index=False)

        with (
            patch("src.cli.train.train_candidate_models") as mock_train,
            patch("src.cli.train.calibrate_frozen_classifier") as mock_calib,
            patch("src.cli.train.validate_processed_data", return_value=MagicMock(passed=True, errors=[])),
            patch("src.cli.train.validate_row_counts", return_value={"ok": True, "passed": True, "difference": 0, "train": 200, "cal": 200, "test": 200, "dataset_rows": 600, "split_total": 600}),
            patch("src.cli.train.generate_reference_stats", return_value={}),
            patch("src.cli.train.generate_reference_categories", return_value={}),
            patch("src.cli.train.generate_reference_correlations", return_value={}),
            patch("src.cli.train.run_validation_profile", return_value=MagicMock(summary={}, passed=True, hard_failures=[])),
            patch("src.cli.train.ExperimentTracker") as mock_tracker_cls,
            patch("joblib.dump"),
            patch("src.cli.train.json_write"),
            patch("src.cli.train.sha256_file", return_value="sha-abc"),
            patch("src.cli.train.copy_to_latest"),
            patch("src.cli.train.mark_latest"),
        ):
            train_result = MagicMock()
            train_result.cv_scores = np.array([0.85])
            train_result.feature_spec.to_dict.return_value = {}
            train_result.feature_spec.numeric = []
            train_result.feature_spec.categorical = []
            train_result.feature_spec.all_features = []
            train_result.feature_dtypes = {}
            mock_train.return_value = {"XGBoost": train_result}
            mock_calib.return_value = MagicMock(calibrated_model=MagicMock(), metrics={})
            mock_tracker = MagicMock()
            mock_tracker_cls.return_value = mock_tracker
            mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

            run_id = cmd_train(paths, cfg)

        assert run_id is not None
        assert len(run_id) > 0

    def test_cmd_train_no_models_raises(self, paths, cfg):
        from src.cli.train import cmd_train

        df = _make_dummy_df()
        for name in ("train", "cal", "test", "dataset"):
            df.to_parquet(paths.data_processed / f"{name}.parquet", index=False)

        with (
            patch("src.cli.train.train_candidate_models", return_value={}),
            patch("src.cli.train.validate_processed_data", return_value=MagicMock(passed=True, errors=[])),
            patch("src.cli.train.validate_row_counts", return_value=MagicMock(ok=True)),
            patch("src.cli.train.generate_reference_stats", return_value={}),
            patch("src.cli.train.generate_reference_categories", return_value={}),
            patch("src.cli.train.generate_reference_correlations", return_value={}),
            patch("src.cli.train.run_validation_profile", return_value=MagicMock(summary={})),
            patch("src.cli.train.ExperimentTracker") as mock_tracker_cls,
        ):
            mock_tracker = MagicMock()
            mock_tracker_cls.return_value = mock_tracker
            mock_tracker.start_run.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracker.start_run.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises((ValueError, RuntimeError, Exception)):
                cmd_train(paths, cfg, run_id="test-run-002")
