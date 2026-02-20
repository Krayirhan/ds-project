"""Tests for src/cli/preprocess.py — cmd_preprocess."""

from __future__ import annotations

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
    p.data_raw.mkdir(parents=True, exist_ok=True)
    p.data_processed.mkdir(parents=True, exist_ok=True)
    p.reports_metrics.mkdir(parents=True, exist_ok=True)
    return p


def _make_hotel_df() -> pd.DataFrame:
    """Minimal hotel_bookings-like DataFrame."""
    rng = np.random.default_rng(1)
    n = 200
    return pd.DataFrame(
        {
            "lead_time": rng.integers(0, 300, n),
            "adr": rng.uniform(0.0, 500.0, n),
            "arrival_date_month": rng.choice(
                [
                    "January", "February", "March", "April",
                    "May", "June", "July", "August",
                ], n
            ),
            "arrival_date_week_number": rng.integers(1, 53, n),
            "arrival_date_day_of_month": rng.integers(1, 31, n),
            "stays_in_weekend_nights": rng.integers(0, 5, n),
            "stays_in_week_nights": rng.integers(0, 10, n),
            "adults": rng.integers(1, 4, n),
            "children": rng.integers(0, 3, n).astype(float),
            "babies": rng.integers(0, 2, n),
            "previous_cancellations": rng.integers(0, 5, n),
            "previous_bookings_not_canceled": rng.integers(0, 5, n),
            "booking_changes": rng.integers(0, 5, n),
            "days_in_waiting_list": rng.integers(0, 100, n),
            "required_car_parking_spaces": rng.integers(0, 2, n),
            "total_of_special_requests": rng.integers(0, 5, n),
            "is_canceled": rng.integers(0, 2, n),
        }
    )


class TestCmdPreprocess:
    def _patch_all_validation(self):
        """Return a context-manager-compatible list of patches for validation fns."""
        return [
            patch("src.cli.preprocess.check_data_staleness", return_value=MagicMock(is_stale=False, summary="")),
            patch("src.cli.preprocess.basic_schema_checks"),
            patch("src.cli.preprocess.validate_target_labels"),
            patch("src.cli.preprocess.null_ratio_report", return_value={}),
            patch("src.cli.preprocess.detect_duplicates", return_value=MagicMock(n_duplicates=0)),
            patch("src.cli.preprocess.detect_row_anomalies", return_value=MagicMock(n_anomalies=0)),
            patch("src.cli.preprocess.validate_raw_data", return_value=MagicMock(passed=True, errors=[])),
            patch("src.cli.preprocess.validate_processed_data", return_value=MagicMock(passed=True, errors=[])),
            patch("src.cli.preprocess.validate_data_volume", return_value=MagicMock(ok=True)),
            patch("src.cli.preprocess.generate_reference_stats", return_value={}),
            patch("src.cli.preprocess.generate_reference_categories", return_value={}),
            patch("src.cli.preprocess.generate_reference_correlations", return_value={}),
            patch("src.cli.preprocess.run_validation_profile", return_value=MagicMock(summary={})),
            patch("src.cli.preprocess.get_schema_fingerprint", return_value={"fingerprint": "fp123", "n_columns": 5}),
            patch("src.cli.preprocess.sha256_file", return_value="sha-abc"),
            patch("src.cli.preprocess.infer_feature_spec", return_value=MagicMock(to_dict=lambda: {}, version="v1")),
            patch("src.cli.preprocess.json_write"),
        ]

    def test_raises_if_raw_csv_missing(self, paths, cfg):
        from src.cli.preprocess import cmd_preprocess

        with pytest.raises(Exception):
            cmd_preprocess(paths, cfg)

    def test_cmd_preprocess_success(self, paths, cfg):
        from src.cli.preprocess import cmd_preprocess

        df = _make_hotel_df()
        csv_path = paths.data_raw / "hotel_bookings.csv"
        df.to_csv(csv_path, index=False)

        patches = self._patch_all_validation()

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patches[10],
            patches[11],
            patches[12],
            patches[13],
            patches[14],
            patches[15],
            patches[16],
            patch("src.cli.preprocess.write_parquet"),
            patch("src.cli.preprocess.preprocess_basic", return_value=df),
        ):
            # Should not raise
            cmd_preprocess(paths, cfg)

    def test_stale_data_hard_fail_raises(self, paths, cfg):
        from src.cli.preprocess import cmd_preprocess

        df = _make_hotel_df()
        csv_path = paths.data_raw / "hotel_bookings.csv"
        df.to_csv(csv_path, index=False)

        stale_result = MagicMock(is_stale=True, summary="Data is 400 days old")

        with (
            patch("src.cli.preprocess.check_data_staleness", return_value=stale_result),
        ):
            # When staleness severity is hard_fail (check cfg)
            if cfg.validation.staleness.severity == "hard_fail":
                with pytest.raises(ValueError, match="Stale data"):
                    cmd_preprocess(paths, cfg)
            else:
                # soft_fail / warn — should log and continue, not raise
                with (
                    patch("src.cli.preprocess.basic_schema_checks"),
                    patch("src.cli.preprocess.validate_target_labels"),
                    patch("src.cli.preprocess.null_ratio_report", return_value={}),
                    patch("src.cli.preprocess.detect_duplicates", return_value=MagicMock(n_duplicates=0)),
                    patch("src.cli.preprocess.detect_row_anomalies", return_value=MagicMock(n_anomalies=0)),
                    patch("src.cli.preprocess.validate_raw_data", return_value=MagicMock(passed=True, errors=[])),
                    patch("src.cli.preprocess.validate_processed_data", return_value=MagicMock(passed=True, errors=[])),
                    patch("src.cli.preprocess.validate_data_volume", return_value=MagicMock(ok=True)),
                    patch("src.cli.preprocess.generate_reference_stats", return_value={}),
                    patch("src.cli.preprocess.generate_reference_categories", return_value={}),
                    patch("src.cli.preprocess.generate_reference_correlations", return_value={}),
                    patch("src.cli.preprocess.run_validation_profile", return_value=MagicMock(summary={})),
                    patch("src.cli.preprocess.get_schema_fingerprint", return_value={"fingerprint": "fp123", "n_columns": 5}),
                    patch("src.cli.preprocess.sha256_file", return_value="sha-abc"),
                    patch("src.cli.preprocess.infer_feature_spec", return_value=MagicMock(to_dict=lambda: {}, version="v1")),
                    patch("src.cli.preprocess.json_write"),
                    patch("src.cli.preprocess.write_parquet"),
                    patch("src.cli.preprocess.preprocess_basic", return_value=df),
                ):
                    cmd_preprocess(paths, cfg)

    def test_duplicate_hard_fail_raises(self, paths, cfg):
        from src.cli.preprocess import cmd_preprocess

        df = _make_hotel_df()
        csv_path = paths.data_raw / "hotel_bookings.csv"
        df.to_csv(csv_path, index=False)

        with (
            patch("src.cli.preprocess.check_data_staleness", return_value=MagicMock(is_stale=False)),
            patch("src.cli.preprocess.basic_schema_checks"),
            patch("src.cli.preprocess.validate_target_labels"),
            patch("src.cli.preprocess.null_ratio_report", return_value={}),
            patch("src.cli.preprocess.detect_duplicates", return_value=MagicMock(n_duplicates=500)),
            patch("src.cli.preprocess.detect_row_anomalies", return_value=MagicMock(n_anomalies=0)),
        ):
            if cfg.validation.duplicate.severity == "hard_fail":
                with pytest.raises(ValueError, match="Duplicate"):
                    cmd_preprocess(paths, cfg)
