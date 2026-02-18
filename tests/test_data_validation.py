"""
Tests for Pandera-based data validation framework.
"""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.data_validation import (
    build_raw_schema,
    build_processed_schema,
    build_inference_schema,
    validate_raw_data,
    validate_processed_data,
    validate_inference_payload,
    validate_distributions,
    generate_reference_stats,
    DistributionReport,
)


# ─── Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def valid_raw_row() -> dict:
    return {
        "hotel": "City Hotel",
        "arrival_date_year": 2025,
        "arrival_date_month": "July",
        "arrival_date_week_number": 28,
        "arrival_date_day_of_month": 15,
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 5,
        "adults": 2,
        "children": 0.0,
        "babies": 0,
        "lead_time": 120,
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "booking_changes": 1,
        "adr": 89.5,
        "is_canceled": "no",
    }


@pytest.fixture
def valid_raw_df(valid_raw_row) -> pd.DataFrame:
    return pd.DataFrame([valid_raw_row] * 10)


@pytest.fixture
def valid_processed_df() -> pd.DataFrame:
    return pd.DataFrame({
        "is_canceled": [0, 1, 0, 1, 0],
        "lead_time": [10, 200, 50, 300, 0],
        "adr": [80.0, 120.0, 95.0, 200.0, 60.0],
        "hotel": ["City Hotel"] * 5,
    })


# ─── Raw Schema Tests ──────────────────────────────────────────────────
class TestRawSchema:
    def test_valid_data_passes(self, valid_raw_df):
        errors = validate_raw_data(valid_raw_df, raise_on_error=False)
        assert errors is None

    def test_negative_lead_time_fails(self, valid_raw_row):
        row = valid_raw_row.copy()
        row["lead_time"] = -5
        df = pd.DataFrame([row])
        errors = validate_raw_data(df, raise_on_error=False)
        assert errors is not None

    def test_invalid_hotel_fails(self, valid_raw_row):
        row = valid_raw_row.copy()
        row["hotel"] = "Unknown Hotel"
        df = pd.DataFrame([row])
        errors = validate_raw_data(df, raise_on_error=False)
        assert errors is not None

    def test_invalid_target_label_fails(self, valid_raw_row):
        row = valid_raw_row.copy()
        row["is_canceled"] = "maybe"
        df = pd.DataFrame([row])
        errors = validate_raw_data(df, raise_on_error=False)
        assert errors is not None

    def test_extra_columns_allowed(self, valid_raw_row):
        row = valid_raw_row.copy()
        row["extra_col"] = "some_value"
        df = pd.DataFrame([row])
        errors = validate_raw_data(df, raise_on_error=False)
        assert errors is None

    def test_raises_on_error(self, valid_raw_row):
        row = valid_raw_row.copy()
        row["lead_time"] = -1
        df = pd.DataFrame([row])
        with pytest.raises(SchemaErrors):
            validate_raw_data(df, raise_on_error=True)


# ─── Processed Schema Tests ────────────────────────────────────────────
class TestProcessedSchema:
    def test_valid_processed_data(self, valid_processed_df):
        errors = validate_processed_data(
            valid_processed_df,
            numeric_cols=["lead_time", "adr"],
            categorical_cols=["hotel"],
            raise_on_error=False,
        )
        assert errors is None

    def test_invalid_target_type(self):
        df = pd.DataFrame({"is_canceled": ["yes", "no"]})
        errors = validate_processed_data(df, raise_on_error=False)
        assert errors is not None


# ─── Inference Schema Tests ─────────────────────────────────────────────
class TestInferenceSchema:
    def test_valid_payload(self):
        spec = {"numeric": ["lead_time", "adr"], "categorical": ["hotel"]}
        df = pd.DataFrame({
            "lead_time": [100],
            "adr": [95.0],
            "hotel": ["City Hotel"],
        })
        errors = validate_inference_payload(df, spec, raise_on_error=False)
        assert errors is None


# ─── Distribution Tests ────────────────────────────────────────────────
class TestDistributions:
    def test_no_drift(self):
        ref_stats = {
            "lead_time": {"mean": 100.0, "std": 50.0, "min": 0, "max": 500},
        }
        df = pd.DataFrame({"lead_time": [90, 100, 110, 95, 105]})
        report = validate_distributions(df, ref_stats)
        assert report.passed is True
        assert len(report.violations) == 0

    def test_mean_drift_detected(self):
        ref_stats = {
            "lead_time": {"mean": 100.0, "std": 10.0, "min": 0, "max": 200},
        }
        df = pd.DataFrame({"lead_time": [500, 600, 700]})
        report = validate_distributions(df, ref_stats)
        assert report.passed is False
        drift_violations = [v for v in report.violations if v["check"] == "mean_drift"]
        assert len(drift_violations) > 0

    def test_missing_column(self):
        ref_stats = {"missing_col": {"mean": 0, "std": 1}}
        df = pd.DataFrame({"other": [1, 2, 3]})
        report = validate_distributions(df, ref_stats)
        assert report.passed is False

    def test_generate_reference_stats(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        stats = generate_reference_stats(df, ["a", "b"])
        assert "a" in stats
        assert "b" in stats
        assert abs(stats["a"]["mean"] - 3.0) < 1e-6
        assert "std" in stats["a"]
        assert "q25" in stats["a"]
