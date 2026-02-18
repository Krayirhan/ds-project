import pandas as pd
import pytest

from src.validate import basic_schema_checks, null_ratio_report, validate_target_labels


class TestBasicSchemaChecks:
    def test_passes_on_valid_df(self):
        df = pd.DataFrame({"is_canceled": [0, 1], "lead_time": [10, 20]})
        basic_schema_checks(df, "is_canceled")  # should not raise

    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame({"is_canceled": pd.Series(dtype=int)})
        with pytest.raises(ValueError, match="empty"):
            basic_schema_checks(df, "is_canceled")

    def test_raises_on_missing_target_column(self):
        df = pd.DataFrame({"lead_time": [10, 20]})
        with pytest.raises(ValueError, match="not found"):
            basic_schema_checks(df, "is_canceled")

    def test_raises_on_duplicate_columns(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "a"])
        with pytest.raises(ValueError, match="Duplicate"):
            basic_schema_checks(df, "a")


class TestValidateTargetLabels:
    def test_passes_with_allowed_labels(self):
        df = pd.DataFrame({"y": ["yes", "no", "yes"]})
        validate_target_labels(df, "y", allowed={"yes", "no"})

    def test_raises_on_unexpected_label(self):
        df = pd.DataFrame({"y": ["yes", "no", "maybe"]})
        with pytest.raises(ValueError, match="Unexpected labels"):
            validate_target_labels(df, "y", allowed={"yes", "no"})

    def test_handles_case_insensitive(self):
        df = pd.DataFrame({"y": ["Yes", "NO"]})
        validate_target_labels(df, "y", allowed={"yes", "no"})

    def test_handles_whitespace_in_labels(self):
        df = pd.DataFrame({"y": [" yes ", "no "]})
        validate_target_labels(df, "y", allowed={"yes", "no"})


class TestNullRatioReport:
    def test_returns_series(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
        result = null_ratio_report(df, top_k=5)
        assert isinstance(result, pd.Series)

    def test_highest_ratio_first(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
        result = null_ratio_report(df)
        assert result.index[0] == "b"
        assert result.iloc[0] == pytest.approx(2 / 3)

    def test_zero_nulls(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = null_ratio_report(df)
        assert (result == 0).all()
