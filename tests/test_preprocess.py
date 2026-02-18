import numpy as np
import pandas as pd
import pytest

from src.preprocess import preprocess_basic


def _base_df():
    return pd.DataFrame(
        {
            "hotel": ["Resort Hotel", "City Hotel", "Resort Hotel"],
            "lead_time": [10, 20, 30],
            "is_canceled": ["yes", "no", "yes"],
            "reservation_status": ["Check-Out", "Canceled", "Check-Out"],
        }
    )


LABEL_MAP = {"yes": 1, "no": 0}


class TestLabelMapping:
    def test_maps_yes_no_to_int(self):
        df = _base_df()
        out = preprocess_basic(df, "is_canceled", LABEL_MAP, drop_cols=["reservation_status"])
        assert out["is_canceled"].dtype in (np.int32, np.int64, int)
        assert set(out["is_canceled"].unique()) == {0, 1}

    def test_case_insensitive_mapping(self):
        df = _base_df()
        df["is_canceled"] = ["Yes", "NO", "YES"]
        out = preprocess_basic(df, "is_canceled", LABEL_MAP, drop_cols=["reservation_status"])
        assert list(out["is_canceled"]) == [1, 0, 1]

    def test_unknown_label_raises(self):
        df = _base_df()
        df.loc[0, "is_canceled"] = "maybe"
        with pytest.raises(ValueError, match="Unknown target labels"):
            preprocess_basic(df, "is_canceled", LABEL_MAP, drop_cols=["reservation_status"])


class TestLeakageDrop:
    def test_drops_specified_columns(self):
        df = _base_df()
        out = preprocess_basic(df, "is_canceled", LABEL_MAP, drop_cols=["reservation_status"])
        assert "reservation_status" not in out.columns

    def test_drops_extra_blocked_columns(self):
        df = _base_df()
        out = preprocess_basic(
            df,
            "is_canceled",
            LABEL_MAP,
            drop_cols=["reservation_status"],
            extra_blocked_cols=["hotel"],
        )
        assert "hotel" not in out.columns
        assert "reservation_status" not in out.columns

    def test_ignores_nonexistent_drop_cols(self):
        df = _base_df()
        out = preprocess_basic(
            df, "is_canceled", LABEL_MAP, drop_cols=["reservation_status", "nonexistent_col"]
        )
        assert "reservation_status" not in out.columns
        assert len(out) == 3


class TestEmptyColumnDrop:
    def test_drops_all_nan_column(self):
        df = _base_df()
        df["empty_col"] = np.nan
        out = preprocess_basic(df, "is_canceled", LABEL_MAP, drop_cols=["reservation_status"])
        assert "empty_col" not in out.columns

    def test_keeps_column_with_some_values(self):
        df = _base_df()
        df["partial"] = [np.nan, 5.0, np.nan]
        out = preprocess_basic(df, "is_canceled", LABEL_MAP, drop_cols=["reservation_status"])
        assert "partial" in out.columns


class TestMissingFill:
    def test_numeric_missing_filled_with_median(self):
        df = _base_df()
        df["lead_time"] = [10.0, np.nan, 30.0]
        out = preprocess_basic(df, "is_canceled", LABEL_MAP, drop_cols=["reservation_status"])
        assert out["lead_time"].isna().sum() == 0
        assert out["lead_time"].iloc[1] == 20.0  # median of 10, 30

    def test_categorical_missing_filled_with_mode(self):
        df = _base_df()
        df["hotel"] = ["Resort Hotel", "Resort Hotel", np.nan]
        out = preprocess_basic(df, "is_canceled", LABEL_MAP, drop_cols=["reservation_status"])
        assert out["hotel"].isna().sum() == 0
        assert out["hotel"].iloc[2] == "Resort Hotel"


class TestColumnNormalization:
    def test_strips_whitespace_from_column_names(self):
        df = _base_df()
        df.columns = [" hotel ", " lead_time", "is_canceled", "reservation_status "]
        out = preprocess_basic(df, "is_canceled", LABEL_MAP, drop_cols=["reservation_status"])
        assert "hotel" in out.columns
        assert "lead_time" in out.columns
