import pandas as pd
import pytest

from src.predict import validate_and_prepare_features


def _spec_payload():
    return {
        "numeric": ["x1", "x2"],
        "categorical": ["c1"],
        "all_features": ["x1", "x2", "c1"],
    }


def test_missing_columns_fail_fast():
    df = pd.DataFrame({"x1": [1, 2], "c1": ["a", "b"]})
    with pytest.raises(ValueError):
        validate_and_prepare_features(df, _spec_payload(), fail_on_missing=True)


def test_extra_columns_are_ignored_and_order_frozen():
    df = pd.DataFrame(
        {
            "c1": ["a", "b"],
            "x2": [10, 20],
            "x1": [1, 2],
            "extra": [999, 999],
        }
    )
    out, report = validate_and_prepare_features(df, _spec_payload(), fail_on_missing=True)
    assert list(out.columns) == ["x1", "x2", "c1"]
    assert report["extra_columns"] == ["extra"]
