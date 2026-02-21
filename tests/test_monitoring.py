import numpy as np
import pandas as pd
import pytest

from src.features import FeatureSpec
from src.monitoring import (
    AlertThresholds,
    _safe_hist,
    build_alerts,
    data_drift_report,
    outcome_monitoring_report,
    psi_categorical,
    psi_numeric,
    prediction_drift_report,
)


def test_data_drift_report_structure():
    df_ref = pd.DataFrame({"n1": [1, 2, 3], "c1": ["a", "b", "a"]})
    df_cur = pd.DataFrame({"n1": [2, 3, 4], "c1": ["a", "a", "c"]})
    spec = FeatureSpec(numeric=["n1"], categorical=["c1"])
    report = data_drift_report(df_ref, df_cur, spec)
    assert report["n_features_compared"] == 2
    assert "max_psi" in report
    assert "n1" in report["features"]


def test_prediction_drift_report_basic():
    ref = np.array([0.1, 0.2, 0.3, 0.4])
    cur = np.array([0.2, 0.3, 0.4, 0.5])
    report = prediction_drift_report(ref, cur)
    assert "psi" in report
    assert "ks_stat" in report
    assert report["ks_stat"] >= 0.0


def test_build_alerts_profit_drop_and_action_rate():
    alerts = build_alerts(
        data_drift={"max_psi": 0.25},
        prediction_drift={"psi": 0.22},
        outcome_report={"realized_profit": 50.0, "action_rate": 0.35},
        policy={"expected_net_profit": 100.0, "max_action_rate": 0.20},
        thresholds=AlertThresholds(
            data_drift_psi_threshold=0.2,
            prediction_drift_psi_threshold=0.2,
            profit_drop_ratio_alert=0.2,
            action_rate_tolerance=0.05,
        ),
    )
    assert alerts["data_drift"] is True
    assert alerts["prediction_drift"] is True
    assert alerts["profit_drop"] is True
    assert alerts["action_rate_deviation"] is True
    assert alerts["any_alert"] is True


def test_outcome_monitoring_report_happy_path():
    actions_df = pd.DataFrame(
        {
            "proba": [0.8, 0.2, 0.9, 0.1],
            "action": [1, 0, 1, 0],
        }
    )
    outcome_df = pd.DataFrame({"target": [1, 0, 1, 0]})
    policy = {
        "cost_matrix": {"tp_value": 10, "fp_value": -2, "fn_value": -5, "tn_value": 0}
    }
    rep = outcome_monitoring_report(
        actions_df, outcome_df, actual_col="target", policy=policy
    )
    assert rep["n_rows"] == 4
    assert "realized_profit" in rep


def test_outcome_monitoring_report_missing_actual_column():
    actions_df = pd.DataFrame({"proba": [0.5], "action": [1]})
    outcome_df = pd.DataFrame({"other": [1]})
    with pytest.raises(ValueError):
        outcome_monitoring_report(
            actions_df, outcome_df, actual_col="target", policy={}
        )


def test_safe_hist_zero_total_returns_uniform():
    bins = np.array([0.0, 0.5, 1.0])
    out = _safe_hist(np.array([], dtype=float), bins)
    assert out.shape[0] == 2
    assert np.allclose(out, np.array([0.5, 0.5]))


def test_psi_numeric_empty_and_fallback_edges():
    assert psi_numeric(pd.Series([np.nan, np.nan]), pd.Series([np.nan])) == 0.0

    # Constant arrays force quantile edges fallback branch.
    ref = pd.Series([1.0] * 10)
    cur = pd.Series([2.0] * 10)
    value = psi_numeric(ref, cur, bins=5)
    assert isinstance(value, float)
    assert value >= 0.0


def test_psi_categorical_empty_and_prediction_drift_empty():
    assert (
        psi_categorical(pd.Series([], dtype=object), pd.Series([], dtype=object)) == 0.0
    )

    rep = prediction_drift_report(np.array([], dtype=float), np.array([], dtype=float))
    assert rep["ks_stat"] == 0.0
    assert rep["ref_mean_proba"] == 0.0
    assert rep["cur_mean_proba"] == 0.0


def test_outcome_monitoring_report_zero_rows():
    actions_df = pd.DataFrame({"proba": [], "action": []})
    outcome_df = pd.DataFrame({"target": []})
    rep = outcome_monitoring_report(
        actions_df,
        outcome_df,
        actual_col="target",
        policy={"cost_matrix": {}},
    )
    assert rep == {"n_rows": 0}
