from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.config import ValidationPolicy
from src.data_validation import (
    _js_divergence,
    _psi_score,
    assert_no_nans_after_imputation,
    check_data_staleness,
    compute_psi,
    detect_correlation_drift,
    detect_duplicates,
    detect_feature_importance_drift,
    detect_label_drift,
    detect_row_anomalies,
    detect_training_serving_skew,
    detect_unseen_categories,
    generate_reference_categories,
    generate_reference_correlations,
    generate_reference_stats,
    get_schema_fingerprint,
    run_validation_profile,
    validate_data_volume,
    validate_distributions,
    validate_inference_payload,
    validate_model_output,
    validate_processed_data,
    validate_row_counts,
)


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "is_canceled": [0, 1, 0, 1, 0, 1],
            "lead_time": [10, 20, 30, 40, 50, 60],
            "adr": [80.0, 85.0, 90.0, 95.0, 100.0, 105.0],
            "adults": [2, 2, 2, 2, 2, 2],
            "children": [0, 0, 0, 0, 0, 0],
            "babies": [0, 0, 0, 0, 0, 0],
            "stays_in_weekend_nights": [1, 1, 1, 1, 1, 1],
            "stays_in_week_nights": [2, 2, 2, 2, 2, 2],
            "hotel": ["City Hotel", "Resort Hotel", "City Hotel", "Resort Hotel", "City Hotel", "Resort Hotel"],
        }
    )


def test_detect_row_anomalies_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "adults": [0, 2, 60, 1, 1],
            "children": [0, 0, 0, 0, 0],
            "babies": [0, 0, 0, 0, 0],
            "adr": [50.0, -20.0, 90.0, 100.0, 100.0],
            "stays_in_weekend_nights": [1, 2, 3, 400, 1],
            "stays_in_week_nights": [1, 2, 3, 10, 1],
            "lead_time": [10, 20, 30, 40, 900],
        }
    )
    anomaly_report = detect_row_anomalies(df)
    assert anomaly_report.n_anomalies > 0
    assert "zero_guests" in anomaly_report.anomaly_types
    assert "negative_adr" in anomaly_report.anomaly_types
    assert "extreme_stay" in anomaly_report.anomaly_types
    assert "extreme_lead_time" in anomaly_report.anomaly_types
    assert "extreme_adults" in anomaly_report.anomaly_types

    dup_df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
    dup_report = detect_duplicates(dup_df)
    assert dup_report.n_duplicates == 1
    subset_report = detect_duplicates(dup_df, subset=["a"])
    assert subset_report.n_duplicates == 1


def test_assert_no_nans_after_imputation() -> None:
    df = pd.DataFrame({"a": [1, np.nan], "b": [1, 2], "c": [np.nan, np.nan]})
    counts = assert_no_nans_after_imputation(df, exclude_cols=["c"])
    assert counts == {"a": 1}


def test_label_and_category_checks() -> None:
    label_df = pd.DataFrame({"is_canceled": [1, 1, 1, 0, 1]})
    drift = detect_label_drift(label_df, target_col="is_canceled", ref_positive_rate=0.2, tolerance=0.1)
    assert drift.is_drifted is True

    cats_df = pd.DataFrame({"hotel": ["City Hotel", "Unknown"], "meal": ["BB", "HB"]})
    cat_report = detect_unseen_categories(cats_df, {"hotel": ["City Hotel"], "meal": ["BB"]})
    assert cat_report.n_unseen_total == 2
    assert "hotel" in cat_report.unseen


def test_validate_output_volume_and_row_counts() -> None:
    ok = validate_model_output(np.array([0.1, 0.5, 0.9]))
    assert ok.passed is True

    bad = validate_model_output(np.array([0.2, np.nan, 1.5]))
    assert bad.passed is False
    assert bad.n_nan == 1
    assert bad.n_out_of_range == 1

    vol_ok = validate_data_volume(_base_df(), expected_rows=6, tolerance_ratio=0.5)
    assert vol_ok.is_anomalous is False

    vol_bad = validate_data_volume(_base_df(), expected_rows=100, tolerance_ratio=0.2)
    assert vol_bad.is_anomalous is True

    row_ok = validate_row_counts(dataset_rows=100, train_rows=70, cal_rows=15, test_rows=15, tolerance=1)
    assert row_ok["passed"] is True
    row_bad = validate_row_counts(dataset_rows=100, train_rows=10, cal_rows=10, test_rows=10, tolerance=1)
    assert row_bad["passed"] is False


def test_check_data_staleness(tmp_path: Path) -> None:
    f = tmp_path / "data.parquet"
    f.write_text("x", encoding="utf-8")
    old_ts = time.time() - (120 * 24 * 3600)
    os.utime(f, (old_ts, old_ts))

    stale = check_data_staleness(str(f), max_age_days=90)
    assert stale.is_stale is True
    assert stale.file_modified is not None

    missing = check_data_staleness(str(tmp_path / "missing.parquet"), max_age_days=90)
    assert missing.file_modified is None


def test_schema_fingerprint_with_stats() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    fp1 = get_schema_fingerprint(df, include_stats=False)
    fp2 = get_schema_fingerprint(df, include_stats=True)
    assert fp1["n_columns"] == 2
    assert "n_rows" in fp2
    assert "fingerprint" in fp1


def test_distribution_processed_and_inference_error_paths() -> None:
    ref_stats = {"lead_time": {"mean": 100.0, "std": 10.0, "min": 0.0, "max": 200.0}}

    empty_series_df = pd.DataFrame({"lead_time": ["x", "y", None]})
    report_empty = validate_distributions(empty_series_df, ref_stats)
    assert report_empty.passed is False

    range_violation_df = pd.DataFrame({"lead_time": [-100, -50, -10]})
    report_range = validate_distributions(range_violation_df, ref_stats)
    assert report_range.passed is False

    bad_processed = pd.DataFrame({"is_canceled": ["bad", "nope"]})
    with pytest.raises(SchemaErrors):
        validate_processed_data(bad_processed, raise_on_error=True)

    payload = pd.DataFrame({"lead_time": [10], "extra_col": [1]})
    with pytest.raises(SchemaErrors):
        validate_inference_payload(
            payload,
            {"numeric": ["lead_time"], "categorical": []},
            raise_on_error=True,
            strict=True,
        )


def test_correlation_and_training_serving_skew() -> None:
    df_ref = _base_df()
    ref_corr = {"lead_time__adr": 0.0}
    df_cur = df_ref.copy()
    df_cur["adr"] = [200, 180, 160, 140, 120, 100]

    drift = detect_correlation_drift(
        df_cur,
        reference_corr=ref_corr,
        numeric_cols=["lead_time", "adr"],
        threshold=0.1,
    )
    assert drift.n_drifted >= 1

    not_enough = detect_correlation_drift(
        df_cur,
        reference_corr=ref_corr,
        numeric_cols=["lead_time"],
    )
    assert not_enough.n_drifted == 0

    ref_stats = generate_reference_stats(df_ref, ["lead_time", "adr"])
    skew = detect_training_serving_skew(
        pd.DataFrame({"lead_time": [1000, 1200, 900], "adr": [500, 550, 600]}),
        reference_stats=ref_stats,
        numeric_cols=["lead_time", "adr"],
        tolerance=1.0,
    )
    assert skew.n_skewed >= 1


def test_importance_categories_psi_and_js() -> None:
    scipy = pytest.importorskip("scipy")
    assert scipy is not None

    imp_report = detect_feature_importance_drift(
        current_importance={"a": 0.1, "b": 0.6, "c": 0.2, "d": 0.3},
        reference_importance={"a": 0.6, "b": 0.5, "c": 0.4, "d": 0.3},
        top_k=3,
        rank_drop_threshold=1,
    )
    assert imp_report.n_changed >= 1
    assert imp_report.rank_correlation is None or -1.0 <= imp_report.rank_correlation <= 1.0

    categories = generate_reference_categories(
        pd.DataFrame({"hotel": ["City Hotel", "Resort Hotel"], "meal": ["BB", "HB"]}),
        ["hotel", "meal", "missing"],
    )
    assert "hotel" in categories and "meal" in categories

    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 1.0, 1.0])
    assert _psi_score(a, b) == 0.0
    assert _js_divergence(a, b) == 0.0

    df_reference = pd.DataFrame({"lead_time": np.linspace(0, 1, 200), "adr": np.linspace(10, 20, 200)})
    df_current = pd.DataFrame({"lead_time": np.linspace(2, 3, 200), "adr": np.linspace(10, 20, 200)})

    psi_report = compute_psi(
        df_reference,
        df_current,
        numeric_cols=["lead_time", "adr"],
        warn_threshold=0.05,
        block_threshold=0.10,
        metric="psi",
        critical_columns=["lead_time"],
    )
    assert "lead_time" in psi_report.drift_cols

    js_report = compute_psi(
        df_reference,
        df_current,
        numeric_cols=["lead_time", "adr"],
        warn_threshold=0.01,
        block_threshold=0.05,
        metric="js",
        column_thresholds={"adr": 0.02},
    )
    assert isinstance(js_report.scores, dict)


def test_reference_correlations_generation() -> None:
    df = _base_df()
    corr = generate_reference_correlations(
        df=df,
        numeric_cols=["lead_time", "adr", "adults"],
        target_col="is_canceled",
        top_k=3,
    )
    assert len(corr) > 0


def test_run_validation_profile_default_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DS_ENV", "development")
    df = _base_df().drop_duplicates().reset_index(drop=True)
    ref_stats = generate_reference_stats(df, ["lead_time", "adr"])
    ref_stats["n_rows"] = len(df)
    ref_stats["label_positive_rate"] = float(df["is_canceled"].mean())

    report = run_validation_profile(
        df,
        target_col="is_canceled",
        numeric_cols=["lead_time", "adr"],
        reference_stats=ref_stats,
        reference_df=df,
        phase="monitor",
        policy=None,
    )
    assert isinstance(report.blocked_by, list)
    assert report.passed is True


def test_run_validation_profile_severity_buckets() -> None:
    policy = ValidationPolicy.for_env("staging")

    df_cur = pd.DataFrame(
        {
            "is_canceled": [1, 1, 1, 1, 0, 1],
            "lead_time": [900, 850, 900, 850, 900, 850],
            "adr": [300.0, 320.0, 300.0, 320.0, 300.0, 320.0],
            "adults": [0, 0, 2, 2, 2, 2],
            "children": [0, 0, 0, 0, 0, 0],
            "babies": [0, 0, 0, 0, 0, 0],
            "stays_in_weekend_nights": [1, 1, 1, 1, 1, 1],
            "stays_in_week_nights": [2, 2, 2, 2, 2, 2],
        }
    )
    df_ref = pd.DataFrame(
        {
            "is_canceled": [0, 0, 0, 1, 0, 0],
            "lead_time": [10, 20, 30, 40, 50, 60],
            "adr": [70.0, 75.0, 80.0, 85.0, 90.0, 95.0],
        }
    )

    ref_stats = generate_reference_stats(df_ref, ["lead_time", "adr"])
    ref_stats["n_rows"] = 1000
    ref_stats["label_positive_rate"] = float(df_ref["is_canceled"].mean())

    report = run_validation_profile(
        df_cur,
        target_col="is_canceled",
        numeric_cols=["lead_time", "adr"],
        reference_stats=ref_stats,
        reference_df=df_ref,
        policy=policy,
        phase="monitor",
    )
    assert report.passed is False
    assert "volume" in report.hard_failures
    assert len(report.soft_failures) >= 1
    assert isinstance(report.warnings, list)
