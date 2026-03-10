"""Anomaly and data-quality validation primitives.

This module contains row-level anomaly detection and data quality checks.
It is part of the official validation import surface: ``src.validation``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils import get_logger

logger = get_logger("validation.anomaly")


@dataclass
class AnomalyReport:
    """Row-level anomaly detection results."""

    n_anomalies: int
    anomaly_types: Dict[str, int]
    sample_indices: List[int]
    summary: str


def detect_row_anomalies(df: pd.DataFrame) -> AnomalyReport:
    """
    Domain-aware row-level anomaly detection for hotel bookings.

    Kurallar:
    - Sıfır misafir: adults=0 AND children=0 AND babies=0
    - Negatif ADR (< -10)
    - Aşırı konaklama (> 365 gece toplam)
    - Lead time > 800 (non-realistic)
    - adults > 50 (veri girişi hatası)
    """
    anomaly_flags: Dict[str, pd.Series] = {}

    # Sıfır misafir — mantıksal olarak imkansız kayıt
    if all(c in df.columns for c in ("adults", "children", "babies")):
        adults = pd.to_numeric(df["adults"], errors="coerce").fillna(0)
        children = pd.to_numeric(df["children"], errors="coerce").fillna(0)
        babies = pd.to_numeric(df["babies"], errors="coerce").fillna(0)
        anomaly_flags["zero_guests"] = (adults == 0) & (children == 0) & (babies == 0)

    # Negatif ADR
    if "adr" in df.columns:
        adr = pd.to_numeric(df["adr"], errors="coerce")
        anomaly_flags["negative_adr"] = adr < -10

    # Aşırı konaklama süresi
    if "stays_in_weekend_nights" in df.columns and "stays_in_week_nights" in df.columns:
        weekend = pd.to_numeric(df["stays_in_weekend_nights"], errors="coerce").fillna(
            0
        )
        week = pd.to_numeric(df["stays_in_week_nights"], errors="coerce").fillna(0)
        anomaly_flags["extreme_stay"] = (weekend + week) > 365

    # Aşırı lead time
    if "lead_time" in df.columns:
        lt = pd.to_numeric(df["lead_time"], errors="coerce")
        anomaly_flags["extreme_lead_time"] = lt > 800

    # Aşırı yetişkin sayısı
    if "adults" in df.columns:
        adults_v = pd.to_numeric(df["adults"], errors="coerce")
        anomaly_flags["extreme_adults"] = adults_v > 50

    any_anomaly = pd.Series(False, index=df.index)
    type_counts: Dict[str, int] = {}
    for name, mask in anomaly_flags.items():
        mask = mask.fillna(False)
        cnt = int(mask.sum())
        if cnt > 0:
            type_counts[name] = cnt
            any_anomaly = any_anomaly | mask

    n_total = int(any_anomaly.sum())
    sample = list(any_anomaly[any_anomaly].head(20).index)

    summary = (
        f"Row anomaly scan: {n_total} anomalous row(s) across "
        f"{len(type_counts)} type(s) in {len(df)} rows"
    )
    if type_counts:
        logger.warning(summary)
        for name, cnt in type_counts.items():
            logger.warning(f"  → {name}: {cnt} row(s)")
    else:
        logger.info(f"Row anomaly scan: 0 anomalies in {len(df)} rows ✅")

    return AnomalyReport(
        n_anomalies=n_total,
        anomaly_types=type_counts,
        sample_indices=sample,
        summary=summary,
    )


@dataclass
class DuplicateReport:
    """Duplicate row detection results."""

    n_duplicates: int
    n_total: int
    duplicate_ratio: float
    summary: str


def detect_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> DuplicateReport:
    """
    Tam veya kısmi duplicate satır tespiti.
    """
    dup_mask = df.duplicated(subset=subset, keep="first")
    n_dup = int(dup_mask.sum())
    ratio = n_dup / len(df) if len(df) > 0 else 0.0
    summary = f"Duplicates: {n_dup}/{len(df)} ({ratio:.2%})"

    if n_dup > 0:
        logger.warning(f"⚠️ {summary}")
    else:
        logger.info(f"✅ {summary}")

    return DuplicateReport(
        n_duplicates=n_dup,
        n_total=len(df),
        duplicate_ratio=ratio,
        summary=summary,
    )


def assert_no_nans_after_imputation(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    İmputation sonrası hiç NaN kalmamalı.
    Kalan varsa uyarı loglar ve sütun→count dict döner.
    """
    exclude = set(exclude_cols or [])
    cols = [c for c in df.columns if c not in exclude]
    nan_counts: Dict[str, int] = {}
    for c in cols:
        n = int(df[c].isna().sum())
        if n > 0:
            nan_counts[c] = n

    if nan_counts:
        logger.warning(
            f"⚠️ Post-imputation NaN found in {len(nan_counts)} column(s): {nan_counts}"
        )
    else:
        logger.info(
            f"✅ Post-imputation NaN check passed: 0 NaN in {len(cols)} columns"
        )

    return nan_counts


@dataclass
class CardinalityReport:
    """Unseen category detection results."""

    unseen: Dict[str, List[str]]
    n_unseen_total: int
    summary: str


def detect_unseen_categories(
    df: pd.DataFrame,
    reference_categories: Dict[str, List[str]],
) -> CardinalityReport:
    """
    Eğitimde görmediğimiz yeni kategori değeri geldiğinde tespit.

    reference_categories: {"hotel": ["Resort Hotel", "City Hotel"], ...}
    """
    unseen: Dict[str, List[str]] = {}
    total = 0

    for col, known in reference_categories.items():
        if col not in df.columns:
            continue
        current_cats = set(df[col].dropna().astype(str).unique())
        known_set = set(str(v) for v in known)
        new_cats = sorted(current_cats - known_set)
        if new_cats:
            unseen[col] = new_cats
            total += len(new_cats)
            logger.warning(f"⚠️ Unseen categories in '{col}': {new_cats}")

    summary = f"Cardinality check: {total} unseen category value(s) in {len(unseen)} column(s)"
    if total == 0:
        logger.info(f"✅ {summary}")

    return CardinalityReport(unseen=unseen, n_unseen_total=total, summary=summary)


@dataclass
class OutputValidationReport:
    """Model output validation results."""

    n_rows: int
    n_out_of_range: int
    n_nan: int
    min_val: float
    max_val: float
    passed: bool
    summary: str


def validate_model_output(
    proba: np.ndarray,
    tolerance: float = 1e-6,
) -> OutputValidationReport:
    """
    Model predict_proba çıktısının [0,1] aralığında ve NaN-free olduğunu doğrula.
    Kalibrasyon sonrası bozuk olasılık üretilebilir.
    """
    arr = np.asarray(proba, dtype=float)
    n_nan = int(np.isnan(arr).sum())
    valid = arr[~np.isnan(arr)]
    n_out = int(np.sum((valid < -tolerance) | (valid > 1.0 + tolerance)))
    min_v = float(np.min(valid)) if len(valid) > 0 else float("nan")
    max_v = float(np.max(valid)) if len(valid) > 0 else float("nan")
    passed = n_nan == 0 and n_out == 0

    summary = (
        f"Model output validation: n={len(arr)}, "
        f"NaN={n_nan}, out_of_range={n_out}, "
        f"min={min_v:.6f}, max={max_v:.6f} → "
        f"{'PASS ✅' if passed else 'FAIL ⚠️'}"
    )
    if not passed:
        logger.warning(summary)
    else:
        logger.info(summary)

    return OutputValidationReport(
        n_rows=len(arr),
        n_out_of_range=n_out,
        n_nan=n_nan,
        min_val=min_v,
        max_val=max_v,
        passed=passed,
        summary=summary,
    )


@dataclass
class VolumeReport:
    """Data volume anomaly results."""

    current_rows: int
    expected_range: Tuple[int, int]
    is_anomalous: bool
    summary: str


def validate_data_volume(
    df: pd.DataFrame,
    expected_rows: int,
    tolerance_ratio: float = 0.5,
) -> VolumeReport:
    """
    Gelen veri satır sayısı normalin çok altında veya üstündeyse uyarı.

    expected_rows: Referans satır sayısı (ör: eğitim verisinden)
    tolerance_ratio: ±%50 varsayılan
    """
    lo = max(1, int(expected_rows * (1 - tolerance_ratio)))
    hi = int(expected_rows * (1 + tolerance_ratio))
    n = len(df)
    is_anom = n < lo or n > hi

    summary = (
        f"Data volume: {n} rows (expected [{lo}, {hi}]) → "
        f"{'ANOMALOUS ⚠️' if is_anom else 'OK ✅'}"
    )
    if is_anom:
        logger.warning(summary)
    else:
        logger.info(summary)

    return VolumeReport(
        current_rows=n,
        expected_range=(lo, hi),
        is_anomalous=is_anom,
        summary=summary,
    )


@dataclass
class StalenessReport:
    """Data staleness results."""

    file_modified: Optional[str]
    age_days: Optional[float]
    is_stale: bool
    summary: str


def check_data_staleness(
    file_path: str,
    max_age_days: float = 90.0,
) -> StalenessReport:
    """
    Ham veri dosyasının yaşını kontrol et.
    """
    try:
        mtime = os.path.getmtime(file_path)
        mod_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        age = (datetime.now(timezone.utc) - mod_dt).total_seconds() / 86400
        is_stale = age > max_age_days

        summary = (
            f"Data staleness: {file_path} is {age:.1f} days old "
            f"(max={max_age_days}) → "
            f"{'STALE ⚠️' if is_stale else 'FRESH ✅'}"
        )
        if is_stale:
            logger.warning(summary)
        else:
            logger.info(summary)

        return StalenessReport(
            file_modified=mod_dt.isoformat(),
            age_days=age,
            is_stale=is_stale,
            summary=summary,
        )
    except Exception as e:
        summary = f"Data staleness: cannot check {file_path} — {e}"
        logger.warning(summary)
        return StalenessReport(
            file_modified=None,
            age_days=None,
            is_stale=False,
            summary=summary,
        )


def validate_row_counts(
    dataset_rows: int,
    train_rows: int,
    cal_rows: int,
    test_rows: int,
    tolerance: int = 5,
) -> Dict[str, Any]:
    """
    İşlenmiş veri → split sonrası satır sayısı tutarlılığı.
    """
    split_total = train_rows + cal_rows + test_rows
    diff = abs(dataset_rows - split_total)
    passed = diff <= tolerance

    result = {
        "dataset_rows": dataset_rows,
        "split_total": split_total,
        "train": train_rows,
        "cal": cal_rows,
        "test": test_rows,
        "difference": diff,
        "passed": passed,
    }

    if not passed:
        logger.warning(
            f"⚠️ Row count mismatch: dataset={dataset_rows} vs splits={split_total} (Δ={diff})"
        )
    else:
        logger.info(
            f"✅ Row count consistent: dataset={dataset_rows} = splits={split_total}"
        )

    return result


__all__ = [
    "AnomalyReport",
    "detect_row_anomalies",
    "DuplicateReport",
    "detect_duplicates",
    "assert_no_nans_after_imputation",
    "CardinalityReport",
    "detect_unseen_categories",
    "OutputValidationReport",
    "validate_model_output",
    "VolumeReport",
    "validate_data_volume",
    "StalenessReport",
    "check_data_staleness",
    "validate_row_counts",
]
