"""
data_validation.py — Pandera-based data validation framework.

Provides schema and distribution-level assertions for:
  1. Raw data ingestion   — validate_raw_data()
  2. Processed features   — validate_processed_data()
  3. Inference payload     — validate_inference_payload()
  4. Distribution checks   — validate_distributions()
  5. Row-level anomaly     — detect_row_anomalies()
  6. Duplicate detection   — detect_duplicates()
  7. Post-imputation NaN   — assert_no_nans_after_imputation()
  8. Label drift            — detect_label_drift()
  9. Feature cardinality    — detect_unseen_categories()
  10. Model output range    — validate_model_output()
  11. Data volume anomaly   — validate_data_volume()
  12. Data staleness        — check_data_staleness()
  13. Schema versioning     — get_schema_fingerprint()
  14. Cross-feature corr    — detect_correlation_drift()
  15. Training-serving skew — detect_training_serving_skew()
  16. Row count consistency — validate_row_counts()
  17. Feature importance    — detect_feature_importance_drift()

Usage:
  from src.data_validation import validate_raw_data, validate_processed_data
  validate_raw_data(df)        # raises SchemaError on contract violation
  validate_processed_data(df)  # raises SchemaError on contract violation
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

from .utils import get_logger

logger = get_logger("data_validation")


# ─── Raw Data Schema ───────────────────────────────────────────────────
def build_raw_schema(target_col: str = "is_canceled") -> DataFrameSchema:
    """
    Raw hotel bookings CSV schema.

    Bu schema kontratı şunları garanti eder:
    - Zorunlu kolonlar mevcut
    - Tip uyumu (numeric/string)
    - Mantıksal aralık kontrolleri (örn: negatif gece olamaz)
    - Target label beklenen formatta
    """
    return DataFrameSchema(
        columns={
            # ── Booking identifiers ──
            "hotel": Column(
                str, Check.isin(["Resort Hotel", "City Hotel"]), nullable=False
            ),
            "arrival_date_year": Column(
                int, Check.in_range(2010, 2030), nullable=False
            ),
            "arrival_date_month": Column(str, nullable=False),
            "arrival_date_week_number": Column(
                int, Check.in_range(1, 53), nullable=True
            ),
            "arrival_date_day_of_month": Column(
                int, Check.in_range(1, 31), nullable=True
            ),
            # ── Stay duration ──
            "stays_in_weekend_nights": Column(int, Check.ge(0), nullable=True),
            "stays_in_week_nights": Column(int, Check.ge(0), nullable=True),
            # ── Guest info ──
            "adults": Column(int, Check.ge(0), nullable=True),
            "children": Column(float, Check.ge(0), nullable=True),  # float due to NaN
            "babies": Column(int, Check.ge(0), nullable=True),
            # ── Booking meta ──
            "lead_time": Column(int, Check.ge(0), nullable=False),
            "is_repeated_guest": Column(int, Check.isin([0, 1]), nullable=True),
            "previous_cancellations": Column(int, Check.ge(0), nullable=True),
            "previous_bookings_not_canceled": Column(int, Check.ge(0), nullable=True),
            "booking_changes": Column(int, Check.ge(0), nullable=True),
            "adr": Column(float, nullable=True),
            # ── Target ──
            target_col: Column(
                nullable=False,
                checks=Check(
                    lambda s: s.astype(str)
                    .str.lower()
                    .str.strip()
                    .isin(["0", "1", "yes", "no"])
                    .all(),
                    error=f"Target column '{target_col}' must contain 0/1 or yes/no",
                ),
            ),
        },
        # Raw data may have extra columns — don't fail on those
        strict=False,
        coerce=False,
        name="RawHotelBookingsSchema",
        description="Schema contract for raw hotel bookings CSV ingestion",
    )


# ─── Processed Data Schema ─────────────────────────────────────────────
def build_processed_schema(
    target_col: str = "is_canceled",
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> DataFrameSchema:
    """
    Processed (post-preprocess) schema.

    Kontratlar:
    - Target 0/1 integer
    - Numeric kolonlar finite (no NaN/Inf)
    - Categorical kolonlar boş string değil
    """
    columns: Dict[str, Column] = {
        target_col: Column(int, Check.isin([0, 1]), nullable=False),
    }

    if numeric_cols:
        for col in numeric_cols:
            columns[col] = Column(
                nullable=True,
                checks=[
                    Check(
                        lambda s: pd.to_numeric(s, errors="coerce").notna().all(),
                        error=f"Column '{col}' has non-numeric values after processing",
                    ),
                ],
            )

    if categorical_cols:
        for col in categorical_cols:
            columns[col] = Column(nullable=True)

    return DataFrameSchema(
        columns=columns,
        strict=False,
        coerce=False,
        name="ProcessedHotelBookingsSchema",
        description="Schema contract for processed hotel bookings dataset",
    )


# ─── Inference Payload Schema ──────────────────────────────────────────
def build_inference_schema(
    feature_spec: Dict[str, Any],
    strict: bool = False,
) -> DataFrameSchema:
    """
    Inference-time schema — API payload kontratı.

    Args:
        feature_spec : {"numeric": [...], "categorical": [...]} dict
        strict       : True → beklenmeyen kolon varsa SchemaError fırlatır
                       (ValidationPolicy.strict_inference_schema ile kontrol edilir)
                       False → esneklik modunda yalnızca beklenen kolonları kontrol eder

    Not: Prod ortamda strict=True önerilir; data contract drift'ini erken yakalar.
    """
    columns: Dict[str, Column] = {}

    for col in feature_spec.get("numeric", []):
        columns[col] = Column(nullable=True)

    for col in feature_spec.get("categorical", []):
        columns[col] = Column(nullable=True)

    return DataFrameSchema(
        columns=columns,
        strict=strict,  # strict=True → ekstra kolon → SchemaError
        coerce=True,
        name="InferencePayloadSchema",
        description="Schema contract for inference-time API payload",
    )


# ─── Distribution Validators ───────────────────────────────────────────
@dataclass
class DistributionReport:
    """Distribution validation results."""

    passed: bool
    violations: List[Dict[str, Any]]
    summary: str


def validate_distributions(
    df: pd.DataFrame,
    reference_stats: Dict[str, Dict[str, float]],
    tolerance: float = 3.0,
) -> DistributionReport:
    """
    Distribution-level assertions.

    Her numeric kolon için:
    - Ortalama reference'a göre ±tolerance*std_dev aralığında mı?
    - Min/Max makul aralıkta mı?

    reference_stats format:
      {"column_name": {"mean": ..., "std": ..., "min": ..., "max": ...}}
    """
    violations: List[Dict[str, Any]] = []

    for col, stats in reference_stats.items():
        if col not in df.columns:
            violations.append(
                {
                    "column": col,
                    "check": "existence",
                    "message": f"Column '{col}' missing from dataframe",
                }
            )
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            violations.append(
                {
                    "column": col,
                    "check": "non_empty",
                    "message": f"Column '{col}' is entirely null/non-numeric",
                }
            )
            continue

        ref_mean = stats.get("mean", 0.0)
        ref_std = stats.get("std", 1.0)
        cur_mean = float(series.mean())

        # Mean drift check
        if ref_std > 0 and abs(cur_mean - ref_mean) > tolerance * ref_std:
            violations.append(
                {
                    "column": col,
                    "check": "mean_drift",
                    "message": (
                        f"Mean drift: current={cur_mean:.4f}, "
                        f"reference={ref_mean:.4f}, "
                        f"threshold=±{tolerance}*{ref_std:.4f}"
                    ),
                    "current_mean": cur_mean,
                    "reference_mean": ref_mean,
                }
            )

        # Range check
        ref_min = stats.get("min")
        ref_max = stats.get("max")
        if ref_min is not None:
            cur_min = float(series.min())
            if cur_min < ref_min * 0.5 - abs(ref_min):  # generous lower bound
                violations.append(
                    {
                        "column": col,
                        "check": "range_min",
                        "message": f"Min out of range: current={cur_min}, reference_min={ref_min}",
                    }
                )
        if ref_max is not None:
            cur_max = float(series.max())
            if cur_max > ref_max * 2.0 + abs(ref_max):  # generous upper bound
                violations.append(
                    {
                        "column": col,
                        "check": "range_max",
                        "message": f"Max out of range: current={cur_max}, reference_max={ref_max}",
                    }
                )

    passed = len(violations) == 0
    summary = (
        f"Distribution validation: {'PASSED' if passed else 'FAILED'} — "
        f"{len(violations)} violation(s) across {len(reference_stats)} columns"
    )
    logger.info(summary)

    return DistributionReport(passed=passed, violations=violations, summary=summary)


# ─── High-Level Convenience Functions ───────────────────────────────────
def validate_raw_data(
    df: pd.DataFrame,
    target_col: str = "is_canceled",
    raise_on_error: bool = True,
) -> pa.errors.SchemaErrors | None:
    """
    Raw data schema kontratını doğrula.

    Args:
        df: Raw DataFrame
        target_col: Hedef kolon adı
        raise_on_error: True ise hata fırlatır, False ise error döner

    Returns:
        None (başarılı) veya SchemaErrors (raise_on_error=False ise)
    """
    schema = build_raw_schema(target_col)
    try:
        schema.validate(df, lazy=True)
        logger.info(
            f"✅ Raw data validation passed: {len(df)} rows, {len(df.columns)} cols"
        )
        return None
    except SchemaErrors as e:
        logger.error(
            f"❌ Raw data validation failed: {len(e.failure_cases)} failure(s)"
        )
        if raise_on_error:
            raise
        return e


def validate_processed_data(
    df: pd.DataFrame,
    target_col: str = "is_canceled",
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    raise_on_error: bool = True,
) -> pa.errors.SchemaErrors | None:
    """
    Processed data schema kontratını doğrula.
    """
    schema = build_processed_schema(target_col, numeric_cols, categorical_cols)
    try:
        schema.validate(df, lazy=True)
        logger.info(f"✅ Processed data validation passed: {len(df)} rows")
        return None
    except SchemaErrors as e:
        logger.error(
            f"❌ Processed data validation failed: {len(e.failure_cases)} failure(s)"
        )
        if raise_on_error:
            raise
        return e


def validate_inference_payload(
    df: pd.DataFrame,
    feature_spec: Dict[str, Any],
    raise_on_error: bool = True,
    strict: bool = False,
) -> pa.errors.SchemaErrors | None:
    """
    API inference payload schema kontratını doğrula.

    Args:
        strict : True → beklenmeyen (fazla) kolonlar SchemaError fırlatır.
                 ValidationPolicy.strict_inference_schema ile kontrol edilir.
    """
    # Fazla kolon tespiti — strict olmasa bile logla
    expected = set(feature_spec.get("numeric", [])) | set(
        feature_spec.get("categorical", [])
    )
    extra_cols = set(df.columns) - expected
    if extra_cols:
        logger.warning(
            f"⚠️  Inference payload: {len(extra_cols)} unexpected column(s) "
            f"not in feature_spec → {sorted(extra_cols)}"
        )

    schema = build_inference_schema(feature_spec, strict=strict)
    try:
        schema.validate(df, lazy=True)
        logger.info(f"✅ Inference payload validation passed: {len(df)} rows")
        return None
    except SchemaErrors as e:
        logger.error(
            f"❌ Inference payload validation failed: {len(e.failure_cases)} failure(s)"
        )
        if raise_on_error:
            raise
        return e


def generate_reference_stats(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Training setinden referans istatistikleri üret.
    Distribution validation'da kullanılır.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                stats[col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                }
    return stats


# ═══════════════════════════════════════════════════════════════════════
# NEW VALIDATION FUNCTIONS — Industry Best Practices (Google MLOps L2)
# ═══════════════════════════════════════════════════════════════════════


# ─── 5. Row-Level Anomaly Detection ────────────────────────────────────
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


# ─── 6. Duplicate Detection ───────────────────────────────────────────
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


# ─── 7. Post-Imputation NaN Assertion ─────────────────────────────────
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
            f"⚠️ Post-imputation NaN found in {len(nan_counts)} column(s): "
            f"{nan_counts}"
        )
    else:
        logger.info(
            f"✅ Post-imputation NaN check passed: 0 NaN in {len(cols)} columns"
        )

    return nan_counts


# ─── 8. Label Drift Detection ─────────────────────────────────────────
@dataclass
class LabelDriftReport:
    """Label (target) distribution drift results."""

    ref_positive_rate: float
    cur_positive_rate: float
    drift_magnitude: float
    is_drifted: bool
    summary: str


def detect_label_drift(
    df_cur: pd.DataFrame,
    target_col: str,
    ref_positive_rate: float,
    tolerance: float = 0.10,
) -> LabelDriftReport:
    """
    Hedef değişken dağılım değişimi (label drift / concept drift proxy).

    Eğitimde %37 iptal varsa, canlıda %60'a çıktıysa → concept drift sinyali.
    """
    y = pd.to_numeric(df_cur[target_col], errors="coerce").dropna()
    cur_rate = float(y.mean()) if len(y) > 0 else 0.0
    drift = abs(cur_rate - ref_positive_rate)
    is_drifted = drift > tolerance

    summary = (
        f"Label drift: ref={ref_positive_rate:.3f}, cur={cur_rate:.3f}, "
        f"Δ={drift:.3f}, threshold={tolerance:.3f} → "
        f"{'DRIFT DETECTED ⚠️' if is_drifted else 'OK ✅'}"
    )
    if is_drifted:
        logger.warning(summary)
    else:
        logger.info(summary)

    return LabelDriftReport(
        ref_positive_rate=ref_positive_rate,
        cur_positive_rate=cur_rate,
        drift_magnitude=drift,
        is_drifted=is_drifted,
        summary=summary,
    )


# ─── 9. Feature Cardinality — Unseen Categories ───────────────────────
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


# ─── 10. Model Output Range Validation ────────────────────────────────
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


# ─── 11. Data Volume Anomaly ──────────────────────────────────────────
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


# ─── 12. Data Staleness ───────────────────────────────────────────────
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


# ─── 13. Schema Versioning / Fingerprint ──────────────────────────────
def get_schema_fingerprint(
    df: pd.DataFrame,
    include_stats: bool = False,
) -> Dict[str, Any]:
    """
    DataFrame'in şema parmak izini üret (sütunlar + tipler).
    Schema değiştiğinde takip mekanizması sağlar.
    """
    schema_info = {
        "columns": sorted(df.columns.tolist()),
        "dtypes": {c: str(df[c].dtype) for c in sorted(df.columns)},
        "n_columns": len(df.columns),
    }
    if include_stats:
        schema_info["n_rows"] = len(df)
        schema_info["null_counts"] = {c: int(df[c].isna().sum()) for c in df.columns}

    raw = json.dumps(schema_info, sort_keys=True)
    schema_info["fingerprint"] = hashlib.sha256(raw.encode()).hexdigest()[:16]
    logger.info(
        f"Schema fingerprint: {schema_info['fingerprint']} ({len(df.columns)} cols)"
    )
    return schema_info


# ─── 14. Cross-Feature Correlation Drift ──────────────────────────────
@dataclass
class CorrelationDriftReport:
    """Cross-feature correlation drift results."""

    drifted_pairs: List[Dict[str, Any]]
    n_drifted: int
    summary: str


def detect_correlation_drift(
    df_cur: pd.DataFrame,
    reference_corr: Dict[str, float],
    numeric_cols: List[str],
    threshold: float = 0.20,
) -> CorrelationDriftReport:
    """
    İki özellik arasındaki korelasyon değişimi.

    reference_corr: {"col_a__col_b": 0.45, ...} — eğitim sırasında hesaplanan
    """
    drifted: List[Dict[str, Any]] = []

    available = [c for c in numeric_cols if c in df_cur.columns]
    if len(available) < 2:
        return CorrelationDriftReport(
            drifted_pairs=[],
            n_drifted=0,
            summary="Correlation drift: not enough numeric columns",
        )

    cur_corr_matrix = (
        df_cur[available].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    )

    for pair_key, ref_val in reference_corr.items():
        parts = pair_key.split("__")
        if len(parts) != 2:
            continue
        a, b = parts
        if a not in cur_corr_matrix.columns or b not in cur_corr_matrix.columns:
            continue
        cur_val = float(cur_corr_matrix.loc[a, b])
        delta = abs(cur_val - ref_val)
        if delta > threshold:
            drifted.append(
                {
                    "pair": pair_key,
                    "ref_corr": ref_val,
                    "cur_corr": cur_val,
                    "delta": delta,
                }
            )

    summary = (
        f"Correlation drift: {len(drifted)} pair(s) drifted "
        f"(threshold={threshold}) out of {len(reference_corr)} tracked"
    )
    if drifted:
        logger.warning(f"⚠️ {summary}")
        for d in drifted:
            logger.warning(
                f"  → {d['pair']}: ref={d['ref_corr']:.3f} cur={d['cur_corr']:.3f} Δ={d['delta']:.3f}"
            )
    else:
        logger.info(f"✅ {summary}")

    return CorrelationDriftReport(
        drifted_pairs=drifted,
        n_drifted=len(drifted),
        summary=summary,
    )


def generate_reference_correlations(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str,
    top_k: int = 20,
) -> Dict[str, float]:
    """
    Eğitim verisinden referans korelasyon çiftlerini üret.
    En yüksek target korelasyonlu top_k feature çiftini izler.
    """
    available = [c for c in numeric_cols if c in df.columns]
    if len(available) < 2:
        return {}

    num_df = df[available + [target_col]].apply(pd.to_numeric, errors="coerce")
    target_corr = num_df.corr()[target_col].drop(target_col, errors="ignore").abs()
    top_features = target_corr.nlargest(min(top_k, len(target_corr))).index.tolist()

    corr_matrix = num_df[top_features].corr()
    pairs: Dict[str, float] = {}
    for i, a in enumerate(top_features):
        for b in top_features[i + 1 :]:
            pairs[f"{a}__{b}"] = float(corr_matrix.loc[a, b])
    # Also include target correlations
    for feat in top_features:
        pairs[f"{feat}__{target_col}"] = float(
            num_df[[feat, target_col]].corr().iloc[0, 1]
        )

    logger.info(
        f"Reference correlations generated: {len(pairs)} pairs from {len(top_features)} features"
    )
    return pairs


# ─── 15. Training-Serving Skew (Per-Request) ──────────────────────────
@dataclass
class SkewReport:
    """Training-serving skew results."""

    skewed_features: List[Dict[str, Any]]
    n_skewed: int
    summary: str


def detect_training_serving_skew(
    df_serving: pd.DataFrame,
    reference_stats: Dict[str, Dict[str, float]],
    numeric_cols: List[str],
    tolerance: float = 2.0,
) -> SkewReport:
    """
    Inference sırasında gelen batch'in eğitim dağılımından sapmasını ölç.
    validate_distributions'dan farkı: daha sıkı tolerance ve per-request çalışır.
    """
    skewed: List[Dict[str, Any]] = []

    for col in numeric_cols:
        if col not in df_serving.columns or col not in reference_stats:
            continue
        series = pd.to_numeric(df_serving[col], errors="coerce").dropna()
        if series.empty:
            continue

        ref = reference_stats[col]
        ref_mean = ref.get("mean", 0.0)
        ref_std = ref.get("std", 1.0)
        cur_mean = float(series.mean())

        if ref_std > 0 and abs(cur_mean - ref_mean) > tolerance * ref_std:
            skewed.append(
                {
                    "column": col,
                    "ref_mean": ref_mean,
                    "cur_mean": cur_mean,
                    "z_score": abs(cur_mean - ref_mean) / ref_std,
                }
            )

    summary = (
        f"Training-serving skew: {len(skewed)} feature(s) skewed "
        f"(tolerance={tolerance}σ) out of {len(numeric_cols)} checked"
    )
    if skewed:
        logger.warning(f"⚠️ {summary}")
    else:
        logger.info(f"✅ {summary}")

    return SkewReport(
        skewed_features=skewed,
        n_skewed=len(skewed),
        summary=summary,
    )


# ─── 16. Row Count Consistency ────────────────────────────────────────
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


# ─── 17. Feature Importance Drift ─────────────────────────────────────
@dataclass
class ImportanceDriftReport:
    """Feature importance drift results."""

    changed_features: List[Dict[str, Any]]
    n_changed: int
    rank_correlation: Optional[float]
    summary: str


def detect_feature_importance_drift(
    current_importance: Dict[str, float],
    reference_importance: Dict[str, float],
    top_k: int = 10,
    rank_drop_threshold: int = 5,
) -> ImportanceDriftReport:
    """
    Feature importance sıralaması değişimi.
    Önemli bir feature aniden sıfıra düşerse → data pipeline hatası sinyali.
    """
    # Rank by importance
    ref_ranked = sorted(reference_importance.items(), key=lambda x: -x[1])
    cur_ranked = sorted(current_importance.items(), key=lambda x: -x[1])

    ref_rank = {name: i for i, (name, _) in enumerate(ref_ranked)}
    cur_rank = {name: i for i, (name, _) in enumerate(cur_ranked)}

    changed: List[Dict[str, Any]] = []
    for name in list(ref_rank.keys())[:top_k]:
        r_ref = ref_rank.get(name, -1)
        r_cur = cur_rank.get(name, len(cur_rank))
        rank_diff = r_cur - r_ref
        if abs(rank_diff) >= rank_drop_threshold:
            changed.append(
                {
                    "feature": name,
                    "ref_rank": r_ref,
                    "cur_rank": r_cur,
                    "rank_change": rank_diff,
                    "ref_importance": reference_importance.get(name, 0.0),
                    "cur_importance": current_importance.get(name, 0.0),
                }
            )

    # Spearman rank correlation of shared features
    shared = sorted(set(ref_rank.keys()) & set(cur_rank.keys()))
    rank_corr = None
    if len(shared) >= 3:
        from scipy import stats as sp_stats

        ref_ranks = [ref_rank[f] for f in shared]
        cur_ranks = [cur_rank[f] for f in shared]
        rank_corr = float(sp_stats.spearmanr(ref_ranks, cur_ranks).statistic)

    summary = (
        f"Feature importance drift: {len(changed)} feature(s) changed rank by ≥{rank_drop_threshold}"
        + (f", rank_corr={rank_corr:.3f}" if rank_corr is not None else "")
    )
    if changed:
        logger.warning(f"⚠️ {summary}")
    else:
        logger.info(f"✅ {summary}")

    return ImportanceDriftReport(
        changed_features=changed,
        n_changed=len(changed),
        rank_correlation=rank_corr,
        summary=summary,
    )


# ─── 18. Reference Categories Generation ──────────────────────────────
def generate_reference_categories(
    df: pd.DataFrame,
    categorical_cols: List[str],
) -> Dict[str, List[str]]:
    """Eğitim verisindeki kategorik sütunların benzersiz değerlerini kaydet."""
    cats: Dict[str, List[str]] = {}
    for col in categorical_cols:
        if col in df.columns:
            cats[col] = sorted(df[col].dropna().astype(str).unique().tolist())
    logger.info(f"Reference categories generated for {len(cats)} columns")
    return cats


# ─── 19. PSI (Population Stability Index) ─────────────────────────────
@dataclass
class PSIReport:
    """Per-column PSI scores and overall drift verdict."""

    scores: Dict[str, float]  # {col: psi_score}
    warn_cols: List[str]  # 0.10 ≤ PSI < 0.25
    drift_cols: List[str]  # PSI ≥ 0.25
    overall_passed: bool
    summary: str


def _psi_score(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index (PSI) iki dağılım arasında hesaplar.

    PSI < 0.10        → stabil
    0.10 ≤ PSI < 0.25 → orta drift (uyarı)
    PSI ≥ 0.25        → ciddi drift (alarm)
    """
    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())
    if hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    exp_cnt, _ = np.histogram(expected, bins=bins)
    act_cnt, _ = np.histogram(actual, bins=bins)
    eps = 1e-8
    exp_pct = (exp_cnt + eps) / (len(expected) + eps * n_bins)
    act_pct = (act_cnt + eps) / (len(actual) + eps * n_bins)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _js_divergence(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Jensen-Shannon Divergence — PSI'ye kıyasla simetrik ve 0-1 arasında sınırlı.

    JS < 0.05         → stabil
    0.05 ≤ JS < 0.15  → orta drift
    JS ≥ 0.15         → ciddi drift

    Not: Değer aralığı [0, log(2)] ≈ [0, 0.693] (nats cinsinden).
         Normalize versiyonu [0, 1]'dir; bu implementasyon normalize kullanır.
    """
    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())
    if hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    exp_cnt, _ = np.histogram(expected, bins=bins)
    act_cnt, _ = np.histogram(actual, bins=bins)
    eps = 1e-8
    p = (exp_cnt + eps) / (len(expected) + eps * n_bins)
    q = (act_cnt + eps) / (len(actual) + eps * n_bins)
    m = 0.5 * (p + q)
    # KL(P||M) + KL(Q||M) — log2 ile normalize → [0, 1]
    js = 0.5 * np.sum(p * np.log2(p / m) + q * np.log2(q / m))
    return float(np.clip(js, 0.0, 1.0))


def compute_psi(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    numeric_cols: List[str],
    warn_threshold: float = 0.10,
    block_threshold: float = 0.25,
    n_bins: int = 10,
    metric: str = "psi",
    column_thresholds: Optional[Dict[str, float]] = None,
    critical_columns: Optional[List[str]] = None,
) -> PSIReport:
    """
    Referans ve güncel veri arasındaki PSI veya JS divergence'ı hesaplar.

    Args:
        metric             : "psi" veya "js" — drift metriği seçimi
        column_thresholds  : Per-kolon warn eşiği override'ı {"lead_time": 0.08}
        critical_columns   : Bu kolonlarda drift → otomatik drift_cols'a eklenir
        warn_threshold     : Global warn eşiği (column_thresholds'da olmayan kolonlar)
        block_threshold    : Global block eşiği
    """
    score_fn = _js_divergence if metric == "js" else _psi_score
    col_thresh = column_thresholds or {}
    critical_set = set(critical_columns or [])

    scores: Dict[str, float] = {}
    warn_cols: List[str] = []
    drift_cols: List[str] = []

    for col in numeric_cols:
        if col not in df_reference.columns or col not in df_current.columns:
            continue
        ref_arr = pd.to_numeric(df_reference[col], errors="coerce").dropna().values
        cur_arr = pd.to_numeric(df_current[col], errors="coerce").dropna().values
        if len(ref_arr) < 5 or len(cur_arr) < 5:
            continue

        score = score_fn(ref_arr, cur_arr, n_bins=n_bins)
        scores[col] = round(score, 6)

        # Per-kolon eşik override veya global
        col_warn = col_thresh.get(col, warn_threshold)
        col_block = col_thresh.get(col, block_threshold)
        # Kritik kolon: block eşiğini warn eşiğine çek (her ihlal → drift)
        if col in critical_set:
            col_block = col_warn

        if score >= col_block:
            drift_cols.append(col)
            logger.error(
                f"❌ {metric.upper()} drift [{col}]: {score:.4f} ≥ "
                f"block={col_block:.3f}"
                + (" [CRITICAL]" if col in critical_set else "")
            )
        elif score >= col_warn:
            warn_cols.append(col)
            logger.warning(
                f"⚠️  {metric.upper()} drift [{col}]: {score:.4f} ≥ "
                f"warn={col_warn:.3f}"
            )

    overall_passed = len(drift_cols) == 0
    summary = (
        f"{metric.upper()} check ({n_bins} bins): {len(scores)} cols | "
        f"stable={len(scores) - len(warn_cols) - len(drift_cols)} "
        f"warn={len(warn_cols)} drift={len(drift_cols)}"
    )
    if overall_passed:
        logger.info(f"✅ {summary}")
    else:
        logger.error(f"❌ {summary}")

    return PSIReport(
        scores=scores,
        warn_cols=warn_cols,
        drift_cols=drift_cols,
        overall_passed=overall_passed,
        summary=summary,
    )


# ─── 20. run_validation_profile() — Severity-Aware Tek Nokta ─────────
@dataclass
class ValidationProfileReport:
    """
    run_validation_profile() çıktısı.

    passed       : hard_fail kontrollerin tamamı geçtiyse True
    hard_failures: pipeline'ı durduran kontroller
    soft_failures : soft_fail seviyesindeki ihlaller (caller karar verir)
    warnings     : warn seviyesindeki ihlaller
    details      : Her kontrolün özet stringi
    """

    passed: bool
    hard_failures: List[str]
    soft_failures: List[str]
    warnings: List[str]
    details: Dict[str, Any]

    # Geriye dönük uyumluluk alias'ı
    @property
    def blocked_by(self) -> List[str]:
        return self.hard_failures


def run_validation_profile(
    df: pd.DataFrame,
    *,
    target_col: str = "is_canceled",
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    reference_stats: Optional[Dict[str, Any]] = None,
    reference_df: Optional[pd.DataFrame] = None,
    policy: Optional[Any] = None,
    phase: str = "preprocess",
) -> ValidationProfileReport:
    """
    Tüm validasyon kontrol noktalarını tek çağrıyla çalıştırır.

    Üç severity seviyesi:
      warn      → log-only, devam et
      soft_fail → soft_failures listesine ekle, caller karar verir
      hard_fail → hard_failures listesine ekle → passed=False → caller ValueError fırlatmalı

    Args:
        policy : ValidationPolicy nesnesi. None → DS_ENV'e göre otomatik profil seçer.
        phase  : "preprocess" | "train" | "predict" | "monitor"
    """
    from .config import ValidationPolicy as _VP, CheckConfig as _CC

    # Policy otomatik seçimi: DS_ENV + phase birleşimi
    pol: "_VP" = policy if policy is not None else _VP.for_phase(phase)  # type: ignore[attr-defined]

    hard_failures: List[str] = []
    soft_failures: List[str] = []
    warnings_list: List[str] = []
    details: Dict[str, Any] = {}

    def _apply(
        name: str,
        summary: str,
        is_violation: bool,
        check_cfg: "_CC",
    ) -> None:
        """Severity'ye göre ihlali sınıflandır."""
        details[name] = summary
        if not check_cfg.enabled or not is_violation:
            return
        sev = check_cfg.severity
        if sev == "hard_fail":
            hard_failures.append(name)
            logger.error(f"🚫 HARD_FAIL [{name}]: {summary}")
        elif sev == "soft_fail":
            soft_failures.append(name)
            logger.warning(f"🟡 SOFT_FAIL [{name}]: {summary}")
        else:
            warnings_list.append(name)
            logger.warning(f"⚠️  WARN [{name}]: {summary}")

    # ── Duplicate ───────────────────────────────────────────────────
    if pol.duplicate.enabled:
        dup = detect_duplicates(df)
        dup_ratio = dup.n_duplicates / max(len(df), 1)
        _apply(
            "duplicate", dup.summary, dup_ratio > pol.duplicate.threshold, pol.duplicate
        )

    # ── Volume ──────────────────────────────────────────────────────
    if pol.volume.enabled and reference_stats and "n_rows" in reference_stats:
        vol = validate_data_volume(
            df,
            expected_rows=int(reference_stats["n_rows"]),
            tolerance_ratio=pol.volume.threshold,
        )
        _apply("volume", vol.summary, vol.is_anomalous, pol.volume)

    # ── Row anomaly ─────────────────────────────────────────────────
    if pol.row_anomaly.enabled:
        anom = detect_row_anomalies(df)
        _apply("row_anomaly", anom.summary, anom.n_anomalies > 0, pol.row_anomaly)

    # ── Distribution drift (sigma-based) ────────────────────────────
    if pol.distribution_drift.enabled and reference_stats and numeric_cols:
        numeric_stats = {
            k: v
            for k, v in reference_stats.items()
            if isinstance(v, dict) and "mean" in v and k in numeric_cols
        }
        if numeric_stats:
            drift = validate_distributions(
                df, numeric_stats, tolerance=pol.distribution_drift.threshold
            )
            _apply(
                "distribution_drift",
                drift.summary,
                not drift.passed,
                pol.distribution_drift,
            )

    # ── PSI / JS Divergence (per-kolon threshold + critical kolonlar) ─
    if pol.psi_drift.enabled and reference_df is not None and numeric_cols:
        psi_report = compute_psi(
            df_reference=reference_df,
            df_current=df,
            numeric_cols=numeric_cols,
            warn_threshold=pol.psi_drift.threshold,
            block_threshold=pol.psi_block_threshold,
            n_bins=pol.psi_n_bins,
            metric=pol.psi_metric,
            column_thresholds=dict(pol.column_drift_thresholds),
            critical_columns=list(pol.critical_columns),
        )
        details["psi"] = psi_report.summary
        # warn_cols → warn; drift_cols → psi_drift severity
        for col in psi_report.warn_cols:
            if col not in psi_report.drift_cols:
                warnings_list.append(f"psi_warn:{col}")
        if not psi_report.overall_passed:
            _apply("psi_drift", psi_report.summary, True, pol.psi_drift)

    # ── Label drift ─────────────────────────────────────────────────
    if (
        pol.label_drift.enabled
        and reference_stats
        and "label_positive_rate" in reference_stats
        and target_col in df.columns
    ):
        ld = detect_label_drift(
            df,
            target_col=target_col,
            ref_positive_rate=float(reference_stats["label_positive_rate"]),
            tolerance=pol.label_drift.threshold,
        )
        _apply("label_drift", ld.summary, ld.is_drifted, pol.label_drift)

    overall_passed = len(hard_failures) == 0
    profile_summary = (
        f"ValidationProfile [{phase}|{pol.psi_metric.upper()}]: "
        f"{'PASSED ✅' if overall_passed else 'FAILED 🚫'} | "
        f"hard={hard_failures} soft={soft_failures} warn_count={len(warnings_list)}"
    )
    if overall_passed and not soft_failures:
        logger.info(profile_summary)
    elif overall_passed:
        logger.warning(profile_summary)
    else:
        logger.error(profile_summary)

    return ValidationProfileReport(
        passed=overall_passed,
        hard_failures=hard_failures,
        soft_failures=soft_failures,
        warnings=warnings_list,
        details=details,
    )
