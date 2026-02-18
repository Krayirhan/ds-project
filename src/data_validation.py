"""
data_validation.py — Pandera-based data validation framework.

Provides schema and distribution-level assertions for:
  1. Raw data ingestion   — validate_raw_data()
  2. Processed features   — validate_processed_data()
  3. Inference payload     — validate_inference_payload()
  4. Distribution checks   — validate_distributions()

Usage:
  from src.data_validation import validate_raw_data, validate_processed_data
  validate_raw_data(df)        # raises SchemaError on contract violation
  validate_processed_data(df)  # raises SchemaError on contract violation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema, Index
from pandera.errors import SchemaError, SchemaErrors

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
            "hotel": Column(str, Check.isin(["Resort Hotel", "City Hotel"]), nullable=False),
            "arrival_date_year": Column(int, Check.in_range(2010, 2030), nullable=False),
            "arrival_date_month": Column(str, nullable=False),
            "arrival_date_week_number": Column(int, Check.in_range(1, 53), nullable=True),
            "arrival_date_day_of_month": Column(int, Check.in_range(1, 31), nullable=True),

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
                    lambda s: s.astype(str).str.lower().str.strip().isin(
                        ["0", "1", "yes", "no"]
                    ).all(),
                    error=f"Target column '{target_col}' must contain 0/1 or yes/no"
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
                    Check(lambda s: pd.to_numeric(s, errors="coerce").notna().all(),
                           error=f"Column '{col}' has non-numeric values after processing"),
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
) -> DataFrameSchema:
    """
    Inference-time schema — API payload kontratı.

    feature_spec: {"numeric": [...], "categorical": [...]} dict
    """
    columns: Dict[str, Column] = {}

    for col in feature_spec.get("numeric", []):
        columns[col] = Column(nullable=True)

    for col in feature_spec.get("categorical", []):
        columns[col] = Column(nullable=True)

    return DataFrameSchema(
        columns=columns,
        strict=False,
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
            violations.append({
                "column": col,
                "check": "existence",
                "message": f"Column '{col}' missing from dataframe",
            })
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            violations.append({
                "column": col,
                "check": "non_empty",
                "message": f"Column '{col}' is entirely null/non-numeric",
            })
            continue

        ref_mean = stats.get("mean", 0.0)
        ref_std = stats.get("std", 1.0)
        cur_mean = float(series.mean())

        # Mean drift check
        if ref_std > 0 and abs(cur_mean - ref_mean) > tolerance * ref_std:
            violations.append({
                "column": col,
                "check": "mean_drift",
                "message": (
                    f"Mean drift: current={cur_mean:.4f}, "
                    f"reference={ref_mean:.4f}, "
                    f"threshold=±{tolerance}*{ref_std:.4f}"
                ),
                "current_mean": cur_mean,
                "reference_mean": ref_mean,
            })

        # Range check
        ref_min = stats.get("min")
        ref_max = stats.get("max")
        if ref_min is not None:
            cur_min = float(series.min())
            if cur_min < ref_min * 0.5 - abs(ref_min):  # generous lower bound
                violations.append({
                    "column": col,
                    "check": "range_min",
                    "message": f"Min out of range: current={cur_min}, reference_min={ref_min}",
                })
        if ref_max is not None:
            cur_max = float(series.max())
            if cur_max > ref_max * 2.0 + abs(ref_max):  # generous upper bound
                violations.append({
                    "column": col,
                    "check": "range_max",
                    "message": f"Max out of range: current={cur_max}, reference_max={ref_max}",
                })

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
        logger.info(f"✅ Raw data validation passed: {len(df)} rows, {len(df.columns)} cols")
        return None
    except SchemaErrors as e:
        logger.error(f"❌ Raw data validation failed: {len(e.failure_cases)} failure(s)")
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
        logger.error(f"❌ Processed data validation failed: {len(e.failure_cases)} failure(s)")
        if raise_on_error:
            raise
        return e


def validate_inference_payload(
    df: pd.DataFrame,
    feature_spec: Dict[str, Any],
    raise_on_error: bool = True,
) -> pa.errors.SchemaErrors | None:
    """
    API inference payload schema kontratını doğrula.
    """
    schema = build_inference_schema(feature_spec)
    try:
        schema.validate(df, lazy=True)
        logger.info(f"✅ Inference payload validation passed: {len(df)} rows")
        return None
    except SchemaErrors as e:
        logger.error(f"❌ Inference payload validation failed: {len(e.failure_cases)} failure(s)")
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
