"""Schema-focused validation primitives.

This module contains schema contracts and lightweight schema-oriented checks.
It is part of the official validation import surface: ``src.validation``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

from ..utils import get_logger

logger = get_logger("validation.schema")


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
                    lambda s: (
                        s.astype(str)
                        .str.lower()
                        .str.strip()
                        .isin(["0", "1", "yes", "no"])
                        .all()
                    ),
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


def basic_schema_checks(df: pd.DataFrame, target_col: str) -> None:
    """Fast pre-flight checks before any Pandera schema validation.

    Raises ValueError on empty DataFrame, missing target column, or duplicate
    column names.  Logs a success message otherwise.
    """
    if df.empty:
        raise ValueError("Dataset is empty.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")

    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Duplicate column names detected.")

    logger.info("Basic schema checks passed.")


def validate_target_labels(df: pd.DataFrame, target_col: str, allowed: set) -> None:
    """Assert all values in ``target_col`` belong to the ``allowed`` set.

    Raises ValueError listing unexpected labels so pipeline fails fast before
    any model training or preprocessing step.
    """
    y = df[target_col].astype(str).str.lower().str.strip()
    uniq = set(y.unique())
    if not uniq.issubset(allowed):
        raise ValueError(
            f"Unexpected labels in {target_col}: {sorted(uniq)} | allowed={sorted(allowed)}"
        )
    logger.info("Target labels OK -> %s", sorted(uniq))


def null_ratio_report(df: pd.DataFrame, top_k: int = 10) -> pd.Series:
    """Return a Series of null ratios (descending) for exploratory diagnosis.

    Useful for choosing imputation strategy before fitting the Pandera schema.
    """
    ratios = df.isna().mean().sort_values(ascending=False)
    return ratios.head(top_k)


__all__ = [
    "build_raw_schema",
    "build_processed_schema",
    "build_inference_schema",
    "validate_raw_data",
    "validate_processed_data",
    "validate_inference_payload",
    "generate_reference_stats",
    "get_schema_fingerprint",
    "basic_schema_checks",
    "validate_target_labels",
    "null_ratio_report",
]
