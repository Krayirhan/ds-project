"""
predict.py

Batch inference için production-grade yardımcılar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_validation import (
    validate_inference_payload,
    detect_training_serving_skew,
    detect_unseen_categories,
    validate_model_output,
    detect_row_anomalies,
    validate_data_volume,
)
from .features import FeatureSpec
from .policy import DecisionPolicy, apply, compute_incremental_profit_scores
from .utils import get_logger

logger = get_logger("predict")


def load_feature_spec(feature_spec_path: Path) -> Dict[str, Any]:
    if not feature_spec_path.exists():
        raise FileNotFoundError(f"Feature spec not found: {feature_spec_path}")
    payload = json.loads(feature_spec_path.read_text(encoding="utf-8"))
    if "numeric" not in payload or "categorical" not in payload:
        raise ValueError("Invalid feature spec. Required keys: numeric, categorical")
    payload["numeric"] = list(payload.get("numeric", []))
    payload["categorical"] = list(payload.get("categorical", []))
    payload["all_features"] = [*payload["numeric"], *payload["categorical"]]
    return payload


def validate_and_prepare_features(
    df_input: pd.DataFrame,
    feature_spec_payload: Dict[str, Any],
    fail_on_missing: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # ── Pandera inference şema doğrulaması ──
    pandera_errors = validate_inference_payload(
        df_input, feature_spec_payload, raise_on_error=False
    )
    if pandera_errors is not None:
        logger.warning(
            f"Pandera inference validation: {len(pandera_errors.failure_cases)} issue(s) — "
            "proceeding with best-effort feature preparation"
        )

    spec = FeatureSpec.from_dict(feature_spec_payload)

    expected = set(spec.all_features)
    got = set(df_input.columns)

    missing_cols = sorted(expected - got)
    extra_cols = sorted(got - expected)

    if missing_cols and fail_on_missing:
        raise ValueError(
            "Missing required feature columns: "
            f"{missing_cols}. Input schema does not match training schema."
        )

    if extra_cols:
        logger.info(f"Ignoring extra columns: {extra_cols}")

    work = df_input.copy()

    # Missing kolonları (fail_on_missing=False senaryosu için) NaN ile ekle
    for c in missing_cols:
        work[c] = np.nan

    # Feature order freeze
    work = work[spec.all_features]

    # Type checks / coercion
    numeric_cast_failures: List[str] = []
    for c in spec.numeric:
        converted = pd.to_numeric(work[c], errors="coerce")
        # Orijinalde değer var ama sayıya çevrilemeyenleri yakala
        bad_mask = work[c].notna() & converted.isna()
        if bool(bad_mask.any()):
            numeric_cast_failures.append(c)
        work[c] = converted

    if numeric_cast_failures:
        raise ValueError(
            "Numeric type validation failed for columns: "
            f"{numeric_cast_failures}. Non-numeric values detected."
        )

    for c in spec.categorical:
        # Categorical pipeline string/object ile daha güvenli
        if not (
            pd.api.types.is_object_dtype(work[c])
            or pd.api.types.is_string_dtype(work[c])
        ):
            work[c] = work[c].astype("string")

    # ── Training-serving skew detection (per-request) ──
    skew_report = None
    ref_stats = feature_spec_payload.get("_reference_stats")
    if ref_stats:
        skew_report = detect_training_serving_skew(
            df_serving=work, reference_stats=ref_stats,
            numeric_cols=spec.numeric, tolerance=2.0,
        )

    # ── Unseen category detection ──
    cardinality_report = None
    ref_cats = feature_spec_payload.get("_reference_categories")
    if ref_cats:
        cardinality_report = detect_unseen_categories(work, ref_cats)

    # ── Row-level anomaly detection (inference payload) ──
    anomaly_report = detect_row_anomalies(work)

    # ── Data volume anomaly (per-request batch size) ──
    volume_report = None
    expected_rows = feature_spec_payload.get("_reference_volume_rows")
    if isinstance(expected_rows, int) and expected_rows > 0:
        volume_report = validate_data_volume(
            work, expected_rows=expected_rows, tolerance_ratio=0.90
        )

    report = {
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "feature_count_expected": len(spec.all_features),
        "feature_count_input": int(df_input.shape[1]),
        "feature_count_used": int(work.shape[1]),
        "pandera_schema_passed": pandera_errors is None,
        "training_serving_skew": {
            "n_skewed": skew_report.n_skewed if skew_report else 0,
            "skewed_features": [s["column"] for s in skew_report.skewed_features] if skew_report else [],
        },
        "unseen_categories": {
            "n_unseen": cardinality_report.n_unseen_total if cardinality_report else 0,
            "columns": list(cardinality_report.unseen.keys()) if cardinality_report else [],
        },
        "row_anomalies": {
            "n_anomalies": anomaly_report.n_anomalies,
            "anomaly_types": anomaly_report.anomaly_types,
            "sample_indices": anomaly_report.sample_indices,
        },
        "data_volume": {
            "expected_range": list(volume_report.expected_range) if volume_report else None,
            "is_anomalous": volume_report.is_anomalous if volume_report else None,
            "summary": volume_report.summary if volume_report else "skipped",
        },
    }
    return work, report


def predict_with_policy(
    model,
    policy: DecisionPolicy,
    df_input: pd.DataFrame,
    feature_spec_payload: Dict[str, Any],
    model_used: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    X, schema_report = validate_and_prepare_features(
        df_input, feature_spec_payload=feature_spec_payload
    )

    proba = model.predict_proba(X)[:, 1]

    # ── Model output range validation [0,1] ──
    output_report = validate_model_output(proba)
    if not output_report.passed:
        logger.warning(
            f"Model output validation FAILED: {output_report.n_nan} NaN, "
            f"{output_report.n_out_of_range} out of range"
        )
        # Clamp to [0,1] for safety
        proba = np.clip(np.nan_to_num(proba, nan=0.5), 0.0, 1.0)

    ranking_scores = compute_incremental_profit_scores(
        df_input=df_input, proba=proba, policy=policy
    )
    actions = apply(proba=proba, policy=policy, ranking_scores=ranking_scores)

    out = pd.DataFrame(
        {
            "proba": proba.astype(float),
            "action": actions.astype(int),
            "threshold_used": float(policy.threshold),
            "max_action_rate_used": (
                float(policy.max_action_rate)
                if policy.max_action_rate is not None
                else np.nan
            ),
            "model_used": str(model_used or policy.selected_model),
        }
    )

    pred_report = {
        **schema_report,
        "n_rows": int(len(out)),
        "predicted_action_rate": float(out["action"].mean()) if len(out) > 0 else 0.0,
        "threshold_used": float(policy.threshold),
        "max_action_rate_used": (
            float(policy.max_action_rate)
            if policy.max_action_rate is not None
            else None
        ),
        "model_used": str(model_used or policy.selected_model),
        "ranking_mode": str(policy.raw.get("ranking_mode", "proba")),
    }
    return out, pred_report
