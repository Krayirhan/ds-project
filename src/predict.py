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
        if not (pd.api.types.is_object_dtype(work[c]) or pd.api.types.is_string_dtype(work[c])):
            work[c] = work[c].astype("string")

    report = {
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "feature_count_expected": len(spec.all_features),
        "feature_count_input": int(df_input.shape[1]),
        "feature_count_used": int(work.shape[1]),
    }
    return work, report


def predict_with_policy(
    model,
    policy: DecisionPolicy,
    df_input: pd.DataFrame,
    feature_spec_payload: Dict[str, Any],
    model_used: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    X, schema_report = validate_and_prepare_features(df_input, feature_spec_payload=feature_spec_payload)

    proba = model.predict_proba(X)[:, 1]
    ranking_scores = compute_incremental_profit_scores(df_input=df_input, proba=proba, policy=policy)
    actions = apply(proba=proba, policy=policy, ranking_scores=ranking_scores)

    out = pd.DataFrame(
        {
            "proba": proba.astype(float),
            "action": actions.astype(int),
            "threshold_used": float(policy.threshold),
            "max_action_rate_used": (
                float(policy.max_action_rate) if policy.max_action_rate is not None else np.nan
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
            float(policy.max_action_rate) if policy.max_action_rate is not None else None
        ),
        "model_used": str(model_used or policy.selected_model),
        "ranking_mode": str(policy.raw.get("ranking_mode", "proba")),
    }
    return out, pred_report
