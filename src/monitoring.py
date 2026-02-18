from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from .features import FeatureSpec


def _safe_hist(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins)
    hist = hist.astype(float)
    total = hist.sum()
    if total <= 0:
        return np.full_like(hist, 1.0 / len(hist), dtype=float)
    p = hist / total
    eps = 1e-9
    p = np.clip(p, eps, None)
    return p / p.sum()


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref_v = pd.to_numeric(ref, errors="coerce").dropna().to_numpy()
    cur_v = pd.to_numeric(cur, errors="coerce").dropna().to_numpy()
    if ref_v.size == 0 or cur_v.size == 0:
        return 0.0

    q = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref_v, q))
    if edges.shape[0] <= 2:
        lo = min(float(np.min(ref_v)), float(np.min(cur_v)))
        hi = max(float(np.max(ref_v)), float(np.max(cur_v)))
        edges = np.linspace(lo, hi if hi > lo else lo + 1e-6, bins + 1)

    p_ref = _safe_hist(ref_v, edges)
    p_cur = _safe_hist(cur_v, edges)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref_s = ref.astype(str).fillna("<NA>")
    cur_s = cur.astype(str).fillna("<NA>")
    cats = sorted(set(ref_s.unique()).union(set(cur_s.unique())))
    if not cats:
        return 0.0

    ref_counts = (
        ref_s.value_counts(normalize=True)
        .reindex(cats, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    cur_counts = (
        cur_s.value_counts(normalize=True)
        .reindex(cats, fill_value=0.0)
        .to_numpy(dtype=float)
    )

    eps = 1e-9
    ref_counts = np.clip(ref_counts, eps, None)
    cur_counts = np.clip(cur_counts, eps, None)
    ref_counts = ref_counts / ref_counts.sum()
    cur_counts = cur_counts / cur_counts.sum()
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def data_drift_report(
    df_ref: pd.DataFrame, df_cur: pd.DataFrame, feature_spec: FeatureSpec
) -> Dict[str, Any]:
    per_feature: Dict[str, Any] = {}

    for c in feature_spec.numeric:
        if c in df_ref.columns and c in df_cur.columns:
            per_feature[c] = {
                "type": "numeric",
                "psi": psi_numeric(df_ref[c], df_cur[c]),
                "ref_mean": float(pd.to_numeric(df_ref[c], errors="coerce").mean()),
                "cur_mean": float(pd.to_numeric(df_cur[c], errors="coerce").mean()),
            }

    for c in feature_spec.categorical:
        if c in df_ref.columns and c in df_cur.columns:
            per_feature[c] = {
                "type": "categorical",
                "psi": psi_categorical(df_ref[c], df_cur[c]),
            }

    psi_values = [
        v["psi"] for v in per_feature.values() if isinstance(v.get("psi"), (int, float))
    ]
    return {
        "n_features_compared": len(per_feature),
        "max_psi": float(max(psi_values)) if psi_values else 0.0,
        "mean_psi": float(np.mean(psi_values)) if psi_values else 0.0,
        "features": per_feature,
    }


def prediction_drift_report(
    ref_proba: np.ndarray, cur_proba: np.ndarray
) -> Dict[str, Any]:
    ref = np.asarray(ref_proba, dtype=float)
    cur = np.asarray(cur_proba, dtype=float)
    bins = np.linspace(0.0, 1.0, 11)
    p_ref = _safe_hist(ref, bins)
    p_cur = _safe_hist(cur, bins)
    psi = float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))

    # Simple KS distance over empirical CDF
    points = np.sort(np.unique(np.concatenate([ref, cur])))
    if points.size == 0:
        ks = 0.0
    else:
        cdf_ref = np.searchsorted(np.sort(ref), points, side="right") / max(len(ref), 1)
        cdf_cur = np.searchsorted(np.sort(cur), points, side="right") / max(len(cur), 1)
        ks = float(np.max(np.abs(cdf_ref - cdf_cur)))

    return {
        "psi": psi,
        "ks_stat": ks,
        "ref_mean_proba": float(np.mean(ref)) if ref.size > 0 else 0.0,
        "cur_mean_proba": float(np.mean(cur)) if cur.size > 0 else 0.0,
    }


def outcome_monitoring_report(
    actions_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    *,
    actual_col: str,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    if actual_col not in outcome_df.columns:
        raise ValueError(f"Outcome column not found: {actual_col}")

    n = min(len(actions_df), len(outcome_df))
    if n == 0:
        return {"n_rows": 0}

    proba = pd.to_numeric(actions_df["proba"].iloc[:n], errors="coerce").to_numpy(
        dtype=float
    )
    action = (
        pd.to_numeric(actions_df["action"].iloc[:n], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    y = (
        pd.to_numeric(outcome_df[actual_col].iloc[:n], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None
    brier = float(brier_score_loss(y, proba))

    cost = policy.get("cost_matrix", {})
    tp_v = float(cost.get("tp_value", 0.0))
    fp_v = float(cost.get("fp_value", 0.0))
    fn_v = float(cost.get("fn_value", 0.0))
    tn_v = float(cost.get("tn_value", 0.0))

    tp = int(np.sum((action == 1) & (y == 1)))
    fp = int(np.sum((action == 1) & (y == 0)))
    fn = int(np.sum((action == 0) & (y == 1)))
    tn = int(np.sum((action == 0) & (y == 0)))

    realized_profit = float(tp * tp_v + fp * fp_v + fn * fn_v + tn * tn_v)

    return {
        "n_rows": int(n),
        "auc": auc,
        "brier": brier,
        "action_rate": float(np.mean(action)),
        "realized_profit": realized_profit,
        "confusion_like": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


@dataclass(frozen=True)
class AlertThresholds:
    data_drift_psi_threshold: float
    prediction_drift_psi_threshold: float
    profit_drop_ratio_alert: float
    action_rate_tolerance: float


def build_alerts(
    *,
    data_drift: Dict[str, Any],
    prediction_drift: Dict[str, Any],
    outcome_report: Optional[Dict[str, Any]],
    policy: Dict[str, Any],
    thresholds: AlertThresholds,
) -> Dict[str, Any]:
    alerts: Dict[str, Any] = {
        "data_drift": data_drift.get("max_psi", 0.0)
        >= thresholds.data_drift_psi_threshold,
        "prediction_drift": prediction_drift.get("psi", 0.0)
        >= thresholds.prediction_drift_psi_threshold,
        "profit_drop": False,
        "action_rate_deviation": False,
    }

    expected_profit = policy.get("expected_net_profit")
    expected_action_rate = policy.get("max_action_rate")

    if outcome_report and isinstance(expected_profit, (int, float)):
        realized = outcome_report.get("realized_profit")
        if isinstance(realized, (int, float)):
            drop_ratio = (
                1.0 - (realized / expected_profit) if expected_profit != 0 else 0.0
            )
            alerts["profit_drop"] = bool(
                drop_ratio >= thresholds.profit_drop_ratio_alert
            )
            alerts["profit_drop_ratio"] = float(drop_ratio)

    if outcome_report and isinstance(expected_action_rate, (int, float)):
        ar = outcome_report.get("action_rate")
        if isinstance(ar, (int, float)):
            alerts["action_rate_deviation"] = bool(
                abs(ar - expected_action_rate) >= thresholds.action_rate_tolerance
            )
            alerts["action_rate_delta"] = float(ar - expected_action_rate)

    alerts["any_alert"] = bool(any(v for k, v in alerts.items() if isinstance(v, bool)))
    return alerts
