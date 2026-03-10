"""Drift-focused validation primitives.

This module contains drift detection algorithms and the validation profile runner.
It is part of the official validation import surface: ``src.validation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils import get_logger
from .anomaly import detect_duplicates, detect_row_anomalies, validate_data_volume

logger = get_logger("validation.drift")


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
                f"⚠️  {metric.upper()} drift [{col}]: {score:.4f} ≥ warn={col_warn:.3f}"
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
    from ..config import ValidationPolicy as _VP, CheckConfig as _CC

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


__all__ = [
    "DistributionReport",
    "validate_distributions",
    "LabelDriftReport",
    "detect_label_drift",
    "CorrelationDriftReport",
    "detect_correlation_drift",
    "generate_reference_correlations",
    "SkewReport",
    "detect_training_serving_skew",
    "ImportanceDriftReport",
    "detect_feature_importance_drift",
    "generate_reference_categories",
    "PSIReport",
    "_psi_score",
    "_js_divergence",
    "compute_psi",
    "ValidationProfileReport",
    "run_validation_profile",
]
