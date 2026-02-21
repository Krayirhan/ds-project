"""CLI: monitor command."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from ..config import ExperimentConfig, Paths
from ..data_validation import (
    validate_distributions,
    detect_label_drift,
    detect_correlation_drift,
    validate_data_volume,
    detect_feature_importance_drift,
)
from ..features import FeatureSpec
from ..io import read_parquet
from ..monitoring import (
    AlertThresholds,
    build_alerts,
    data_drift_report,
    outcome_monitoring_report,
    prediction_drift_report,
)
from ..policy import load_decision_policy
from ..predict import load_feature_spec, predict_with_policy
from ..utils import get_logger
from ._helpers import (
    json_write,
    mark_latest,
    notify_webhook,
    read_input_dataset,
    resolve_latest_run_id,
)

logger = get_logger("cli.monitor")


def cmd_monitor(
    paths: Paths,
    cfg: ExperimentConfig,
    input_path: Optional[str] = None,
    outcome_path: Optional[str] = None,
    actual_col: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    policy_file = paths.reports_metrics / "decision_policy.json"
    policy = load_decision_policy(policy_file)
    if policy.raw.get("policy_version") != cfg.contract.policy_version:
        raise ValueError(
            f"Policy version mismatch. expected={cfg.contract.policy_version} got={policy.raw.get('policy_version')}"
        )
    resolved_run_id = run_id or str(
        policy.raw.get("run_id")
        or resolve_latest_run_id(paths.reports_metrics / "latest.json")
    )

    model_artifact = policy.selected_model_artifact
    if not model_artifact:
        raise ValueError("Policy does not include selected_model_artifact")
    model = joblib.load(paths.project_root / model_artifact)

    if input_path:
        in_path = Path(input_path)
        if not in_path.is_absolute():
            in_path = paths.project_root / in_path
    else:
        in_path = paths.data_processed / "test.parquet"

    df_cur = read_input_dataset(in_path)
    df_ref = read_parquet(paths.data_processed / "dataset.parquet")

    run_feature_spec = paths.reports_metrics / resolved_run_id / "feature_spec.json"
    feature_spec_file = (
        run_feature_spec
        if run_feature_spec.exists()
        else (paths.reports / "feature_spec.json")
    )
    feature_spec_payload = load_feature_spec(feature_spec_file)
    spec = FeatureSpec.from_dict(feature_spec_payload)

    actions_cur, pred_report = predict_with_policy(
        model=model,
        policy=policy,
        df_input=df_cur,
        feature_spec_payload=feature_spec_payload,
        model_used=model_artifact,
    )

    ref_pred_path = paths.reports_predictions / resolved_run_id / "actions.parquet"
    if ref_pred_path.exists():
        actions_ref = read_parquet(ref_pred_path)
    else:
        df_ref_sample = read_parquet(paths.data_processed / "test.parquet")
        actions_ref, _ = predict_with_policy(
            model=model,
            policy=policy,
            df_input=df_ref_sample,
            feature_spec_payload=feature_spec_payload,
            model_used=model_artifact,
        )

    # ── Pandera dağılım doğrulaması (referans istatistiklerle) ──
    ref_stats_path = paths.reports_metrics / resolved_run_id / "reference_stats.json"
    if not ref_stats_path.exists():
        ref_stats_path = paths.reports_metrics / "reference_stats.json"
    if ref_stats_path.exists():
        import json

        ref_stats = json.loads(ref_stats_path.read_text(encoding="utf-8"))
        dist_report = validate_distributions(df_cur, reference_stats=ref_stats)
        if not dist_report.passed:
            logger.warning(
                f"Distribution validation FAILED: {len(dist_report.violations)} violation(s)"
            )
            for v in dist_report.violations:
                logger.warning(f"  → {v['column']}: {v['message']}")
    else:
        logger.info("No reference_stats.json found — skipping distribution validation")
        dist_report = None

    # ── Label drift detection ──
    label_drift_result = None
    if cfg.target_col in df_ref.columns and cfg.target_col in df_cur.columns:
        ref_rate = float(
            pd.to_numeric(df_ref[cfg.target_col], errors="coerce").dropna().mean()
        )
        label_drift_result = detect_label_drift(
            df_cur,
            target_col=cfg.target_col,
            ref_positive_rate=ref_rate,
            tolerance=0.10,
        )

    # ── Cross-feature correlation drift ──
    corr_drift_result = None
    ref_corr_path = (
        paths.reports_metrics / resolved_run_id / "reference_correlations.json"
    )
    if not ref_corr_path.exists():
        ref_corr_path = paths.reports_metrics / "reference_correlations.json"
    if ref_corr_path.exists():
        import json as _json

        ref_corr = _json.loads(ref_corr_path.read_text(encoding="utf-8"))
        corr_drift_result = detect_correlation_drift(
            df_cur,
            reference_corr=ref_corr,
            numeric_cols=spec.numeric,
            threshold=0.20,
        )

    # ── Data volume anomaly (monitor) ──
    volume_result = validate_data_volume(
        df_cur,
        expected_rows=len(df_ref),
        tolerance_ratio=0.50,
    )

    # ── Feature importance drift (run-vs-previous) ──
    importance_drift_result = None
    cur_imp_path = paths.reports_metrics / resolved_run_id / "feature_importance.json"
    if not cur_imp_path.exists():
        cur_imp_path = paths.reports_metrics / "feature_importance.json"
    ref_imp_path = paths.reports_metrics / "feature_importance.prev.json"
    if cur_imp_path.exists() and ref_imp_path.exists():
        import json as _json

        cur_imp = _json.loads(cur_imp_path.read_text(encoding="utf-8"))
        ref_imp = _json.loads(ref_imp_path.read_text(encoding="utf-8"))
        if isinstance(cur_imp, dict) and isinstance(ref_imp, dict) and ref_imp:
            importance_drift_result = detect_feature_importance_drift(
                current_importance=cur_imp,
                reference_importance=ref_imp,
                top_k=10,
                rank_drop_threshold=5,
            )

    data_drift = data_drift_report(df_ref=df_ref, df_cur=df_cur, feature_spec=spec)
    pred_drift = prediction_drift_report(
        ref_proba=actions_ref["proba"].to_numpy(dtype=float),
        cur_proba=actions_cur["proba"].to_numpy(dtype=float),
    )

    outcome_report = None
    if outcome_path:
        out_path = Path(outcome_path)
        if not out_path.is_absolute():
            out_path = paths.project_root / out_path
        outcome_df = read_input_dataset(out_path)
        outcome_report = outcome_monitoring_report(
            actions_df=actions_cur,
            outcome_df=outcome_df,
            actual_col=actual_col or cfg.target_col,
            policy=policy.raw,
        )
    elif cfg.target_col in df_cur.columns:
        outcome_report = outcome_monitoring_report(
            actions_df=actions_cur,
            outcome_df=df_cur,
            actual_col=cfg.target_col,
            policy=policy.raw,
        )

    alerts = build_alerts(
        data_drift=data_drift,
        prediction_drift=pred_drift,
        outcome_report=outcome_report,
        policy=policy.raw,
        data_volume_is_anomalous=volume_result.is_anomalous,
        thresholds=AlertThresholds(
            data_drift_psi_threshold=cfg.monitoring.data_drift_psi_threshold,
            prediction_drift_psi_threshold=cfg.monitoring.prediction_drift_psi_threshold,
            profit_drop_ratio_alert=cfg.monitoring.profit_drop_ratio_alert,
            action_rate_tolerance=cfg.monitoring.action_rate_tolerance,
        ),
    )

    webhook_url = cfg.monitoring.alert_webhook_url or os.getenv("ALERT_WEBHOOK_URL")
    if alerts.get("any_alert"):
        notify_webhook(
            webhook_url,
            {
                "event": "monitoring_alert",
                "run_id": resolved_run_id,
                "alerts": alerts,
                "policy": policy.raw.get("selected_model"),
            },
            dlq_path=paths.reports_monitoring / "dead_letter_webhooks.jsonl",
        )

    report = {
        "run_id": resolved_run_id,
        "input_path": str(in_path),
        "policy_path": str(policy_file),
        "model_used": model_artifact,
        "prediction_summary": pred_report,
        "distribution_validation": (
            {
                "passed": dist_report.passed if dist_report else None,
                "violations": dist_report.violations if dist_report else [],
                "summary": dist_report.summary if dist_report else "skipped",
            }
            if dist_report is not None
            else {"passed": None, "summary": "no_reference_stats"}
        ),
        "label_drift": {
            "ref_rate": (
                label_drift_result.ref_positive_rate if label_drift_result else None
            ),
            "cur_rate": (
                label_drift_result.cur_positive_rate if label_drift_result else None
            ),
            "is_drifted": label_drift_result.is_drifted if label_drift_result else None,
            "summary": label_drift_result.summary if label_drift_result else "skipped",
        },
        "correlation_drift": {
            "n_drifted": corr_drift_result.n_drifted if corr_drift_result else 0,
            "drifted_pairs": (
                corr_drift_result.drifted_pairs if corr_drift_result else []
            ),
            "summary": corr_drift_result.summary if corr_drift_result else "skipped",
        },
        "data_volume": {
            "current_rows": volume_result.current_rows,
            "expected_range": list(volume_result.expected_range),
            "is_anomalous": volume_result.is_anomalous,
            "summary": volume_result.summary,
        },
        "feature_importance_drift": {
            "n_changed": (
                importance_drift_result.n_changed if importance_drift_result else 0
            ),
            "rank_correlation": (
                importance_drift_result.rank_correlation
                if importance_drift_result
                else None
            ),
            "changed_features": (
                importance_drift_result.changed_features
                if importance_drift_result
                else []
            ),
            "summary": (
                importance_drift_result.summary
                if importance_drift_result
                else "skipped"
            ),
        },
        "data_drift": data_drift,
        "prediction_drift": pred_drift,
        "outcome_monitoring": outcome_report,
        "alerts": alerts,
    }

    out_dir = paths.reports_monitoring / resolved_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_write(out_dir / "monitoring_report.json", report)
    json_write(paths.reports_monitoring / "latest_monitoring_report.json", report)
    mark_latest(
        paths.reports_monitoring,
        resolved_run_id,
        extra={"monitoring_report": str(out_dir / "monitoring_report.json")},
    )

    logger.info(f"Saved monitoring report -> {out_dir / 'monitoring_report.json'}")
    return resolved_run_id
