"""CLI: monitor command."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib

from ..config import ExperimentConfig, Paths
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
