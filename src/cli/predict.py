"""CLI: predict command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib

from ..config import ExperimentConfig, Paths
from ..io import read_parquet, write_parquet
from ..policy import load_decision_policy
from ..predict import load_feature_spec, predict_with_policy
from ..utils import get_logger, sha256_file
from ._helpers import json_write, mark_latest, read_input_dataset, resolve_latest_run_id

logger = get_logger("cli.predict")


def cmd_predict(
    paths: Paths,
    cfg: ExperimentConfig,
    input_path: Optional[str] = None,
    policy_path: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    policy_file = Path(policy_path) if policy_path else (paths.reports_metrics / "decision_policy.json")
    policy = load_decision_policy(policy_file)
    if policy.raw.get("policy_version") != cfg.contract.policy_version:
        raise ValueError(
            f"Policy version mismatch. expected={cfg.contract.policy_version} got={policy.raw.get('policy_version')}"
        )

    resolved_run_id = run_id or str(
        policy.raw.get("run_id") or resolve_latest_run_id(paths.reports_metrics / "latest.json")
    )

    model_artifact = policy.selected_model_artifact
    if not model_artifact:
        raise ValueError("Policy does not include selected_model_artifact")
    model_path = paths.project_root / Path(model_artifact)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    model = joblib.load(model_path)

    expected_sha = policy.raw.get("selected_model_sha256")
    if expected_sha:
        actual_sha = sha256_file(str(model_path))
        if actual_sha != expected_sha:
            raise ValueError("Model checksum mismatch. Artifact integrity validation failed.")

    expected_schema_version = policy.raw.get("feature_schema_version")

    if input_path:
        in_path = Path(input_path)
        if not in_path.is_absolute():
            in_path = paths.project_root / in_path
    else:
        in_path = paths.data_processed / "test.parquet"

    df_input = read_input_dataset(in_path)

    run_feature_spec = paths.reports_metrics / resolved_run_id / "feature_spec.json"
    global_feature_spec = paths.reports / "feature_spec.json"
    feature_spec_file = run_feature_spec if run_feature_spec.exists() else global_feature_spec
    feature_spec = load_feature_spec(feature_spec_file)
    if expected_schema_version and feature_spec.get("schema_version") != expected_schema_version:
        raise ValueError(
            "Feature schema version mismatch. "
            f"policy={expected_schema_version} feature_spec={feature_spec.get('schema_version')}"
        )

    actions_df, pred_report = predict_with_policy(
        model=model,
        policy=policy,
        df_input=df_input,
        feature_spec_payload=feature_spec,
        model_used=model_artifact,
    )

    run_pred_dir = paths.reports_predictions / resolved_run_id
    run_pred_dir.mkdir(parents=True, exist_ok=True)

    run_actions_path = run_pred_dir / "actions.parquet"
    write_parquet(actions_df, run_actions_path)

    root_actions_path = paths.reports_predictions / "actions.parquet"
    write_parquet(actions_df, root_actions_path)

    pred_meta = {
        "run_id": resolved_run_id,
        "input_path": str(in_path),
        "policy_path": str(policy_file),
        "feature_spec_path": str(feature_spec_file),
        "output_actions_path": str(run_actions_path),
        **pred_report,
    }
    json_write(run_pred_dir / "prediction_report.json", pred_meta)
    json_write(paths.reports_predictions / "latest_prediction_report.json", pred_meta)
    mark_latest(paths.reports_predictions, resolved_run_id, extra={"actions_path": str(run_actions_path)})

    logger.info(f"Saved predictions -> {run_actions_path}")
    return resolved_run_id
