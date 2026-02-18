"""CLI: explain command — standalone model explainability."""

from __future__ import annotations

from typing import Any, Dict, Optional


from ..config import ExperimentConfig, Paths
from ..explain import (
    compute_permutation_importance,
    compute_shap_values,
    save_explainability_report,
)
from ..io import read_parquet
from ..utils import get_logger
from ._helpers import json_read, resolve_latest_run_id, safe_load

logger = get_logger("cli.explain")


def cmd_explain(
    paths: Paths,
    cfg: ExperimentConfig,
    run_id: Optional[str] = None,
    sample_size: int = 500,
) -> str:
    """Standalone explainability: permutation importance + optional SHAP."""
    run_id = run_id or resolve_latest_run_id(
        paths.models / "latest.json",
        paths.reports_metrics / "latest.json",
    )

    run_metrics_dir = paths.reports_metrics / run_id

    # Load decision policy to find champion model
    policy_path = run_metrics_dir / "decision_policy.json"
    if not policy_path.exists():
        policy_path = paths.reports_metrics / "decision_policy.json"
    if not policy_path.exists():
        raise FileNotFoundError(f"Decision policy not found for run {run_id}")

    policy = json_read(policy_path)
    champion = policy.get("selected_model")
    artifact = policy.get("selected_model_artifact")
    if not artifact:
        raise ValueError("Policy does not contain selected_model_artifact")

    model_path = paths.project_root / artifact
    model = safe_load(model_path)
    if model is None:
        raise FileNotFoundError(f"Model not found: {model_path}")

    test_df = read_parquet(paths.data_processed / "test.parquet")
    X_test = test_df.drop(columns=[cfg.target_col])
    y_test = test_df[cfg.target_col].astype(int).values

    report: Dict[str, Any] = {"run_id": run_id, "champion_model": champion}

    # Permutation importance
    logger.info("Computing permutation importance...")
    perm_result = compute_permutation_importance(model, X_test, y_test, seed=cfg.seed)
    report["permutation_importance"] = perm_result
    save_explainability_report(
        perm_result, run_metrics_dir / "permutation_importance.json"
    )

    # SHAP (optional)
    logger.info("Computing SHAP values...")
    shap_result = compute_shap_values(model, X_test, max_samples=sample_size)
    if shap_result is not None:
        report["shap"] = shap_result
        save_explainability_report(
            shap_result, run_metrics_dir / "shap_summary.json"
        )

    # Combined report
    save_explainability_report(report, run_metrics_dir / "explainability_report.json")
    logger.info(f"Explainability report saved → {run_metrics_dir / 'explainability_report.json'}")
    return run_id
