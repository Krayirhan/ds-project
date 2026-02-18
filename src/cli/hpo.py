"""CLI: hpo (hyperparameter optimization) command."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..config import ExperimentConfig, Paths
from ..experiment_tracking import ExperimentTracker
from ..hpo import run_hpo
from ..io import read_parquet
from ..utils import get_logger
from ._helpers import json_write, mark_latest, new_run_id

logger = get_logger("cli.hpo")


def cmd_hpo(
    paths: Paths,
    cfg: ExperimentConfig,
    n_trials: int = 50,
    run_id: Optional[str] = None,
) -> str:
    """Optuna HPO çalıştır, best params'ı kaydet."""
    run_id = run_id or new_run_id()

    dataset_path = paths.data_processed / "dataset.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}. Run preprocess first.")
    df = read_parquet(dataset_path)

    logger.info(f"Starting HPO | n_trials={n_trials} run_id={run_id}")

    result = run_hpo(
        df=df,
        target_col=cfg.target_col,
        seed=cfg.seed,
        cv_folds=cfg.cv_folds,
        n_trials=n_trials,
    )

    run_metrics_dir = paths.reports_metrics / run_id
    run_metrics_dir.mkdir(parents=True, exist_ok=True)

    hpo_report = {
        "run_id": run_id,
        "model_type": result.model_type,
        "n_trials": result.n_trials,
        "best_score": result.best_score,
        "best_params": result.best_params,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    json_write(run_metrics_dir / "hpo_result.json", hpo_report)
    json_write(paths.reports_metrics / "latest_hpo_result.json", hpo_report)

    # Optional MLflow logging
    tracker = ExperimentTracker()
    with tracker.start_run(run_name=f"hpo_{run_id}"):
        tracker.log_params(result.best_params)
        tracker.log_metric("hpo_best_roc_auc", result.best_score)
        tracker.log_param("hpo_model_type", result.model_type)
        tracker.log_param("hpo_n_trials", result.n_trials)
        tracker.log_artifact(run_metrics_dir / "hpo_result.json")

    logger.info(
        f"HPO completed | best_score={result.best_score:.4f} "
        f"model_type={result.model_type} params={result.best_params}"
    )
    return run_id
