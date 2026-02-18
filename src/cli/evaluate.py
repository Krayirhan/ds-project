"""CLI: evaluate command — with optional SHAP explainability."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import ExperimentConfig, Paths
from ..cost_matrix import (
    CostMatrix,
    sweep_thresholds_for_profit,
    sweep_thresholds_for_profit_with_constraint,
)
from ..evaluate import evaluate_binary_classifier, sweep_thresholds
from ..experiment_tracking import ExperimentTracker
from ..io import read_parquet
from ..utils import get_logger
from ._helpers import (
    build_policy_payload,
    copy_to_latest,
    json_read,
    json_write,
    mark_latest,
    pick_best_policy,
    resolve_latest_run_id,
    safe_load,
)

logger = get_logger("cli.evaluate")


def cmd_evaluate(
    paths: Paths, cfg: ExperimentConfig, run_id: Optional[str] = None
) -> str:
    run_id = run_id or resolve_latest_run_id(
        paths.models / "latest.json",
        paths.reports_metrics / "latest.json",
    )

    run_metrics_dir = paths.reports_metrics / run_id
    run_metrics_dir.mkdir(parents=True, exist_ok=True)

    test_df = read_parquet(paths.data_processed / "test.parquet")
    model_registry_path = run_metrics_dir / "model_registry.json"
    model_checksums_path = run_metrics_dir / "model_checksums.json"
    if not model_registry_path.exists():
        raise FileNotFoundError(f"Model registry not found: {model_registry_path}")
    model_registry = json_read(model_registry_path)
    model_checksums = (
        json_read(model_checksums_path) if model_checksums_path.exists() else {}
    )

    models: Dict[str, Any] = {}
    for model_name, rel_artifact in model_registry.items():
        p = paths.project_root / rel_artifact
        m = safe_load(p)
        if m is not None:
            models[model_name] = m

    if not models:
        raise ValueError("No model artifacts could be loaded for evaluation.")

    cost = CostMatrix(
        tp_value=cfg.cost.tp_value,
        fp_value=cfg.cost.fp_value,
        fn_value=cfg.cost.fn_value,
        tn_value=cfg.cost.tn_value,
    )
    action_rates = cfg.decision.action_rates
    prefer_models = cfg.decision.prefer_models

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "cost_matrix": cost.__dict__,
        "action_rates": action_rates,
        "models": {},
    }
    decision_policies: Dict[str, Any] = {}

    # ── Optional MLflow tracking ───────────────────────────────────
    tracker = ExperimentTracker()
    with tracker.start_run(run_name=f"evaluate_{run_id}"):
        for model_name, model_obj in models.items():
            safe_name = model_name.replace("/", "_")

            evaluate_binary_classifier(
                model_obj,
                test_df,
                cfg.target_col,
                run_metrics_dir / f"{safe_name}_metrics.json",
                threshold=0.50,
                tag=f"{safe_name}_0.50",
            )
            evaluate_binary_classifier(
                model_obj,
                test_df,
                cfg.target_col,
                run_metrics_dir / f"{safe_name}_decision_metrics.json",
                threshold=cfg.decision_threshold,
                tag=f"{safe_name}_decision_{cfg.decision_threshold:.2f}",
            )
            sweep_thresholds(
                model_obj,
                test_df,
                cfg.target_col,
                run_metrics_dir / f"threshold_sweep_{safe_name}.json",
            )

            unconstrained = sweep_thresholds_for_profit(
                model_obj, test_df, cfg.target_col, cost
            )
            json_write(
                run_metrics_dir / f"profit_sweep_{safe_name}.json",
                {
                    "model": model_name,
                    "cost_matrix": cost.__dict__,
                    "best_threshold": unconstrained.best_threshold,
                    "best_profit": unconstrained.best_profit,
                    "rows": unconstrained.rows,
                },
            )

            out_rows = []
            for r in action_rates:
                constrained = sweep_thresholds_for_profit_with_constraint(
                    model=model_obj,
                    df_test=test_df,
                    target_col=cfg.target_col,
                    cost=cost,
                    max_action_rate=r,
                )
                row = {
                    "max_action_rate": r,
                    "best_threshold": constrained.best_threshold,
                    "best_profit": constrained.best_profit,
                }
                out_rows.append(row)

                json_write(
                    run_metrics_dir
                    / f"profit_sweep_{safe_name}_constrained_{int(r * 100)}.json",
                    {
                        "model": model_name,
                        "constraint": {"max_action_rate": r},
                        "cost_matrix": cost.__dict__,
                        "best_threshold": constrained.best_threshold,
                        "best_profit": constrained.best_profit,
                        "rows": constrained.rows,
                    },
                )

            summary["models"][model_name] = out_rows

            valid_rows = [
                x
                for x in out_rows
                if isinstance(x["best_profit"], (int, float))
                and x["best_profit"] == x["best_profit"]
            ]
            if valid_rows:
                selected_model_policy = max(valid_rows, key=lambda x: x["best_profit"])
                decision_policies[model_name] = build_policy_payload(
                    run_id=run_id,
                    selected={"model": model_name, **selected_model_policy},
                    cost=cost,
                    prefer_models=prefer_models,
                    model_registry=model_registry,
                    model_checksums=model_checksums,
                    debug={"selection_scope": "single_model", "model_name": model_name},
                    uplift_cfg=cfg.uplift,
                    contract_cfg=cfg.contract,
                )
                json_write(
                    run_metrics_dir / f"decision_policy_{safe_name}.json",
                    decision_policies[model_name],
                )

        summary_path = run_metrics_dir / "profit_sweep_constrained_summary.json"
        json_write(summary_path, summary)
        json_write(run_metrics_dir / "decision_policies.json", decision_policies)

        pick = pick_best_policy(summary, prefer_models=prefer_models)
        run_policy_path = run_metrics_dir / "decision_policy.json"

        if pick.get("status") != "ok":
            json_write(
                run_policy_path,
                {
                    "status": "no_policy",
                    "run_id": run_id,
                    "reason": pick.get("reason", "unknown"),
                    "summary_path": str(summary_path),
                },
            )
            copy_to_latest(
                run_policy_path, paths.reports_metrics / "decision_policy.json"
            )
            mark_latest(
                paths.reports_metrics,
                run_id,
                extra={"decision_policy": str(run_policy_path)},
            )
            return run_id

        selected = pick["selected"]
        decision_policy = build_policy_payload(
            run_id=run_id,
            selected=selected,
            cost=cost,
            prefer_models=prefer_models,
            model_registry=model_registry,
            model_checksums=model_checksums,
            debug={
                "max_profit": pick.get("max_profit"),
                "num_candidates": pick.get("num_candidates"),
                "num_top_ties": pick.get("num_top_ties"),
                "summary_path": str(summary_path),
                "champion_selection": "global_profit_under_constraint",
            },
            uplift_cfg=cfg.uplift,
            contract_cfg=cfg.contract,
        )

        json_write(run_policy_path, decision_policy)
        copy_to_latest(run_policy_path, paths.reports_metrics / "decision_policy.json")
        mark_latest(
            paths.reports_metrics,
            run_id,
            extra={"decision_policy": str(run_policy_path)},
        )

        # ── MLflow: log evaluation metrics ─────────────────────────
        tracker.log_metrics(
            {
                "champion_model": 0,  # placeholder — tag below
                "expected_net_profit": float(
                    decision_policy.get("expected_net_profit", 0)
                ),
                "threshold": float(decision_policy.get("threshold", 0.5)),
            }
        )
        tracker.set_tag(
            "champion_model", decision_policy.get("selected_model", "unknown")
        )
        tracker.log_artifact(run_policy_path)

        # ── Auto-generate explainability artifacts ─────────────────
        _generate_explainability(
            paths=paths,
            cfg=cfg,
            run_id=run_id,
            champion_model_name=decision_policy.get("selected_model"),
            models=models,
            test_df=test_df,
            tracker=tracker,
        )

        logger.info(
            "DECISION POLICY -> "
            f"model={decision_policy['selected_model']} "
            f"max_action_rate={decision_policy['max_action_rate']} "
            f"thr={decision_policy['threshold']} "
            f"profit={decision_policy['expected_net_profit']}"
        )

    return run_id


def _generate_explainability(
    *,
    paths: Paths,
    cfg: ExperimentConfig,
    run_id: str,
    champion_model_name: Optional[str],
    models: Dict[str, Any],
    test_df: Any,
    tracker: ExperimentTracker,
) -> None:
    """Post-evaluation: generate permutation importance and optional SHAP."""
    try:
        from ..explain import (
            compute_permutation_importance,
            compute_shap_values,
            save_explainability_report,
        )
    except Exception:
        logger.info("Explainability module not available — skipping.")
        return

    if not champion_model_name or champion_model_name not in models:
        return

    model = models[champion_model_name]
    X_test = test_df.drop(columns=[cfg.target_col])
    y_test = test_df[cfg.target_col].astype(int).values
    run_metrics_dir = paths.reports_metrics / run_id

    # Permutation importance (always works)
    try:
        perm_report = compute_permutation_importance(
            model, X_test, y_test, seed=cfg.seed
        )
        out_perm = run_metrics_dir / "permutation_importance.json"
        save_explainability_report(perm_report, out_perm)
        tracker.log_artifact(out_perm)
        logger.info(f"Permutation importance saved → {out_perm}")
    except Exception as exc:
        logger.warning(f"Permutation importance failed: {exc}")

    # SHAP (optional — requires shap package)
    try:
        shap_report = compute_shap_values(model, X_test, max_samples=500)
        if shap_report is not None:
            out_shap = run_metrics_dir / "shap_summary.json"
            save_explainability_report(shap_report, out_shap)
            tracker.log_artifact(out_shap)
            logger.info(f"SHAP summary saved → {out_shap}")
    except Exception as exc:
        logger.warning(f"SHAP computation failed: {exc}")
