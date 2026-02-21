from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.config import ExperimentConfig, Paths
import src.cli.monitor as monitor


def _paths(tmp_path: Path) -> Paths:
    p = Paths(project_root=tmp_path)
    p.data_processed.mkdir(parents=True, exist_ok=True)
    p.reports_metrics.mkdir(parents=True, exist_ok=True)
    p.reports_predictions.mkdir(parents=True, exist_ok=True)
    p.reports_monitoring.mkdir(parents=True, exist_ok=True)
    p.models.mkdir(parents=True, exist_ok=True)
    return p


def _cfg() -> ExperimentConfig:
    return ExperimentConfig()


def test_cmd_monitor_relative_paths_and_optional_branches(monkeypatch, tmp_path: Path):
    paths = _paths(tmp_path)
    cfg = _cfg()
    run_id = "run-42"

    # model artifact exists
    model_artifact = "models/xgb.joblib"
    (paths.project_root / model_artifact).write_text("bin", encoding="utf-8")

    # optional reference artifacts to exercise branches
    run_dir = paths.reports_metrics / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reference_stats.json").write_text(json.dumps({"lead_time": {"mean": 10}}), encoding="utf-8")
    (run_dir / "reference_correlations.json").write_text(json.dumps({"lead_time|adr": 0.2}), encoding="utf-8")
    (run_dir / "feature_importance.json").write_text(json.dumps({"lead_time": 0.6, "adr": 0.4}), encoding="utf-8")
    (paths.reports_metrics / "feature_importance.prev.json").write_text(
        json.dumps({"lead_time": 0.4, "adr": 0.6}),
        encoding="utf-8",
    )

    ref_pred_path = paths.reports_predictions / run_id / "actions.parquet"
    ref_pred_path.parent.mkdir(parents=True, exist_ok=True)
    ref_pred_path.write_text("stub", encoding="utf-8")

    df_cur = pd.DataFrame(
        {
            "lead_time": [1, 2, 3],
            "adr": [100.0, 200.0, 300.0],
            "arrival_date_month": ["January", "February", "March"],
            "is_canceled": [0, 1, 0],
        }
    )
    actions_cur = pd.DataFrame({"proba": [0.2, 0.8, 0.4], "action": [0, 1, 0]})
    actions_ref = pd.DataFrame({"proba": [0.1, 0.6, 0.3], "action": [0, 1, 0]})

    read_input_calls: list[Path] = []
    read_parquet_calls: list[Path] = []
    drift_called = {"corr": 0, "importance": 0}

    def _read_input_dataset(path: Path):
        read_input_calls.append(path)
        return df_cur

    def _read_parquet(path: Path):
        read_parquet_calls.append(path)
        if path == paths.data_processed / "dataset.parquet":
            return df_cur
        if path == ref_pred_path:
            return actions_ref
        raise AssertionError(f"unexpected read_parquet path: {path}")

    def _corr_drift(*args, **kwargs):
        drift_called["corr"] += 1
        return SimpleNamespace(n_drifted=1, drifted_pairs=[["lead_time", "adr"]], summary="corr")

    def _imp_drift(*args, **kwargs):
        drift_called["importance"] += 1
        return SimpleNamespace(
            n_changed=2,
            rank_correlation=0.7,
            changed_features=["lead_time"],
            summary="importance",
        )

    policy = SimpleNamespace(
        raw={
            "policy_version": cfg.contract.policy_version,
            "run_id": run_id,
            "selected_model": "xgb",
        },
        selected_model_artifact=model_artifact,
    )

    monkeypatch.setattr(monitor, "load_decision_policy", lambda _: policy)
    monkeypatch.setattr(monitor.joblib, "load", lambda _: object())
    monkeypatch.setattr(monitor, "read_input_dataset", _read_input_dataset)
    monkeypatch.setattr(monitor, "read_parquet", _read_parquet)
    monkeypatch.setattr(
        monitor,
        "load_feature_spec",
        lambda _: {"numeric": ["lead_time", "adr"], "categorical": ["arrival_date_month"]},
    )
    monkeypatch.setattr(monitor, "predict_with_policy", lambda **kwargs: (actions_cur, {"n_rows": 3}))
    monkeypatch.setattr(
        monitor,
        "validate_distributions",
        lambda *_args, **_kwargs: SimpleNamespace(
            passed=False,
            violations=[{"column": "adr", "message": "drift"}],
            summary="distribution failed",
        ),
    )
    monkeypatch.setattr(
        monitor,
        "detect_label_drift",
        lambda *_args, **_kwargs: SimpleNamespace(
            ref_positive_rate=0.3,
            cur_positive_rate=0.4,
            is_drifted=True,
            summary="label drift",
        ),
    )
    monkeypatch.setattr(monitor, "detect_correlation_drift", _corr_drift)
    monkeypatch.setattr(
        monitor,
        "validate_data_volume",
        lambda *_args, **_kwargs: SimpleNamespace(
            current_rows=3,
            expected_range=(1, 10),
            is_anomalous=False,
            summary="ok",
        ),
    )
    monkeypatch.setattr(monitor, "detect_feature_importance_drift", _imp_drift)
    monkeypatch.setattr(monitor, "data_drift_report", lambda **kwargs: {"psi_scores": {}})
    monkeypatch.setattr(monitor, "prediction_drift_report", lambda **kwargs: {"psi": 0.01})
    monkeypatch.setattr(monitor, "outcome_monitoring_report", lambda **kwargs: {"profit": 100.0})
    monkeypatch.setattr(monitor, "build_alerts", lambda **kwargs: {"any_alert": False})
    monkeypatch.setattr(monitor, "json_write", lambda *args, **kwargs: None)
    monkeypatch.setattr(monitor, "mark_latest", lambda *args, **kwargs: None)
    monkeypatch.setattr(monitor, "notify_webhook", lambda *args, **kwargs: None)

    out = monitor.cmd_monitor(
        paths=paths,
        cfg=cfg,
        input_path="data/current.parquet",
        outcome_path="data/outcome.csv",
        actual_col="is_canceled",
        run_id=run_id,
    )
    assert out == run_id

    # relative paths should be resolved under project root
    assert read_input_calls[0] == (paths.project_root / "data/current.parquet")
    assert read_input_calls[1] == (paths.project_root / "data/outcome.csv")
    # use pre-existing reference predictions file branch
    assert ref_pred_path in read_parquet_calls
    # optional drift branches should run when artifacts exist
    assert drift_called["corr"] == 1
    assert drift_called["importance"] == 1
