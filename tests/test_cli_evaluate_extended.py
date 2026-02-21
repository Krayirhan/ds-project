from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.config import ExperimentConfig, Paths
import src.cli.evaluate as cli_eval


def _paths(tmp_path: Path) -> Paths:
    p = Paths(project_root=tmp_path)
    p.data_processed.mkdir(parents=True, exist_ok=True)
    p.models.mkdir(parents=True, exist_ok=True)
    p.reports_metrics.mkdir(parents=True, exist_ok=True)
    return p


def _cfg() -> ExperimentConfig:
    return ExperimentConfig()


def _model():
    m = SimpleNamespace()
    m.predict_proba = lambda X: np.column_stack([np.full(len(X), 0.7), np.full(len(X), 0.3)])
    return m


def test_cmd_evaluate_no_policy_branch(monkeypatch, tmp_path: Path):
    paths = _paths(tmp_path)
    cfg = _cfg()
    run_id = "run-nopol"
    run_dir = paths.reports_metrics / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_registry.json").write_text("{}", encoding="utf-8")
    (run_dir / "model_checksums.json").write_text("{}", encoding="utf-8")

    (paths.data_processed / "test.parquet").write_text("stub", encoding="utf-8")
    (paths.models / "xgb.joblib").write_text("bin", encoding="utf-8")

    monkeypatch.setattr(cli_eval, "read_parquet", lambda *_args, **_kwargs: pd.DataFrame({"f": [1, 2], "is_canceled": [0, 1]}))
    monkeypatch.setattr(cli_eval, "json_read", lambda p: {"xgb": "models/xgb.joblib"} if p.name == "model_registry.json" else {"xgb": "sha"})
    monkeypatch.setattr(cli_eval, "safe_load", lambda *_args, **_kwargs: _model())
    monkeypatch.setattr(cli_eval, "evaluate_binary_classifier", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_eval, "sweep_thresholds", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli_eval,
        "sweep_thresholds_for_profit",
        lambda *args, **kwargs: SimpleNamespace(best_threshold=0.5, best_profit=1.0, rows=[]),
    )
    monkeypatch.setattr(
        cli_eval,
        "sweep_thresholds_for_profit_with_constraint",
        lambda *args, **kwargs: SimpleNamespace(best_threshold=0.5, best_profit=float("nan"), rows=[]),
    )
    monkeypatch.setattr(cli_eval, "pick_best_policy", lambda *args, **kwargs: {"status": "no_valid_candidate", "reason": "none"})
    monkeypatch.setattr(cli_eval, "json_write", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_eval, "copy_to_latest", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_eval, "mark_latest", lambda *args, **kwargs: None)

    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    tracker = SimpleNamespace(start_run=lambda **kwargs: _Ctx())
    monkeypatch.setattr(cli_eval, "ExperimentTracker", lambda: tracker)

    out = cli_eval.cmd_evaluate(paths, cfg, run_id=run_id)
    assert out == run_id


def test_generate_explainability_import_error_and_missing_champion(monkeypatch, tmp_path: Path):
    paths = _paths(tmp_path)
    cfg = _cfg()
    tracker = SimpleNamespace(log_artifact=lambda *_args, **_kwargs: None)
    test_df = pd.DataFrame({"f1": [1, 2], "is_canceled": [0, 1]})

    # import failure branch
    monkeypatch.setitem(__import__("sys").modules, "src.explain", None)
    cli_eval._generate_explainability(
        paths=paths,
        cfg=cfg,
        run_id="r1",
        champion_model_name="xgb",
        models={"xgb": object()},
        test_df=test_df,
        tracker=tracker,
    )

    # champion missing branch
    monkeypatch.delitem(__import__("sys").modules, "src.explain", raising=False)
    import src.explain as explain_mod

    monkeypatch.setitem(__import__("sys").modules, "src.explain", explain_mod)
    cli_eval._generate_explainability(
        paths=paths,
        cfg=cfg,
        run_id="r2",
        champion_model_name="missing",
        models={"xgb": object()},
        test_df=test_df,
        tracker=tracker,
    )


def test_generate_explainability_success_and_shap_exception(monkeypatch, tmp_path: Path):
    paths = _paths(tmp_path)
    cfg = _cfg()
    run_id = "r3"
    (paths.reports_metrics / run_id).mkdir(parents=True, exist_ok=True)
    tracker_calls = {"n": 0}
    tracker = SimpleNamespace(log_artifact=lambda *_args, **_kwargs: tracker_calls.__setitem__("n", tracker_calls["n"] + 1))
    test_df = pd.DataFrame({"f1": [1, 2], "is_canceled": [0, 1]})

    import sys
    import types

    fake_explain = types.ModuleType("src.explain")
    fake_explain.compute_permutation_importance = lambda *args, **kwargs: {"kind": "perm"}
    fake_explain.compute_shap_values = lambda *args, **kwargs: {"kind": "shap"}
    fake_explain.save_explainability_report = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "src.explain", fake_explain)

    cli_eval._generate_explainability(
        paths=paths,
        cfg=cfg,
        run_id=run_id,
        champion_model_name="xgb",
        models={"xgb": _model()},
        test_df=test_df,
        tracker=tracker,
    )
    assert tracker_calls["n"] >= 1

    # shap exception branch
    fake_explain.compute_shap_values = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("shap down"))
    cli_eval._generate_explainability(
        paths=paths,
        cfg=cfg,
        run_id=run_id,
        champion_model_name="xgb",
        models={"xgb": _model()},
        test_df=test_df,
        tracker=tracker,
    )
