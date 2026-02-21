from __future__ import annotations

import sys
import types

from src.config import ValidationPolicy, load_experiment_config


def test_validation_policy_for_env_profiles():
    dev = ValidationPolicy.for_env("dev")
    stg = ValidationPolicy.for_env("staging")
    prod = ValidationPolicy.for_env("prod")
    fallback = ValidationPolicy.for_env("unknown-env")

    assert dev.staleness.severity == "warn"
    assert stg.volume.severity in {"hard_fail", "soft_fail"}
    assert prod.strict_inference_schema is True
    assert fallback.staleness.severity == dev.staleness.severity


def test_validation_policy_for_phase_profiles():
    train_phase = ValidationPolicy.for_phase("train", env="prod")
    assert train_phase.inference_schema.enabled is False
    assert train_phase.serving_skew.enabled is False

    predict_phase = ValidationPolicy.for_phase("predict", env="prod")
    assert predict_phase.raw_schema.enabled is False
    assert predict_phase.strict_inference_schema is True
    assert predict_phase.inference_schema.severity == "hard_fail"

    monitor_phase = ValidationPolicy.for_phase("monitor", env="prod")
    assert monitor_phase.psi_drift.severity == "hard_fail"
    assert monitor_phase.label_drift.severity == "hard_fail"

    preprocess_phase = ValidationPolicy.for_phase("preprocess", env="dev")
    assert preprocess_phase.duplicate.enabled is True

    # unknown phase -> base policy
    unknown = ValidationPolicy.for_phase("other", env="dev")  # type: ignore[arg-type]
    assert unknown.duplicate.enabled is True


def test_load_experiment_config_missing_file_returns_defaults(tmp_path):
    cfg = load_experiment_config(tmp_path / "missing.yaml")
    assert cfg.target_col == "is_canceled"
    assert cfg.cost.tp_value > 0


def test_load_experiment_config_default_path_executes():
    cfg = load_experiment_config()
    assert cfg.target_col


def test_load_experiment_config_yaml_importerror_returns_defaults(tmp_path, monkeypatch):
    p = tmp_path / "params.yaml"
    p.write_text("experiment:\n  seed: 123\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "yaml", None)
    cfg = load_experiment_config(p)
    assert cfg.seed == 42


def test_load_experiment_config_yaml_parse_error_returns_defaults(tmp_path, monkeypatch):
    p = tmp_path / "params.yaml"
    p.write_text("experiment: [bad", encoding="utf-8")

    yaml_mod = types.SimpleNamespace(
        safe_load=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad yaml"))
    )
    monkeypatch.setitem(sys.modules, "yaml", yaml_mod)

    cfg = load_experiment_config(p)
    assert cfg.seed == 42


def test_load_experiment_config_success(tmp_path, monkeypatch):
    p = tmp_path / "params.yaml"
    p.write_text(
        "\n".join(
            [
                "experiment:",
                "  target_col: cancelled",
                "  test_size: 0.3",
                "  seed: 7",
                "  cv_folds: 3",
                "cost_matrix:",
                "  tp_value: 200",
                "  fp_value: -10",
                "  fn_value: -150",
                "  tn_value: 1",
                "decision:",
                "  action_rates: [0.1, 0.2]",
            ]
        ),
        encoding="utf-8",
    )

    import yaml

    monkeypatch.setitem(sys.modules, "yaml", yaml)
    cfg = load_experiment_config(p)
    assert cfg.target_col == "cancelled"
    assert cfg.test_size == 0.3
    assert cfg.seed == 7
    assert cfg.cv_folds == 3
    assert cfg.cost.tp_value == 200.0
    assert cfg.decision.action_rates == [0.1, 0.2]
