from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.config import CheckConfig, ExperimentConfig, Paths, ValidationPolicy
import src.cli.preprocess as prep


def _paths(tmp_path: Path) -> Paths:
    p = Paths(project_root=tmp_path)
    p.data_raw.mkdir(parents=True, exist_ok=True)
    p.data_processed.mkdir(parents=True, exist_ok=True)
    p.reports_metrics.mkdir(parents=True, exist_ok=True)
    return p


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lead_time": [10, 20, 30, 40],
            "adr": [100.0, 120.0, 130.0, 140.0],
            "arrival_date_month": ["January", "February", "March", "April"],
            "hotel": ["City", "Resort", "City", "City"],
            "reservation_status": ["A", "B", "C", "D"],
            "is_canceled": ["no", "yes", "no", "yes"],
        }
    )


def _cfg(*, duplicate_severity: str = "warn", duplicate_threshold: float = 0.02, staleness_severity: str = "warn") -> ExperimentConfig:
    cfg = ExperimentConfig()
    policy = ValidationPolicy(
        duplicate=CheckConfig(severity=duplicate_severity, enabled=True, threshold=duplicate_threshold),
        staleness=CheckConfig(severity=staleness_severity, enabled=True, threshold=180.0),
    )
    object.__setattr__(cfg, "validation", policy)
    return cfg


def _patch_common(
    monkeypatch,
    *,
    df: pd.DataFrame,
    staleness: SimpleNamespace | None = None,
    dup_count: int = 0,
    profile_passed: bool = True,
    volume_fn=None,
):
    monkeypatch.setattr(prep, "read_csv", lambda *_args, **_kwargs: df)
    monkeypatch.setattr(
        prep,
        "check_data_staleness",
        lambda *_args, **_kwargs: staleness
        or SimpleNamespace(is_stale=False, age_days=1, summary="fresh"),
    )
    monkeypatch.setattr(prep, "basic_schema_checks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prep, "validate_target_labels", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prep, "null_ratio_report", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(prep, "detect_duplicates", lambda *_args, **_kwargs: SimpleNamespace(n_duplicates=dup_count))
    monkeypatch.setattr(prep, "detect_row_anomalies", lambda *_args, **_kwargs: SimpleNamespace(n_anomalies=0))
    monkeypatch.setattr(prep, "validate_raw_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        prep,
        "preprocess_basic",
        lambda **_kwargs: df.assign(is_canceled=[0, 1, 0, 1]).drop(columns=["reservation_status"]),
    )
    monkeypatch.setattr(
        prep,
        "infer_feature_spec",
        lambda *_args, **_kwargs: SimpleNamespace(
            numeric=["lead_time", "adr"],
            categorical=["arrival_date_month", "hotel"],
        ),
    )
    monkeypatch.setattr(prep, "validate_processed_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        prep,
        "run_validation_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            passed=profile_passed,
            hard_failures=([] if profile_passed else ["duplicate"]),
        ),
    )
    monkeypatch.setattr(prep, "generate_reference_stats", lambda *_args, **_kwargs: {"lead_time": {"mean": 25.0}})
    monkeypatch.setattr(prep, "generate_reference_categories", lambda *_args, **_kwargs: {"hotel": ["City"]})
    monkeypatch.setattr(prep, "generate_reference_correlations", lambda *_args, **_kwargs: {"lead_time|adr": 0.2})
    monkeypatch.setattr(prep, "get_schema_fingerprint", lambda *_args, **_kwargs: {"fingerprint": "fp-1", "n_columns": 5})
    monkeypatch.setattr(prep, "write_parquet", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prep, "json_write", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prep, "sha256_file", lambda *_args, **_kwargs: "sha-1")
    monkeypatch.setattr(
        prep,
        "validate_data_volume",
        volume_fn
        or (lambda *_args, **_kwargs: SimpleNamespace(
            current_rows=4,
            expected_range=(2, 8),
            is_anomalous=False,
            summary="ok",
        )),
    )


def test_preprocess_staleness_hard_fail_raises(monkeypatch, tmp_path: Path):
    paths = _paths(tmp_path)
    cfg = _cfg(staleness_severity="hard_fail")
    _patch_common(
        monkeypatch,
        df=_df(),
        staleness=SimpleNamespace(is_stale=True, age_days=999, summary="old"),
    )
    with pytest.raises(ValueError, match="Stale data blocked by policy"):
        prep.cmd_preprocess(paths, cfg)


def test_preprocess_staleness_soft_fail_and_duplicate_soft_fail(monkeypatch, tmp_path: Path):
    paths = _paths(tmp_path)
    cfg = _cfg(duplicate_severity="soft_fail", duplicate_threshold=0.10, staleness_severity="soft_fail")
    _patch_common(
        monkeypatch,
        df=_df(),
        staleness=SimpleNamespace(is_stale=True, age_days=200, summary="stale"),
        dup_count=2,  # 2/4 > 10%
    )
    prep.cmd_preprocess(paths, cfg)


def test_preprocess_duplicate_warn_branch(monkeypatch, tmp_path: Path):
    paths = _paths(tmp_path)
    cfg = _cfg(duplicate_severity="warn", duplicate_threshold=0.10, staleness_severity="warn")
    _patch_common(monkeypatch, df=_df(), dup_count=2)
    prep.cmd_preprocess(paths, cfg)


def test_preprocess_duplicate_hard_fail_branch(monkeypatch, tmp_path: Path):
    paths = _paths(tmp_path)
    cfg = _cfg(duplicate_severity="hard_fail", duplicate_threshold=0.10, staleness_severity="warn")
    _patch_common(monkeypatch, df=_df(), dup_count=2)
    with pytest.raises(ValueError, match="Duplicate check blocked by policy"):
        prep.cmd_preprocess(paths, cfg)


def test_preprocess_validation_profile_failure_raises(monkeypatch, tmp_path: Path):
    paths = _paths(tmp_path)
    cfg = _cfg()
    _patch_common(monkeypatch, df=_df(), profile_passed=False)
    with pytest.raises(ValueError, match="Validation profile FAILED"):
        prep.cmd_preprocess(paths, cfg)


def test_preprocess_lineage_expected_rows_from_previous_run(monkeypatch, tmp_path: Path):
    paths = _paths(tmp_path)
    cfg = _cfg()
    (paths.reports_metrics / "data_lineage_preprocess.json").write_text(
        '{"processed_rows": 123}',
        encoding="utf-8",
    )

    captured = {"expected_rows": None}

    def _volume(df, *, expected_rows, tolerance_ratio=0.50):
        captured["expected_rows"] = expected_rows
        return SimpleNamespace(
            current_rows=len(df),
            expected_range=(100, 150),
            is_anomalous=False,
            summary="ok",
        )

    _patch_common(monkeypatch, df=_df(), volume_fn=_volume)
    prep.cmd_preprocess(paths, cfg)
    assert captured["expected_rows"] == 123


def test_preprocess_lineage_invalid_json_falls_back_to_current_len(monkeypatch, tmp_path: Path):
    paths = _paths(tmp_path)
    cfg = _cfg()
    df = _df()
    (paths.reports_metrics / "data_lineage_preprocess.json").write_text(
        "{invalid-json",
        encoding="utf-8",
    )

    captured = {"expected_rows": None}

    def _volume(df_input, *, expected_rows, tolerance_ratio=0.50):
        captured["expected_rows"] = expected_rows
        return SimpleNamespace(
            current_rows=len(df_input),
            expected_range=(1, 10),
            is_anomalous=False,
            summary="ok",
        )

    _patch_common(monkeypatch, df=df, volume_fn=_volume)
    prep.cmd_preprocess(paths, cfg)
    assert captured["expected_rows"] == len(df)
