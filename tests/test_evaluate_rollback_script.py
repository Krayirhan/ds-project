import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/evaluate_rollback.py")


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), *args]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd, check=False, text=True, capture_output=True, env=merged_env
    )


def test_no_report_defaults_to_no_rollback(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    p = _run("--report-path", str(missing))
    assert p.returncode == 0
    assert "rollback_required=False" in p.stdout
    assert "rollback_reasons=['none']" in p.stdout


def test_profit_drop_without_ops_correlation_does_not_trigger_rollback(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    ops = tmp_path / "ops.json"
    report.write_text(
        json.dumps({"alerts": {"profit_drop": True}}, ensure_ascii=True),
        encoding="utf-8",
    )
    ops.write_text(
        json.dumps(
            {
                "ops_signals_available": True,
                "high_5xx_rate": False,
                "high_p95_latency": False,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    p = _run("--report-path", str(report), "--ops-signals-path", str(ops))
    assert p.returncode == 0
    assert "rollback_required=False" in p.stdout
    assert "model_quality_without_ops_correlation" in p.stdout


def test_correlated_signals_trigger_rollback(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    ops = tmp_path / "ops.json"
    report.write_text(
        json.dumps(
            {
                "alerts": {
                    "data_drift": True,
                    "action_rate_deviation": True,
                }
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    ops.write_text(
        json.dumps(
            {
                "ops_signals_available": True,
                "high_5xx_rate": True,
                "high_p95_latency": True,
                "five_xx_ratio": 0.0023,
                "p95_latency_seconds": 0.41,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    p = _run("--report-path", str(report), "--ops-signals-path", str(ops))
    assert p.returncode == 0
    assert "rollback_required=True" in p.stdout
    assert "correlated_5xx+latency+model_quality" in p.stdout
    assert "data_drift+action_rate_deviation" in p.stdout


def test_data_volume_anomaly_does_not_trigger_rollback(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    ops = tmp_path / "ops.json"
    report.write_text(
        json.dumps({"alerts": {"data_volume_anomaly": True}}, ensure_ascii=True),
        encoding="utf-8",
    )
    ops.write_text(
        json.dumps(
            {
                "ops_signals_available": True,
                "high_5xx_rate": False,
                "high_p95_latency": False,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    p = _run("--report-path", str(report), "--ops-signals-path", str(ops))
    assert p.returncode == 0
    assert "rollback_required=False" in p.stdout
    assert "rollback_reasons=['none']" in p.stdout
    assert "non_rollback_signals=['data_volume_anomaly']" in p.stdout


def test_missing_ops_signals_blocks_rollback_even_with_model_alert(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    missing_ops = tmp_path / "ops-missing.json"
    report.write_text(
        json.dumps({"alerts": {"prediction_drift": True}}, ensure_ascii=True),
        encoding="utf-8",
    )
    p = _run("--report-path", str(report), "--ops-signals-path", str(missing_ops))
    assert p.returncode == 0
    assert "rollback_required=False" in p.stdout
    assert "ops_signals_unavailable" in p.stdout
    assert "model_quality_without_ops_correlation" in p.stdout


def test_fail_on_rollback_returns_42(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    ops = tmp_path / "ops.json"
    report.write_text(
        json.dumps({"alerts": {"prediction_drift": True}}, ensure_ascii=True),
        encoding="utf-8",
    )
    ops.write_text(
        json.dumps(
            {
                "ops_signals_available": True,
                "high_5xx_rate": True,
                "high_p95_latency": True,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    p = _run(
        "--report-path",
        str(report),
        "--ops-signals-path",
        str(ops),
        "--fail-on-rollback",
    )
    assert p.returncode == 42


def test_github_output_written(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    ops = tmp_path / "ops.json"
    report.write_text(
        json.dumps({"alerts": {"prediction_drift": True}}, ensure_ascii=True),
        encoding="utf-8",
    )
    ops.write_text(
        json.dumps(
            {
                "ops_signals_available": True,
                "high_5xx_rate": True,
                "high_p95_latency": True,
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "gh_output.txt"
    p = _run(
        "--report-path",
        str(report),
        "--ops-signals-path",
        str(ops),
        env={"GITHUB_OUTPUT": str(out)},
    )
    assert p.returncode == 0
    text = out.read_text(encoding="utf-8")
    assert "rollback_required=true" in text
    assert (
        "rollback_reasons=correlated_5xx+latency+model_quality,prediction_drift" in text
    )
    assert "non_rollback_signals=none" in text
