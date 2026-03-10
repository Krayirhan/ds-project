"""Evaluate rollback criteria from latest monitoring report.

This utility is used by CI workflows and can also be run locally.
It writes GitHub Action outputs when GITHUB_OUTPUT is present.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _model_quality_reasons(alerts: dict[str, Any]) -> list[str]:
    reasons: list[str] = []

    # Critical model/business signals from latest monitoring report.
    if bool(alerts.get("profit_drop", False)):
        reasons.append("profit_drop")
    if bool(alerts.get("prediction_drift", False)):
        reasons.append("prediction_drift")
    if bool(alerts.get("data_drift", False)) and bool(
        alerts.get("action_rate_deviation", False)
    ):
        reasons.append("data_drift+action_rate_deviation")

    return reasons


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _ops_signals(payload: dict[str, Any]) -> dict[str, Any]:
    # Payload is intentionally tolerant to support future schema changes.
    return {
        "available": bool(payload.get("ops_signals_available", False)),
        "high_5xx_rate": bool(payload.get("high_5xx_rate", False)),
        "high_p95_latency": bool(payload.get("high_p95_latency", False)),
        "error_budget_burn_fast": bool(payload.get("error_budget_burn_fast", False)),
        "five_xx_ratio": payload.get("five_xx_ratio"),
        "p95_latency_seconds": payload.get("p95_latency_seconds"),
    }


def _evaluate_rollback_matrix(
    *,
    model_quality_reasons: list[str],
    ops: dict[str, Any],
) -> list[str]:
    """Correlated rollback matrix.

    Rollback requires multi-signal correlation:
      - operational 5xx breach
      - operational p95 latency breach
      - at least one model/business quality breach
    """
    reasons: list[str] = []
    if (
        bool(model_quality_reasons)
        and bool(ops.get("high_5xx_rate", False))
        and bool(ops.get("high_p95_latency", False))
    ):
        reasons.append("correlated_5xx+latency+model_quality")
        reasons.extend(model_quality_reasons)
    return reasons


def _evaluate_non_rollback_signals(
    alerts: dict[str, Any],
    *,
    model_quality_reasons: list[str],
    ops: dict[str, Any],
) -> list[str]:
    signals: list[str] = []

    # Operational signal only: this should not trigger policy rollback by itself.
    if bool(alerts.get("data_volume_anomaly", False)):
        signals.append("data_volume_anomaly")

    if not bool(ops.get("available", False)):
        signals.append("ops_signals_unavailable")
        if model_quality_reasons:
            signals.append("model_quality_without_ops_correlation")
        return signals

    ops_pair = bool(ops.get("high_5xx_rate", False)) and bool(
        ops.get("high_p95_latency", False)
    )
    if model_quality_reasons and not ops_pair:
        signals.append("model_quality_without_ops_correlation")
    if ops_pair and not model_quality_reasons:
        signals.append("ops_correlation_without_model_quality")

    if bool(ops.get("error_budget_burn_fast", False)):
        signals.append("error_budget_burn_fast")

    return signals


def _write_github_output(
    output_path: Path,
    *,
    rollback_required: bool,
    rollback_reasons: list[str],
    non_rollback_signals: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(f"rollback_required={'true' if rollback_required else 'false'}\n")
        fh.write(
            "rollback_reasons="
            + (",".join(rollback_reasons) if rollback_reasons else "none")
            + "\n"
        )
        fh.write(
            "non_rollback_signals="
            + (",".join(non_rollback_signals) if non_rollback_signals else "none")
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate whether rollback is required from monitoring report."
    )
    parser.add_argument(
        "--report-path",
        default="reports/monitoring/latest_monitoring_report.json",
        help="Path to monitoring report JSON",
    )
    parser.add_argument(
        "--ops-signals-path",
        default="reports/monitoring/ops_signals.json",
        help="Path to operational (5xx/latency) signal JSON",
    )
    parser.add_argument(
        "--fail-on-rollback",
        action="store_true",
        help="Exit with status 42 when rollback is required.",
    )
    args = parser.parse_args()

    report_path = Path(args.report_path)
    rollback_required = False
    rollback_reasons: list[str] = []
    non_rollback_signals: list[str] = []

    if not report_path.exists():
        print(f"No monitoring report found: {report_path}. Skipping rollback check.")
    else:
        payload = _load_json(report_path)
        alerts = payload.get("alerts", {})
        if not isinstance(alerts, dict):
            alerts = {}
        ops = _ops_signals(_load_json(Path(args.ops_signals_path)))

        print(f"alerts={alerts}")
        print(f"ops_signals={ops}")

        model_reasons = _model_quality_reasons(alerts)
        rollback_reasons = _evaluate_rollback_matrix(
            model_quality_reasons=model_reasons,
            ops=ops,
        )
        non_rollback_signals = _evaluate_non_rollback_signals(
            alerts,
            model_quality_reasons=model_reasons,
            ops=ops,
        )
        rollback_required = bool(rollback_reasons)

    print(f"rollback_required={rollback_required}")
    print(f"rollback_reasons={rollback_reasons if rollback_reasons else ['none']}")
    print(
        f"non_rollback_signals={non_rollback_signals if non_rollback_signals else ['none']}"
    )

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        _write_github_output(
            Path(github_output),
            rollback_required=rollback_required,
            rollback_reasons=rollback_reasons,
            non_rollback_signals=non_rollback_signals,
        )

    if args.fail_on_rollback and rollback_required:
        return 42
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
