"""
cli._helpers

Shared private helper functions used across CLI subcommands.
Extracted from the original monolithic main.py.
"""

from __future__ import annotations

import json
import shutil
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from ..io import read_csv, read_parquet
from ..utils import get_logger

logger = get_logger("cli")


# ── JSON helpers ────────────────────────────────────────────────────
def json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def json_read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Run-id management ──────────────────────────────────────────────
def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def mark_latest(
    base_dir: Path, run_id: str, extra: Optional[Dict[str, Any]] = None
) -> None:
    payload = {
        "run_id": run_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    json_write(base_dir / "latest.json", payload)


def resolve_latest_run_id(*latest_paths: Path) -> str:
    for p in latest_paths:
        if p.exists():
            payload = json_read(p)
            rid = payload.get("run_id")
            if rid:
                return str(rid)
    raise FileNotFoundError("No latest run pointer found.")


# ── File helpers ────────────────────────────────────────────────────
def copy_to_latest(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def safe_load(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


def read_input_dataset(input_path: Path):
    if input_path.suffix.lower() == ".parquet":
        return read_parquet(input_path)
    if input_path.suffix.lower() == ".csv":
        return read_csv(input_path)
    raise ValueError(
        f"Unsupported input format: {input_path.suffix}. Use .parquet or .csv"
    )


# ── Webhook / DLQ ──────────────────────────────────────────────────
def append_dead_letter(dlq_path: Path, payload: Dict[str, Any]) -> None:
    dlq_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "payload": payload,
    }
    with dlq_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def notify_webhook(
    url: Optional[str],
    payload: Dict[str, Any],
    dlq_path: Optional[Path] = None,
) -> None:
    if not url:
        return

    max_attempts = 3
    backoff_seconds = 1.0
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5).read()
            return
        except Exception as e:
            last_error = e
            logger.warning(
                f"Webhook notify attempt {attempt}/{max_attempts} failed: {e}"
            )
            if attempt < max_attempts:
                time.sleep(backoff_seconds)
                backoff_seconds *= 2.0

    logger.warning(f"Webhook notify failed after {max_attempts} attempts: {last_error}")
    if dlq_path is not None:
        append_dead_letter(dlq_path, payload)


# ── Policy selection ───────────────────────────────────────────────
def pick_best_policy(
    summary: Dict[str, Any], prefer_models: List[str]
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for model_name, rows in summary.get("models", {}).items():
        for row in rows:
            candidates.append(
                {
                    "model": model_name,
                    "max_action_rate": row["max_action_rate"],
                    "best_threshold": row["best_threshold"],
                    "best_profit": row["best_profit"],
                }
            )

    candidates = [
        c
        for c in candidates
        if isinstance(c["best_profit"], (int, float))
        and c["best_profit"] == c["best_profit"]
    ]
    if not candidates:
        return {
            "status": "no_valid_candidate",
            "reason": "No non-NaN constrained profit found.",
        }

    max_profit = max(c["best_profit"] for c in candidates)
    top = [c for c in candidates if c["best_profit"] == max_profit]

    if len(top) == 1:
        selected = top[0]
    else:
        selected = None
        for pm in prefer_models:
            for c in top:
                if c["model"] == pm:
                    selected = c
                    break
            if selected is not None:
                break
        if selected is None:
            selected = top[0]

    return {
        "status": "ok",
        "selected": selected,
        "max_profit": max_profit,
        "num_candidates": len(candidates),
        "num_top_ties": len(top),
    }


def build_policy_payload(
    *,
    run_id: str,
    selected: Dict[str, Any],
    cost,
    prefer_models: List[str],
    model_registry: Dict[str, str],
    model_checksums: Dict[str, str],
    debug: Dict[str, Any],
    uplift_cfg,
    contract_cfg,
) -> Dict[str, Any]:
    return {
        "status": "ok",
        "policy_version": contract_cfg.policy_version,
        "feature_schema_version": contract_cfg.feature_schema_version,
        "run_id": run_id,
        "selection_basis": "maximize_net_profit_under_capacity_constraint",
        "selected_model": selected["model"],
        "selected_model_artifact": model_registry.get(selected["model"]),
        "selected_model_sha256": model_checksums.get(selected["model"]),
        "max_action_rate": selected["max_action_rate"],
        "threshold": selected["best_threshold"],
        "expected_net_profit": selected["best_profit"],
        "cost_matrix": cost.__dict__,
        "ranking_mode": uplift_cfg.ranking_mode,
        "uplift": {
            "segment_col": uplift_cfg.segment_col,
            "tp_value_by_segment": uplift_cfg.tp_value_by_segment,
            "default_tp_value": cost.tp_value,
            "fp_value": cost.fp_value,
            "fn_value": cost.fn_value,
            "tn_value": cost.tn_value,
        },
        "tie_break_preference": prefer_models,
        "debug": debug,
    }
