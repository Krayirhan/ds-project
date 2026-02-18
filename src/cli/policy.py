"""CLI: promote-policy, rollback-policy, retry-webhook-dlq commands."""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from typing import Dict, List, Optional

from ..config import Paths
from ..utils import get_logger
from ._helpers import json_write, mark_latest

logger = get_logger("cli.policy")


def cmd_promote_policy(paths: Paths, run_id: str, slot: str = "default") -> None:
    src = paths.reports_metrics / run_id / "decision_policy.json"
    if not src.exists():
        raise FileNotFoundError(f"Run policy not found: {src}")
    if slot not in {"default", "blue", "green"}:
        raise ValueError("slot must be one of: default, blue, green")

    dst = (
        paths.reports_metrics / "decision_policy.json"
        if slot == "default"
        else paths.reports_metrics / f"decision_policy.{slot}.json"
    )
    bak = (
        paths.reports_metrics / "decision_policy.previous.json"
        if slot == "default"
        else paths.reports_metrics / f"decision_policy.{slot}.previous.json"
    )
    if dst.exists():
        shutil.copyfile(dst, bak)
    shutil.copyfile(src, dst)
    if slot in {"blue", "green"}:
        json_write(paths.reports_metrics / "active_slot.json", {"active_slot": slot})
    mark_latest(
        paths.reports_metrics,
        run_id,
        extra={"decision_policy": str(src), "promoted": True, "slot": slot},
    )
    logger.info(f"Promoted decision policy -> {src} to slot={slot}")


def cmd_rollback_policy(paths: Paths, slot: str = "default") -> None:
    if slot not in {"default", "blue", "green"}:
        raise ValueError("slot must be one of: default, blue, green")
    dst = (
        paths.reports_metrics / "decision_policy.json"
        if slot == "default"
        else paths.reports_metrics / f"decision_policy.{slot}.json"
    )
    bak = (
        paths.reports_metrics / "decision_policy.previous.json"
        if slot == "default"
        else paths.reports_metrics / f"decision_policy.{slot}.previous.json"
    )
    if not bak.exists():
        raise FileNotFoundError("No previous policy backup found.")
    shutil.copyfile(bak, dst)
    if slot in {"blue", "green"}:
        json_write(paths.reports_metrics / "active_slot.json", {"active_slot": slot})
    logger.info(f"Rolled back decision policy from backup for slot={slot}")


def cmd_retry_webhook_dlq(
    paths: Paths, webhook_url: Optional[str] = None
) -> Dict[str, int]:
    dlq_path = paths.reports_monitoring / "dead_letter_webhooks.jsonl"
    url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")
    if not dlq_path.exists():
        return {"retried": 0, "success": 0, "failed": 0}
    if not url:
        raise ValueError("webhook url is required (arg or ALERT_WEBHOOK_URL env)")

    lines = dlq_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {"retried": 0, "success": 0, "failed": 0}

    survivors: List[str] = []
    retried = 0
    success = 0
    failed = 0

    for line in lines:
        retried += 1
        try:
            row = json.loads(line)
            payload = row.get("payload", {})
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5).read()
            success += 1
        except Exception:
            failed += 1
            survivors.append(line)

    if survivors:
        dlq_path.write_text("\n".join(survivors) + "\n", encoding="utf-8")
    else:
        dlq_path.unlink(missing_ok=True)

    result = {"retried": retried, "success": success, "failed": failed}
    logger.info(f"Webhook DLQ retry result: {result}")
    return result
