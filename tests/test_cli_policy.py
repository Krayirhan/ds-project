"""Tests for src/cli/policy.py — cmd_promote_policy, cmd_rollback_policy, cmd_retry_webhook_dlq."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Paths


@pytest.fixture()
def paths(tmp_path: Path) -> Paths:
    p = Paths(project_root=tmp_path)
    p.reports_metrics.mkdir(parents=True, exist_ok=True)
    p.reports_monitoring.mkdir(parents=True, exist_ok=True)
    return p


def _write_run_policy(paths: Paths, run_id: str) -> Path:
    run_dir = paths.reports_metrics / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    policy = {"run_id": run_id, "threshold": 0.5, "policy_version": "v1"}
    policy_file = run_dir / "decision_policy.json"
    policy_file.write_text(json.dumps(policy))
    return policy_file


# ── cmd_promote_policy ────────────────────────────────────────────────────────


class TestCmdPromotePolicy:
    def test_raises_if_run_policy_not_found(self, paths):
        from src.cli.policy import cmd_promote_policy

        with pytest.raises(FileNotFoundError, match="Run policy not found"):
            cmd_promote_policy(paths, run_id="nonexistent-run")

    def test_raises_on_invalid_slot(self, paths):
        from src.cli.policy import cmd_promote_policy

        _write_run_policy(paths, "run-001")
        with pytest.raises(ValueError, match="slot must be one of"):
            cmd_promote_policy(paths, run_id="run-001", slot="canary")

    def test_promote_to_default_slot(self, paths):
        from src.cli.policy import cmd_promote_policy

        _write_run_policy(paths, "run-001")

        with patch("src.cli.policy.mark_latest"):
            cmd_promote_policy(paths, run_id="run-001", slot="default")

        active_policy = paths.reports_metrics / "decision_policy.json"
        assert active_policy.exists()
        data = json.loads(active_policy.read_text())
        assert data["run_id"] == "run-001"

    def test_promote_creates_backup_if_existing(self, paths):
        from src.cli.policy import cmd_promote_policy

        # Create an existing active policy
        existing = {"run_id": "old-run", "threshold": 0.6}
        (paths.reports_metrics / "decision_policy.json").write_text(
            json.dumps(existing)
        )

        _write_run_policy(paths, "run-002")

        with patch("src.cli.policy.mark_latest"):
            cmd_promote_policy(paths, run_id="run-002", slot="default")

        backup = paths.reports_metrics / "decision_policy.previous.json"
        assert backup.exists()
        backup_data = json.loads(backup.read_text())
        assert backup_data["run_id"] == "old-run"

    def test_promote_to_blue_slot(self, paths):
        from src.cli.policy import cmd_promote_policy

        _write_run_policy(paths, "run-blue")

        with (
            patch("src.cli.policy.mark_latest"),
            patch("src.cli.policy.json_write") as mock_write,
        ):
            cmd_promote_policy(paths, run_id="run-blue", slot="blue")

        # Should write active_slot.json
        written_paths = [str(c.args[0]) for c in mock_write.call_args_list]
        assert any("active_slot" in p for p in written_paths)

    def test_promote_to_green_slot(self, paths):
        from src.cli.policy import cmd_promote_policy

        _write_run_policy(paths, "run-green")

        with (
            patch("src.cli.policy.mark_latest"),
            patch("src.cli.policy.json_write") as mock_write,
        ):
            cmd_promote_policy(paths, run_id="run-green", slot="green")

        written_paths = [str(c.args[0]) for c in mock_write.call_args_list]
        assert any("active_slot" in p for p in written_paths)


# ── cmd_rollback_policy ───────────────────────────────────────────────────────


class TestCmdRollbackPolicy:
    def test_raises_if_no_backup(self, paths):
        from src.cli.policy import cmd_rollback_policy

        with pytest.raises(FileNotFoundError, match="No previous policy backup"):
            cmd_rollback_policy(paths, slot="default")

    def test_raises_on_invalid_slot(self, paths):
        from src.cli.policy import cmd_rollback_policy

        with pytest.raises(ValueError, match="slot must be one of"):
            cmd_rollback_policy(paths, slot="canary")

    def test_rollback_restores_backup(self, paths):
        from src.cli.policy import cmd_rollback_policy

        # Create backup
        backup = {"run_id": "old-run", "threshold": 0.6}
        (paths.reports_metrics / "decision_policy.previous.json").write_text(
            json.dumps(backup)
        )
        # Create current policy
        current = {"run_id": "new-run", "threshold": 0.5}
        (paths.reports_metrics / "decision_policy.json").write_text(json.dumps(current))

        cmd_rollback_policy(paths, slot="default")

        restored = json.loads(
            (paths.reports_metrics / "decision_policy.json").read_text()
        )
        assert restored["run_id"] == "old-run"


# ── cmd_retry_webhook_dlq ─────────────────────────────────────────────────────


class TestCmdRetryWebhookDlq:
    def test_returns_zeros_if_no_dlq_file(self, paths):
        from src.cli.policy import cmd_retry_webhook_dlq

        result = cmd_retry_webhook_dlq(paths, webhook_url="http://example.com/hook")
        assert result == {"retried": 0, "success": 0, "failed": 0}

    def test_raises_if_no_url_and_no_env(self, paths):
        from src.cli.policy import cmd_retry_webhook_dlq

        dlq = paths.reports_monitoring / "dead_letter_webhooks.jsonl"
        dlq.write_text(json.dumps({"payload": {"alert": True}}) + "\n")

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("ALERT_WEBHOOK_URL", None)
            with pytest.raises(ValueError, match="webhook url is required"):
                cmd_retry_webhook_dlq(paths, webhook_url=None)

    def test_returns_zeros_on_empty_dlq_file(self, paths):
        from src.cli.policy import cmd_retry_webhook_dlq

        dlq = paths.reports_monitoring / "dead_letter_webhooks.jsonl"
        dlq.write_text("")

        result = cmd_retry_webhook_dlq(paths, webhook_url="http://example.com/hook")
        assert result == {"retried": 0, "success": 0, "failed": 0}

    def test_successful_retry_removes_from_dlq(self, paths):
        from src.cli.policy import cmd_retry_webhook_dlq

        dlq = paths.reports_monitoring / "dead_letter_webhooks.jsonl"
        dlq.write_text(json.dumps({"payload": {"alert": "drift_detected"}}) + "\n")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b"ok"
            mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_response)
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = cmd_retry_webhook_dlq(paths, webhook_url="http://example.com/hook")

        assert result["retried"] == 1
        assert result["success"] == 1
        assert result["failed"] == 0
        # DLQ should be cleaned up
        assert not dlq.exists()

    def test_failed_retry_keeps_in_dlq(self, paths):
        from src.cli.policy import cmd_retry_webhook_dlq

        dlq = paths.reports_monitoring / "dead_letter_webhooks.jsonl"
        dlq.write_text(json.dumps({"payload": {"alert": "drift"}}) + "\n")

        with patch(
            "urllib.request.urlopen", side_effect=Exception("Connection refused")
        ):
            result = cmd_retry_webhook_dlq(paths, webhook_url="http://example.com/hook")

        assert result["retried"] == 1
        assert result["failed"] == 1
        assert result["success"] == 0
        # Failed item should remain in DLQ
        assert dlq.exists()
