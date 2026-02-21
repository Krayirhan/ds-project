"""Tests for src/cli/serve.py."""

from __future__ import annotations

from unittest.mock import patch

from src.cli.serve import cmd_serve_api


def test_cmd_serve_api_uses_defaults() -> None:
    with patch("src.cli.serve.uvicorn.run") as mock_run:
        cmd_serve_api()

    mock_run.assert_called_once_with(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        timeout_graceful_shutdown=30,
    )


def test_cmd_serve_api_uses_overrides() -> None:
    with patch("src.cli.serve.uvicorn.run") as mock_run:
        cmd_serve_api(host="127.0.0.1", port=9000, graceful_shutdown_seconds=12)

    mock_run.assert_called_once_with(
        "src.api:app",
        host="127.0.0.1",
        port=9000,
        reload=False,
        timeout_graceful_shutdown=12,
    )
