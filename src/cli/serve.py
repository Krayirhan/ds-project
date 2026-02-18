"""CLI: serve-api command."""

from __future__ import annotations

import uvicorn


def cmd_serve_api(
    host: str = "0.0.0.0", port: int = 8000, graceful_shutdown_seconds: int = 30
) -> None:
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=False,
        timeout_graceful_shutdown=graceful_shutdown_seconds,
    )
