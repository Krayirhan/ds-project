from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api_shared import (
    get_or_create_reload_lock,
    reload_serving_state_for_app,
    require_admin_key,
)


def _request(headers: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(headers=headers or {})


def test_require_admin_key_enforces_when_configured(monkeypatch):
    monkeypatch.setenv("DS_ADMIN_KEY", "admin-secret")
    with pytest.raises(HTTPException) as ex:
        require_admin_key(_request({"x-admin-key": "wrong"}))
    assert ex.value.status_code == 403


def test_require_admin_key_allows_missing_config(monkeypatch):
    monkeypatch.delenv("DS_ADMIN_KEY", raising=False)
    require_admin_key(_request())


def test_reload_serving_state_for_app_sets_state_and_reuses_lock():
    lock = asyncio.Lock()
    app = SimpleNamespace(state=SimpleNamespace(_reload_lock=lock, serving=None))
    loaded = object()

    result = asyncio.run(reload_serving_state_for_app(app, loader=lambda: loaded))

    assert result is loaded
    assert app.state.serving is loaded
    assert get_or_create_reload_lock(app) is lock
