from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import src.chat.router as router


class _HealthyClient:
    def __init__(self, ok: bool):
        self._ok = ok
        self.model = "llama"

    async def health(self):
        return self._ok


class _Orchestrator:
    def __init__(self):
        self.raise_start = None
        self.raise_message = None
        self.raise_summary = None

    async def start_session(self, **kwargs):
        if self.raise_start:
            raise self.raise_start
        return "s-1", "welcome"

    async def quick_actions(self, **kwargs):
        return [{"label": "a", "message": "b"}]

    async def send_message(self, **kwargs):
        if self.raise_message:
            raise self.raise_message
        return "reply"

    async def summary(self, **kwargs):
        if self.raise_summary:
            raise self.raise_summary
        return {
            "session_id": "s-1",
            "risk_score": 0.7,
            "risk_label": "high",
            "message_count": 2,
            "created_at": 1.0,
            "last_active": 2.0,
        }


def test_chat_health_ok_and_degraded(monkeypatch):
    monkeypatch.setattr(router, "get_ollama_client", lambda: _HealthyClient(True))
    r1 = asyncio.run(router.chat_health())
    assert r1["status"] == "ok"
    assert r1["model"] == "llama"

    monkeypatch.setattr(router, "get_ollama_client", lambda: _HealthyClient(False))
    r2 = asyncio.run(router.chat_health())
    assert r2["status"] == "degraded"


def test_start_session_success_and_error(monkeypatch):
    orch = _Orchestrator()
    monkeypatch.setattr(router, "get_orchestrator", lambda: orch)
    body = router.StartSessionRequest(
        customer_data={"lead_time": 10},
        risk_score=0.6,
        risk_label="mid",
    )

    out = asyncio.run(router.start_session(body))
    assert out.session_id == "s-1"
    assert out.quick_actions[0]["label"] == "a"

    orch.raise_start = RuntimeError("boom")
    with pytest.raises(HTTPException) as ex:
        asyncio.run(router.start_session(body))
    assert ex.value.status_code == 500


def test_message_success_404_and_500(monkeypatch):
    orch = _Orchestrator()
    monkeypatch.setattr(router, "get_orchestrator", lambda: orch)
    body = router.ChatMessageRequest(session_id="s-1", message="hello")

    out = asyncio.run(router.message(body))
    assert out.bot_message == "reply"

    orch.raise_message = ValueError("not found")
    with pytest.raises(HTTPException) as ex1:
        asyncio.run(router.message(body))
    assert ex1.value.status_code == 404

    orch.raise_message = RuntimeError("crash")
    with pytest.raises(HTTPException) as ex2:
        asyncio.run(router.message(body))
    assert ex2.value.status_code == 500


def test_summary_success_and_404(monkeypatch):
    orch = _Orchestrator()
    monkeypatch.setattr(router, "get_orchestrator", lambda: orch)

    out = asyncio.run(router.summary("s-1"))
    assert out.session_id == "s-1"

    orch.raise_summary = ValueError("missing")
    with pytest.raises(HTTPException) as ex:
        asyncio.run(router.summary("s-1"))
    assert ex.value.status_code == 404
