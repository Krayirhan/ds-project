from __future__ import annotations

import asyncio

import pytest

import src.chat.ollama_client as oc


class _Resp:
    def __init__(self, *, status_code: int = 200, content: str = "ok"):
        self.status_code = status_code
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return {"message": {"content": self._content}}


class _Client:
    def __init__(self):
        self.is_closed = False
        self.last_post = None
        self.last_get = None
        self.post_resp = _Resp(content="  merhaba  ")
        self.get_resp = _Resp(status_code=200)

    async def post(self, url, json):
        self.last_post = (url, json)
        return self.post_resp

    async def get(self, url):
        self.last_get = url
        return self.get_resp

    async def aclose(self):
        self.is_closed = True


def test_get_client_lazy_init_and_aclose(monkeypatch):
    created = {"n": 0}
    fake_client = _Client()

    def _factory(*args, **kwargs):
        created["n"] += 1
        return fake_client

    monkeypatch.setattr(oc.httpx, "AsyncClient", _factory)
    c = oc.OllamaClient(base_url="http://x", model="m", timeout_seconds=5)
    assert c._get_client() is fake_client
    assert c._get_client() is fake_client
    assert created["n"] == 1

    asyncio.run(c.aclose())
    assert fake_client.is_closed is True
    assert c._client is None


def test_chat_success_and_error(monkeypatch):
    c = oc.OllamaClient(base_url="http://ollama", model="m")
    fake_client = _Client()
    monkeypatch.setattr(c, "_get_client", lambda: fake_client)

    out = asyncio.run(c.chat(messages=[{"role": "user", "content": "hi"}]))
    assert out == "merhaba"
    assert fake_client.last_post[0].endswith("/api/chat")

    async def _boom(*args, **kwargs):
        raise RuntimeError("down")

    fake_client.post = _boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="Ollama"):
        asyncio.run(c.chat(messages=[{"role": "user", "content": "hi"}]))


def test_health_success_and_exception(monkeypatch):
    c = oc.OllamaClient(base_url="http://ollama", model="m")
    fake_client = _Client()
    monkeypatch.setattr(c, "_get_client", lambda: fake_client)
    assert asyncio.run(c.health()) is True

    fake_client.get_resp = _Resp(status_code=503)
    assert asyncio.run(c.health()) is False

    async def _boom(*args, **kwargs):
        raise RuntimeError("down")

    fake_client.get = _boom  # type: ignore[assignment]
    assert asyncio.run(c.health()) is False


def test_get_ollama_client_singleton(monkeypatch):
    oc._client = None
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "llama")
    monkeypatch.setenv("OLLAMA_TIMEOUT_SECONDS", "9")
    c1 = oc.get_ollama_client()
    c2 = oc.get_ollama_client()
    assert c1 is c2
    assert c1.base_url == "http://localhost:11434"
    assert c1.model == "llama"
    assert c1.timeout_seconds == 9.0
