from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import src.chat.memory as mem


class FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.expire_calls: list[tuple[str, int]] = []

    def setex(self, key: str, ttl: int, value: str):
        self.data[key] = value
        self.ttls[key] = ttl

    def get(self, key: str):
        return self.data.get(key)

    def expire(self, key: str, ttl: int):
        self.ttls[key] = ttl
        self.expire_calls.append((key, ttl))

    def ttl(self, key: str) -> int:
        return self.ttls.get(key, -1)

    def ping(self):
        return True


@pytest.fixture(autouse=True)
def _reset_store():
    mem._store = None
    yield
    mem._store = None


def test_chat_session_add_message_and_ollama_messages(monkeypatch):
    session = mem.ChatSession(
        session_id="s-1",
        customer_data={"lead_time": 10},
        risk_score=0.8,
        risk_label="high",
    )
    monkeypatch.setattr(mem.time, "time", lambda: 1234.0)
    session.add_message(role="user", content="hello")
    assert session.messages[-1].role == "user"
    assert session.last_active == 1234.0

    msgs = session.to_ollama_messages(system_prompt="system")
    assert msgs[0] == {"role": "system", "content": "system"}
    assert msgs[1]["content"] == "hello"


def test_chat_session_expiry(monkeypatch):
    session = mem.ChatSession(
        session_id="s-2",
        customer_data={},
        risk_score=0.2,
        risk_label="low",
    )
    session.last_active = 100.0
    monkeypatch.setattr(mem.time, "time", lambda: 200.0)
    assert session.is_expired(ttl_seconds=50) is True
    assert session.is_expired(ttl_seconds=150) is False


def test_inmemory_store_create_get_trim_and_cleanup():
    store = mem.SessionStore(ttl_seconds=3600, max_history=4)
    s1 = store.create_session(customer_data={}, risk_score=0.3, risk_label="low")
    assert store.get_session(session_id=s1.session_id) is s1
    assert store.get_session(session_id="missing") is None

    # trim no-op path
    store.trim_history(session=s1)
    assert len(s1.messages) == 0

    # trim active path
    for i in range(8):
        s1.add_message(role="user", content=f"m{i}")
    store.trim_history(session=s1)
    assert len(s1.messages) == 4
    assert s1.messages[0].content == "m0"
    assert s1.messages[1].content == "m1"

    # cleanup path
    s1.last_active = 0.0
    s2 = store.create_session(customer_data={}, risk_score=0.7, risk_label="high")
    s2.last_active = 9999999999.0
    store.ttl_seconds = 1
    store._cleanup_expired()
    assert s1.session_id not in store._sessions
    assert s2.session_id in store._sessions


def test_inmemory_store_expired_session_is_evicted_on_get():
    store = mem.SessionStore(ttl_seconds=-1, max_history=4)
    session = store.create_session(customer_data={}, risk_score=0.5, risk_label="mid")
    assert store.get_session(session_id=session.session_id) is None
    assert session.session_id not in store._sessions


def test_redis_store_roundtrip_and_save_ttl_paths():
    redis = FakeRedis()
    store = mem.RedisSessionStore(redis, ttl_seconds=120, max_history=4)

    session = store.create_session(customer_data={"a": 1}, risk_score=0.9, risk_label="high")
    key = store._key(session.session_id)
    assert key in redis.data
    assert redis.ttls[key] == 120

    loaded = store.get_session(session_id=session.session_id)
    assert loaded is not None
    assert loaded.session_id == session.session_id
    assert redis.expire_calls[-1] == (key, 120)

    # save path with positive remaining ttl uses at least 60s
    redis.ttls[key] = 30
    store.save_session(loaded)
    assert redis.ttls[key] == 60

    # save path with expired/unknown ttl falls back to default ttl
    redis.ttls[key] = -1
    store.save_session(loaded)
    assert redis.ttls[key] == 120

    # trim path
    for i in range(7):
        loaded.add_message(role="assistant", content=f"x{i}")
    store.trim_history(session=loaded)
    assert len(loaded.messages) == 4

    # explicit no-op branch
    store._cleanup_expired()


def test_redis_store_invalid_payload_returns_none():
    redis = FakeRedis()
    store = mem.RedisSessionStore(redis, ttl_seconds=120, max_history=4)
    redis.data[store._key("bad")] = "{not-json"
    assert store.get_session(session_id="bad") is None


def test_get_session_store_returns_cached_instance():
    cached = mem.SessionStore()
    mem._store = cached
    assert mem.get_session_store() is cached


def test_get_session_store_redis_backend_success(monkeypatch):
    client = FakeRedis()

    class _RedisFactory:
        @staticmethod
        def from_url(*args, **kwargs):
            return client

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("CHAT_SESSION_TTL_SECONDS", "1800")
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=_RedisFactory))

    store = mem.get_session_store()
    assert isinstance(store, mem.RedisSessionStore)
    assert store.ttl_seconds == 1800


def test_get_session_store_falls_back_to_inmemory_when_redis_fails(monkeypatch):
    class _RedisFactory:
        @staticmethod
        def from_url(*args, **kwargs):
            raise RuntimeError("redis down")

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=_RedisFactory))

    store = mem.get_session_store()
    assert isinstance(store, mem.SessionStore)


def test_redis_serializer_roundtrip():
    redis = FakeRedis()
    store = mem.RedisSessionStore(redis, ttl_seconds=60, max_history=6)
    session = mem.ChatSession(
        session_id="s-raw",
        customer_data={"hotel": "city"},
        risk_score=0.42,
        risk_label="mid",
    )
    session.add_message(role="user", content="hello")
    raw = store._serialize(session)
    parsed = json.loads(raw)
    assert parsed["session_id"] == "s-raw"
    restored = store._deserialize(raw)
    assert restored.messages[0].content == "hello"
