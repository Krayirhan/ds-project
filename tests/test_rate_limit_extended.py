from __future__ import annotations

import time
import types

from src.rate_limit import InMemoryRateLimiter, RedisRateLimiter, build_rate_limiter


def test_inmemory_evict_stale_locked_and_allow_trigger(monkeypatch):
    rl = InMemoryRateLimiter()
    now = 1000.0
    one_min_ago = now - 60.0

    rl._bucket = {
        "stale_a": [100.0],
        "stale_b": [200.0],
        "fresh": [950.0],
    }
    rl._MAX_KEYS = 1
    monkeypatch.setattr(time, "time", lambda: now)

    assert rl.allow("new", 10) is True
    assert "stale_a" not in rl._bucket
    assert "stale_b" not in rl._bucket
    assert "new" in rl._bucket

    # direct eviction path for stale list + logger branch
    rl._bucket["x"] = [100.0]
    rl._evict_stale_locked(one_min_ago)
    assert "x" not in rl._bucket


class _FakeRedisClient:
    def register_script(self, script):
        def _run(keys, args):
            _key = keys[0]
            _now, _window, limit = args
            return 1 if int(limit) > 0 else 0

        return _run

    def ping(self):
        return True


def test_build_rate_limiter_redis_success(monkeypatch):
    fake_client = _FakeRedisClient()

    class _RedisFactory:
        @staticmethod
        def from_url(url, decode_responses=True):
            return fake_client

    redis_mod = types.SimpleNamespace(Redis=_RedisFactory)
    monkeypatch.setitem(__import__("sys").modules, "redis", redis_mod)

    rl = build_rate_limiter(
        backend="redis",
        redis_url="redis://localhost:6379/0",
        key_prefix="ds",
    )
    assert isinstance(rl, RedisRateLimiter)
    assert rl.allow("client-1", 1) is True
