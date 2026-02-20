from src.rate_limit import InMemoryRateLimiter, RedisRateLimiter, build_rate_limiter


def test_inmemory_rate_limiter_blocks_after_limit():
    rl = InMemoryRateLimiter()
    key = "client-1"
    assert rl.allow(key, 2) is True
    assert rl.allow(key, 2) is True
    assert rl.allow(key, 2) is False


def test_build_rate_limiter_falls_back_to_memory_without_redis_url():
    rl = build_rate_limiter(backend="redis", redis_url=None, key_prefix="x")
    assert isinstance(rl, InMemoryRateLimiter)


class FakeRedis:
    """Fake Redis client supporting Lua script registration for RedisRateLimiter tests."""

    def __init__(self, count: int):
        # `count` simulates how many requests are already in the sorted set
        self._count = count

    def register_script(self, script: str):
        """Return a callable that simulates the Lua rate-limit script.

        Lua script behaviour:
        - ARGV = [now, window_seconds, limit]
        - Returns 1 (allow) if count < limit, else 0 (block)
        """
        count = self._count

        def _run(keys, args):
            _now, _window, limit = args
            return 1 if count < int(limit) else 0

        return _run


def test_redis_rate_limiter_allows_under_limit():
    rl = RedisRateLimiter(redis_client=FakeRedis(count=1), key_prefix="k")
    assert rl.allow("client-1", limit_per_minute=2) is True


def test_redis_rate_limiter_blocks_at_limit():
    rl = RedisRateLimiter(redis_client=FakeRedis(count=2), key_prefix="k")
    assert rl.allow("client-1", limit_per_minute=2) is False
