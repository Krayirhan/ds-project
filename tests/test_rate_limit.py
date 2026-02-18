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


class FakePipeline:
    def __init__(self, count):
        self._count = count

    def zremrangebyscore(self, *args, **kwargs):
        return self

    def zcard(self, *args, **kwargs):
        return self

    def zadd(self, *args, **kwargs):
        return self

    def expire(self, *args, **kwargs):
        return self

    def execute(self):
        return [None, self._count, None, None]


class FakeRedis:
    def __init__(self, count):
        self._count = count

    def pipeline(self):
        return FakePipeline(self._count)


def test_redis_rate_limiter_allows_under_limit():
    rl = RedisRateLimiter(redis_client=FakeRedis(count=1), key_prefix="k")
    assert rl.allow("client-1", limit_per_minute=2) is True


def test_redis_rate_limiter_blocks_at_limit():
    rl = RedisRateLimiter(redis_client=FakeRedis(count=2), key_prefix="k")
    assert rl.allow("client-1", limit_per_minute=2) is False
