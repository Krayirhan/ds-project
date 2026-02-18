from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

from .utils import get_logger

logger = get_logger("rate_limit")


class BaseRateLimiter:
    def allow(
        self, key: str, limit_per_minute: int
    ) -> bool:  # pragma: no cover - interface only
        raise NotImplementedError


class InMemoryRateLimiter(BaseRateLimiter):
    def __init__(self) -> None:
        self._bucket: Dict[str, List[float]] = {}

    def allow(self, key: str, limit_per_minute: int) -> bool:
        now = time.time()
        one_min_ago = now - 60.0
        arr = self._bucket.get(key, [])
        arr = [x for x in arr if x >= one_min_ago]
        if len(arr) >= limit_per_minute:
            self._bucket[key] = arr
            return False
        arr.append(now)
        self._bucket[key] = arr
        return True


@dataclass
class RedisRateLimiter(BaseRateLimiter):
    redis_client: any
    key_prefix: str = "ds:rate"

    def allow(self, key: str, limit_per_minute: int) -> bool:
        # Sliding-ish 60s window via sorted-set score=timestamp
        now = time.time()
        window_start = now - 60.0
        rkey = f"{self.key_prefix}:{key}"

        pipe = self.redis_client.pipeline()
        pipe.zremrangebyscore(rkey, 0, window_start)
        pipe.zcard(rkey)
        pipe.zadd(rkey, {str(now): now})
        pipe.expire(rkey, 61)
        _, count, _, _ = pipe.execute()

        # count is before current request add, so block if already at/over limit
        return int(count) < int(limit_per_minute)


def build_rate_limiter(
    *, backend: str, redis_url: str | None, key_prefix: str
) -> BaseRateLimiter:
    normalized = (backend or "memory").strip().lower()
    if normalized == "redis":
        try:
            import redis

            if not redis_url:
                raise ValueError("redis_url is required for backend='redis'")
            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            logger.info("Using Redis distributed rate limiter")
            return RedisRateLimiter(redis_client=client, key_prefix=key_prefix)
        except Exception as e:
            logger.warning(f"Falling back to in-memory rate limiter. reason={e}")

    logger.info("Using in-memory rate limiter")
    return InMemoryRateLimiter()
