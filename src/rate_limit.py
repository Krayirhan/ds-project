from __future__ import annotations

import threading
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
    """Thread-safe sliding-window rate limiter with automatic stale-key eviction."""

    # Evict stale entries when bucket grows beyond this size (#21 memory leak fix)
    _MAX_KEYS: int = 50_000

    def __init__(self) -> None:
        self._bucket: Dict[str, List[float]] = {}
        self._lock = threading.Lock()  # #20 race condition fix

    def allow(self, key: str, limit_per_minute: int) -> bool:
        now = time.time()
        one_min_ago = now - 60.0
        with self._lock:
            arr = self._bucket.get(key, [])
            # Slide window: drop timestamps older than 60 s
            arr = [x for x in arr if x >= one_min_ago]
            if len(arr) >= limit_per_minute:
                self._bucket[key] = arr
                return False
            arr.append(now)
            self._bucket[key] = arr
            # Periodic eviction: remove keys whose entire window has expired
            if len(self._bucket) > self._MAX_KEYS:
                self._evict_stale_locked(one_min_ago)
        return True

    def _evict_stale_locked(self, one_min_ago: float) -> None:
        """Remove keys with no recent timestamps. Must be called while holding self._lock."""
        stale = [
            k for k, v in self._bucket.items()
            if not v or max(v) < one_min_ago
        ]
        for k in stale:
            del self._bucket[k]
        if stale:
            logger.debug(
                "Rate limiter evicted %d stale keys (bucket size now: %d)",
                len(stale),
                len(self._bucket),
            )


@dataclass
class RedisRateLimiter(BaseRateLimiter):
    redis_client: any
    key_prefix: str = "ds:rate"

    # Atomic sliding-window via Lua: ZREMRANGEBYSCORE + ZCARD + ZADD + EXPIRE in one step.
    # This eliminates the TOCTOU race condition present in pipeline-based approaches.
    _LUA_SCRIPT = """
        local key        = KEYS[1]
        local now        = tonumber(ARGV[1])
        local window     = tonumber(ARGV[2])
        local limit      = tonumber(ARGV[3])
        local window_start = now - window
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
        local count = redis.call('ZCARD', key)
        if count >= limit then
            return 0
        end
        redis.call('ZADD', key, now, tostring(now))
        redis.call('EXPIRE', key, math.ceil(window) + 1)
        return 1
    """

    def __post_init__(self) -> None:
        self._script = self.redis_client.register_script(self._LUA_SCRIPT)

    def allow(self, key: str, limit_per_minute: int) -> bool:
        now = time.time()
        rkey = f"{self.key_prefix}:{key}"
        result = self._script(keys=[rkey], args=[now, 60.0, limit_per_minute])
        return bool(result)


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
