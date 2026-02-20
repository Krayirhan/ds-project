# ADR-008: Lua-Based Distributed Rate Limiting

## Status
Accepted

## Date
2026-02-19

## Context

The API needs rate limiting to prevent abuse and ensure fair resource allocation across multiple users. The initial implementation used a simple in-process token bucket which:
- **Did not work horizontally**: each pod had its own counter, so N pods allowed N× the configured rate
- **Had a race condition**: the READ-then-WRITE pattern (`ZCARD` + `ZADD`) could allow bursts beyond the limit under concurrent load
- **Was not auditable**: no way to inspect current sliding window state

Requirements:
1. Rate limit must be enforced globally across all API pods
2. Must be atomic — no race conditions under concurrent requests
3. Must use a sliding window (not fixed window, which allows 2× bursts at boundary)
4. Must degrade gracefully when Redis is unavailable

## Decision

Implement rate limiting using **Redis + Lua script** with a **sliding window log** algorithm:

```lua
-- Single atomic operation: remove expired entries, count, conditionally add
redis.call('ZREMRANGEBYSCORE', key, 0, now - window_ms)
local count = redis.call('ZCARD', key)
if count < limit then
  redis.call('ZADD', key, now, now)
  redis.call('PEXPIRE', key, window_ms)
  return 0  -- allowed
end
return 1  -- denied
```

Key properties:
- **Atomicity**: Lua scripts execute as a single Redis transaction (no MULTI/EXEC needed)
- **Sliding window**: sorted set scored by timestamp; expired entries pruned on each request
- **Global enforcement**: all pods share the same Redis key per API key
- **Graceful degradation**: if Redis is unreachable, requests are allowed through (fail-open policy)

The Lua script is registered once at startup via `redis.register_script()` and cached as a callable for zero-overhead repeated invocation.

## Consequences

### Positive
- ✅ Correct sliding window semantics across all pods
- ✅ Zero race conditions (atomicity guaranteed by Redis single-threaded execution)
- ✅ Low latency: single Redis round-trip per request
- ✅ Inspectable: `ZRANGE <key> 0 -1 WITHSCORES` shows the sliding window
- ✅ Automatic TTL cleanup via `PEXPIRE`

### Negative
- ❌ Redis becomes a hard dependency for rate limiting (mitigated by fail-open)
- ❌ Slightly higher complexity than a simple counter (`INCR` + `EXPIRE`)
- ❌ Sorted set memory: O(requests per window) per key (acceptable at typical request rates)

### Neutral
- Redis sorted set memory usage: ~100 bytes per entry × requests/window. At 100 req/min limit, ~10 KB per API key. Acceptable.

## Alternatives Considered

| Alternative | Rejected Because |
|-------------|-----------------|
| In-process token bucket | Not distributed; each pod allows full limit |
| Redis `INCR` + `EXPIRE` (fixed window) | Allows 2× burst at window boundary |
| Redis `SETNX` + token bucket | Still has race condition without Lua |
| API Gateway rate limiting (Kong, etc.) | Adds infrastructure dependency; overkill for current scale |
| `redis.pipeline()` WATCH/MULTI/EXEC | More complex; `WATCH` requires retry on contention |

## References
- [Redis Lua scripting](https://redis.io/docs/manual/programmability/eval-intro/)
- [Sliding window rate limiting with Redis](https://engineering.classdojo.com/blog/2015/02/06/rolling-rate-limiter/)
- `src/rate_limit.py`: implementation
- `tests/test_rate_limit.py`: test coverage
