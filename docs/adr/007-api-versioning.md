# ADR-007: API Versioning Strategy

## Status
Accepted — 2026-02-18

## Context
As the API evolves (richer response metadata, new endpoints, breaking changes),
we need a versioning strategy that:

1. Preserves backward compatibility for existing consumers
2. Allows iterative improvement of response contracts
3. Provides a clear deprecation/sunset path
4. Minimizes code duplication

## Decision

### URL-Path Prefix Versioning
```
/v1/predict_proba    ← Original contract
/v1/decide
/v1/reload

/v2/predict_proba    ← Enhanced responses (+ meta block)
/v2/decide
/v2/reload
```

**Why URL-path over header-based or query-param versioning?**
- Explicit and discoverable (visible in logs, metrics, traces)
- Easy to route in ingress/load balancer
- OpenAPI spec clearly separates versions
- Curl-friendly for debugging

### Implementation
- **FastAPI `APIRouter`**: Each version is a separate router module
  - `src/api_v1.py` → `router_v1 = APIRouter(prefix="/v1")`
  - `src/api_v2.py` → `router_v2 = APIRouter(prefix="/v2")`
- **Root endpoints preserved**: `/predict_proba`, `/decide` still work (backward compat)
- **Shared serving state**: Both versions use the same `app.state.serving`

### V2 Enhancements
| Field | V1 | V2 |
|-------|----|----|
| `meta.api_version` | — | `"v2"` |
| `meta.model_used` | — | Model artifact name |
| `meta.latency_ms` | — | Server-side latency |
| `meta.request_id` | — | Propagated request ID |

### Deprecation Strategy
1. V1 will be marked with `Sunset` header when V3 is introduced
2. Minimum 6-month deprecation window
3. Deprecation announced via CHANGELOG + API response header

## Consequences

### Positive
- Zero disruption for existing V1 consumers
- V2 provides richer observability metadata
- Clear upgrade path (add `meta` handling in client)
- Metrics labeled by endpoint version (`v1.decide`, `v2.decide`)

### Negative
- Some code duplication between V1 and V2 routers
- Three sets of endpoints active simultaneously (root + V1 + V2)
- Testing surface increases with each version

### Future
- V3 could introduce streaming responses for large batch inference
- Consider GraphQL or gRPC for high-throughput internal consumers
