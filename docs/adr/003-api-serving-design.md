# ADR-003: API Serving Design

## Status
Accepted — 2026-02-17

## Context
The ML model must serve real-time predictions via HTTP API. Requirements:
- Low latency (<300ms p95)
- Authentication + rate limiting
- Graceful degradation during model reload
- Prometheus metrics for observability
- Health/readiness probes for Kubernetes

## Decision

### Framework: FastAPI
Chosen for async support, automatic OpenAPI docs, Pydantic validation, and Python ecosystem fit.

### Endpoint Design
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe (model loaded) |
| `/metrics` | GET | Prometheus metrics |
| `/predict_proba` | POST | Raw probability scores |
| `/decide` | POST | Policy-applied actions |
| `/reload` | POST | Hot-reload model + policy |

### Middleware Stack
1. **Request ID** — UUID propagation via `x-request-id` header
2. **API Key Auth** — Header-based (`x-api-key`), configurable
3. **Rate Limiting** — Token bucket, memory or Redis backend
4. **Payload Guard** — 8MB max, row count limit
5. **Graceful Shutdown** — 503 during drain period

### Model Loading
- Loaded at startup via `lifespan` context manager
- Hot-reload via `/reload` endpoint (atomic swap)
- Policy contract version check on load
- SHA256 integrity verification of model artifact

### Security Headers
All responses include: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer`, `Cache-Control: no-store`

## Consequences

### Positive
- Sub-100ms typical latency for single-record inference
- Zero-downtime model updates via `/reload`
- Production-grade security posture
- Full observability via Prometheus + structured logging

### Negative
- Single-process serving (no built-in horizontal scaling at app level)
- Model loaded in memory (limits model size)
- Rate limiter state lost on restart (memory mode)
