# SLO / SLI

## SLI Definitions
- Availability: successful responses / total responses
- Latency: p95 request latency on `/predict_proba` and `/decide`
- Correctness proxy: schema validation failure rate
- Business: action-rate deviation and realized-profit drop

## SLO Targets
- Availability: >= 99.9% (30d)
- p95 latency: <= 300ms for payload <= 200 records
- 5xx rate: <= 0.1%
- Action-rate deviation: <= 5%
- Profit drop alert threshold: >= 20%

## Alert Rules
- Any `any_alert=true` in monitoring report
- p95 latency breach for 15m
- 5xx rate breach for 10m

## Error Budget Policy
- On burn > 50% monthly budget: freeze feature releases
- On burn > 80%: rollback to previous stable policy/model
