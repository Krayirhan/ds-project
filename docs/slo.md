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
- `DSProjectHighP95Latency`: p95 latency > 300ms for 15m (warning)
- `DSProjectHigh5xxRate`: 5xx ratio > 0.1% for 10m (critical)
- `DSProjectErrorBudgetBurnFast`: 5xx ratio > 1.44% for 5m (critical)
- `DSProjectErrorBudgetBurnSlow`: 5xx ratio > 0.6% for 30m (warning)
- `DSProjectHighPSI`: feature PSI > 0.2 for 5m (warning)
- `DSProjectLowAUC`: ROC-AUC < 0.65 for 10m (critical)
- `DSProjectActionRateAnomaly`: action rate deviates > 15pp from 7-day baseline for 15m (warning)

## Burn-Rate Policy
- Burn budget base: monthly 5xx budget = 0.1% error ratio.
- Fast burn alert (14.4x): `1.44%` 5xx ratio over 5m, treated as immediate rollback/halt signal.
- Slow burn alert (6x): `0.6%` 5xx ratio over 30m, treated as release-freeze signal.

## Error Budget Policy
- On burn > 50% monthly budget: freeze feature releases
- On burn > 80%: rollback to previous stable policy/model

## Automated Response
- Canary auto-halt/rollback source: `DSProjectErrorBudgetBurnFast` and critical model/business alerts in runbook.
- Scheduled monitor rollback source: `.github/workflows/monitor.yml` evaluates `latest_monitoring_report.json` and triggers rollback for critical conditions.
