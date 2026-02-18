# Production Runbook

## Deploy
1. Build and push container image.
2. Run `python main.py train`, `python main.py evaluate`.
3. Ensure Redis is reachable (`REDIS_URL`) for distributed rate limit.
3. Promote policy with slot:
   - `python main.py promote-policy --run-id <run_id> --slot blue`
4. Point traffic to blue (or set active slot in metrics policy pointer).
5. Validate `/ready`, `/metrics`, and monitoring report.

## Canary Rollout
- Apply canary manifests:
  - `kubectl apply -f deploy/k8s/canary-deployment.yaml`
  - `kubectl apply -f deploy/k8s/canary-ingress.yaml`
- Start with canary weight 10% and monitor SLO/alerts.
- Increase weight gradually after stable windows.

## Rollback
- `python main.py rollback-policy --slot blue`
- If needed, switch active slot to green by promoting known-good run:
  - `python main.py promote-policy --run-id <known_good_run> --slot green`

## Incidents
- Check API health: `/health`, `/ready`
- Check metrics: `/metrics`
- Check latest monitoring:
  - `reports/monitoring/latest_monitoring_report.json`
- If alerts active and profit degrades, rollback policy.
- Retry dead-letter webhook events:
  - `python main.py retry-webhook-dlq --url <webhook_url>`

## Secret Rotation
- Rotate `DS_API_KEY` in secret manager.
- Restart service with new secret.

## Recovery
- Restore model and policy artifacts from immutable storage backup.
- Validate checksum and contract versions before serving.
