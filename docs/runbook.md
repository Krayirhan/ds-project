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

## DVC Remote Storage

### Architecture
Data versioning uses DVC with S3-compatible object storage:

| Remote   | URL                       | Use case                        |
|----------|---------------------------|---------------------------------|
| `minio`  | `s3://ds-dvc-bucket`      | Local dev via Docker Compose    |
| `prod`   | `s3://ds-dvc-bucket`      | CI/CD & production (real S3)    |
| `gcs`    | `gs://ds-dvc-bucket`      | Optional GCS backend            |
| `azure`  | `azure://ds-dvc-bucket`   | Optional Azure Blob backend     |

### Local Dev Setup (MinIO)
```bash
# Start MinIO alongside all other services
docker compose -f docker-compose.dev.yml up minio -d

# MinIO Console: http://localhost:9001  (minioadmin / minioadmin)
# Bucket `ds-dvc-bucket` is auto-created on first start.

# Push local data to MinIO remote
make dvc-push

# Pull data on another machine / fresh checkout
make dvc-pull
```

### CI/CD
Three GitHub Actions secrets must be set in the repository:

| Secret                  | Value                                        |
|-------------------------|----------------------------------------------|
| `DVC_ACCESS_KEY_ID`     | AWS / MinIO access key                       |
| `DVC_SECRET_ACCESS_KEY` | AWS / MinIO secret key                       |
| `DVC_S3_ENDPOINT_URL`   | Endpoint URL (empty = real AWS S3)           |

The `ci.yml` `test` job runs `dvc pull data/raw data/processed` automatically.
If the remote is unreachable the step warns and continues (non-blocking).

### Switching the Default Remote
```bash
# Use production AWS S3 instead of MinIO
dvc remote default prod

# Or temporarily override for a single command
dvc push --remote prod
```

### Adding a New Data Version
```bash
# After modifying data/raw/ …
dvc add data/raw/hotel_bookings.csv
git add data/raw/hotel_bookings.csv.dvc
git commit -m "data: update raw dataset vX.Y"
dvc push
```

### Credential Rotation
1. Generate new credentials in AWS IAM / MinIO Console.
2. Update `DVC_ACCESS_KEY_ID` and `DVC_SECRET_ACCESS_KEY` in GitHub Secrets.
3. Update `.dvc/config.local` on developer machines.
4. Verify with `dvc status --cloud`.

---

## Model Health Monitoring

### Prometheus Alerts

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| `DSProjectHighPSI` | `ds_feature_psi > 0.2` for 5m | warning | Investigate feature drift; consider retraining |
| `DSProjectLowAUC` | `ds_model_roc_auc < 0.65` for 10m | critical | Rollback model; retrain urgently |
| `DSProjectActionRateAnomaly` | Action rate deviates > 15pp from 7-day avg for 15m | warning | Check threshold config; inspect recent predictions |

### Manual Model Health Check

```bash
# Run full monitoring pipeline — updates Prometheus gauges
python main.py monitor

# Check PSI scores for all features
cat reports/monitoring/latest_monitoring_report.json | python -m json.tool | grep psi

# Check model AUC on labeled test set
cat reports/metrics/latest_metrics.json
```

**Gauge update schedule**: run `python main.py monitor` every 4 hours in production.

---

## Security Runbook

### Rate Limit Exceeded (HTTP 429)

```bash
# Inspect current sliding window for an API key
redis-cli ZRANGE "rate_limit:<api_key>" 0 -1 WITHSCORES

# Clear rate limit for a key (emergency reset)
redis-cli DEL "rate_limit:<api_key>"
```

### Container Security

All pods enforce:
- `runAsUser: 1000`, `runAsNonRoot: true`
- `readOnlyRootFilesystem: true` (only `/tmp` writable via emptyDir)
- `allowPrivilegeEscalation: false`
- `capabilities.drop: [ALL]`
- `seccompProfile: RuntimeDefault`

If a pod fails `CrashLoopBackOff` due to permission errors, check `kubectl describe pod` for security context violation events.
