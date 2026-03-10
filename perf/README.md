# Performance Tests

Bu dizin API ve altyapı performans testlerini içerir.

## Araçlar

| Dosya | Araç | Amaç |
|-------|------|------|
| `k6_smoke.js` | [k6](https://k6.io/) | Hızlı smoke test — `/health` ve `/decide` endpoint'leri |
| `locustfile.py` | [Locust](https://locust.io/) | Kapsamlı yük testi — login, dashboard, predict, decide, chat |

## Çalıştırma

### k6 Smoke Test

```bash
# k6 kurulu olmalı: https://k6.io/docs/getting-started/installation/
k6 run perf/k6_smoke.js
```

Varsayılan hedef: `http://localhost:8000`  
`K6_BASE_URL` ortam değişkeni ile değiştirilebilir.

### Locust Yük Testi

```bash
# API sunucusu çalışır durumda olmalı
locust -f perf/locustfile.py --host http://localhost:8000

# Headless mod (CI için):
locust -f perf/locustfile.py \
  --host http://localhost:8000 \
  --headless \
  --users 50 \
  --spawn-rate 5 \
  --run-time 60s
```

**Ortam Değişkenleri:**
- `DASHBOARD_USER` — Dashboard kullanıcı adı (varsayılan: `admin`)
- `DASHBOARD_PASS` — Dashboard şifresi (**zorunlu** — `replace-me` değerini değiştirin)

## Başarı Kriterleri

| Metrik | Hedef |
|--------|-------|
| `/health` p95 latency | < 50ms |
| `/decide` p95 latency | < 500ms (tek satır) |
| `/predict_proba` p95 latency | < 500ms (tek satır) |
| Hata oranı | < 1% |
| Throughput | ≥ 50 req/s (tek worker) |

## CI Entegrasyonu

`.github/workflows/load-test.yml` workflow'u ile otomatik çalışır.
Detaylı sonuçlar Grafana dashboard'unda izlenebilir.
