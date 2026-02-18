# ds_project — Üretim Seviyesi Veri Bilimi Pipeline'ı

Bu proje, otel rezervasyon iptal tahmini için uçtan uca bir veri bilimi ve MLOps hattı sunar. 
Amaç; modeli **tekrarlanabilir**, **izlenebilir**, **güvenli** ve **üretime uygun** şekilde geliştirmek, yayınlamak ve işletmektir.

## İçindekiler

- [Proje Özeti](#proje-ozeti)
- [Temel Yetenekler](#temel-yetenekler)
- [Teknoloji ve Mimari](#teknoloji-ve-mimari)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Model Eğitimi ve Tahmin Akışı](#model-eğitimi-ve-tahmin-akışı)
- [API Servisi](#api-servisi)
- [İzleme ve Alarm Yönetimi](#izleme-ve-alarm-yönetimi)
- [Rollout / Rollback](#rollout--rollback)
- [Test Stratejisi](#test-stratejisi)
- [Container ve Kubernetes Dağıtımı](#container-ve-kubernetes-dağıtımı)
- [CI/CD](#cicd)
- [Performans ve Yük Testleri](#performans-ve-yük-testleri)
- [Veri Soygeçmişi ve Versiyonlama](#veri-soygeçmişi-ve-versiyonlama)
- [Operasyonel Dokümantasyon](#operasyonel-dokümantasyon)
- [Ortam Değişkenleri](#ortam-değişkenleri)

## Proje Özeti

`ds_project`, klasik modelleme adımlarını (ön işleme, eğitim, değerlendirme, tahmin) üretim ihtiyaçlarıyla birleştirir:

- politika tabanlı karar mekanizması,
- kalibrasyon ve maliyet duyarlı değerlendirme,
- API üzerinden online inference,
- gözlemlenebilirlik (health/readiness/metrics),
- drift ve performans izleme,
- otomatik rollback senaryoları,
- konteyner ve Kubernetes dağıtımı.

## Temel Yetenekler

- Uçtan uca ML yaşam döngüsü (preprocess → train → evaluate → predict)
- API tabanlı skor ve karar servisleme
- Prometheus metrikleri ve operasyonel health endpoint'leri
- PSI/KS tabanlı drift analizi
- AUC, Brier ve realize edilen kârlılık takibi
- Canary dağıtım ve blue/green politika geçişleri
- DVC ile pipeline tekrarlanabilirliği ve soygeçmiş takibi

## Teknoloji ve Mimari

- **Dil/Runtime:** Python 3.10+
- **Paketleme:** [pyproject.toml](pyproject.toml)
- **Pipeline:** [dvc.yaml](dvc.yaml)
- **API ve servis kodu:** [src](src), [apps/backend](apps/backend)
- **Frontend (fullstack UI):** [apps/frontend](apps/frontend)
- **Dağıtım artefaktları:** [deploy](deploy)
- **Testler:** [tests](tests)
- **Raporlar ve metrikler:** [reports](reports)

## Hızlı Başlangıç

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Model Eğitimi ve Tahmin Akışı

```bash
python main.py preprocess
python main.py train
python main.py evaluate
python main.py predict
```

Bu akış sonrası model artefaktları [models](models) altında, metrik ve değerlendirme çıktıları ise [reports](reports) altında üretilir.

## API Servisi

Önce gerekli ortam değişkenlerini tanımlayın:

```bash
set DS_API_KEY=your-secret-key
set RATE_LIMIT_BACKEND=redis
set REDIS_URL=redis://localhost:6379/0
python main.py serve-api --host 0.0.0.0 --port 8000
```

### Endpoint'ler

- `GET /health` → liveness
- `GET /ready` → model ve policy yüklü mü kontrolü
- `GET /metrics` → Prometheus metrikleri
- `POST /predict_proba` → olasılık çıktısı
- `POST /decide` → politika tabanlı karar
- `POST /reload` → servis yeniden başlatmadan model/policy yenileme
- `GET /dashboard/api/overview` → dashboard veri endpoint'i (train/test metrikleri)
- `GET /dashboard/api/runs` → run listesi
- `GET /dashboard/api/db-status` → veritabanı bağlantı durumu
- `POST /auth/login` → dashboard giriş
- `POST /auth/logout` → dashboard çıkış
- `GET /auth/me` → aktif oturum bilgisi

İsteklerde header:

- `x-api-key: <DS_API_KEY>`

Rate limit backend seçenekleri:

- `memory` → tek pod / geliştirme ortamı
- `redis` → dağıtık ve çok replika üretim ortamı

## Fullstack Web (Önerilen Mimari)

Bu projede önerilen yapı:

- Backend: FastAPI (`src`) + entrypoint (`apps/backend/main.py`)
- Frontend: React/Vite (`apps/frontend`)
- ML çekirdeği: mevcut pipeline (`src/train.py`, `src/evaluate.py`, `src/predict.py`)

Frontend'i lokal çalıştırma:

```bash
cd apps/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Varsayılan URL:

- Frontend UI: `http://localhost:5173`
- Backend API: `http://localhost:8000`

Kurumsal dashboard sayfaları:

- Genel Bakış
- Model Analizi
- Run Geçmişi
- Sistem ve Veritabanı

Dashboard giriş bilgileri (dev ortamı):

- kullanıcı adı: `admin`
- şifre: `ChangeMe123!`

## İzleme ve Alarm Yönetimi

İzleme işini çalıştırmak için:

```bash
python main.py monitor
```

Çıktılar:

- `reports/monitoring/<run_id>/monitoring_report.json`
- `reports/monitoring/latest_monitoring_report.json`

Kapsanan kontroller:

- veri drift (PSI)
- tahmin drift (PSI + KS)
- outcome metrikleri (AUC, Brier, realized profit)
- alarm bayrakları

Webhook alarmı için:

- `ALERT_WEBHOOK_URL` değişkenini tanımlayın.

Dead-letter queue yeniden deneme:

```bash
python main.py retry-webhook-dlq --url https://example.com/webhook
```

## Rollout / Rollback

Belirli bir run policy'sini promote etmek:

```bash
python main.py promote-policy --run-id 20260217_220731
```

Blue/green slot bazlı promote:

```bash
python main.py promote-policy --run-id 20260217_220731 --slot blue
python main.py promote-policy --run-id 20260217_220731 --slot green
```

Önceki policy'e rollback:

```bash
python main.py rollback-policy
```

Slot bazlı rollback:

```bash
python main.py rollback-policy --slot blue
```

## Test Stratejisi

```bash
pytest
```

Test kapsamı:

- policy birim testleri
- şema doğrulama testleri
- uçtan uca smoke testler

## Container ve Kubernetes Dağıtımı

### Docker

```bash
docker build -t ds-project:latest .
docker run -e DS_API_KEY=your-secret-key -p 8000:8000 ds-project:latest
```

### Docker Compose (API + Frontend + Redis + PostgreSQL + Monitoring)

```bash
docker compose -f docker-compose.dev.yml up --build
```

Önemli URL'ler:

- API: `http://localhost:8000`
- Frontend Dashboard: `http://localhost:5173`
- PostgreSQL: `localhost:5432` (`ds_dashboard`)

### Kubernetes

Manifestler: [deploy/k8s](deploy/k8s)

İçerik:

- namespace, deployment, service
- HPA
- network policy
- PDB
- secret örneği
- canary deployment ve canary ingress

Örnek uygulama sırası:

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/secrets.example.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/hpa.yaml
kubectl apply -f deploy/k8s/network-policy.yaml
kubectl apply -f deploy/k8s/pdb.yaml
kubectl apply -f deploy/k8s/ingress.yaml
kubectl apply -f deploy/k8s/canary-deployment.yaml
kubectl apply -f deploy/k8s/canary-ingress.yaml
```

Canary trafik bölüşümü [deploy/k8s/canary-ingress.yaml](deploy/k8s/canary-ingress.yaml) içindeki NGINX annotation'ları ile yönetilir (varsayılan ağırlık: %10).

## CI/CD

Başlıca workflow dosyaları:

- [.github/workflows/ci.yml](.github/workflows/ci.yml)
- [.github/workflows/deploy.yml](.github/workflows/deploy.yml)
- [.github/workflows/monitor.yml](.github/workflows/monitor.yml)
- [.github/workflows/security.yml](.github/workflows/security.yml)

Not: [monitor.yml](.github/workflows/monitor.yml), izleme raporunda `any_alert=true` olduğunda otomatik policy rollback akışını tetikler.

## Performans ve Yük Testleri

### Locust

```bash
locust -f perf/locustfile.py --host http://127.0.0.1:8000
```

### k6

```bash
k6 run perf/k6_smoke.js
```

SLO kontrolleri k6 threshold'larında tanımlıdır (ör. `p95 < 300ms`, `p99 < 800ms`).

## Veri Soygeçmişi ve Versiyonlama

- Pipeline tanımı: [dvc.yaml](dvc.yaml)
- Parametreler: [params.yaml](params.yaml)
- Preprocess lineage artefaktı: `reports/metrics/data_lineage_preprocess.json`
- Train lineage artefaktı: `reports/metrics/<run_id>/data_lineage.json`

Pipeline'ı yeniden üretmek için:

```bash
dvc repro
```

## Operasyonel Dokümantasyon

- Mimari: [docs/architecture.md](docs/architecture.md)
- Runbook: [docs/runbook.md](docs/runbook.md)
- SLO: [docs/slo.md](docs/slo.md)

## Ortam Değişkenleri

Detaylı değişken listesi için [\.env.example](.env.example) dosyasını kullanın.
