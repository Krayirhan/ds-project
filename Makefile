.PHONY: setup hooks lint test test-cov train evaluate predict monitor serve hpo explain load-locust dev-up dev-down helm-lint helm-template

setup:
	python -m venv .venv
	.venv\\Scripts\\python.exe -m pip install --upgrade pip
	.venv\\Scripts\\python.exe -m pip install -r requirements.txt

hooks:
	.venv\\Scripts\\python.exe -m pre_commit install --hook-type pre-commit --hook-type pre-push

lint:
	.venv\\Scripts\\python.exe -m pre_commit run --all-files

test:
	.venv\\Scripts\\python.exe -m pytest -q

test-cov:
	.venv\\Scripts\\python.exe -m pytest --cov=src --cov-report=term-missing

train:
	.venv\\Scripts\\python.exe main.py train

evaluate:
	.venv\\Scripts\\python.exe main.py evaluate

predict:
	.venv\\Scripts\\python.exe main.py predict

monitor:
	.venv\\Scripts\\python.exe main.py monitor

serve:
	.venv\\Scripts\\python.exe main.py serve-api --host 0.0.0.0 --port 8000

load-locust:
	.venv\\Scripts\\python.exe -m locust -f perf/locustfile.py --host http://127.0.0.1:8000

hpo:
	.venv\\Scripts\\python.exe main.py hpo --n-trials 50

explain:
	.venv\\Scripts\\python.exe main.py explain

# ── Dev Stack (docker-compose) ──────────────────────────────────────
dev-up:
	docker compose -f docker-compose.dev.yml up --build -d
	@echo "API:        http://localhost:8000"
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000  (admin/admin)"
	@echo "Jaeger UI:  http://localhost:16686"

dev-down:
	docker compose -f docker-compose.dev.yml down -v

# ── Helm ────────────────────────────────────────────────────────────
helm-lint:
	helm lint deploy/helm/ds-project

helm-template:
	helm template ds-project deploy/helm/ds-project --values deploy/helm/ds-project/values.yaml
