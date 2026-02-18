FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json \
    DS_API_KEY=change-me

WORKDIR /app

COPY requirements-prod.txt /app/requirements-prod.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r /app/requirements-prod.txt

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json \
    DS_API_KEY=change-me

WORKDIR /app

RUN useradd -m appuser

COPY --from=builder /install /usr/local

COPY . /app

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["python", "main.py", "serve-api", "--host", "0.0.0.0", "--port", "8000"]
