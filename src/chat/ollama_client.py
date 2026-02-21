from __future__ import annotations

import asyncio
import json
import os
import time
from typing import AsyncGenerator

import httpx

from ..utils import get_logger

logger = get_logger("chat_ollama")

# ── Circuit breaker constants ──────────────────────────────────────────────────
_CB_FAILURE_THRESHOLD = 5   # consecutive failures before opening circuit
_CB_RECOVERY_SECONDS = 30   # seconds to wait before attempting half-open probe
_RETRY_ATTEMPTS = 3          # total attempts (1 original + 2 retries)
_RETRY_BASE_DELAY = 1.0      # exponential backoff base (seconds)


class OllamaClient:
    """Async Ollama API client with a persistent connection pool (#28),
    exponential-backoff retry, and a simple circuit breaker.

    Circuit states:
      CLOSED  — normal operation
      OPEN    — fast-fail after _CB_FAILURE_THRESHOLD consecutive failures
      HALF-OPEN — single probe request after _CB_RECOVERY_SECONDS
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        # Persistent client — created lazily, shared across all requests
        self._client: httpx.AsyncClient | None = None
        # Circuit breaker state
        self._cb_failures: int = 0
        self._cb_opened_at: float = 0.0
        self._cb_open: bool = False

    def _get_client(self) -> httpx.AsyncClient:
        """Return (or create) the shared async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout_seconds,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        """Gracefully close the underlying connection pool (call at app shutdown)."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _cb_check(self) -> None:
        """Raise immediately if circuit is OPEN (not yet recovery time)."""
        if not self._cb_open:
            return
        if time.monotonic() - self._cb_opened_at >= _CB_RECOVERY_SECONDS:
            logger.info("Circuit breaker: entering HALF-OPEN — sending probe request")
            self._cb_open = False  # allow one probe
        else:
            raise RuntimeError(
                f"Ollama circuit breaker OPEN — retry after {_CB_RECOVERY_SECONDS}s"
            )

    def _cb_success(self) -> None:
        self._cb_failures = 0
        self._cb_open = False

    def _cb_failure(self) -> None:
        self._cb_failures += 1
        if self._cb_failures >= _CB_FAILURE_THRESHOLD:
            self._cb_open = True
            self._cb_opened_at = time.monotonic()
            logger.warning(
                "Circuit breaker OPEN after %d consecutive failures", self._cb_failures
            )

    async def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
    ) -> str:
        self._cb_check()
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
            },
        }
        last_exc: Exception | None = None
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                client = self._get_client()
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                self._cb_success()
                return str(data.get("message", {}).get("content", "")).strip()
            except Exception as exc:
                last_exc = exc
                self._cb_failure()
                if attempt < _RETRY_ATTEMPTS - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama chat attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1, _RETRY_ATTEMPTS, exc, delay,
                    )
                    await asyncio.sleep(delay)
        logger.error("Ollama chat failed after %d attempts: %s", _RETRY_ATTEMPTS, last_exc)
        raise RuntimeError("Ollama yanıt veremedi") from last_exc

    async def health(self) -> bool:
        try:
            client = self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def chat_stream(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response tokens via Ollama's streaming API.

        Yields individual content tokens as they arrive from the model.
        Applies circuit breaker and raises ``RuntimeError`` on complete failure.
        """
        self._cb_check()
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
            },
        }
        try:
            client = self._get_client()
            async with client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        done = data.get("done", False)
                        if token:
                            yield token
                        if done:
                            break
                    except json.JSONDecodeError:
                        continue
            self._cb_success()
        except Exception as exc:
            self._cb_failure()
            logger.error("Ollama chat_stream failed: %s", exc)
            raise RuntimeError("Ollama streaming yanıt veremedi") from exc


_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
        )
    return _client

# ── Ollama Embedding Client (RAG vector search) ───────────────────────────────


class OllamaEmbeddingClient:
    """Ollama embedding client for building and querying a vector similarity index.

    Uses Ollama's ``/api/embed`` endpoint.  Configure the embedding model via
    ``OLLAMA_EMBED_MODEL`` env var (default: ``nomic-embed-text``, ~274 MB).
    Falls back silently when the model is unavailable — TF-IDF is used instead.
    """

    def __init__(self, *, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]] | None:
        """Synchronously embed a list of texts — called once at startup."""
        try:
            results: list[list[float]] = []
            with httpx.Client(timeout=120.0) as client:
                for text in texts:
                    r = client.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.model, "input": text},
                    )
                    r.raise_for_status()
                    embeddings = r.json().get("embeddings", [])
                    if not embeddings:
                        return None
                    results.append(embeddings[0])
            return results
        except Exception as exc:
            logger.info(
                "Embedding batch unavailable (model=%s): %s — using TF-IDF fallback",
                self.model,
                exc,
            )
            return None

    def embed_sync(self, text: str) -> list[float] | None:
        """Synchronously embed a single text. Run in executor inside async contexts."""
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text},
                )
                r.raise_for_status()
                embeddings = r.json().get("embeddings", [])
                return embeddings[0] if embeddings else None
        except Exception:
            return None


_embed_client: OllamaEmbeddingClient | None = None


def get_embedding_client() -> OllamaEmbeddingClient:
    global _embed_client
    if _embed_client is None:
        _embed_client = OllamaEmbeddingClient(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        )
    return _embed_client