from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import src.chat.router as router


class _MockKnowledgeDb:
    def __init__(self):
        self._chunks: list[dict] = []
        self._next_id = 1

    def create_chunk(
        self,
        *,
        chunk_id: str,
        category: str,
        tags: list[str],
        title: str,
        content: str,
        priority: int = 5,
    ) -> dict:
        row = {
            "id": self._next_id,
            "chunk_id": chunk_id,
            "category": category,
            "tags": tags,
            "title": title,
            "content": content,
            "priority": priority,
            "is_active": True,
            "has_embedding": True,
        }
        self._next_id += 1
        self._chunks.append(row)
        return row

    def list_chunks(self, *, include_inactive: bool = False) -> list[dict]:
        if include_inactive:
            return list(self._chunks)
        return [row for row in self._chunks if row.get("is_active", True)]


class _Limiter:
    def __init__(self, allow_result: bool):
        self.allow_result = allow_result
        self.calls: list[tuple[str, int]] = []

    def allow(self, key: str, limit: int) -> bool:
        self.calls.append((key, limit))
        return self.allow_result


@pytest.fixture(autouse=True)
def _admin_and_clean_rate_state(monkeypatch):
    monkeypatch.setenv("DS_ADMIN_KEY", "admin-secret")
    router._ingest_rate_fallback.clear()


def _request(
    *,
    headers: dict[str, str] | None = None,
    host: str = "127.0.0.1",
    limiter=None,
):
    return SimpleNamespace(
        headers=headers or {},
        client=SimpleNamespace(host=host),
        app=SimpleNamespace(state=SimpleNamespace(rate_limiter=limiter)),
    )


def _admin_request(*, limiter=None, host: str = "127.0.0.1"):
    return _request(
        headers={"x-admin-key": "admin-secret"},
        host=host,
        limiter=limiter,
    )


def _ingest_payload(**overrides) -> router.KnowledgeIngestRequest:
    base = {
        "source_name": "policy-update",
        "source_type": "text",
        "content": " ".join(["retention"] * 80),
        "category": "playbook",
        "tags": ["retention", "ops"],
        "priority": 4,
        "chunk_size": 300,
        "chunk_overlap": 40,
    }
    base.update(overrides)
    return router.KnowledgeIngestRequest(**base)


def test_ingest_knowledge_no_db_raises_503(monkeypatch):
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(router.ingest_knowledge(_ingest_payload(), _admin_request()))

    assert exc.value.status_code == 503


def test_ingest_knowledge_success(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: db)

    payload = _ingest_payload(source_name="new-policy", category="cancellation")
    result = asyncio.run(router.ingest_knowledge(payload, _admin_request()))

    assert result["source_name"] == "new-policy"
    assert result["source_type"] == "text"
    assert result["chunks_created"] >= 1
    assert result["chunks_failed"] == 0
    assert len(db.list_chunks(include_inactive=True)) == result["chunks_created"]


def test_ingest_knowledge_all_chunk_creates_fail_returns_400(monkeypatch):
    bad_db = MagicMock()
    bad_db.create_chunk.side_effect = RuntimeError("DB exploded")
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: bad_db)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(router.ingest_knowledge(_ingest_payload(), _admin_request()))

    assert exc.value.status_code == 400
    assert "DB exploded" in str(exc.value.detail)


def test_ingest_knowledge_rate_limited_returns_429(monkeypatch):
    db = _MockKnowledgeDb()
    limiter = _Limiter(allow_result=False)
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: db)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.ingest_knowledge(
                _ingest_payload(),
                _admin_request(limiter=limiter, host="10.0.0.44"),
            )
        )

    assert exc.value.status_code == 429
    assert limiter.calls[0][0] == "chat:knowledge_ingest:10.0.0.44"


def test_ingest_knowledge_text_rejects_content_base64(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: db)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.ingest_knowledge(
                _ingest_payload(content_base64="Zm9vYmFy"),
                _admin_request(),
            )
        )

    assert exc.value.status_code == 400
    assert "content_base64" in str(exc.value.detail)


def test_ingest_knowledge_text_payload_too_large_returns_413(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: db)
    monkeypatch.setattr(router, "_MAX_INGEST_TEXT_CHARS", 100)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.ingest_knowledge(
                _ingest_payload(content="x" * 101),
                _admin_request(),
            )
        )

    assert exc.value.status_code == 413


def test_ingest_knowledge_chunk_limit_returns_413(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: db)
    monkeypatch.setattr(router, "_MAX_INGEST_CHUNKS", 1)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.ingest_knowledge(
                _ingest_payload(
                    content=" ".join(["rule"] * 500),
                    chunk_size=200,
                    chunk_overlap=20,
                ),
                _admin_request(),
            )
        )

    assert exc.value.status_code == 413


def test_ingest_knowledge_pdf_requires_content_base64(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: db)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.ingest_knowledge(
                _ingest_payload(source_type="pdf", content_base64=None, content=None),
                _admin_request(),
            )
        )

    assert exc.value.status_code == 400


def test_ingest_knowledge_pdf_mime_validation(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: db)
    encoded = base64.b64encode(b"%PDF-1.7 fake payload").decode("ascii")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.ingest_knowledge(
                _ingest_payload(
                    source_type="pdf",
                    content=None,
                    content_base64=f"data:text/plain;base64,{encoded}",
                ),
                _admin_request(),
            )
        )

    assert exc.value.status_code == 400
    assert "MIME" in str(exc.value.detail)


def test_knowledge_stats_no_db_returns_503(monkeypatch):
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(router.knowledge_stats(_admin_request()))

    assert exc.value.status_code == 503


def test_knowledge_stats_reflects_current_store(monkeypatch):
    db = _MockKnowledgeDb()
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: db)
    req = _admin_request()

    asyncio.run(router.ingest_knowledge(_ingest_payload(category="cancellation"), req))
    asyncio.run(router.ingest_knowledge(_ingest_payload(category="upsell"), req))

    stats = asyncio.run(router.knowledge_stats(req))
    assert stats["total_chunks"] >= 2
    assert stats["active_chunks"] >= 2
    assert stats["embedded_chunks"] >= 2
    assert stats["categories"]["cancellation"] >= 1
    assert stats["categories"]["upsell"] >= 1


def test_list_knowledge_no_db_returns_503(monkeypatch):
    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            router.list_knowledge_chunks(_admin_request(), include_inactive=True)
        )

    assert exc.value.status_code == 503


def test_list_knowledge_returns_only_active_when_requested(monkeypatch):
    db = _MockKnowledgeDb()
    db.create_chunk(
        chunk_id="a-1",
        category="ops",
        tags=["ops"],
        title="A",
        content="A chunk content",
        priority=3,
    )
    row = db.create_chunk(
        chunk_id="b-2",
        category="ops",
        tags=["ops"],
        title="B",
        content="B chunk content",
        priority=4,
    )
    row["is_active"] = False

    monkeypatch.setattr(router, "get_knowledge_db_store_dep", lambda: db)
    req = _admin_request()

    active_only = asyncio.run(router.list_knowledge_chunks(req, include_inactive=False))
    all_rows = asyncio.run(router.list_knowledge_chunks(req, include_inactive=True))

    assert len(active_only) == 1
    assert len(all_rows) == 2
