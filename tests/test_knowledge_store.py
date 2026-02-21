from __future__ import annotations

import sys

import src.chat.knowledge.store as ks


def test_store_build_fallback_when_tfidf_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", None)
    store = ks.KnowledgeStore()
    assert store._vectorizer is None
    assert store._tfidf_matrix is None


def test_retrieve_by_text_fallback_without_vectorizer():
    store = ks.KnowledgeStore()
    store._vectorizer = None
    store._tfidf_matrix = None
    out = store.retrieve_by_text(query="iptal", top_k=2)
    assert len(out) == 2
    assert out[0].priority <= out[1].priority


def test_retrieve_by_text_exception_falls_back():
    class _BadVectorizer:
        def transform(self, *_args, **_kwargs):
            raise RuntimeError("bad vectorizer")

    store = ks.KnowledgeStore()
    store._vectorizer = _BadVectorizer()
    store._tfidf_matrix = object()
    out = store.retrieve_by_text(query="online", top_k=3)
    assert len(out) == 3


def test_retrieve_tag_overlap_and_extract_tags_paths():
    store = ks.KnowledgeStore()
    out = store.retrieve(tags=["online", "depozito", "non_refund"], top_k=3)
    assert len(out) >= 1
    assert out[0].chunk_id in {"pol_001", "pol_002", "pol_005"}

    low_risk_tags = store._extract_tags(
        customer_data={"deposit_type": "Non Refund", "lead_time": 1, "previous_cancellations": 0},
        risk_score=0.2,
    )
    assert any("risk" in t for t in low_risk_tags)
    assert "non_refund" in low_risk_tags


def test_retrieve_by_customer_fallback_to_tag_scoring():
    store = ks.KnowledgeStore()
    store._vectorizer = None
    out = store.retrieve_by_customer(
        customer_data={
            "deposit_type": "No Deposit",
            "lead_time": 220,
            "previous_cancellations": 1,
            "market_segment": "Online TA",
        },
        risk_score=0.9,
        top_k=3,
    )
    assert len(out) >= 1
