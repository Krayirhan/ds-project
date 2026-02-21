from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import src.chat.orchestrator as orc
from src.chat.memory import ChatSession
from src.chat.pipeline.intent_classifier import ClassifiedIntent, Intent


class _Store:
    def __init__(self):
        self.sessions: dict[str, ChatSession] = {}
        self.saved = 0
        self.trimmed = 0

    def create_session(self, *, customer_data, risk_score, risk_label):
        s = ChatSession(
            session_id="sess-1",
            customer_data=customer_data,
            risk_score=risk_score,
            risk_label=risk_label,
        )
        self.sessions[s.session_id] = s
        return s

    def get_session(self, *, session_id):
        return self.sessions.get(session_id)

    def save_session(self, session):
        self.sessions[session.session_id] = session
        self.saved += 1

    def trim_history(self, *, session):
        self.trimmed += 1


class _Knowledge:
    def retrieve_by_customer(self, **kwargs):
        return [SimpleNamespace(title="T", content="C")]

    def retrieve_by_text(self, **kwargs):
        return [SimpleNamespace(title="T2", content="C2")]


class _Ollama:
    def __init__(self, outputs=None, exc: Exception | None = None):
        self.outputs = list(outputs or ["yanit"])
        self.exc = exc
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.outputs.pop(0)


def _make_orchestrator(monkeypatch, *, knowledge=None, ollama=None, store=None):
    st = store or _Store()
    kn = knowledge or _Knowledge()
    ol = ollama or _Ollama()
    monkeypatch.setattr(orc, "get_session_store", lambda: st)
    monkeypatch.setattr(orc, "get_knowledge_store", lambda: kn)
    monkeypatch.setattr(orc, "get_ollama_client", lambda: ol)
    return orc.ChatOrchestrator(), st, kn, ol


def test_start_session_success(monkeypatch):
    orchestrator, store, _kn, _ol = _make_orchestrator(monkeypatch)
    monkeypatch.setattr(
        orc,
        "build_customer_context",
        lambda **kwargs: SimpleNamespace(risk_percent=80.0),
    )
    monkeypatch.setattr(orc, "assemble_first_prompt", lambda **kwargs: "first prompt")
    monkeypatch.setattr(orc, "validate_response", lambda text: SimpleNamespace(is_valid=True, cleaned_response=text, issues=[]))

    sid, reply = asyncio.run(
        orchestrator.start_session(
            customer_data={"hotel": "City"},
            risk_score=0.8,
            risk_label="high",
        )
    )
    assert sid == "sess-1"
    assert isinstance(reply, str)
    assert store.saved == 1


def test_send_message_missing_session_raises(monkeypatch):
    orchestrator, _store, _kn, _ol = _make_orchestrator(monkeypatch)
    with pytest.raises(ValueError):
        asyncio.run(orchestrator.send_message(session_id="missing", user_message="hello"))


def test_send_message_with_text_retrieval(monkeypatch):
    orchestrator, store, _kn, _ol = _make_orchestrator(monkeypatch)
    session = store.create_session(customer_data={"lead_time": 10}, risk_score=0.5, risk_label="mid")
    _mock_intent = ClassifiedIntent(intent=Intent.RISK_EXPLANATION, confidence=0.8, hint_for_llm="hint")
    monkeypatch.setattr(orc, "classify_intent", lambda _: _mock_intent)
    monkeypatch.setattr(
        orc,
        "build_customer_context",
        lambda **kwargs: SimpleNamespace(risk_percent=50.0),
    )
    monkeypatch.setattr(orc, "assemble_user_prompt", lambda **kwargs: "user prompt")

    async def _ask(*args, **kwargs):
        return "assistant reply"

    monkeypatch.setattr(orchestrator, "_ask_llm", _ask)
    out = asyncio.run(orchestrator.send_message(session_id=session.session_id, user_message="what now"))
    assert out == "assistant reply"
    assert store.saved == 1
    assert store.trimmed == 1


def test_send_message_fallback_to_customer_retrieval(monkeypatch):
    class _KnowledgeNoText:
        def retrieve_by_customer(self, **kwargs):
            return [SimpleNamespace(title="A", content="B")]

    orchestrator, store, _kn, _ol = _make_orchestrator(monkeypatch, knowledge=_KnowledgeNoText())
    session = store.create_session(customer_data={"lead_time": 10}, risk_score=0.5, risk_label="mid")
    _mock_intent = ClassifiedIntent(intent=Intent.RISK_EXPLANATION, confidence=0.8, hint_for_llm="hint")
    monkeypatch.setattr(orc, "classify_intent", lambda _: _mock_intent)
    monkeypatch.setattr(
        orc,
        "build_customer_context",
        lambda **kwargs: SimpleNamespace(risk_percent=50.0),
    )
    monkeypatch.setattr(orc, "assemble_user_prompt", lambda **kwargs: "user prompt")

    async def _ask(*args, **kwargs):
        return "assistant reply"

    monkeypatch.setattr(orchestrator, "_ask_llm", _ask)
    out = asyncio.run(orchestrator.send_message(session_id=session.session_id, user_message="what now"))
    assert out == "assistant reply"


def test_ask_llm_valid_retry_fallback_and_exception(monkeypatch):
    orchestrator, store, _kn, _ol = _make_orchestrator(monkeypatch, ollama=_Ollama(outputs=["ilk", "ikinci"]))
    session = store.create_session(customer_data={}, risk_score=0.7, risk_label="high")

    monkeypatch.setattr(orc, "SYSTEM_PROMPT", "SYSTEM")
    monkeypatch.setattr(
        orc,
        "validate_response",
        lambda text: SimpleNamespace(is_valid=True, cleaned_response=f"CLEAN:{text}", issues=[]),
    )
    out1 = asyncio.run(orchestrator._ask_llm(session=session, risk_percent=70))
    assert out1.startswith("CLEAN:")

    # mostly_english retry branch
    orchestrator.ollama = _Ollama(outputs=["english", "turkce"])
    calls = {"n": 0}

    def _validate_retry(text):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(is_valid=False, cleaned_response=text, issues=["mostly_english"])
        return SimpleNamespace(is_valid=True, cleaned_response="temiz", issues=[])

    monkeypatch.setattr(orc, "validate_response", _validate_retry)
    out2 = asyncio.run(orchestrator._ask_llm(session=session, risk_percent=70))
    assert out2 == "temiz"

    # invalid without retry-success -> fallback
    monkeypatch.setattr(
        orc,
        "validate_response",
        lambda text: SimpleNamespace(is_valid=False, cleaned_response=text, issues=["too_short"]),
    )
    monkeypatch.setattr(orc, "fallback_response", lambda pct, intent=None: f"fallback-{pct}")
    out3 = asyncio.run(orchestrator._ask_llm(session=session, risk_percent=55))
    assert out3 == "fallback-55"

    # exception path -> fallback
    orchestrator.ollama = _Ollama(exc=RuntimeError("llm down"))
    out4 = asyncio.run(orchestrator._ask_llm(session=session, risk_percent=40))
    assert out4 == "fallback-40"


def test_quick_actions_and_summary_paths(monkeypatch):
    orchestrator, store, _kn, _ol = _make_orchestrator(monkeypatch)

    assert asyncio.run(orchestrator.quick_actions(session_id="missing")) == []

    s_high = store.create_session(customer_data={}, risk_score=0.9, risk_label="high")
    high_actions = asyncio.run(orchestrator.quick_actions(session_id=s_high.session_id))
    assert len(high_actions) == 3

    s_mid = ChatSession(session_id="mid", customer_data={}, risk_score=0.5, risk_label="mid")
    store.sessions[s_mid.session_id] = s_mid
    mid_actions = asyncio.run(orchestrator.quick_actions(session_id=s_mid.session_id))
    assert len(mid_actions) == 2

    s_low = ChatSession(session_id="low", customer_data={}, risk_score=0.2, risk_label="low")
    store.sessions[s_low.session_id] = s_low
    low_actions = asyncio.run(orchestrator.quick_actions(session_id=s_low.session_id))
    assert len(low_actions) == 2

    summary = asyncio.run(orchestrator.summary(session_id=s_high.session_id))
    assert summary["session_id"] == s_high.session_id
    with pytest.raises(ValueError):
        asyncio.run(orchestrator.summary(session_id="missing"))


def test_get_orchestrator_singleton(monkeypatch):
    # Use monkeypatch for the singleton so it's restored after the test
    monkeypatch.setattr(orc, "_orchestrator", None)
    created = {"n": 0}

    class _DummyOrchestrator:
        def __init__(self):
            created["n"] += 1

    monkeypatch.setattr(orc, "ChatOrchestrator", _DummyOrchestrator)
    o1 = orc.get_orchestrator()
    o2 = orc.get_orchestrator()
    assert o1 is o2
    assert created["n"] == 1
