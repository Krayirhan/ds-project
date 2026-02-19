from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatSession:
    session_id: str
    customer_data: dict[str, Any]
    risk_score: float
    risk_label: str
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def add_message(self, *, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self.last_active = time.time()

    def to_ollama_messages(self, *, system_prompt: str) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for msg in self.messages:
            payload.append({"role": msg.role, "content": msg.content})
        return payload

    def is_expired(self, *, ttl_seconds: int) -> bool:
        return (time.time() - self.last_active) > ttl_seconds


class SessionStore:
    def __init__(self, *, ttl_seconds: int = 3600, max_history: int = 20) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_history = max_history
        self._sessions: dict[str, ChatSession] = {}

    def create_session(
        self,
        *,
        customer_data: dict[str, Any],
        risk_score: float,
        risk_label: str,
    ) -> ChatSession:
        session = ChatSession(
            session_id=str(uuid.uuid4()),
            customer_data=customer_data,
            risk_score=risk_score,
            risk_label=risk_label,
        )
        self._sessions[session.session_id] = session
        self._cleanup_expired()
        return session

    def get_session(self, *, session_id: str) -> ChatSession | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired(ttl_seconds=self.ttl_seconds):
            del self._sessions[session_id]
            return None
        return session

    def trim_history(self, *, session: ChatSession) -> None:
        if len(session.messages) <= self.max_history:
            return
        # Son konuşmayı koru, ilk uzun bağlamı da çok azaltma
        session.messages = session.messages[:2] + session.messages[-(self.max_history - 2) :]

    def _cleanup_expired(self) -> None:
        expired = [
            sid
            for sid, session in self._sessions.items()
            if session.is_expired(ttl_seconds=self.ttl_seconds)
        ]
        for sid in expired:
            self._sessions.pop(sid, None)


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
