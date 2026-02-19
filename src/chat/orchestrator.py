from __future__ import annotations

from typing import Any

from ..utils import get_logger
from .knowledge import get_knowledge_store
from .memory import ChatSession, get_session_store
from .ollama_client import get_ollama_client
from .pipeline import (
    SYSTEM_PROMPT,
    assemble_first_prompt,
    assemble_user_prompt,
    build_customer_context,
    classify_intent,
    fallback_response,
    validate_response,
)

logger = get_logger("chat_orchestrator")


class ChatOrchestrator:
    def __init__(self) -> None:
        self.store = get_session_store()
        self.knowledge = get_knowledge_store()
        self.ollama = get_ollama_client()

    async def start_session(
        self,
        *,
        customer_data: dict[str, Any],
        risk_score: float,
        risk_label: str,
    ) -> tuple[str, str]:
        chunks = self.knowledge.retrieve_by_customer(
            customer_data=customer_data,
            risk_score=risk_score,
            top_k=3,
        )
        retrieved = "\n\n".join(f"{c.title}: {c.content}" for c in chunks)
        ctx = build_customer_context(
            customer_data=customer_data,
            risk_score=risk_score,
            risk_label=risk_label,
            retrieved_chunks_text=retrieved,
        )

        session = self.store.create_session(
            customer_data=customer_data,
            risk_score=risk_score,
            risk_label=risk_label,
        )

        first_prompt = assemble_first_prompt(ctx=ctx)
        session.add_message(role="user", content=first_prompt)

        reply = await self._ask_llm(session=session, risk_percent=ctx.risk_percent)
        session.add_message(role="assistant", content=reply)
        self.store.save_session(session)  # persist mutations (#25)
        return session.session_id, reply

    async def send_message(self, *, session_id: str, user_message: str) -> str:
        session = self.store.get_session(session_id=session_id)
        if session is None:
            raise ValueError("Oturum bulunamadı veya süresi doldu")

        intent = classify_intent(user_message)

        # Hybrid retrieval: use message text (TF-IDF) when available, tag fallback otherwise
        chunks = self.knowledge.retrieve_by_text(
            query=user_message, top_k=2
        ) if hasattr(self.knowledge, "retrieve_by_text") else self.knowledge.retrieve_by_customer(
            customer_data=session.customer_data,
            risk_score=session.risk_score,
            top_k=2,
        )

        retrieved = "\n\n".join(f"{c.title}: {c.content}" for c in chunks)
        ctx = build_customer_context(
            customer_data=session.customer_data,
            risk_score=session.risk_score,
            risk_label=session.risk_label,
            retrieved_chunks_text=retrieved,
        )

        prompt = assemble_user_prompt(
            ctx=ctx,
            classified_intent=intent,
            user_message=user_message,
        )
        session.add_message(role="user", content=prompt)
        self.store.trim_history(session=session)

        reply = await self._ask_llm(session=session, risk_percent=ctx.risk_percent)
        session.add_message(role="assistant", content=reply)
        self.store.save_session(session)  # persist mutations (#25)
        return reply

    async def _ask_llm(self, *, session: ChatSession, risk_percent: float) -> str:
        messages = session.to_ollama_messages(system_prompt=SYSTEM_PROMPT)
        try:
            raw = await self.ollama.chat(messages=messages, temperature=0.3)
            checked = validate_response(raw)
            if checked.is_valid:
                return checked.cleaned_response
            if "mostly_english" in checked.issues:
                retry_messages = session.to_ollama_messages(
                    system_prompt=SYSTEM_PROMPT + "\nKesinlikle yalnızca Türkçe yaz."
                )
                raw_retry = await self.ollama.chat(messages=retry_messages, temperature=0.2)
                checked_retry = validate_response(raw_retry)
                if checked_retry.is_valid:
                    return checked_retry.cleaned_response
            return fallback_response(risk_percent)
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return fallback_response(risk_percent)

    async def quick_actions(self, *, session_id: str) -> list[dict[str, str]]:
        session = self.store.get_session(session_id=session_id)
        if session is None:
            return []
        if session.risk_score >= 0.6:
            return [
                {"label": "Ne yapmalıyım?", "message": "Bu müşteri için şimdi ne yapmalıyım?"},
                {"label": "Neden riskli?", "message": "Risk neden yüksek görünüyor?"},
                {"label": "Arama metni", "message": "Müşteriyi ararken ne söylemeliyim?"},
            ]
        if session.risk_score >= 0.35:
            return [
                {"label": "Önlem gerekli mi?", "message": "Bu müşteri için hemen önlem almalı mıyım?"},
                {"label": "Kritik nokta", "message": "En kritik risk faktörü nedir?"},
            ]
        return [
            {"label": "Upsell öner", "message": "Bu müşteriye uygun ek hizmet önerisi ver."},
            {"label": "Güvenli mi?", "message": "Bu rezervasyon güvenli mi?"},
        ]

    async def summary(self, *, session_id: str) -> dict[str, Any]:
        session = self.store.get_session(session_id=session_id)
        if session is None:
            raise ValueError("Oturum bulunamadı")
        return {
            "session_id": session.session_id,
            "risk_score": session.risk_score,
            "risk_label": session.risk_label,
            "message_count": len(session.messages),
            "created_at": session.created_at,
            "last_active": session.last_active,
        }


_orchestrator: ChatOrchestrator | None = None


def get_orchestrator() -> ChatOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ChatOrchestrator()
    return _orchestrator
