from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .ollama_client import get_ollama_client
from .orchestrator import get_orchestrator

router_chat = APIRouter(prefix="/chat", tags=["chat"])


class StartSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_data: dict[str, Any] = Field(default_factory=dict)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_label: str = "unknown"


class StartSessionResponse(BaseModel):
    session_id: str
    bot_message: str
    quick_actions: list[dict[str, str]]
    risk_score: float
    risk_label: str


class ChatMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    message: str = Field(min_length=1, max_length=2000)


class ChatMessageResponse(BaseModel):
    session_id: str
    bot_message: str
    quick_actions: list[dict[str, str]]


class SessionSummaryResponse(BaseModel):
    session_id: str
    risk_score: float
    risk_label: str
    message_count: int
    created_at: float
    last_active: float


@router_chat.get("/health")
async def chat_health() -> dict[str, str]:
    client = get_ollama_client()
    alive = await client.health()
    return {
        "status": "ok" if alive else "degraded",
        "ollama": "ok" if alive else "unreachable",
        "model": client.model,
    }


@router_chat.post("/session", response_model=StartSessionResponse)
async def start_session(body: StartSessionRequest) -> StartSessionResponse:
    orchestrator = get_orchestrator()
    try:
        session_id, bot_message = await orchestrator.start_session(
            customer_data=body.customer_data,
            risk_score=body.risk_score,
            risk_label=body.risk_label,
        )
        actions = await orchestrator.quick_actions(session_id=session_id)
        return StartSessionResponse(
            session_id=session_id,
            bot_message=bot_message,
            quick_actions=actions,
            risk_score=body.risk_score,
            risk_label=body.risk_label,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router_chat.post("/message", response_model=ChatMessageResponse)
async def message(body: ChatMessageRequest) -> ChatMessageResponse:
    orchestrator = get_orchestrator()
    try:
        bot_message = await orchestrator.send_message(
            session_id=body.session_id,
            user_message=body.message,
        )
        actions = await orchestrator.quick_actions(session_id=body.session_id)
        return ChatMessageResponse(
            session_id=body.session_id,
            bot_message=bot_message,
            quick_actions=actions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router_chat.get("/session/{session_id}/summary", response_model=SessionSummaryResponse)
async def summary(session_id: str) -> SessionSummaryResponse:
    orchestrator = get_orchestrator()
    try:
        payload = await orchestrator.summary(session_id=session_id)
        return SessionSummaryResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
