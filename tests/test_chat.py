import os

from fastapi.testclient import TestClient

from src.api import app


def _headers() -> dict[str, str]:
    return {"x-api-key": os.environ.get("DS_API_KEY", "test-key")}


def test_chat_session_and_message_flow():
    os.environ.setdefault("DS_API_KEY", "test-key")
    payload = {
        "customer_data": {
            "hotel": "City Hotel",
            "lead_time": 210,
            "deposit_type": "No Deposit",
            "previous_cancellations": 1,
            "market_segment": "Online TA",
            "adults": 2,
            "children": 0,
            "stays_in_week_nights": 2,
            "stays_in_weekend_nights": 1,
        },
        "risk_score": 0.73,
        "risk_label": "high",
    }

    with TestClient(app) as client:
        created = client.post("/chat/session", json=payload, headers=_headers())
        assert created.status_code == 200
        body = created.json()
        assert "session_id" in body
        assert isinstance(body.get("bot_message"), str)

        session_id = body["session_id"]
        msg = client.post(
            "/chat/message",
            json={
                "session_id": session_id,
                "message": "Bu müşteri için ilk adım ne olmalı?",
            },
            headers=_headers(),
        )
        assert msg.status_code == 200
        msg_body = msg.json()
        assert msg_body["session_id"] == session_id
        assert isinstance(msg_body.get("bot_message"), str)


def test_chat_requires_api_key():
    os.environ.setdefault("DS_API_KEY", "test-key")
    payload = {
        "customer_data": {"lead_time": 10},
        "risk_score": 0.2,
        "risk_label": "low",
    }
    with TestClient(app) as client:
        r = client.post("/chat/session", json=payload)
        assert r.status_code == 401
