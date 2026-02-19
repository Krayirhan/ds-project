from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    RISK_EXPLANATION = "risk_explanation"
    ACTION_REQUEST = "action_request"
    UPSELL_QUERY = "upsell_query"
    CUSTOMER_PROFILE = "customer_profile"
    POLICY_QUESTION = "policy_question"
    GENERAL_CHAT = "general_chat"


@dataclass(frozen=True)
class ClassifiedIntent:
    intent: Intent
    confidence: float
    hint_for_llm: str


_RULES: list[tuple[list[str], Intent, str]] = [
    (
        ["neden", "niye", "risk", "ne anlama"],
        Intent.RISK_EXPLANATION,
        "Risk nedenlerini sade ve sayısal örneklerle açıkla.",
    ),
    (
        ["ne yap", "önle", "aksiyon", "adım", "öner", "tavsiye"],
        Intent.ACTION_REQUEST,
        "Temsilciye 3 somut adım ver.",
    ),
    (
        ["upsell", "ek hizmet", "upgrade", "satış", "ek gelir"],
        Intent.UPSELL_QUERY,
        "Müşteri profiline uygun ek hizmet öner.",
    ),
    (
        ["profil", "müşteri kim", "nasıl biri", "hakkında"],
        Intent.CUSTOMER_PROFILE,
        "Müşteri profilini kısa özetle.",
    ),
    (
        ["politika", "kural", "iade", "prosedür", "şart"],
        Intent.POLICY_QUESTION,
        "Politikayı net ve kısa açıkla.",
    ),
]


def classify_intent(user_message: str) -> ClassifiedIntent:
    """Score each intent by fraction of its keywords that appear in the message.

    Unlike first-match-wins, this approach:
    - Handles messages that match multiple rule sets (picks highest score)
    - Gives partial credit for partial matches
    - Produces a calibrated confidence value in [0.5, 1.0]
    """
    msg = user_message.lower()
    best_intent: Intent = Intent.GENERAL_CHAT
    best_score: float = 0.0
    best_hint: str = "Soruyu müşteri hizmetleri bağlamında kısa ve net yanıtla."

    for keywords, intent, hint in _RULES:
        matches = sum(1 for k in keywords if k in msg)
        if matches == 0:
            continue
        # Normalize by keyword count so short rules don't dominate
        score = matches / len(keywords)
        if score > best_score:
            best_score = score
            best_intent = intent
            best_hint = hint

    # Map raw score [0, 1] → confidence [0.5, 1.0]
    confidence = round(min(1.0, 0.5 + best_score * 0.5), 2) if best_score > 0 else 0.5
    return ClassifiedIntent(
        intent=best_intent,
        confidence=confidence,
        hint_for_llm=best_hint,
    )
