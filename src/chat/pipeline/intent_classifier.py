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
    msg = user_message.lower()
    for keywords, intent, hint in _RULES:
        if any(k in msg for k in keywords):
            return ClassifiedIntent(intent=intent, confidence=1.0, hint_for_llm=hint)
    return ClassifiedIntent(
        intent=Intent.GENERAL_CHAT,
        confidence=0.5,
        hint_for_llm="Soruyu müşteri hizmetleri bağlamında kısa ve net yanıtla.",
    )
