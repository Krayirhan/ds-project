from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    cleaned_response: str
    issues: list[str]


_ENGLISH_PATTERNS = [
    r"\bthe\b",
    r"\bthis\b",
    r"\bcustomer\b",
    r"\bplease\b",
    r"\bshould\b",
    r"\bhotel\b",
    r"\bbooking\b",
    r"\bcancel\b",
]


def validate_response(text: str) -> ValidationResult:
    cleaned = text.strip()
    issues: list[str] = []

    if len(cleaned) < 20:
        issues.append("too_short")

    english_hits = 0
    for pattern in _ENGLISH_PATTERNS:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            english_hits += 1
    if english_hits >= 2:
        issues.append("mostly_english")

    if not cleaned:
        issues.append("empty")

    return ValidationResult(
        is_valid=not any(x in issues for x in ["empty", "mostly_english"]),
        cleaned_response=cleaned,
        issues=issues,
    )


def fallback_response(risk_percent: float) -> str:
    if risk_percent >= 65:
        return (
            f"Bu müşteri için iptal riski yüksek (%{risk_percent:.0f}). "
            "İlk olarak bugün kısa bir teyit araması yapın ve küçük bir avantaj teklif edin."
        )
    if risk_percent >= 35:
        return (
            f"Bu müşteri için iptal riski orta (%{risk_percent:.0f}). "
            "Check-in tarihine 7 gün kala hatırlatma ve teyit mesajı gönderin."
        )
    return (
        f"Bu müşteri için iptal riski düşük (%{risk_percent:.0f}). "
        "Ek hizmet önerisi ile bağlılık ve gelir artırabilirsiniz."
    )
