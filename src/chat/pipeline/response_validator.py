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


def fallback_response(risk_percent: float, intent: str | None = None) -> str:
    """Intent-aware rule-based response for when LLM is unavailable."""
    pct = f"{risk_percent:.0f}"
    high = risk_percent >= 65
    medium = risk_percent >= 35

    if intent == "risk_explanation":
        if high:
            return (
                f"Risk %{pct} yüksek. Temel nedenler: depozito alınmamış (No Deposit), "
                f"Online TA kanalında iptal oranı yüksek, geçmiş iptal geçmişi var. "
                f"Bu faktörlerin bir arada bulunması istatistiksel olarak en yüksek iptal riskine işaret eder. "
                f"Depozito talep etmek tek başına riski %20-30 düşürebilir."
            )
        elif medium:
            return (
                f"Risk %{pct} orta seviyede. En belirleyici faktörler: depozito türü ve rezervasyon kanalı. "
                f"Lead time 30 günden fazlaysa iptal olasılığı artar. "
                f"Proaktif iletişim ve küçük bir avantaj teklifi bu riski düşürebilir."
            )
        else:
            return (
                f"Risk %{pct} düşük. Güvenilir müşteri profili: sadık müşteri, "
                f"güçlü depozito veya düşük iptal geçmişi bu seviyeye katkı sağlıyor. "
                f"Standart prosedürleri uygulayın, ek tedbir gerekmez."
            )

    elif intent == "action_request":
        if high:
            return (
                f"İptal riski %{pct} yüksek. Önerilen 3 adım:\n"
                f"1. Bugün kısa bir teyit araması yapın ve küçük bir avantaj teklif edin (ücretsiz oda yükseltme vb.).\n"
                f"2. Depozito politikasını nazikçe hatırlatın veya iade garantisi önerin.\n"
                f"3. Check-in'e 3 gün kala kişiselleştirilmiş SMS veya e-posta hatırlatması gönderin."
            )
        elif medium:
            return (
                f"İptal riski %{pct} orta. Önerilen 3 adım:\n"
                f"1. Check-in tarihine 7 gün kala hatırlatma ve teyit mesajı gönderin.\n"
                f"2. Kişiselleştirilmiş bir karşılama notu veya erken check-in teklifi ekleyin.\n"
                f"3. Gerekirse esnek iade seçeneği veya küçük bir avantaj sunmayı değerlendirin."
            )
        else:
            return (
                f"Risk %{pct} düşük, acil önlem gerekmez. Yine de:\n"
                f"1. Oda yükseltme veya ek hizmet önerin (upsell fırsatı).\n"
                f"2. Sadakat programı teklifini iletin.\n"
                f"3. Konaklamadan sonra memnuniyet anketi planlayın."
            )

    elif intent == "customer_profile":
        level = "yüksek" if high else "orta" if medium else "düşük"
        return (
            f"Bu müşteri %{pct} iptal riski taşıyor — {level} seviye. "
            f"Profili belirleyen başlıca etkenler: rezervasyon kanalı, depozito türü ve geçmiş iptal sayısı. "
            f"Müşteri ilk kez rezervasyon yapıyorsa risk daha yüksektir. "
            f"Detaylı analiz için müşterinin geçmiş rezervasyon geçmişine bakmanızı öneririm."
        )

    elif intent == "policy_question":
        return (
            "İptal politikası özeti:\n"
            "• No Deposit: Ücretsiz iptal, ancak iptal riski en yüksek grup.\n"
            "• Non Refund: İade edilmez — iptal oranı düşük ama müşteri memnuniyeti riski var.\n"
            "• Refundable: Check-in'den 48 saat öncesine kadar ücretsiz iptal hakkı tanır.\n"
            "Politika değişikliği veya istisna için otel yöneticisi ile iletişime geçin."
        )

    elif intent == "upsell_query":
        return (
            "Müşteri profiline uygun upsell önerileri:\n"
            "1. Oda yükseltme — suite veya deniz/havuz manzaralı oda\n"
            "2. Yarım pansiyon veya tam pansiyon geçişi\n"
            "3. Spa, aktivite paketi veya havalimanı transfer hizmeti\n"
            "Not: Düşük riskli ve sadık müşterilerde upsell kabul oranı belirgin biçimde daha yüksektir."
        )

    else:  # GENERAL_CHAT veya tanımlanamayan niyet
        if high:
            return (
                f"Bu müşteri için iptal riski yüksek (%{pct}). "
                "İlk olarak bugün kısa bir teyit araması yapın ve küçük bir avantaj teklif edin."
            )
        elif medium:
            return (
                f"Bu müşteri için iptal riski orta (%{pct}). "
                "Check-in tarihine 7 gün kala hatırlatma ve teyit mesajı gönderin."
            )
        else:
            return (
                f"Bu müşteri için iptal riski düşük (%{pct}). "
                "Ek hizmet önerisi ile bağlılık ve gelir artırabilirsiniz."
            )
