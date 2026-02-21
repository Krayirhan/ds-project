from __future__ import annotations

from .context_builder import CustomerContext
from .intent_classifier import ClassifiedIntent


SYSTEM_PROMPT = """Sen bir otel müşteri hizmetleri uzmanısın. Görevin, otel temsilcisine müşteri rezervasyon iptal riskini yönetmesinde somut destek vermektir.

KATİ KURALLAR:
• Yanıtlarını YALNIZCA Türkçe yaz — tek bir İngilizce kelime kullanma.
• Yanıtı 120 kelimeyi geçme; kısa, öz ve uygulanabilir ol.
• Somut ve numaralı adımlar ver: “1. … 2. … 3. …”
• Teknik terim veya istatistik jargonu kullanma; sade otel dili kullan.
• Her yanıtın sonuna “🎯 İlk Adım:” etiketiyle tek bir öncelikli eylem ekle.
• Müşteri verisi ve risk yüzdesı sana verilmektedir; bunlara dayanan önerilerde bulun.
• Otel temsilcisine doğrudan ve arkadaşça hitap et.
• ÖNEMLİ: "Mevcut Durum" bölümündeki bilgilere dikkat et. Zaten uygulanmış olan önlemleri tekrar önerme.
  - Depozito zaten "İade Edilmez" ise bunu önerme; bunun yerine iptali engellemeye yönelik başka önlemler öner.
  - Sadık müşteri ise "sadık müşteri avantajı sun" önerisi anlamsızdır; farklı bir yaklaşım öner.
  - İptal geçmişi yoksa "geçmiş iptallere dikkat et" demekten kaçın.
• KRİTİK — Model skoru ile çelişme: ML modeli yüksek risk gösteriyorsa bunu yanılış veya "İade edilmez depozito varken iptal olmaz" gibi çelişkili açıklamalar yapma.
  Gerçek otel verilerinde iade edilmez depozitolu müşteriler de yüksek oranda iptal edebilmektedir; bu nedenle model skoru geçerlidir.
  Görevin: riski sorgulamak değil, riski düşürmek için somut ve uygulanabilir adımlar önermektir."""


def assemble_first_prompt(*, ctx: CustomerContext) -> str:
    factors = "\n".join(f"• {f}" for f in ctx.key_risk_factors)
    retrieval = ctx.retrieved_chunks_text.strip()
    retrieval_block = f"\n\n📋 Politika referansları:\n{retrieval}" if retrieval else ""
    current_state = _current_state_block(ctx.raw_data)
    return (
        f"📊 Müşteri Profili:\n{ctx.profile_summary_tr}\n\n"
        f"⚠️ Risk Seviyesi: {ctx.risk_level_tr} (%{ctx.risk_percent:.0f})\n\n"
        f"🔍 Risk Faktörleri:\n{factors}"
        f"{current_state}"
        f"{retrieval_block}\n\n"
        f"Bu müşteri için {ctx.risk_level_tr} riske karşı 3 somut ve öncelikli aksiyon öner. "
        f"Mevcut durumda zaten uygulanmış önlemleri tekrar önerme."
    )


def assemble_user_prompt(
    *,
    ctx: CustomerContext,
    classified_intent: ClassifiedIntent,
    user_message: str,
) -> str:
    factors = "\n".join(f"• {f}" for f in ctx.key_risk_factors)
    retrieval = ctx.retrieved_chunks_text.strip()
    retrieval_block = f"\n\n📋 Politika referansları:\n{retrieval}" if retrieval else ""
    current_state = _current_state_block(ctx.raw_data)
    return (
        f"📊 Müşteri: {ctx.profile_summary_tr}\n"
        f"⚠️ Risk: {ctx.risk_level_tr} (%{ctx.risk_percent:.0f})\n"
        f"🔍 Faktörler:\n{factors}"
        f"{current_state}"
        f"{retrieval_block}\n\n"
        f"💡 Yanıt odağı: {classified_intent.hint_for_llm}\n\n"
        f"❓ Soru: {user_message}"
    )


def _current_state_block(customer_data: dict) -> str:
    """LLM'e mevcut rezervasyon durumunu açıklar; zaten uygulanmış önlemleri tekrar önermesini engeller."""
    lines: list[str] = []

    dep = str(customer_data.get("deposit_type", ""))
    if dep == "Non Refund":
        lines.append("✅ Depozito: İade Edilmez (zaten uygulanmış — tekrar önerme)")
    elif dep == "Refundable":
        lines.append("🟡 Depozito: İade Edilebilir")
    else:
        lines.append("❌ Depozito: Yok (alınmamış)")

    if int(customer_data.get("is_repeated_guest", 0) or 0) == 1:
        lines.append("✅ Sadık müşteri (daha önce konaklamış — sadakat avanıtajını tekrar sunma)")
    else:
        lines.append("🔴 İlk ziyaret")

    prev = int(customer_data.get("previous_cancellations", 0) or 0)
    if prev > 0:
        lines.append(f"❌ Geçmiş iptal: {prev} adet")
    else:
        lines.append("✅ Geçmiş iptal yok")

    block = "\n".join(f"  {l}" for l in lines)
    return f"\n\n📌 Mevcut Durum (bunları tekrar önerme):\n{block}"
