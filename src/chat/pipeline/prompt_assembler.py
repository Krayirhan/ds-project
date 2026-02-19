from __future__ import annotations

from .context_builder import CustomerContext
from .intent_classifier import ClassifiedIntent


SYSTEM_PROMPT = """Sen bir otel müşteri hizmetleri karar asistanısın.
Yalnızca Türkçe yaz.
Teknik terim kullanma.
Kısa, net ve uygulanabilir öneriler ver.
Yanıtın sonunda temsilcinin atacağı ilk adımı söyle."""


def assemble_first_prompt(*, ctx: CustomerContext) -> str:
    factors = "\n".join(f"- {f}" for f in ctx.key_risk_factors)
    retrieval = ctx.retrieved_chunks_text.strip()
    retrieval_block = f"\n\nİlgili politikalar:\n{retrieval}" if retrieval else ""
    return (
        f"Müşteri özeti:\n{ctx.profile_summary_tr}\n\n"
        f"Risk faktörleri:\n{factors}"
        f"{retrieval_block}\n\n"
        "Bu müşteri için 3 somut aksiyon öner."
    )


def assemble_user_prompt(
    *,
    ctx: CustomerContext,
    classified_intent: ClassifiedIntent,
    user_message: str,
) -> str:
    factors = "\n".join(f"- {f}" for f in ctx.key_risk_factors)
    retrieval = ctx.retrieved_chunks_text.strip()
    retrieval_block = f"\n\nİlgili politikalar:\n{retrieval}" if retrieval else ""
    return (
        f"Müşteri özeti:\n{ctx.profile_summary_tr}\n\n"
        f"Risk faktörleri:\n{factors}"
        f"{retrieval_block}\n\n"
        f"Yanıt yönlendirmesi: {classified_intent.hint_for_llm}\n\n"
        f"Kullanıcı sorusu: {user_message}"
    )
