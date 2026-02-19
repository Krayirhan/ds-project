from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    category: str
    tags: list[str]
    title: str
    content: str
    priority: int = 5


KNOWLEDGE_BASE: list[KnowledgeChunk] = [
    KnowledgeChunk(
        chunk_id="pol_001",
        category="cancellation_policy",
        tags=["depozito", "non_refund", "iptal", "iade"],
        title="İade edilmez depozito",
        content=(
            "İade edilmez depozito olan rezervasyonlarda iptal eğilimi daha düşüktür. "
            "Müşteriye rezervasyonun korunduğunu, tarih değişikliği seçeneğinin iptalden daha iyi olduğunu anlatın."
        ),
        priority=1,
    ),
    KnowledgeChunk(
        chunk_id="pol_002",
        category="cancellation_policy",
        tags=["depozito", "no_deposit", "iptal", "yüksek_risk"],
        title="Depozitosuz rezervasyon",
        content=(
            "Depozitosuz rezervasyonlarda müşteri finansal bağlılık hissetmeyebilir. "
            "Küçük bir avantaj karşılığında rezervasyonu teyit ettirmek iptali azaltabilir."
        ),
        priority=1,
    ),
    KnowledgeChunk(
        chunk_id="pol_003",
        category="retention",
        tags=["lead_time", "uzun", "erken_rezervasyon", "yüksek_risk"],
        title="Uzun süre önce yapılan rezervasyon",
        content=(
            "Çok erken yapılan rezervasyonlarda plan değişikliği riski artar. "
            "Check-in tarihine yaklaşırken hatırlatma ve esnek tarih değişikliği önerin."
        ),
        priority=2,
    ),
    KnowledgeChunk(
        chunk_id="pol_004",
        category="retention",
        tags=["geçmiş_iptal", "previous_cancellations", "yüksek_risk"],
        title="Geçmişte iptal etmiş müşteri",
        content=(
            "Daha önce iptal etmiş müşteri için kişisel temas etkilidir. "
            "24 saat içinde kısa bir arama yaparak rezervasyon niyetini teyit edin."
        ),
        priority=1,
    ),
    KnowledgeChunk(
        chunk_id="pol_005",
        category="segment",
        tags=["online", "ota", "online_ta", "yüksek_risk"],
        title="Online acente rezervasyonu",
        content=(
            "Online acente kanalında müşteri karşılaştırma yapabilir. "
            "Doğrudan iletişim kurup net değer önerisi sunmak iptal riskini azaltır."
        ),
        priority=2,
    ),
    KnowledgeChunk(
        chunk_id="pol_006",
        category="upsell",
        tags=["düşük_risk", "upsell", "ek_hizmet"],
        title="Düşük riskte ek gelir",
        content=(
            "Düşük riskli müşteriye oda yükseltme, transfer veya paket kahvaltı önerilebilir. "
            "Bu yaklaşım hem gelir hem bağlılık sağlar."
        ),
        priority=3,
    ),
]
