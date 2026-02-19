from __future__ import annotations

from .policies import KNOWLEDGE_BASE, KnowledgeChunk


class KnowledgeStore:
    def __init__(self) -> None:
        self._chunks = KNOWLEDGE_BASE

    def retrieve(self, *, tags: list[str], top_k: int = 3) -> list[KnowledgeChunk]:
        query_tags = {t.lower() for t in tags}
        scored: list[tuple[float, KnowledgeChunk]] = []
        for chunk in self._chunks:
            chunk_tags = {t.lower() for t in chunk.tags}
            overlap = len(query_tags & chunk_tags)
            if overlap == 0:
                continue
            score = overlap * (6 - max(1, min(5, chunk.priority)))
            scored.append((float(score), chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def retrieve_by_customer(
        self, *, customer_data: dict, risk_score: float, top_k: int = 3
    ) -> list[KnowledgeChunk]:
        tags = self._extract_tags(customer_data=customer_data, risk_score=risk_score)
        return self.retrieve(tags=tags, top_k=top_k)

    def _extract_tags(self, *, customer_data: dict, risk_score: float) -> list[str]:
        tags: list[str] = []
        if risk_score >= 0.6:
            tags.append("yüksek_risk")
        elif risk_score < 0.35:
            tags.extend(["düşük_risk", "upsell"]) 

        dep = str(customer_data.get("deposit_type", ""))
        if dep == "No Deposit":
            tags.extend(["depozito", "no_deposit"])
        elif dep == "Non Refund":
            tags.extend(["depozito", "non_refund"])

        lead_time = int(customer_data.get("lead_time", 0) or 0)
        if lead_time > 180:
            tags.extend(["lead_time", "uzun", "erken_rezervasyon"])

        prev_cancel = int(customer_data.get("previous_cancellations", 0) or 0)
        if prev_cancel > 0:
            tags.extend(["geçmiş_iptal", "previous_cancellations"])

        segment = str(customer_data.get("market_segment", ""))
        if "Online" in segment:
            tags.extend(["online", "ota", "online_ta"])

        return tags


_store: KnowledgeStore | None = None


def get_knowledge_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store
