from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CustomerContext:
    raw_data: dict[str, Any]
    risk_score: float
    risk_label: str
    risk_percent: float
    risk_level_tr: str
    profile_summary_tr: str
    key_risk_factors: list[str]
    retrieved_chunks_text: str


def build_customer_context(
    *,
    customer_data: dict[str, Any],
    risk_score: float,
    risk_label: str,
    retrieved_chunks_text: str,
) -> CustomerContext:
    risk_percent = round(risk_score * 100.0, 1)
    if risk_percent >= 65:
        level = "YÜKSEK"
    elif risk_percent >= 35:
        level = "ORTA"
    else:
        level = "DÜŞÜK"

    factors = _extract_risk_factors(customer_data=customer_data)
    summary = _profile_summary(
        customer_data=customer_data,
        risk_percent=risk_percent,
        level=level,
    )
    return CustomerContext(
        raw_data=customer_data,
        risk_score=risk_score,
        risk_label=risk_label,
        risk_percent=risk_percent,
        risk_level_tr=level,
        profile_summary_tr=summary,
        key_risk_factors=factors,
        retrieved_chunks_text=retrieved_chunks_text,
    )


def _extract_risk_factors(*, customer_data: dict[str, Any]) -> list[str]:
    factors: list[str] = []
    lead_time = int(customer_data.get("lead_time", 0) or 0)
    if lead_time > 180:
        factors.append(f"Uzun ön rezervasyon ({lead_time} gün)")
    elif lead_time <= 7:
        factors.append(f"Kısa ön rezervasyon ({lead_time} gün)")

    dep = str(customer_data.get("deposit_type", ""))
    if dep == "No Deposit":
        factors.append("Depozito yok")
    elif dep == "Non Refund":
        factors.append("İade edilmez depozito var")

    prev_cancel = int(customer_data.get("previous_cancellations", 0) or 0)
    if prev_cancel > 0:
        factors.append(f"Geçmiş iptal sayısı: {prev_cancel}")

    if int(customer_data.get("is_repeated_guest", 0) or 0) == 1:
        factors.append("Sadık müşteri")
    else:
        factors.append("İlk ziyaret")

    segment = str(customer_data.get("market_segment", ""))
    if segment:
        factors.append(f"Kanal: {segment}")

    return factors or ["Belirgin risk faktörü bulunamadı"]


def _profile_summary(
    *, customer_data: dict[str, Any], risk_percent: float, level: str
) -> str:
    hotel = str(customer_data.get("hotel", "Otel"))
    lead_time = int(customer_data.get("lead_time", 0) or 0)
    adults = int(customer_data.get("adults", 1) or 1)
    children = int(customer_data.get("children", 0) or 0)
    week = int(customer_data.get("stays_in_week_nights", 0) or 0)
    weekend = int(customer_data.get("stays_in_weekend_nights", 0) or 0)
    nights = week + weekend

    group = f"{adults} yetişkin"
    if children > 0:
        group += f", {children} çocuk"

    return (
        f"{hotel} için {lead_time} gün önce yapılan, {group} ve {nights} gecelik "
        f"rezervasyon. Tahmini iptal olasılığı %{risk_percent} ({level})."
    )
