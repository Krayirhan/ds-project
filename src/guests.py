"""guests.py — Hotel guest management API router.

Endpoints:
  POST   /guests           — create guest + auto risk calculation
  GET    /guests           — list guests (search + pagination)
  GET    /guests/{id}      — single guest
  PATCH  /guests/{id}      — partial update + re-calculate risk if booking fields changed

All endpoints are protected by the x-api-key middleware in api.py.
Personal info (name, email, …) is stored to DB only.
Booking fields (hotel, lead_time, …) are also passed to the ML model.
"""
from __future__ import annotations

import datetime
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router_guests = APIRouter(prefix="/guests", tags=["guests"])

# ── Booking/model feature field names (used for selective risk re-calc) ────────
_MODEL_FIELDS = frozenset({
    "hotel", "lead_time", "deposit_type", "market_segment",
    "adults", "children", "babies",
    "stays_in_week_nights", "stays_in_weekend_nights",
    "is_repeated_guest", "previous_cancellations", "adr",
})


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class GuestCreate(BaseModel):
    # ── Personal info (DB only) ───────────────────────────────────────────────
    first_name:  str  = Field(min_length=1, max_length=100)
    last_name:   str  = Field(min_length=1, max_length=100)
    email:       str | None = None
    phone:       str | None = None
    nationality: str | None = None  # ISO-3166 alpha-3
    identity_no: str | None = None  # TC Kimlik / Pasaport
    birth_date:  str | None = None  # YYYY-MM-DD string
    gender:      str | None = None  # M / F / other
    vip_status:  bool       = False
    notes:       str | None = None

    # ── Booking / model features ──────────────────────────────────────────────
    hotel:                   str        = "City Hotel"
    lead_time:               int        = 0
    deposit_type:            str        = "No Deposit"
    market_segment:          str        = "Online TA"
    adults:                  int        = 2
    children:                int        = 0
    babies:                  int        = 0
    stays_in_week_nights:    int        = 2
    stays_in_weekend_nights: int        = 1
    is_repeated_guest:       int        = 0
    previous_cancellations:  int        = 0
    adr:                     float | None = None

    model_config = {"extra": "ignore"}


class GuestUpdate(BaseModel):
    # ── Personal info (all optional) ─────────────────────────────────────────
    first_name:  str  | None = None
    last_name:   str  | None = None
    email:       str  | None = None
    phone:       str  | None = None
    nationality: str  | None = None
    identity_no: str  | None = None
    birth_date:  str  | None = None
    gender:      str  | None = None
    vip_status:  bool | None = None
    notes:       str  | None = None

    # ── Booking / model features (all optional) ───────────────────────────────
    hotel:                   str        | None = None
    lead_time:               int        | None = None
    deposit_type:            str        | None = None
    market_segment:          str        | None = None
    adults:                  int        | None = None
    children:                int        | None = None
    babies:                  int        | None = None
    stays_in_week_nights:    int        | None = None
    stays_in_weekend_nights: int        | None = None
    is_repeated_guest:       int        | None = None
    previous_cancellations:  int        | None = None
    adr:                     float      | None = None

    model_config = {"extra": "ignore"}


class GuestResponse(BaseModel):
    id:           int
    first_name:   str
    last_name:    str
    email:        str | None
    phone:        str | None
    nationality:  str | None
    identity_no:  str | None
    birth_date:   str | None
    gender:       str | None
    vip_status:   bool
    notes:        str | None
    hotel:                   str
    lead_time:               int
    deposit_type:            str
    market_segment:          str
    adults:                  int
    children:                int
    babies:                  int
    stays_in_week_nights:    int
    stays_in_weekend_nights: int
    is_repeated_guest:       int
    previous_cancellations:  int
    adr:          float | None
    risk_score:   float | None
    risk_label:   str   | None
    created_at:   str
    updated_at:   str


class GuestListResponse(BaseModel):
    total: int
    items: list[GuestResponse]


# ── Helper: calculate risk from booking fields ────────────────────────────────

def _calculate_risk(request: Request, fields: dict[str, Any]) -> tuple[float, str]:
    """Run the ML model on booking fields. Returns (risk_score, risk_label).
    Gracefully returns (0.5, 'medium') if model is unavailable."""
    import pandas as pd
    from .predict import validate_and_prepare_features

    serving = getattr(request.app.state, "serving", None)
    if serving is None:
        logger.warning("Model not loaded — risk defaults to medium")
        return 0.5, "medium"

    arrival = datetime.date.today() + datetime.timedelta(
        days=int(fields.get("lead_time", 0))
    )
    record: dict[str, Any] = {
        "hotel":                      fields.get("hotel", "City Hotel"),
        "lead_time":                  fields.get("lead_time", 0),
        "deposit_type":               fields.get("deposit_type", "No Deposit"),
        "market_segment":             fields.get("market_segment", "Online TA"),
        "adults":                     fields.get("adults", 2),
        "children":                   fields.get("children", 0),
        "babies":                     fields.get("babies", 0),
        "stays_in_week_nights":       fields.get("stays_in_week_nights", 0),
        "stays_in_weekend_nights":    fields.get("stays_in_weekend_nights", 1),
        "previous_cancellations":     fields.get("previous_cancellations", 0),
        "is_repeated_guest":          fields.get("is_repeated_guest", 0),
        "arrival_date_year":          arrival.year,
        "arrival_date_month":         arrival.strftime("%B"),
        "arrival_date_week_number":   arrival.isocalendar()[1],
        "arrival_date_day_of_month":  arrival.day,
        "previous_bookings_not_canceled": 0,
        "booking_changes":            0,
        "agent":                      0,
        "company":                    0,
        "days_in_waiting_list":       0,
        "adr":                        fields.get("adr") or 100.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests":  0,
        "meal":                       "BB",
        "country":                    "PRT",
        "distribution_channel":       "TA/TO",
        "reserved_room_type":         "A",
        "assigned_room_type":         "A",
        "customer_type":              "Transient",
    }
    try:
        df = pd.DataFrame([record])
        X, _ = validate_and_prepare_features(
            df, serving.feature_spec, fail_on_missing=False
        )
        proba = float(serving.model.predict_proba(X)[0, 1])
    except Exception as exc:
        logger.warning("Risk calculation failed: %s", exc)
        return 0.5, "medium"

    label = "high" if proba >= 0.65 else ("medium" if proba >= 0.35 else "low")
    return round(proba, 4), label


def _row_to_response(row: dict[str, Any]) -> GuestResponse:
    """Convert a DB row dict to a GuestResponse."""
    def _dt(val: Any) -> str:
        return val.isoformat() if hasattr(val, "isoformat") else str(val)

    return GuestResponse(
        id=row["id"],
        first_name=row["first_name"],
        last_name=row["last_name"],
        email=row.get("email"),
        phone=row.get("phone"),
        nationality=row.get("nationality"),
        identity_no=row.get("identity_no"),
        birth_date=str(row["birth_date"]) if row.get("birth_date") else None,
        gender=row.get("gender"),
        vip_status=bool(row.get("vip_status", False)),
        notes=row.get("notes"),
        hotel=row["hotel"],
        lead_time=row["lead_time"],
        deposit_type=row["deposit_type"],
        market_segment=row["market_segment"],
        adults=row["adults"],
        children=row["children"],
        babies=row["babies"],
        stays_in_week_nights=row["stays_in_week_nights"],
        stays_in_weekend_nights=row["stays_in_weekend_nights"],
        is_repeated_guest=row["is_repeated_guest"],
        previous_cancellations=row["previous_cancellations"],
        adr=row.get("adr"),
        risk_score=row.get("risk_score"),
        risk_label=row.get("risk_label"),
        created_at=_dt(row["created_at"]),
        updated_at=_dt(row["updated_at"]),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router_guests.post("", response_model=GuestResponse, status_code=201)
async def create_guest(body: GuestCreate, request: Request) -> GuestResponse:
    """Yeni misafir kaydı oluşturur. Booking alanlarından otomatik risk hesaplanır."""
    from .guest_store import get_guest_store

    data = body.model_dump()

    # birth_date: string → Python date object
    if data.get("birth_date"):
        try:
            data["birth_date"] = datetime.date.fromisoformat(data["birth_date"])
        except ValueError:
            data["birth_date"] = None

    # Compute risk from booking fields
    risk_score, risk_label = _calculate_risk(request, data)
    data["risk_score"] = risk_score
    data["risk_label"] = risk_label

    store = get_guest_store()
    row = store.create_guest(data)
    return _row_to_response(row)


@router_guests.get("", response_model=GuestListResponse)
async def list_guests(
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> GuestListResponse:
    """Misafir listesi. `search` isim/email'e göre filtreler."""
    from .guest_store import get_guest_store

    store = get_guest_store()
    total = store.count_guests(search=search)
    items = store.list_guests(search=search, limit=min(limit, 200), offset=offset)
    return GuestListResponse(total=total, items=[_row_to_response(r) for r in items])


@router_guests.get("/{guest_id}", response_model=GuestResponse)
async def get_guest(guest_id: int) -> GuestResponse:
    """Tek misafir getir."""
    from .guest_store import get_guest_store

    store = get_guest_store()
    row = store.get_guest(guest_id)
    if not row:
        raise HTTPException(status_code=404, detail="Misafir bulunamadı")
    return _row_to_response(row)


@router_guests.delete("/{guest_id}", status_code=204)
async def delete_guest(guest_id: int) -> None:
    """Misafiri kalıcı olarak siler."""
    from .guest_store import get_guest_store

    store = get_guest_store()
    deleted = store.delete_guest(guest_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Misafir bulunamadı")


@router_guests.patch("/{guest_id}", response_model=GuestResponse)
async def update_guest(
    guest_id: int, body: GuestUpdate, request: Request
) -> GuestResponse:
    """Kısmi güncelleme. Booking alanı değişmişse risk yeniden hesaplanır."""
    from .guest_store import get_guest_store

    store = get_guest_store()
    existing = store.get_guest(guest_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Misafir bulunamadı")

    # Only non-None values are applied
    updates: dict[str, Any] = {
        k: v for k, v in body.model_dump().items() if v is not None
    }

    # birth_date string → date object
    if "birth_date" in updates:
        try:
            updates["birth_date"] = datetime.date.fromisoformat(updates["birth_date"])
        except ValueError:
            updates.pop("birth_date", None)

    # Re-calculate risk if any booking/model field changed
    if updates.keys() & _MODEL_FIELDS:
        merged = {
            **{k: existing.get(k) for k in _MODEL_FIELDS},
            **{k: v for k, v in updates.items() if k in _MODEL_FIELDS},
        }
        risk_score, risk_label = _calculate_risk(request, merged)
        updates["risk_score"] = risk_score
        updates["risk_label"] = risk_label

    row = store.update_guest(guest_id, updates)
    if not row:
        raise HTTPException(status_code=404, detail="Güncelleme başarısız")
    return _row_to_response(row)
