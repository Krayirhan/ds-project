"""guest_store.py — PostgreSQL-backed hotel guest management.

Provides a SQLAlchemy-based GuestStore that:
  - stores guests in the `hotel_guests` table (Alembic migration 854e7dedec10)
  - separates personal info (DB only) from model features (used for prediction)
  - exposes CRUD helpers used by guests.py router

Usage:
    from src.guest_store import init_guest_store, get_guest_store

    # At startup (api.py lifespan):
    init_guest_store(engine)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    or_,
    select,
    update,
    desc,
)
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_guest_store: Optional["GuestStore"] = None


class GuestStore:
    """SQLAlchemy-backed hotel guest repository."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.metadata = MetaData()
        self.guests = Table(
            "hotel_guests",
            self.metadata,
            # ── Identity ──────────────────────────────────────────────────────
            Column("id",          Integer,       primary_key=True, autoincrement=True),
            Column("first_name",  String(100),   nullable=False),
            Column("last_name",   String(100),   nullable=False),
            Column("email",       String(200),   nullable=True),
            Column("phone",       String(30),    nullable=True),
            Column("nationality", String(3),     nullable=True),  # ISO-3166 alpha-3
            Column("identity_no", String(50),    nullable=True),  # TC / Pasaport
            Column("birth_date",  Date(),        nullable=True),
            Column("gender",      String(10),    nullable=True),  # M / F / other
            Column("vip_status",  Boolean(),     nullable=False),
            Column("notes",       Text(),        nullable=True),
            # ── Model features (used for risk prediction) ─────────────────────
            Column("hotel",                   String(50),  nullable=False),
            Column("lead_time",               Integer(),   nullable=False),
            Column("deposit_type",            String(30),  nullable=False),
            Column("market_segment",          String(30),  nullable=False),
            Column("adults",                  Integer(),   nullable=False),
            Column("children",                Integer(),   nullable=False),
            Column("babies",                  Integer(),   nullable=False),
            Column("stays_in_week_nights",    Integer(),   nullable=False),
            Column("stays_in_weekend_nights", Integer(),   nullable=False),
            Column("is_repeated_guest",       Integer(),   nullable=False),
            Column("previous_cancellations",  Integer(),   nullable=False),
            Column("adr",                     Float(),     nullable=True),
            # ── Prediction result ─────────────────────────────────────────────
            Column("risk_score",  Float(),      nullable=True),
            Column("risk_label",  String(10),   nullable=True),
            # ── Meta ──────────────────────────────────────────────────────────
            Column("created_at",  DateTime(timezone=True), nullable=False),
            Column("updated_at",  DateTime(timezone=True), nullable=False),
        )

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def create_guest(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a new guest row and return the complete row dict."""
        now = datetime.now(timezone.utc)
        row_data = {**data, "created_at": now, "updated_at": now}
        with self.engine.connect() as conn:
            result = conn.execute(
                insert(self.guests).values(**row_data).returning(self.guests)
            )
            conn.commit()
            new_row = result.mappings().first()
        return dict(new_row)

    def list_guests(
        self,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return guests ordered by newest first, with optional name/email search."""
        stmt = select(self.guests).order_by(desc(self.guests.c.created_at))
        if search:
            pattern = f"%{search}%"
            stmt = stmt.where(
                or_(
                    self.guests.c.first_name.ilike(pattern),
                    self.guests.c.last_name.ilike(pattern),
                    self.guests.c.email.ilike(pattern),
                )
            )
        stmt = stmt.limit(limit).offset(offset)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

    def count_guests(self, search: str | None = None) -> int:
        """Return total guest count (for pagination)."""
        stmt = select(func.count()).select_from(self.guests)
        if search:
            pattern = f"%{search}%"
            stmt = stmt.where(
                or_(
                    self.guests.c.first_name.ilike(pattern),
                    self.guests.c.last_name.ilike(pattern),
                    self.guests.c.email.ilike(pattern),
                )
            )
        with self.engine.connect() as conn:
            return conn.execute(stmt).scalar() or 0

    def get_guest(self, guest_id: int) -> Dict[str, Any] | None:
        """Return a single guest by id, or None."""
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(self.guests).where(self.guests.c.id == guest_id)
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def update_guest(self, guest_id: int, data: Dict[str, Any]) -> Dict[str, Any] | None:
        """Partial update. Returns updated row dict, or None if not found."""
        now = datetime.now(timezone.utc)
        update_data = {**data, "updated_at": now}
        with self.engine.connect() as conn:
            result = conn.execute(
                update(self.guests)
                .where(self.guests.c.id == guest_id)
                .values(**update_data)
                .returning(self.guests)
            )
            conn.commit()
            row = result.mappings().first()
        return dict(row) if row else None

    def delete_guest(self, guest_id: int) -> bool:
        """Delete a guest by id. Returns True if deleted, False if not found."""
        from sqlalchemy import delete as sa_delete
        with self.engine.connect() as conn:
            result = conn.execute(
                sa_delete(self.guests).where(self.guests.c.id == guest_id)
            )
            conn.commit()
        return result.rowcount > 0


# ── Module-level singleton ────────────────────────────────────────────────────

def init_guest_store(engine: Engine) -> None:
    """Initialize the global GuestStore with an existing SQLAlchemy engine."""
    global _guest_store
    _guest_store = GuestStore(engine)
    logger.info("GuestStore initialized")


def get_guest_store() -> GuestStore:
    if _guest_store is None:
        raise RuntimeError(
            "GuestStore is not initialized. Call init_guest_store() first."
        )
    return _guest_store
