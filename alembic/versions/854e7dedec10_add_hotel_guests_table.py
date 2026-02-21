"""add_hotel_guests_table

Revision ID: 854e7dedec10
Revises: 3a7f9b2c4d8e
Create Date: 2026-02-21 17:37:06.502255

Adds the `hotel_guests` table for full customer profile storage.
Personal info + booking fields + model prediction result.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '854e7dedec10'
down_revision: Union[str, Sequence[str], None] = '3a7f9b2c4d8e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "hotel_guests",
        # ── Kimlik bilgileri ──────────────────────────────────────────────────
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("first_name",   sa.String(100), nullable=False),
        sa.Column("last_name",    sa.String(100), nullable=False),
        sa.Column("email",        sa.String(200), nullable=True),
        sa.Column("phone",        sa.String(30),  nullable=True),
        sa.Column("nationality",  sa.String(3),   nullable=True),   # ISO-3166 alpha-3
        sa.Column("identity_no",  sa.String(50),  nullable=True),   # TC / Pasaport
        sa.Column("birth_date",   sa.Date(),      nullable=True),
        sa.Column("gender",       sa.String(10),  nullable=True),   # M / F / other
        sa.Column("vip_status",   sa.Boolean(),   nullable=False, server_default="false"),
        sa.Column("notes",        sa.Text(),      nullable=True),
        # ── Model özellikleri (tahmin için kullanılır) ─────────────────────────
        sa.Column("hotel",                    sa.String(50),  nullable=False, server_default="City Hotel"),
        sa.Column("lead_time",                sa.Integer(),   nullable=False, server_default="0"),
        sa.Column("deposit_type",             sa.String(30),  nullable=False, server_default="No Deposit"),
        sa.Column("market_segment",           sa.String(30),  nullable=False, server_default="Online TA"),
        sa.Column("adults",                   sa.Integer(),   nullable=False, server_default="2"),
        sa.Column("children",                 sa.Integer(),   nullable=False, server_default="0"),
        sa.Column("babies",                   sa.Integer(),   nullable=False, server_default="0"),
        sa.Column("stays_in_week_nights",     sa.Integer(),   nullable=False, server_default="0"),
        sa.Column("stays_in_weekend_nights",  sa.Integer(),   nullable=False, server_default="1"),
        sa.Column("is_repeated_guest",        sa.Integer(),   nullable=False, server_default="0"),
        sa.Column("previous_cancellations",   sa.Integer(),   nullable=False, server_default="0"),
        sa.Column("adr",                      sa.Float(),     nullable=True),
        # ── Model tahmini (kayıt sırasında hesaplanır) ─────────────────────────
        sa.Column("risk_score",  sa.Float(),   nullable=True),
        sa.Column("risk_label",  sa.String(10), nullable=True),
        # ── Meta ──────────────────────────────────────────────────────────────
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_hotel_guests_last_name", "hotel_guests", ["last_name"])
    op.create_index("ix_hotel_guests_email",     "hotel_guests", ["email"])


def downgrade() -> None:
    op.drop_index("ix_hotel_guests_email",     table_name="hotel_guests")
    op.drop_index("ix_hotel_guests_last_name", table_name="hotel_guests")
    op.drop_table("hotel_guests")
