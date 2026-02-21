"""add_users_table

Revision ID: 3a7f9b2c4d8e
Revises: 221e77090362
Create Date: 2026-02-21 00:00:00.000000

Adds the `users` table for DB-backed authentication.
Replaces the previous env-var credential system.

Columns:
  id            — surrogate PK
  username      — unique login name
  password_hash — bcrypt hash ($2b$…)
  role          — 'admin' | 'viewer'
  is_active     — soft-delete flag
  created_at    — UTC timestamp
  updated_at    — UTC timestamp
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "3a7f9b2c4d8e"
down_revision: Union[str, Sequence[str], None] = "221e77090362"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(64), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(256), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
