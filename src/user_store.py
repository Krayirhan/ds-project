"""user_store.py — PostgreSQL-backed user management.

Provides a SQLAlchemy-based UserStore that:
  - stores users in the `users` table (created by Alembic migration 3a7f9b2c4d8e)
  - hashes passwords with bcrypt
  - exposes CRUD helpers used by dashboard_auth.py
  - seeds the initial admin user from env at startup

Usage:
    from src.user_store import init_user_store, get_user_store, seed_admin

    # At startup (api.py lifespan):
    init_user_store(database_url)
    seed_admin()
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import bcrypt
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

_user_store: Optional["UserStore"] = None


class UserStore:
    """SQLAlchemy-backed user repository."""

    def __init__(self, database_url: str) -> None:
        self.engine = create_engine(database_url, pool_pre_ping=True, future=True)
        self.metadata = MetaData()
        self.users = Table(
            "users",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("username", String(64), nullable=False, unique=True),
            Column("password_hash", String(256), nullable=False),
            Column("role", String(32), nullable=False, server_default="viewer"),
            Column("is_active", Boolean, nullable=False, server_default="true"),
            Column("created_at", DateTime(timezone=True), nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False),
        )

    def create_schema(self) -> None:
        """Create the users table if it does not exist."""
        self.metadata.create_all(self.engine)

    # ── Query helpers ────────────────────────────────────────────────────────

    def get_user(self, username: str) -> Dict[str, Any] | None:
        """Return a user dict or None if not found."""
        with self.engine.connect() as conn:
            row = (
                conn.execute(
                    select(self.users).where(self.users.c.username == username)
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def list_users(self) -> List[Dict[str, Any]]:
        """Return all users (password hash excluded)."""
        with self.engine.connect() as conn:
            rows = conn.execute(select(self.users)).mappings().all()
        return [
            {
                "id": r["id"],
                "username": r["username"],
                "role": r["role"],
                "is_active": r["is_active"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
            }
            for r in rows
        ]

    # ── Password helpers ─────────────────────────────────────────────────────

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def verify_password(self, username: str, password: str) -> bool:
        """Return True if username exists, is active, and password matches."""
        user = self.get_user(username)
        if user is None or not user.get("is_active"):
            return False
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"),
                user["password_hash"].encode("utf-8"),
            )
        except Exception:
            return False

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def create_user(self, username: str, password: str) -> bool:
        """Create a new user.  Returns True on success, False if username exists."""
        now = datetime.now(timezone.utc)
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(self.users).values(
                        username=username,
                        password_hash=self.hash_password(password),
                        role="admin",
                        is_active=True,
                        created_at=now,
                        updated_at=now,
                    )
                )
            logger.info("User '%s' created.", username)
            return True
        except IntegrityError:
            logger.debug("create_user: username '%s' already exists.", username)
            return False
        except Exception as exc:
            logger.warning("create_user failed for '%s': %s", username, exc)
            return False

    def update_password(self, username: str, new_password: str) -> bool:
        """Update the password for an existing user."""
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            result = conn.execute(
                update(self.users)
                .where(self.users.c.username == username)
                .values(password_hash=self.hash_password(new_password), updated_at=now)
            )
        if result.rowcount == 0:
            logger.warning("update_password: user '%s' not found.", username)
            return False
        logger.info("Password updated for user '%s'.", username)
        return True

    def update_role(self, username: str, role: str) -> bool:
        """Update the role for an existing user."""
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            result = conn.execute(
                update(self.users)
                .where(self.users.c.username == username)
                .values(role=role, updated_at=now)
            )
        return result.rowcount > 0

    def set_active(self, username: str, is_active: bool) -> bool:
        """Soft-delete or re-enable a user."""
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            result = conn.execute(
                update(self.users)
                .where(self.users.c.username == username)
                .values(is_active=is_active, updated_at=now)
            )
        return result.rowcount > 0

    def delete_user(self, username: str) -> bool:
        """Hard-delete a user row."""
        from sqlalchemy import delete as sa_delete

        with self.engine.begin() as conn:
            result = conn.execute(
                sa_delete(self.users).where(self.users.c.username == username)
            )
        return result.rowcount > 0


# ── Module-level singleton ────────────────────────────────────────────────────


def get_user_store() -> UserStore | None:
    """Return the initialized UserStore, or None if not yet initialized."""
    return _user_store


def init_user_store(database_url: str) -> UserStore:
    """Initialize the global UserStore and create the schema."""
    global _user_store
    _user_store = UserStore(database_url=database_url)
    _user_store.create_schema()
    logger.info("UserStore initialized: %s", database_url.split("@")[-1])
    return _user_store


def seed_admin() -> None:
    """Ensure the admin user exists in the DB.

    Called once at API startup. Reads the password from env:
      DASHBOARD_ADMIN_PASSWORD_ADMIN  → password for 'admin'

    If not set, defaults to 'admin123' (change immediately!).
    """
    store = get_user_store()
    if store is None:
        logger.warning("UserStore not initialized; skipping admin seed.")
        return

    admin_pass = os.getenv("DASHBOARD_ADMIN_PASSWORD_ADMIN", "").strip()
    if not admin_pass:
        admin_pass = "admin123"
        logger.warning(
            "DASHBOARD_ADMIN_PASSWORD_ADMIN not set. "
            "Seeding 'admin' with default password 'admin123'. Change this immediately!"
        )

    if store.get_user("admin") is None:
        store.create_user("admin", admin_pass)
        logger.info("Admin user seeded into database.")
    else:
        logger.debug("Admin user already exists in database.")
