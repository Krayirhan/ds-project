from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from .utils import get_logger

logger = get_logger("db_bootstrap")

# Core tables that must exist before API requests are accepted.
CORE_REQUIRED_TABLES: tuple[str, ...] = (
    "experiment_runs",
    "model_metrics",
    "users",
)

OPTIONAL_TABLE_FLAGS: dict[str, str] = {
    "hotel_guests": "DB_REQUIRE_GUESTS_TABLE",
    "knowledge_chunks": "DB_REQUIRE_KNOWLEDGE_TABLE",
}

# Backward-compatible strict default set.
REQUIRED_TABLES: tuple[str, ...] = CORE_REQUIRED_TABLES + tuple(
    OPTIONAL_TABLE_FLAGS.keys()
)


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def resolve_required_tables_from_env() -> tuple[str, ...]:
    """Resolve required table set from core tables + optional feature flags."""
    required = list(CORE_REQUIRED_TABLES)
    for table_name, flag_name in OPTIONAL_TABLE_FLAGS.items():
        if _env_flag_enabled(flag_name, default=True):
            required.append(table_name)
    return tuple(required)


@contextmanager
def _temp_database_url(database_url: str):
    old_value = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old_value


def run_migrations(database_url: str) -> None:
    """Apply Alembic migrations up to head for the given DB URL."""
    project_root = Path(__file__).resolve().parent.parent
    alembic_ini = project_root / "alembic.ini"
    if not alembic_ini.exists():
        raise RuntimeError(f"Alembic config missing: {alembic_ini}")

    cfg = Config(str(alembic_ini))
    cfg.set_main_option("sqlalchemy.url", database_url)

    with _temp_database_url(database_url):
        command.upgrade(cfg, "head")

    logger.info("Database migrations applied (head).")


def ensure_required_tables(
    database_url: str,
    *,
    required_tables: Iterable[str] | None = None,
) -> None:
    """Raise if one or more required tables are missing."""
    effective_required = (
        tuple(required_tables)
        if required_tables is not None
        else resolve_required_tables_from_env()
    )

    engine = create_engine(database_url, pool_pre_ping=True, future=True)
    try:
        with engine.connect() as conn:
            insp = inspect(conn)
            existing = set(insp.get_table_names())
    finally:
        engine.dispose()

    missing = [table for table in effective_required if table not in existing]
    if missing:
        raise RuntimeError(f"Missing required tables: {', '.join(sorted(missing))}")

    logger.info("Required database tables verified: %s", ", ".join(effective_required))
