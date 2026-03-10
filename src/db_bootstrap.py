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
REQUIRED_TABLES: tuple[str, ...] = (
    "experiment_runs",
    "model_metrics",
    "users",
    "hotel_guests",
    "knowledge_chunks",
)


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
    required_tables: Iterable[str] = REQUIRED_TABLES,
) -> None:
    """Raise if one or more required tables are missing."""
    engine = create_engine(database_url, pool_pre_ping=True, future=True)
    try:
        with engine.connect() as conn:
            insp = inspect(conn)
            existing = set(insp.get_table_names())
    finally:
        engine.dispose()

    missing = [table for table in required_tables if table not in existing]
    if missing:
        raise RuntimeError(f"Missing required tables: {', '.join(sorted(missing))}")

    logger.info("Required database tables verified.")
