"""Root conftest.py — ensures ``src`` is importable regardless of install mode.

When pytest discovers this file in the project root it automatically inserts
the directory containing it into ``sys.path``.  This makes ``from src.xxx``
imports work in CI, tox, and bare ``pytest`` invocations without requiring an
editable install.
"""

import sys
from pathlib import Path

import pytest

# Guarantee project root is on sys.path
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


@pytest.fixture(autouse=True)
def _isolate_test_db_and_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use an isolated SQLite DB per test and avoid flaky parallel CPU detection."""
    db_path = tmp_path / "dashboard_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
