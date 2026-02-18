"""Root conftest.py — ensures ``src`` is importable regardless of install mode.

When pytest discovers this file in the project root it automatically inserts
the directory containing it into ``sys.path``.  This makes ``from src.xxx``
imports work in CI, tox, and bare ``pytest`` invocations without requiring an
editable install.
"""

import sys
from pathlib import Path

# Guarantee project root is on sys.path
_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
