"""Integration smoke test — gerçek CLI pipeline.

Bu test data/raw/hotel_bookings.csv'nin varlığını gerektirir.
Dosya yoksa (DVC pull yapılmamışsa) otomatik olarak atlanır.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DATA_CSV = _ROOT / "data" / "raw" / "hotel_bookings.csv"


@pytest.mark.skipif(
    not _DATA_CSV.exists(),
    reason="hotel_bookings.csv bulunamadı — önce 'dvc pull data/raw' çalıştırın",
)
def test_pipeline_smoke_cli():
    py = sys.executable

    for cmd in (
        [py, "main.py", "preprocess"],
        [py, "main.py", "train"],
        [py, "main.py", "evaluate"],
        [py, "main.py", "predict"],
    ):
        proc = subprocess.run(
            cmd,
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=300,  # 5 dk timeout — sonsuz takılma önlemi
        )
        assert proc.returncode == 0, (
            f"failed: {' '.join(cmd)}\n"
            f"STDOUT:{proc.stdout[-3000:]}\n"  # son 3000 karakter — log taşmasını önle
            f"STDERR:{proc.stderr[-3000:]}"
        )
