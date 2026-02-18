"""
io.py

Veri okuma/yazma standardı.

Neden ayrı?
- Veri formatı tek yerden kontrol edilir.
- Pipeline boyunca tutarlılık sağlar.

Neden Parquet?
- CSV'den hızlı
- Tipleri daha iyi korur
"""

from pathlib import Path
import pandas as pd
from .utils import get_logger

logger = get_logger("io")


def read_csv(path: Path) -> pd.DataFrame:
    logger.info(f"Reading CSV: {path}")
    return pd.read_csv(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing Parquet: {path} | rows={len(df)} cols={df.shape[1]}")
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    logger.info(f"Reading Parquet: {path}")
    return pd.read_parquet(path)
