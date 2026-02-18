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


# Bilinen hotel bookings string kolonları — pandas type çıkarımını hızlandırır
# Not: int kolon tipleri (lead_time, is_repeated_guest vb.) burada belirtilmez;
#      pandas CSV'den doğal çıkarım ile int64 okur — Pandera kontratıyla uyumlu kalır.
_HOTEL_BOOKINGS_DTYPES: dict = {
    "hotel": "object",
    "is_canceled": "object",
    "arrival_date_month": "object",
    "meal": "object",
    "country": "object",
    "market_segment": "object",
    "distribution_channel": "object",
    "reserved_room_type": "object",
    "assigned_room_type": "object",
    "deposit_type": "object",
    "agent": "object",
    "company": "object",
    "customer_type": "object",
    "reservation_status": "object",
    "reservation_status_date": "object",
    # Gerçekten float olabilecek kolonlar (NaN içerebilir)
    "children": "float64",
    "adr": "float64",
}


def read_csv(path: Path, dtype: dict | None = None) -> pd.DataFrame:
    """
    CSV okur. hotel_bookings.csv için otomatik dtype hint uygulanır.

    dtype parametresi ile özel şema geçilebilir;
    None bırakılırsa dosya adına göre hotel bookings şeması uygulanır.
    """
    logger.info(f"Reading CSV: {path}")
    # Dosya adı hotel_bookings ise bilinen şemayı kullan
    effective_dtype: dict | None = dtype
    if effective_dtype is None and "hotel_bookings" in Path(path).stem:
        effective_dtype = _HOTEL_BOOKINGS_DTYPES
    return pd.read_csv(path, dtype=effective_dtype, low_memory=False)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing Parquet: {path} | rows={len(df)} cols={df.shape[1]}")
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    logger.info(f"Reading Parquet: {path}")
    return pd.read_parquet(path)
