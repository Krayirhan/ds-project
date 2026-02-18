"""
preprocess.py

Temel temizlik + dataset kontratı standardizasyonu.

Ne yapıyoruz?
1) Leakage/blocked kolonları drop ediyoruz (AUC=1.0 sahte başarıyı engeller)
2) Target label'ı normalize ediyoruz (yes/no -> 1/0)
3) Basit missing fill (baseline)

Neden target mapping burada?
- Train/Evaluate boyunca y tipi tutarlı olsun.
- "y_true string, y_pred int" hatası burada kökten biter.
"""

from typing import Optional, Dict, List
import pandas as pd
from .utils import get_logger

logger = get_logger("preprocess")


def preprocess_basic(
    df: pd.DataFrame,
    target_col: str,
    label_map: Dict[str, int],
    drop_cols: Optional[List[str]] = None,
    extra_blocked_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = df.copy()

    # Kolon isimlerini normalize etmek iyi pratik:
    # - Fazladan boşluklar, tutarsız isimler sorun çıkarabilir.
    df.columns = [c.strip() for c in df.columns]

    # 1) Leakage / blocked kolonları düşür
    blocked: List[str] = []
    if drop_cols:
        blocked.extend(drop_cols)
    if extra_blocked_cols:
        blocked.extend(extra_blocked_cols)

    if blocked:
        existing = [c for c in blocked if c in df.columns]
        if existing:
            logger.info(f"Dropping blocked columns: {existing}")
            df = df.drop(columns=existing)

    # 2) Target label normalize (yes/no -> 1/0)
    y = df[target_col].astype(str).str.lower().str.strip()
    unknown = set(y.unique()) - set(label_map.keys())
    if unknown:
        raise ValueError(
            f"Unknown target labels: {sorted(unknown)} | label_map keys={sorted(label_map.keys())}"
        )
    df[target_col] = y.map(label_map).astype(int)

    # 3) Tamamen boş kolonları düşür (baseline)
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        logger.info(f"Dropping empty columns: {empty_cols}")
        df = df.drop(columns=empty_cols)

    # 4) Basit missing fill (baseline)
    # Not: Bu başlangıç içindir. Sonra domain-aware/advanced strateji ekleyeceğiz.
    feature_cols = [c for c in df.columns if c != target_col]
    for c in feature_cols:
        if df[c].dtype.kind in "biufc":
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].median())
        else:
            if df[c].isna().any():
                mode = df[c].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else "UNKNOWN"
                df[c] = df[c].fillna(fill)

    return df
