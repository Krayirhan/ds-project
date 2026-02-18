"""
validate.py

Veri kalitesi ve şema kontrolleri.

Neden gerekli?
- Production'da en çok patlayan şey veri kontratıdır:
  - hedef label formatı değişir
  - kolon isimleri değişir
  - dataset boş gelir
Validasyon bunları erken yakalar (fail fast).
"""

import pandas as pd
from .utils import get_logger

logger = get_logger("validate")


def basic_schema_checks(df: pd.DataFrame, target_col: str) -> None:
    if df.empty:
        raise ValueError("Dataset is empty.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")

    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Duplicate column names detected.")

    logger.info("Basic schema checks passed.")


def validate_target_labels(
    df: pd.DataFrame, target_col: str, allowed: set[str]
) -> None:
    """
    Target label check.

    Neden?
    - Senin dataset'te is_canceled: yes/no
    - Başka sürümde 0/1 olabilir
    - Beklenmeyen label gelirse pipeline sessizce yanlış çalışmasın.
    """
    y = df[target_col].astype(str).str.lower().str.strip()
    uniq = set(y.unique())
    if not uniq.issubset(allowed):
        raise ValueError(
            f"Unexpected labels in {target_col}: {sorted(uniq)} | allowed={sorted(allowed)}"
        )
    logger.info(f"Target labels OK -> {sorted(uniq)}")


def null_ratio_report(df: pd.DataFrame, top_k: int = 10) -> pd.Series:
    """
    Missing value oranlarını raporlar.

    Neden?
    - Missing stratejisi seçmeden önce teşhis gerekir.
    """
    ratios = df.isna().mean().sort_values(ascending=False)
    logger.info(f"Top null ratios:\n{ratios.head(top_k)}")
    return ratios
