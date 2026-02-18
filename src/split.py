"""
split.py

Train/test split burada.

Neden stratify?
- Target oranı train/test'te benzer olsun.
- Aksi halde test set dengesiz olur ve metrikler yanıltır.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import get_logger

logger = get_logger("split")

def stratified_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[target_col]
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    logger.info(f"Split done | train={len(train_df)} test={len(test_df)}")
    return train_df, test_df
