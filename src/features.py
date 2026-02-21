"""
features.py

Feature dönüşümleri burada.

Neden Pipeline/ColumnTransformer?
- Dönüşümler train ve predict'te aynı olmalı.
- Notebook içinde manual dönüşüm yapmak training-serving skew doğurur.

Burada:
- FeatureEngineer: domain-aware ön dönüşüm katmanı
  * arrival_date_month → sin/cos cyclic encoding (ay sürekliliğini korur)
  * country → frequency encoding (100+ kardinaliteyi çözer)
- numeric: median impute + StandardScaler
- categorical: most_frequent impute + OneHotEncoder
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .utils import get_logger

logger = get_logger("features")

# ─── Ay adı → numara eşlemesi ───────────────────────────────────────────
_MONTH_MAP: Dict[str, float] = {
    "january": 1.0,
    "february": 2.0,
    "march": 3.0,
    "april": 4.0,
    "may": 5.0,
    "june": 6.0,
    "july": 7.0,
    "august": 8.0,
    "september": 9.0,
    "october": 10.0,
    "november": 11.0,
    "december": 12.0,
    # Kısaltmalar
    "jan": 1.0,
    "feb": 2.0,
    "mar": 3.0,
    "apr": 4.0,
    "jun": 6.0,
    "jul": 7.0,
    "aug": 8.0,
    "sep": 9.0,
    "oct": 10.0,
    "nov": 11.0,
    "dec": 12.0,
}

# Yüksek kardinaliteli kolonlar (frequency encoding uygulanır)
_HIGH_CARDINALITY_COLS: Tuple[str, ...] = ("country",)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-aware feature engineering adımı.

    İki dönüşüm uygular:
    1) arrival_date_month → arrival_date_month_sin + arrival_date_month_cos
       - Ocak ve Aralık ay olarak birbirinin komşusudur ama OHE bunu göremez.
       - Cyclic encoding bu sürekliliği korur.

    2) country → frekans (0–1 arası float)
       - 100+ kategori OHE ile aşırı seyrek matris üretir.
       - Frekans encoding boyutu tek bir numeric kolona düşürür.
       - Eğitimde görülmeyen ülkeler 0.0 alır.

    sklearn Pipeline uyumludur: fit() eğitimde, transform() hem eğitim
    hem inference'ta çalışır.
    """

    def __init__(
        self,
        month_col: str = "arrival_date_month",
        high_card_cols: Tuple[str, ...] = _HIGH_CARDINALITY_COLS,
    ):
        self.month_col = month_col
        self.high_card_cols = high_card_cols
        self._freq_maps: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in self.high_card_cols:
            if col in df.columns:
                freq = df[col].astype(str).value_counts(normalize=True).to_dict()
                self._freq_maps[col] = freq
                logger.info(f"FreqEncoder fitted: col={col} unique_values={len(freq)}")
        # Compute and cache the output feature names for sklearn set_output API
        cols = list(df.columns)
        if self.month_col in cols:
            cols.remove(self.month_col)
            cols = cols + [f"{self.month_col}_sin", f"{self.month_col}_cos"]
        self._feature_names_out: List[str] = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()

        # 1) Cyclic ay encoding
        if self.month_col in df.columns:
            months = (
                df[self.month_col].astype(str).str.lower().str.strip().map(_MONTH_MAP)
            )
            # Sayısal ay değerleri zaten varsa (1-12) doğrudan kullan
            numeric_fallback = pd.to_numeric(df[self.month_col], errors="coerce")
            months = (
                months.where(months.notna(), numeric_fallback).fillna(0.0).astype(float)
            )

            df[f"{self.month_col}_sin"] = np.sin(2 * np.pi * months / 12.0)
            df[f"{self.month_col}_cos"] = np.cos(2 * np.pi * months / 12.0)
            df = df.drop(columns=[self.month_col])
            logger.debug(
                f"CyclicEncoder: '{self.month_col}' → "
                f"'{self.month_col}_sin', '{self.month_col}_cos'"
            )

        # 2) Frequency encoding
        for col in self.high_card_cols:
            if col in df.columns:
                freq_map = self._freq_maps.get(col, {})
                df[col] = df[col].astype(str).map(freq_map).fillna(0.0).astype(float)
                logger.debug(f"FreqEncoder: '{col}' → float frequency")

        return df

    def get_feature_names_out(self, input_features=None):
        """sklearn set_output API compatibility — returns output column names."""
        if hasattr(self, "_feature_names_out"):
            import numpy as np

            return np.asarray(self._feature_names_out, dtype=object)
        return None


@dataclass(frozen=True)
class FeatureSpec:
    numeric: List[str]
    categorical: List[str]

    @property
    def all_features(self) -> List[str]:
        return [*self.numeric, *self.categorical]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "all_features": self.all_features,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "FeatureSpec":
        return FeatureSpec(
            numeric=list(payload.get("numeric", [])),
            categorical=list(payload.get("categorical", [])),
        )


def infer_feature_spec(df: pd.DataFrame, target_col: str) -> FeatureSpec:
    """
    DataFrame kolon tiplerinden feature spec çıkarır (FeatureEngineer öncesi).

    Neden pre-FE spec?
    - Bu spec API payload validasyonu ve validate_processed_data için kullanılır.
    - Kullanıcı country (string), arrival_date_month (string) gibi ham kolonlar gönderir.
    - build_preprocessor() içinde FE sonrası ColumnTransformer kolonları hesaplanır.
    """
    X = df.drop(columns=[target_col])
    feature_cols = list(X.columns)
    numeric = [c for c in feature_cols if X[c].dtype.kind in "biufc"]
    categorical = [c for c in feature_cols if c not in numeric]
    logger.info(f"Feature spec | numeric={len(numeric)} categorical={len(categorical)}")
    return FeatureSpec(numeric=numeric, categorical=categorical)


# FeatureEngineer'ın hangi kolonları dönüştürdüğünü build_preprocessor'a bildir
_FE_MONTH_COL: str = "arrival_date_month"  # Kaldırılır → _sin + _cos eklenir
_FE_FREQ_COLS: frozenset = frozenset(_HIGH_CARDINALITY_COLS)  # In-place float'a dönüşür


def build_preprocessor(spec: FeatureSpec) -> Pipeline:
    """
    Tam önişleme pipeline'ı (pre-FE spec alır).

    Adımlar:
    1. FeatureEngineer: cyclic ay encoding + country frequency encoding
    2. ColumnTransformer:
       - numeric: median impute → StandardScaler
       - categorical: most_frequent impute → OneHotEncoder

    FeatureEngineer sonrası ColumnTransformer kolon düzenlemesi:
    - arrival_date_month (categorical) → kaldırılır; _sin + _cos numeric olarak eklenir
    - high_cardinality kolonlar (country) → aynı isimde kalır ama float'a döner → numeric

    Neden StandardScaler?
    - LogisticRegression gibi lineer modellerde ölçek farkı optimizasyonu zorlar.
    """
    # ── Post-FE ColumnTransformer kolon listeleri ──────────────────────────
    ct_numeric: List[str] = list(spec.numeric)
    # FE month → sin + cos ekle
    if _FE_MONTH_COL in spec.categorical:
        ct_numeric.extend([f"{_FE_MONTH_COL}_sin", f"{_FE_MONTH_COL}_cos"])
    # FE freq kolonlar: aynı isimle ama artık float (numeric'e taşı)
    for col in _FE_FREQ_COLS:
        if col in spec.categorical:
            ct_numeric.append(col)
    # FE tarafından işlenen kategorikler ColumnTransformer'a verilmez
    ct_categorical: List[str] = [
        c for c in spec.categorical if c != _FE_MONTH_COL and c not in _FE_FREQ_COLS
    ]

    # ── Alt pipeline'lar ──────────────────────────────────────────────────
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, ct_numeric),
            ("cat", categorical_pipe, ct_categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )
