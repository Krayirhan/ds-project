"""
features.py

Feature dönüşümleri burada.

Neden Pipeline/ColumnTransformer?
- Dönüşümler train ve predict'te aynı olmalı.
- Notebook içinde manual dönüşüm yapmak training-serving skew doğurur.

Burada:
- numeric: median impute + StandardScaler
- categorical: most_frequent impute + OneHotEncoder
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .utils import get_logger

logger = get_logger("features")


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
    Otomatik feature type inference.

    Neden?
    - Hızlı baseline için.
    - Sonra bu listeleri manuel optimize edeceğiz.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    numeric = [c for c in feature_cols if df[c].dtype.kind in "biufc"]
    categorical = [c for c in feature_cols if c not in numeric]
    logger.info(f"Feature spec | numeric={len(numeric)} categorical={len(categorical)}")
    return FeatureSpec(numeric=numeric, categorical=categorical)


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """
    Dönüşüm blokları.

    Neden StandardScaler?
    - LogisticRegression gibi lineer modellerde ölçek farkı optimizasyonu zorlar.
    - Scaling ile convergence ve stabilite artar.
    """
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, spec.numeric),
            ("cat", categorical_pipe, spec.categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor
