"""Model training with champion/challenger support."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .features import FeatureSpec, build_preprocessor, infer_feature_spec
from .utils import get_logger

logger = get_logger("train")


@dataclass(frozen=True)
class TrainResult:
    model: Pipeline
    cv_scores: np.ndarray
    feature_spec: FeatureSpec
    feature_dtypes: Dict[str, str]


def _build_model_pipeline(spec: FeatureSpec, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(spec)),
            ("clf", estimator),
        ]
    )


def _build_baseline_estimator(seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=3000, solver="lbfgs", random_state=seed)


def _build_first_available_challenger(seed: int) -> Tuple[str, object]:
    # Priority: XGBoost -> LightGBM -> CatBoost -> sklearn fallback
    try:
        from xgboost import XGBClassifier

        return (
            "challenger_xgboost",
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        return (
            "challenger_lightgbm",
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary",
                random_state=seed,
            ),
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        return (
            "challenger_catboost",
            CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                random_seed=seed,
                verbose=False,
            ),
        )
    except Exception:
        pass

    logger.info("No external GBM package found. Falling back to HistGradientBoosting.")
    return (
        "challenger_histgb",
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=400,
            random_state=seed,
        ),
    )


def _fit_one(
    *,
    name: str,
    estimator,
    spec: FeatureSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int,
    seed: int,
    feature_dtypes: Dict[str, str],
) -> TrainResult:
    model = _build_model_pipeline(spec, estimator)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    logger.info(
        f"[{name}] CV ROC-AUC | mean={scores.mean():.4f} std={scores.std():.4f}"
    )
    model.fit(X, y)
    return TrainResult(
        model=model,
        cv_scores=scores,
        feature_spec=spec,
        feature_dtypes=feature_dtypes,
    )


def train_candidate_models(
    df: pd.DataFrame,
    target_col: str,
    seed: int,
    cv_folds: int,
    include_challenger: bool = True,
) -> Dict[str, TrainResult]:
    spec = infer_feature_spec(df, target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int).values
    feature_dtypes = {c: str(X[c].dtype) for c in spec.all_features if c in X.columns}

    results: Dict[str, TrainResult] = {}

    results["baseline"] = _fit_one(
        name="baseline",
        estimator=_build_baseline_estimator(seed),
        spec=spec,
        X=X,
        y=y,
        cv_folds=cv_folds,
        seed=seed,
        feature_dtypes=feature_dtypes,
    )

    if include_challenger:
        challenger_name, challenger_estimator = _build_first_available_challenger(seed)
        results[challenger_name] = _fit_one(
            name=challenger_name,
            estimator=challenger_estimator,
            spec=spec,
            X=X,
            y=y,
            cv_folds=cv_folds,
            seed=seed,
            feature_dtypes=feature_dtypes,
        )

    return results


def train_baseline(
    df: pd.DataFrame, target_col: str, seed: int, cv_folds: int
) -> TrainResult:
    """Backward-compatible wrapper."""
    return train_candidate_models(
        df=df,
        target_col=target_col,
        seed=seed,
        cv_folds=cv_folds,
        include_challenger=False,
    )["baseline"]
