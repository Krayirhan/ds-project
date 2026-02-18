"""
hpo.py

Optuna tabanlı hyperparameter optimization.

Neden HPO?
- Default parametreler sadece baseline içindir.
- Grid search yüksek boyutta çok yavaş; Optuna Bayesian + pruning ile verimli.
- Cross-validation objective overfitting'i kontrol eder.

Kullanım:
    python main.py hpo --n-trials 50
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .features import build_preprocessor, infer_feature_spec
from .utils import get_logger

logger = get_logger("hpo")


@dataclass(frozen=True)
class HPOResult:
    """Hyperparameter optimization sonucu."""

    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    model_type: str


def _detect_best_model_type() -> str:
    """Mevcut en iyi GBM kütüphanesini seç (XGBoost > LightGBM > HistGB)."""
    try:
        import xgboost  # noqa: F401

        return "xgboost"
    except ImportError:
        pass
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        pass
    return "histgb"


def _get_search_space(trial: Any, model_type: str) -> Dict[str, Any]:
    """Optuna search space tanımları (model tipine göre)."""
    if model_type == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
    if model_type == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    # histgb fallback
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 1e-8, 10.0, log=True
        ),
    }


def _build_estimator(model_type: str, params: Dict[str, Any], seed: int) -> Any:
    """HPO parametreleriyle estimator oluşturur."""
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **params, objective="binary", random_state=seed, verbosity=-1
        )
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(**params, random_state=seed)


def run_hpo(
    df: pd.DataFrame,
    target_col: str,
    seed: int,
    cv_folds: int,
    n_trials: int = 50,
    model_type: Optional[str] = None,
) -> HPOResult:
    """
    Optuna ile hyperparameter optimization çalıştırır.

    Returns:
        HPOResult with best_params, best_score, n_trials, model_type
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required for HPO. Install: pip install optuna")

    resolved_type = model_type or _detect_best_model_type()
    spec = infer_feature_spec(df, target_col)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int).values

    def objective(trial: Any) -> float:
        params = _get_search_space(trial, resolved_type)
        estimator = _build_estimator(resolved_type, params, seed)
        pipeline = Pipeline(
            [
                ("preprocess", build_preprocessor(spec)),
                ("clf", estimator),
            ]
        )
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=f"hpo_{resolved_type}", direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        f"HPO completed | model_type={resolved_type} n_trials={n_trials} "
        f"best_score={study.best_value:.4f} best_params={study.best_params}"
    )

    return HPOResult(
        best_params=study.best_params,
        best_score=float(study.best_value),
        n_trials=n_trials,
        model_type=resolved_type,
    )
