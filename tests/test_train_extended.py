from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd

import src.train as train
from src.features import FeatureSpec


def test_build_model_pipeline_structure():
    spec = FeatureSpec(numeric=["x1"], categorical=["c1"])
    est = train._build_baseline_estimator(seed=42)
    pipe = train._build_model_pipeline(spec, est)
    assert "preprocess" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_build_first_available_challenger_xgboost(monkeypatch):
    mod = types.ModuleType("xgboost")

    class XGBClassifier:  # noqa: N801
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mod.XGBClassifier = XGBClassifier
    monkeypatch.setitem(sys.modules, "xgboost", mod)

    name, est = train._build_first_available_challenger(seed=7)
    assert name == "challenger_xgboost"
    assert est.__class__.__name__ == "XGBClassifier"


def test_build_first_available_challenger_lightgbm(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    mod = types.ModuleType("lightgbm")

    class LGBMClassifier:  # noqa: N801
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mod.LGBMClassifier = LGBMClassifier
    monkeypatch.setitem(sys.modules, "lightgbm", mod)

    name, est = train._build_first_available_challenger(seed=7)
    assert name == "challenger_lightgbm"
    assert est.__class__.__name__ == "LGBMClassifier"


def test_build_first_available_challenger_catboost(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    mod = types.ModuleType("catboost")

    class CatBoostClassifier:  # noqa: N801
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mod.CatBoostClassifier = CatBoostClassifier
    monkeypatch.setitem(sys.modules, "catboost", mod)

    name, est = train._build_first_available_challenger(seed=7)
    assert name == "challenger_catboost"
    assert est.__class__.__name__ == "CatBoostClassifier"


def test_build_first_available_challenger_fallback_histgb(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    monkeypatch.setitem(sys.modules, "catboost", None)

    name, est = train._build_first_available_challenger(seed=11)
    assert name == "challenger_histgb"
    assert est.__class__.__name__ == "HistGradientBoostingClassifier"


def test_fit_one_returns_train_result(monkeypatch):
    spec = FeatureSpec(numeric=["x"], categorical=[])
    X = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}).drop(columns=["y"])
    y = np.array([0, 1, 0, 1], dtype=int)

    class DummyModel:
        def fit(self, Xv, yv):
            self.fitted_ = True
            return self

    monkeypatch.setattr(train, "_build_model_pipeline", lambda spec, estimator: DummyModel())
    monkeypatch.setattr(train, "cross_val_score", lambda model, X, y, cv, scoring: np.array([0.7, 0.8]))

    out = train._fit_one(
        name="baseline",
        estimator=object(),
        spec=spec,
        X=X,
        y=y,
        cv_folds=2,
        seed=42,
        feature_dtypes={"x": "int64"},
    )
    assert out.cv_scores.shape[0] == 2
    assert out.feature_dtypes["x"] == "int64"
