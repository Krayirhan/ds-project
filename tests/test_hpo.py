"""Tests for src/hpo.py — Optuna hyperparameter optimization."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "lead_time": rng.integers(0, 100, n),
            "adr": rng.uniform(50, 300, n),
            "arrival_date_month": rng.choice(["January", "March", "July"], n),
            "is_canceled": rng.integers(0, 2, n),
        }
    )


class TestDetectBestModelType:
    def test_returns_xgboost_when_available(self):
        from src.hpo import _detect_best_model_type

        xgb_mock = MagicMock()
        with patch.dict("sys.modules", {"xgboost": xgb_mock}):
            result = _detect_best_model_type()
        assert result == "xgboost"

    def test_falls_back_to_lightgbm_when_no_xgboost(self):
        from src.hpo import _detect_best_model_type

        lgb_mock = MagicMock()
        with patch.dict("sys.modules", {"xgboost": None, "lightgbm": lgb_mock}):
            result = _detect_best_model_type()
        assert result == "lightgbm"

    def test_falls_back_to_histgb_when_no_gbm(self):
        from src.hpo import _detect_best_model_type

        with patch.dict("sys.modules", {"xgboost": None, "lightgbm": None}):
            result = _detect_best_model_type()
        assert result == "histgb"


class TestGetSearchSpace:
    def _make_trial(self) -> MagicMock:
        trial = MagicMock()
        trial.suggest_int.return_value = 100
        trial.suggest_float.return_value = 0.1
        return trial

    def test_xgboost_search_space_has_expected_keys(self):
        from src.hpo import _get_search_space

        trial = self._make_trial()
        params = _get_search_space(trial, "xgboost")

        assert "n_estimators" in params
        assert "learning_rate" in params
        assert "max_depth" in params
        assert "subsample" in params
        assert "colsample_bytree" in params

    def test_lightgbm_search_space_has_expected_keys(self):
        from src.hpo import _get_search_space

        trial = self._make_trial()
        params = _get_search_space(trial, "lightgbm")

        assert "n_estimators" in params
        assert "learning_rate" in params
        assert "num_leaves" in params

    def test_histgb_search_space_has_expected_keys(self):
        from src.hpo import _get_search_space

        trial = self._make_trial()
        params = _get_search_space(trial, "histgb")

        assert "learning_rate" in params
        assert "max_depth" in params
        assert "max_iter" in params


class TestBuildEstimator:
    def test_builds_xgboost_classifier(self):
        from src.hpo import _build_estimator

        try:
            est = _build_estimator(
                "xgboost",
                {
                    "n_estimators": 50,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "reg_alpha": 0.01,
                    "min_child_weight": 1,
                },
                seed=42,
            )
            assert hasattr(est, "fit")
        except ImportError:
            pytest.skip("xgboost not installed")

    def test_builds_histgb_classifier(self):
        from src.hpo import _build_estimator

        est = _build_estimator(
            "histgb",
            {
                "learning_rate": 0.1,
                "max_depth": 4,
                "max_iter": 100,
                "min_samples_leaf": 10,
                "l2_regularization": 1.0,
            },
            seed=42,
        )
        assert hasattr(est, "fit")

    def test_builds_lightgbm_classifier_with_stub(self, monkeypatch):
        from src.hpo import _build_estimator

        mod = types.ModuleType("lightgbm")

        class LGBMClassifier:  # noqa: N801
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        mod.LGBMClassifier = LGBMClassifier
        monkeypatch.setitem(__import__("sys").modules, "lightgbm", mod)

        est = _build_estimator(
            "lightgbm", {"n_estimators": 100, "learning_rate": 0.1}, seed=17
        )
        assert est.kwargs["objective"] == "binary"
        assert est.kwargs["random_state"] == 17


class TestRunHPO:
    def test_raises_import_error_without_optuna(self):
        from src.hpo import run_hpo

        df = _make_df()
        with patch.dict("sys.modules", {"optuna": None}):
            with pytest.raises(ImportError, match="optuna"):
                run_hpo(df, "is_canceled", seed=42, cv_folds=2, n_trials=1)

    def test_run_hpo_returns_hpo_result(self):
        from src.hpo import run_hpo, HPOResult

        df = _make_df()

        mock_optuna = MagicMock()
        mock_study = MagicMock()
        mock_study.best_value = 0.82
        mock_study.best_params = {
            "learning_rate": 0.1,
            "max_depth": 4,
            "max_iter": 100,
            "min_samples_leaf": 10,
            "l2_regularization": 1.0,
        }
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.logging = MagicMock()
        mock_optuna.logging.WARNING = 30

        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            result = run_hpo(
                df, "is_canceled", seed=42, cv_folds=2, n_trials=3, model_type="histgb"
            )

        assert isinstance(result, HPOResult)
        assert result.best_score == 0.82
        assert result.n_trials == 3
        assert result.model_type == "histgb"

    def test_run_hpo_calls_optimize_with_n_trials(self):
        from src.hpo import run_hpo

        df = _make_df()

        mock_optuna = MagicMock()
        mock_study = MagicMock()
        mock_study.best_value = 0.75
        mock_study.best_params = {
            "learning_rate": 0.05,
            "max_depth": 3,
            "max_iter": 200,
            "min_samples_leaf": 20,
            "l2_regularization": 0.5,
        }
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.logging = MagicMock()
        mock_optuna.logging.WARNING = 30

        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            run_hpo(
                df, "is_canceled", seed=42, cv_folds=2, n_trials=7, model_type="histgb"
            )

        mock_study.optimize.assert_called_once()
        call_kwargs = mock_study.optimize.call_args[1]
        assert call_kwargs["n_trials"] == 7

    def test_run_hpo_detects_model_type_automatically(self):
        from src.hpo import run_hpo

        df = _make_df()

        mock_optuna = MagicMock()
        mock_study = MagicMock()
        mock_study.best_value = 0.78
        mock_study.best_params = {
            "learning_rate": 0.1,
            "max_depth": 4,
            "max_iter": 100,
            "min_samples_leaf": 5,
            "l2_regularization": 1.0,
        }
        mock_optuna.create_study.return_value = mock_study
        mock_optuna.logging = MagicMock()
        mock_optuna.logging.WARNING = 30

        # Force histgb by removing xgboost and lightgbm from sys.modules
        with (
            patch.dict(
                "sys.modules",
                {"optuna": mock_optuna, "xgboost": None, "lightgbm": None},
            ),
        ):
            result = run_hpo(df, "is_canceled", seed=42, cv_folds=2, n_trials=2)

        assert result.model_type == "histgb"

    def test_run_hpo_executes_objective_with_cv_path(self):
        from src.hpo import run_hpo

        df = _make_df(n=60)

        trial = MagicMock()
        trial.suggest_int.side_effect = lambda _name, low, _high, step=None: (
            low if step is None else low
        )
        trial.suggest_float.side_effect = lambda _name, low, _high, log=False: low

        class _FakeStudy:
            def __init__(self):
                self.best_value = 0.88
                self.best_params = {"learning_rate": 0.1}
                self.objective_score = None

            def optimize(self, objective, n_trials, show_progress_bar):
                assert n_trials == 1
                assert show_progress_bar is True
                self.objective_score = objective(trial)

        fake_study = _FakeStudy()
        fake_optuna = MagicMock()
        fake_optuna.create_study.return_value = fake_study
        fake_optuna.logging = MagicMock()
        fake_optuna.logging.WARNING = 30

        with (
            patch.dict("sys.modules", {"optuna": fake_optuna}),
            patch("src.hpo.cross_val_score", return_value=np.array([0.70, 0.80])),
        ):
            result = run_hpo(
                df=df,
                target_col="is_canceled",
                seed=42,
                cv_folds=2,
                n_trials=1,
                model_type="histgb",
            )

        assert fake_study.objective_score == pytest.approx(0.75)
        assert result.best_score == pytest.approx(0.88)
