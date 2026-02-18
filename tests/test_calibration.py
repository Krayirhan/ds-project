import numpy as np
import pandas as pd
import pytest

from src.calibration import (
    CalibrationResult,
    _reliability_table,
    calibrate_frozen_classifier,
    calibrate_prefit_classifier,
)


def _fitted_pipeline():
    """Return a fitted sklearn Pipeline with enough samples for 5-fold CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(42)
    n = 40  # 20 per class — enough for StratifiedKFold(5) inside CalibratedClassifierCV
    X = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
    y = np.array([0] * 20 + [1] * 20)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)
    return pipe, X, y


class TestReliabilityTable:
    def test_returns_correct_number_of_bins(self):
        y = np.array([0, 1, 0, 1])
        proba = np.array([0.1, 0.9, 0.3, 0.8])
        table = _reliability_table(y, proba, bins=5)
        assert len(table) == 5

    def test_all_bins_have_required_keys(self):
        y = np.array([0, 1, 0, 1, 1])
        proba = np.array([0.05, 0.95, 0.25, 0.85, 0.65])
        table = _reliability_table(y, proba, bins=10)
        for row in table:
            assert "bin" in row
            assert "left" in row
            assert "right" in row
            assert "count" in row

    def test_empty_bin_has_none_values(self):
        y = np.array([0, 1])
        proba = np.array([0.01, 0.99])
        table = _reliability_table(y, proba, bins=10)
        empty_bins = [r for r in table if r["count"] == 0]
        assert len(empty_bins) > 0
        for eb in empty_bins:
            assert eb["avg_pred"] is None
            assert eb["empirical_rate"] is None


class TestCalibrateFrozenClassifier:
    def test_sigmoid_returns_calibration_result(self):
        model, X, y = _fitted_pipeline()
        result = calibrate_frozen_classifier(model, X, y, method="sigmoid")
        assert isinstance(result, CalibrationResult)
        assert result.metrics["calibration_method"] == "sigmoid"
        assert 0.0 <= result.metrics["brier"] <= 1.0
        assert result.metrics["n_cal"] == len(y)

    def test_isotonic_returns_calibration_result(self):
        model, X, y = _fitted_pipeline()
        result = calibrate_frozen_classifier(model, X, y, method="isotonic")
        assert isinstance(result, CalibrationResult)
        assert result.metrics["calibration_method"] == "isotonic"

    def test_calibrated_model_has_predict_proba(self):
        model, X, y = _fitted_pipeline()
        result = calibrate_frozen_classifier(model, X, y, method="sigmoid")
        proba = result.calibrated_model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_metrics_contain_reliability_bins(self):
        model, X, y = _fitted_pipeline()
        result = calibrate_frozen_classifier(model, X, y, method="sigmoid")
        assert "reliability_bins" in result.metrics
        assert isinstance(result.metrics["reliability_bins"], list)
        assert len(result.metrics["reliability_bins"]) == 10

    def test_metrics_contain_log_loss(self):
        model, X, y = _fitted_pipeline()
        result = calibrate_frozen_classifier(model, X, y, method="sigmoid")
        assert "log_loss" in result.metrics
        assert result.metrics["log_loss"] > 0.0

    def test_positive_rate_correct(self):
        model, X, y = _fitted_pipeline()
        result = calibrate_frozen_classifier(model, X, y, method="sigmoid")
        assert result.metrics["positive_rate_cal"] == pytest.approx(np.mean(y))

    def test_invalid_method_raises(self):
        model, X, y = _fitted_pipeline()
        with pytest.raises(ValueError, match="method must be"):
            calibrate_frozen_classifier(model, X, y, method="platt")


class TestCalibratePrefitClassifier:
    def test_backward_compat_wrapper(self):
        model, X, y = _fitted_pipeline()
        result = calibrate_prefit_classifier(model, X, y, method="sigmoid")
        assert isinstance(result, CalibrationResult)
        assert result.metrics["calibration_method"] == "sigmoid"
