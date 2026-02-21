from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate_binary_classifier, sweep_thresholds


class DummyModel:
    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        p1 = self._proba[: len(X)]
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def test_evaluate_binary_classifier_writes_metrics(tmp_path: Path):
    df = pd.DataFrame({"f1": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    model = DummyModel([0.2, 0.8, 0.3, 0.9])
    out = tmp_path / "metrics.json"
    metrics = evaluate_binary_classifier(model, df, "target", out, threshold=0.5)
    assert out.exists()
    assert metrics["n_test"] == 4
    assert metrics["threshold"] == 0.5


def test_sweep_thresholds_rule_mode_precision_given_recall(tmp_path: Path):
    df = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "target": [0, 1, 1, 0, 1]})
    model = DummyModel([0.1, 0.9, 0.7, 0.2, 0.8])
    out = tmp_path / "sweep.json"
    res = sweep_thresholds(
        model,
        df,
        "target",
        out,
        thresholds=np.array([0.2, 0.5, 0.8]),
        rule_mode="maximize_precision_given_recall",
        min_recall=0.5,
    )
    assert out.exists()
    assert res["best_by_f1"]["threshold"] in {0.2, 0.5, 0.8}
    assert "selection" in res["best_by_rule"]


def test_sweep_thresholds_default_thresholds_and_recall_given_precision(tmp_path: Path):
    df = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "target": [0, 1, 1, 0, 1]})
    model = DummyModel([0.1, 0.9, 0.7, 0.2, 0.8])
    out = tmp_path / "sweep_default.json"
    res = sweep_thresholds(
        model,
        df,
        "target",
        out,
        rule_mode="maximize_recall_given_precision",
        min_precision=0.5,
    )
    assert out.exists()
    assert len(res["rows"]) > 0
    assert res["best_by_rule"]["selection"]["threshold"] is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rule_mode": "maximize_precision_given_recall", "min_recall": None},
        {"rule_mode": "maximize_recall_given_precision", "min_precision": None},
    ],
)
def test_sweep_thresholds_rule_validation_errors(tmp_path: Path, kwargs):
    df = pd.DataFrame({"f1": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    model = DummyModel([0.2, 0.8, 0.3, 0.9])
    out = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        sweep_thresholds(
            model,
            df,
            "target",
            out,
            thresholds=np.array([0.5]),
            **kwargs,
        )


def test_sweep_thresholds_rule_no_candidate(tmp_path: Path):
    df = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "target": [0, 1, 1, 0, 1]})
    model = DummyModel([0.1, 0.9, 0.7, 0.2, 0.8])
    out = tmp_path / "no_candidate.json"
    res = sweep_thresholds(
        model,
        df,
        "target",
        out,
        thresholds=np.array([0.2, 0.5, 0.8]),
        rule_mode="maximize_precision_given_recall",
        min_recall=1.1,
    )
    assert out.exists()
    assert res["best_by_rule"]["selection"]["threshold"] is None
