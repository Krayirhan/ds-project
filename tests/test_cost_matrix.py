import numpy as np
import pandas as pd

from src.cost_matrix import (
    CostMatrix,
    compute_profit_from_confusion,
    default_cost_matrix_example,
    sweep_thresholds_for_profit,
    sweep_thresholds_for_profit_with_constraint,
)


class DummyModel:
    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        p1 = self._proba[: len(X)]
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def test_compute_profit_from_confusion_basic():
    cost = CostMatrix(tp_value=10.0, fp_value=-2.0, fn_value=-5.0, tn_value=0.0)
    profit = compute_profit_from_confusion(tn=5, fp=2, fn=1, tp=3, cost=cost)
    assert profit == 3 * 10.0 + 2 * -2.0 + 1 * -5.0


def test_profit_sweep_with_constraint_respects_rate():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [0, 1, 0, 1, 1]})
    model = DummyModel([0.9, 0.8, 0.7, 0.3, 0.2])
    cost = CostMatrix(tp_value=100, fp_value=-20, fn_value=-50, tn_value=0)
    res = sweep_thresholds_for_profit_with_constraint(
        model=model,
        df_test=df,
        target_col="y",
        cost=cost,
        max_action_rate=0.4,
    )
    assert 0.0 < res.best_threshold < 1.0
    assert len(res.rows) > 0
    feasible_rows = [r for r in res.rows if r.get("feasible")]
    assert feasible_rows


def test_profit_sweep_unconstrained_returns_rows():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    model = DummyModel([0.9, 0.8, 0.2, 0.1])
    cost = CostMatrix(tp_value=10, fp_value=-1, fn_value=-2, tn_value=0)
    res = sweep_thresholds_for_profit(
        model=model,
        df_test=df,
        target_col="y",
        cost=cost,
        thresholds=np.array([0.2, 0.5, 0.8]),
    )
    assert res.best_threshold in {0.2, 0.5, 0.8}
    assert len(res.rows) == 3


def test_profit_sweep_quantile_fallback_path():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [0, 1, 0, 1, 1]})
    model = DummyModel([0.9, 0.9, 0.9, 0.9, 0.9])
    cost = CostMatrix(tp_value=10, fp_value=-1, fn_value=-2, tn_value=0)
    res = sweep_thresholds_for_profit_with_constraint(
        model=model,
        df_test=df,
        target_col="y",
        cost=cost,
        max_action_rate=0.2,
        thresholds=np.array([0.1]),
        use_quantile_fallback=True,
    )
    assert len(res.rows) >= 2
    assert any(r.get("selection_strategy") == "quantile_fallback" for r in res.rows)


def test_default_cost_matrix_example_values():
    c = default_cost_matrix_example()
    assert c.tp_value == 180.0
    assert c.fn_value == -200.0
