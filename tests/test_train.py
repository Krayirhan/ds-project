import pandas as pd

from src.train import train_baseline, train_candidate_models


def _tiny_df():
    return pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5, 6, 7, 8],
            "num2": [10, 9, 8, 7, 6, 5, 4, 3],
            "cat1": ["a", "a", "b", "b", "a", "b", "a", "b"],
            "target": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )


def test_train_candidate_models_baseline_only():
    df = _tiny_df()
    results = train_candidate_models(
        df,
        target_col="target",
        seed=42,
        cv_folds=2,
        include_challenger=False,
    )
    assert "baseline" in results
    assert len(results) == 1
    assert results["baseline"].cv_scores.shape[0] == 2


def test_train_candidate_models_with_challenger():
    df = _tiny_df()
    results = train_candidate_models(
        df,
        target_col="target",
        seed=42,
        cv_folds=2,
        include_challenger=True,
    )
    assert "baseline" in results
    assert len(results) >= 2


def test_train_baseline_wrapper():
    df = _tiny_df()
    res = train_baseline(df, target_col="target", seed=42, cv_folds=2)
    assert res.cv_scores.shape[0] == 2
