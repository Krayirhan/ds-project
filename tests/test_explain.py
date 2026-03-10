"""Tests for src/explain.py — permutation importance + SHAP explainability."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_xy(n: int = 150, seed: int = 7):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "lead_time": rng.integers(0, 100, n),
            "adr": rng.uniform(50, 300, n),
            "arrival_date_month": rng.choice(["January", "March"], n),
        }
    )
    y = rng.integers(0, 2, n)
    return X, y


def _make_signal_xy(n: int = 600, seed: int = 123):
    """Synthetic data where lead_time/adr carry strong predictive signal."""
    rng = np.random.default_rng(seed)
    lead_time = rng.integers(0, 365, n)
    adr = rng.uniform(40, 350, n)
    month = rng.choice(["January", "March"], n)

    linear = (
        0.045 * lead_time
        + 0.008 * adr
        + np.where(month == "January", 0.6, -0.3)
        + rng.normal(0, 1.0, n)
        - 9.0
    )
    prob = 1.0 / (1.0 + np.exp(-linear))
    y = (rng.random(n) < prob).astype(int)
    X = pd.DataFrame(
        {
            "lead_time": lead_time,
            "adr": adr,
            "arrival_date_month": month,
        }
    )
    return X, y


def _make_fitted_pipeline(X, y):
    """Build a simple but real scikit-learn pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    num_cols = ["lead_time", "adr"]
    cat_cols = ["arrival_date_month"]
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )
    clf = LogisticRegression(max_iter=200, solver="liblinear", random_state=42)
    pipe = Pipeline([("preprocess", preprocess), ("clf", clf)])
    pipe.fit(X, y)
    return pipe


class TestComputePermutationImportance:
    def test_returns_expected_structure(self):
        from src.explain import compute_permutation_importance

        X, y = _make_xy()
        model = _make_fitted_pipeline(X, y)

        result = compute_permutation_importance(
            model, X, y, n_repeats=3, seed=42, scoring="roc_auc"
        )

        assert result["method"] == "permutation_importance"
        assert result["scoring"] == "roc_auc"
        assert result["n_features"] == 3
        assert isinstance(result["ranking"], list)
        assert len(result["ranking"]) == 3

    def test_ranking_has_expected_keys(self):
        from src.explain import compute_permutation_importance

        X, y = _make_xy()
        model = _make_fitted_pipeline(X, y)

        result = compute_permutation_importance(model, X, y, n_repeats=2)

        for entry in result["ranking"]:
            assert "feature" in entry
            assert "importance_mean" in entry
            assert "importance_std" in entry
            assert isinstance(entry["importance_mean"], float)

    def test_ranking_sorted_descending(self):
        from src.explain import compute_permutation_importance

        X, y = _make_xy()
        model = _make_fitted_pipeline(X, y)

        result = compute_permutation_importance(model, X, y, n_repeats=2)
        importances = [r["importance_mean"] for r in result["ranking"]]
        assert importances == sorted(importances, reverse=True)

    def test_all_features_included(self):
        from src.explain import compute_permutation_importance

        X, y = _make_xy()
        model = _make_fitted_pipeline(X, y)

        result = compute_permutation_importance(model, X, y, n_repeats=2)
        feature_names = {r["feature"] for r in result["ranking"]}
        assert feature_names == set(X.columns)


class TestAggregateShapToOriginal:
    def test_numeric_feature_maps_directly(self):
        from src.explain import _aggregate_shap_to_original

        # shap_values: 5 samples × 2 transformed features (all numeric)
        shap_values = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        )
        transformed_names = ["lead_time", "adr"]
        original_features = ["lead_time", "adr"]

        agg = _aggregate_shap_to_original(
            shap_values, transformed_names, original_features
        )

        np.testing.assert_array_almost_equal(agg["lead_time"], shap_values[:, 0])
        np.testing.assert_array_almost_equal(agg["adr"], shap_values[:, 1])

    def test_ohe_features_aggregated_to_original(self):
        from src.explain import _aggregate_shap_to_original

        # 3 samples × 3 transformed features: 1 numeric + 2 OHE categories
        shap_values = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )
        transformed_names = ["lead_time", "month_January", "month_March"]
        original_features = ["lead_time", "month"]

        agg = _aggregate_shap_to_original(
            shap_values, transformed_names, original_features
        )

        np.testing.assert_array_almost_equal(agg["lead_time"], shap_values[:, 0])
        # month = sum of OHE columns
        expected_month = shap_values[:, 1] + shap_values[:, 2]
        np.testing.assert_array_almost_equal(agg["month"], expected_month)

    def test_unmatched_feature_logged_debug(self):
        from src.explain import _aggregate_shap_to_original

        shap_values = np.array([[0.1, 0.2]])
        transformed_names = ["unknown_feature", "another_unknown"]
        original_features = ["lead_time"]

        # Should not raise; unmatched features are logged but skipped
        agg = _aggregate_shap_to_original(
            shap_values, transformed_names, original_features
        )
        # lead_time should be zero since nothing mapped to it
        np.testing.assert_array_almost_equal(agg["lead_time"], [0.0])


class TestComputeShapValues:
    def test_returns_none_when_shap_not_installed(self):
        from src.explain import compute_shap_values

        X, _ = _make_xy()
        model = _make_fitted_pipeline(X, _)

        with patch.dict("sys.modules", {"shap": None}):
            result = compute_shap_values(model, X)

        assert result is None

    def test_returns_none_when_model_not_pipeline(self):
        from src.explain import compute_shap_values

        X, _ = _make_xy()

        mock_shap = MagicMock()
        mock_model = MagicMock()
        mock_model.named_steps.get.return_value = None  # No preprocess/clf steps

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = compute_shap_values(mock_model, X)

        assert result is None

    def test_returns_none_on_shap_exception(self):
        from src.explain import compute_shap_values

        X, y = _make_xy()
        model = _make_fitted_pipeline(X, y)

        mock_shap = MagicMock()
        mock_shap.TreeExplainer.side_effect = RuntimeError("SHAP failure")
        mock_shap.KernelExplainer.side_effect = RuntimeError("SHAP failure")

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = compute_shap_values(model, X)

        assert result is None

    def test_returns_dict_with_mocked_shap(self):
        from src.explain import compute_shap_values

        X, y = _make_xy()
        model = _make_fitted_pipeline(X, y)
        n_transformed = model.named_steps["preprocess"].transform(X[:5]).shape[1]

        mock_shap_values = np.random.default_rng(42).random((5, n_transformed))

        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = mock_shap_values

        mock_shap = MagicMock()
        mock_shap.sample.side_effect = lambda arr, _k: arr
        mock_shap.KernelExplainer.return_value = mock_explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = compute_shap_values(model, X.head(5))

        assert result is not None
        assert result["method"] == "shap"
        assert "ranking" in result
        assert len(result["ranking"]) > 0

    def test_tree_shap_list_output_and_feature_name_fallback(self):
        from src.explain import compute_shap_values

        X, _ = _make_xy(n=5)

        class _SparseLike:
            def __init__(self, arr):
                self._arr = arr
                self.shape = arr.shape

            def toarray(self):
                return self._arr

        class _Pre:
            def transform(self, _X):
                return _SparseLike(np.array([[0.1, 0.2], [0.2, 0.3]]))

            def get_feature_names_out(self):
                raise RuntimeError("no feature names")

        class _TreeClf:
            feature_importances_ = [0.2, 0.8]

        model = MagicMock()
        model.named_steps.get.side_effect = lambda name: {
            "preprocess": _Pre(),
            "clf": _TreeClf(),
        }.get(name)

        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [
            np.zeros((2, 2)),
            np.array([[0.1, 0.2], [0.2, 0.3]]),
        ]
        mock_shap = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = compute_shap_values(model, X.head(2))

        assert result is not None
        assert result["method"] == "shap"
        assert result["n_original_features"] == len(X.columns)

    def test_kernel_shap_list_output_path(self):
        from src.explain import compute_shap_values

        X, _ = _make_xy(n=20)

        class _Pre:
            def transform(self, _X):
                return np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])

            def get_feature_names_out(self):
                return ["f1", "f2"]

        class _Clf:
            def predict_proba(self, arr):
                n = len(arr)
                return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

        model = MagicMock()
        model.named_steps.get.side_effect = lambda name: {
            "preprocess": _Pre(),
            "clf": _Clf(),
        }.get(name)

        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [
            np.zeros((3, 2)),
            np.array([[0.1, -0.2], [0.2, 0.1], [-0.1, 0.05]]),
        ]
        mock_shap = MagicMock()
        mock_shap.sample.side_effect = lambda arr, _k: arr
        mock_shap.KernelExplainer.return_value = mock_explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = compute_shap_values(model, X.head(3))

        assert result is not None
        assert result["method"] == "shap"
        assert len(result["ranking"]) == len(X.columns)


class TestSaveExplainabilityReport:
    def test_saves_json_file(self, tmp_path):
        from src.explain import save_explainability_report

        report = {
            "method": "permutation_importance",
            "n_features": 3,
            "ranking": [
                {
                    "feature": "lead_time",
                    "importance_mean": 0.05,
                    "importance_std": 0.01,
                }
            ],
        }
        out_path = tmp_path / "reports" / "explain.json"

        save_explainability_report(report, out_path)

        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded["method"] == "permutation_importance"
        assert len(loaded["ranking"]) == 1

    def test_creates_parent_dirs(self, tmp_path):
        from src.explain import save_explainability_report

        out_path = tmp_path / "a" / "b" / "c" / "explain.json"
        save_explainability_report({"method": "shap", "ranking": []}, out_path)

        assert out_path.exists()


class TestExplainabilityConsistency:
    def test_permutation_importance_statistical_stability(self):
        from src.explain import compute_permutation_importance

        X, y = _make_signal_xy()
        model = _make_fitted_pipeline(X, y)

        res_a = compute_permutation_importance(
            model, X, y, n_repeats=8, seed=11, scoring="roc_auc"
        )
        res_b = compute_permutation_importance(
            model, X, y, n_repeats=8, seed=97, scoring="roc_auc"
        )

        imp_a = pd.Series(
            {row["feature"]: row["importance_mean"] for row in res_a["ranking"]}
        ).sort_index()
        imp_b = pd.Series(
            {row["feature"]: row["importance_mean"] for row in res_b["ranking"]}
        ).sort_index()

        spearman_rank_corr = imp_a.rank().corr(imp_b.rank())
        assert spearman_rank_corr is not None
        assert spearman_rank_corr >= 0.5

        top_a = res_a["ranking"][0]["feature"]
        top_b = res_b["ranking"][0]["feature"]
        assert top_a in {"lead_time", "adr"}
        assert top_b in {"lead_time", "adr"}

        # Ensure we have non-trivial signal on at least one major driver.
        assert max(imp_a["lead_time"], imp_a["adr"]) > 0.01

    def test_permutation_and_shap_ranking_consistency_with_kernel_stub(self):
        from src.explain import compute_permutation_importance, compute_shap_values

        X, y = _make_signal_xy(n=400, seed=777)
        model = _make_fitted_pipeline(X, y)
        perm = compute_permutation_importance(
            model, X, y, n_repeats=6, seed=42, scoring="roc_auc"
        )
        perm_map = {
            row["feature"]: float(row["importance_mean"]) for row in perm["ranking"]
        }

        preprocess = model.named_steps["preprocess"]
        clf = model.named_steps["clf"]
        X_t = preprocess.transform(X.head(120))
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        coef = np.asarray(clf.coef_)[0]

        class _KernelExplainer:
            def __init__(self, _predict_proba, _background):
                pass

            def shap_values(self, arr, nsamples=100):  # noqa: ARG002
                values = np.asarray(arr) * coef
                return [np.zeros_like(values), values]

        class _ShapStub:
            @staticmethod
            def sample(arr, k):
                return np.asarray(arr)[:k]

            KernelExplainer = _KernelExplainer

        with patch.dict("sys.modules", {"shap": _ShapStub()}):
            shap_result = compute_shap_values(model, X.head(120))

        assert shap_result is not None
        shap_map = {
            row["feature"]: float(row["mean_abs_shap"])
            for row in shap_result["ranking"]
        }

        common = sorted(set(perm_map).intersection(shap_map))
        perm_rank = pd.Series({f: perm_map[f] for f in common}).rank()
        shap_rank = pd.Series({f: shap_map[f] for f in common}).rank()
        rank_corr = perm_rank.corr(shap_rank)
        assert rank_corr is not None
        assert rank_corr >= 0.5

        top_perm = max(perm_map.items(), key=lambda kv: kv[1])[0]
        top_shap = max(shap_map.items(), key=lambda kv: kv[1])[0]
        assert top_perm in {"lead_time", "adr"}
        assert top_shap in {"lead_time", "adr"}
