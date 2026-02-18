from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.policy import DecisionPolicy
from src.predict import load_feature_spec, predict_with_policy, validate_and_prepare_features


# ── helpers ───────────────────────────────────────────────────────────


class DummyModel:
    """Model that ignores X and returns fixed probabilities."""

    def __init__(self, proba):
        self._proba = np.asarray(proba, dtype=float)

    def predict_proba(self, X):
        p1 = self._proba[: len(X)]
        return np.column_stack([1.0 - p1, p1])


def _feature_spec():
    return {
        "numeric": ["n1", "n2"],
        "categorical": ["c1"],
        "all_features": ["n1", "n2", "c1"],
    }


def _input_df(n: int = 3):
    return pd.DataFrame(
        {
            "n1": list(range(n)),
            "n2": list(range(10, 10 + n)),
            "c1": ["a"] * n,
        }
    )


def _simple_policy(**overrides) -> DecisionPolicy:
    defaults = dict(
        selected_model="test_model",
        selected_model_artifact="models/test.joblib",
        threshold=0.5,
        max_action_rate=None,
        expected_net_profit=100.0,
        raw={"ranking_mode": "proba", "policy_version": "1.0.0"},
    )
    defaults.update(overrides)
    return DecisionPolicy(**defaults)


# ── load_feature_spec ─────────────────────────────────────────────────


class TestLoadFeatureSpec:
    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_feature_spec(Path("/nonexistent/spec.json"))

    def test_invalid_spec_missing_keys(self, tmp_path: Path):
        p = tmp_path / "spec.json"
        p.write_text('{"other_key": []}')
        with pytest.raises(ValueError, match="numeric"):
            load_feature_spec(p)

    def test_valid_spec_returns_dict(self, tmp_path: Path):
        import json

        p = tmp_path / "spec.json"
        p.write_text(json.dumps(_feature_spec()))
        result = load_feature_spec(p)
        assert result["numeric"] == ["n1", "n2"]
        assert result["categorical"] == ["c1"]
        assert "all_features" in result


# ── validate_and_prepare_features extras ──────────────────────────────


class TestValidateAndPrepareExtras:
    def test_numeric_cast_failure_raises(self):
        df = pd.DataFrame({"n1": [1, "not_a_number"], "n2": [3, 4], "c1": ["a", "b"]})
        with pytest.raises(ValueError, match="Numeric type validation"):
            validate_and_prepare_features(df, _feature_spec())

    def test_fail_on_missing_false_fills_nan(self):
        df = pd.DataFrame({"n1": [1, 2], "c1": ["a", "b"]})  # n2 missing
        out, report = validate_and_prepare_features(df, _feature_spec(), fail_on_missing=False)
        assert "n2" in out.columns
        assert out["n2"].isna().all()
        assert "n2" in report["missing_columns"]

    def test_categorical_non_object_gets_coerced(self):
        df = pd.DataFrame({"n1": [1, 2], "n2": [3, 4], "c1": [100, 200]})
        out, _ = validate_and_prepare_features(df, _feature_spec())
        assert out["c1"].dtype.name in ("string", "object")


# ── predict_with_policy full path ─────────────────────────────────────


class TestPredictWithPolicy:
    def test_returns_dataframe_and_report(self):
        df = _input_df(3)
        model = DummyModel([0.8, 0.3, 0.9])
        policy = _simple_policy()
        actions_df, report = predict_with_policy(
            model=model,
            policy=policy,
            df_input=df,
            feature_spec_payload=_feature_spec(),
            model_used="test_artifact",
        )
        assert isinstance(actions_df, pd.DataFrame)
        assert len(actions_df) == 3
        assert set(actions_df.columns) >= {"proba", "action", "threshold_used", "model_used"}
        assert isinstance(report, dict)

    def test_action_column_respects_threshold(self):
        df = _input_df(4)
        model = DummyModel([0.8, 0.3, 0.6, 0.1])
        policy = _simple_policy(threshold=0.5)
        actions_df, _ = predict_with_policy(
            model=model, policy=policy, df_input=df, feature_spec_payload=_feature_spec()
        )
        assert actions_df["action"].tolist() == [1, 0, 1, 0]

    def test_report_contains_required_keys(self):
        df = _input_df(2)
        model = DummyModel([0.7, 0.2])
        policy = _simple_policy()
        _, report = predict_with_policy(
            model=model, policy=policy, df_input=df, feature_spec_payload=_feature_spec()
        )
        for key in (
            "n_rows",
            "predicted_action_rate",
            "threshold_used",
            "model_used",
            "ranking_mode",
            "missing_columns",
            "extra_columns",
        ):
            assert key in report, f"Missing key: {key}"

    def test_model_used_default_from_policy(self):
        df = _input_df(1)
        model = DummyModel([0.5])
        policy = _simple_policy()
        actions_df, report = predict_with_policy(
            model=model, policy=policy, df_input=df, feature_spec_payload=_feature_spec()
        )
        assert actions_df["model_used"].iloc[0] == "test_model"
        assert report["model_used"] == "test_model"

    def test_with_max_action_rate_constraint(self):
        df = _input_df(5)
        model = DummyModel([0.9, 0.8, 0.7, 0.6, 0.55])
        policy = _simple_policy(threshold=0.5, max_action_rate=0.4)
        actions_df, report = predict_with_policy(
            model=model, policy=policy, df_input=df, feature_spec_payload=_feature_spec()
        )
        assert actions_df["action"].sum() <= 2  # floor(0.4*5)=2
        assert report["predicted_action_rate"] <= 0.5

    def test_extra_columns_ignored_in_output(self):
        df = _input_df(2)
        df["extra_col"] = [99, 100]
        model = DummyModel([0.7, 0.3])
        policy = _simple_policy()
        _, report = predict_with_policy(
            model=model, policy=policy, df_input=df, feature_spec_payload=_feature_spec()
        )
        assert "extra_col" in report["extra_columns"]
