import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.policy import (
    DecisionPolicy,
    apply,
    apply_policy_to_proba,
    compute_incremental_profit_scores,
    decide_actions_from_proba,
    load_decision_policy,
)


# ── load_decision_policy ──────────────────────────────────────────────


class TestLoadDecisionPolicy:
    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_decision_policy(tmp_path / "nonexistent.json")

    def test_bad_status_raises(self, tmp_path: Path):
        p = tmp_path / "policy.json"
        p.write_text(json.dumps({"status": "no_policy", "reason": "test"}))
        with pytest.raises(ValueError, match="status"):
            load_decision_policy(p)

    def test_invalid_threshold_raises(self, tmp_path: Path):
        p = tmp_path / "policy.json"
        p.write_text(json.dumps({"status": "ok", "threshold": 1.5}))
        with pytest.raises(ValueError, match="threshold"):
            load_decision_policy(p)

    def test_invalid_max_action_rate_raises(self, tmp_path: Path):
        p = tmp_path / "policy.json"
        p.write_text(json.dumps({"status": "ok", "threshold": 0.5, "max_action_rate": 0.0}))
        with pytest.raises(ValueError, match="max_action_rate"):
            load_decision_policy(p)

    def test_valid_policy_loads(self, tmp_path: Path):
        p = tmp_path / "policy.json"
        p.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "threshold": 0.5,
                    "max_action_rate": 0.3,
                    "selected_model": "baseline",
                    "selected_model_artifact": "models/test.joblib",
                    "expected_net_profit": 100.0,
                }
            )
        )
        policy = load_decision_policy(p)
        assert isinstance(policy, DecisionPolicy)
        assert policy.threshold == 0.5
        assert policy.max_action_rate == 0.3
        assert policy.selected_model == "baseline"

    def test_none_max_action_rate_allowed(self, tmp_path: Path):
        p = tmp_path / "policy.json"
        p.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "threshold": 0.4,
                    "selected_model": "m",
                }
            )
        )
        policy = load_decision_policy(p)
        assert policy.max_action_rate is None


# ── decide_actions_from_proba edge cases ──────────────────────────────


class TestDecideActionsEdgeCases:
    def test_invalid_proba_dimension_raises(self):
        proba = np.array([[0.1, 0.9], [0.3, 0.7]])
        with pytest.raises(ValueError, match="1D"):
            decide_actions_from_proba(proba, threshold=0.5)

    def test_invalid_threshold_in_decide_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            decide_actions_from_proba(np.array([0.5]), threshold=1.5)

    def test_invalid_max_action_rate_in_decide_raises(self):
        with pytest.raises(ValueError, match="max_action_rate"):
            decide_actions_from_proba(np.array([0.6]), threshold=0.5, max_action_rate=0.0)

    def test_ranking_scores_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            decide_actions_from_proba(
                np.array([0.6, 0.7]),
                threshold=0.5,
                ranking_scores=np.array([0.1]),
            )

    def test_negative_ranking_scores_block_action(self):
        proba = np.array([0.6, 0.7, 0.8])
        ranking = np.array([-0.1, 0.5, -0.2])
        actions = decide_actions_from_proba(proba, threshold=0.5, ranking_scores=ranking)
        assert actions.tolist() == [0, 1, 0]

    def test_k_zero_returns_all_zeros(self):
        proba = np.array([0.6, 0.7])
        actions = decide_actions_from_proba(proba, threshold=0.5, max_action_rate=0.01)
        assert actions.sum() == 0

    def test_eligible_leq_k_no_truncation(self):
        proba = np.array([0.1, 0.2, 0.8])
        actions = decide_actions_from_proba(proba, threshold=0.7, max_action_rate=0.5)
        # Only 1 eligible, k=floor(0.5*3)=1, eligible<=k → no truncation
        assert actions.tolist() == [0, 0, 1]

    def test_no_eligible_after_ranking_filter(self):
        proba = np.array([0.6, 0.7])
        ranking = np.array([-1.0, -2.0])
        actions = decide_actions_from_proba(
            proba, threshold=0.5, max_action_rate=0.5, ranking_scores=ranking
        )
        assert actions.sum() == 0


# ── apply / apply_policy_to_proba wrappers ────────────────────────────


def _simple_policy(**overrides) -> DecisionPolicy:
    defaults = dict(
        selected_model="m",
        selected_model_artifact=None,
        threshold=0.5,
        max_action_rate=None,
        expected_net_profit=None,
        raw={},
    )
    defaults.update(overrides)
    return DecisionPolicy(**defaults)


class TestApplyWrappers:
    def test_apply_policy_to_proba_delegates(self):
        result = apply_policy_to_proba(np.array([0.3, 0.7]), _simple_policy())
        assert result.tolist() == [0, 1]

    def test_apply_alias_same_result(self):
        proba = np.array([0.3, 0.7])
        policy = _simple_policy()
        assert apply(proba, policy).tolist() == apply_policy_to_proba(proba, policy).tolist()

    def test_apply_with_max_action_rate(self):
        proba = np.array([0.9, 0.8, 0.7, 0.6, 0.55])
        policy = _simple_policy(threshold=0.5, max_action_rate=0.4)
        actions = apply(proba, policy)
        assert actions.sum() == 2


# ── compute_incremental_profit_scores ─────────────────────────────────


class TestIncrementalProfitScores:
    def test_returns_none_for_proba_mode(self):
        policy = _simple_policy(raw={"ranking_mode": "proba"})
        result = compute_incremental_profit_scores(
            pd.DataFrame({"x": [1]}), np.array([0.5]), policy
        )
        assert result is None

    def test_returns_none_when_ranking_mode_absent(self):
        policy = _simple_policy(raw={})
        result = compute_incremental_profit_scores(
            pd.DataFrame({"x": [1]}), np.array([0.5]), policy
        )
        assert result is None

    def test_computes_scores_without_segment(self):
        policy = _simple_policy(
            raw={
                "ranking_mode": "incremental_profit",
                "uplift": {
                    "default_tp_value": 100.0,
                    "fp_value": -10.0,
                    "fn_value": -50.0,
                    "tn_value": 0.0,
                },
            }
        )
        scores = compute_incremental_profit_scores(
            pd.DataFrame({"x": [1, 2]}), np.array([0.8, 0.2]), policy
        )
        assert scores is not None
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_segment_column_varies_tp_value(self):
        policy = _simple_policy(
            raw={
                "ranking_mode": "incremental_profit",
                "uplift": {
                    "segment_col": "seg",
                    "tp_value_by_segment": {"A": 200.0, "B": 50.0},
                    "default_tp_value": 100.0,
                    "fp_value": -10.0,
                    "fn_value": -50.0,
                    "tn_value": 0.0,
                },
            }
        )
        scores = compute_incremental_profit_scores(
            pd.DataFrame({"seg": ["A", "B"]}), np.array([0.5, 0.5]), policy
        )
        assert scores[0] > scores[1]

    def test_missing_segment_column_raises(self):
        policy = _simple_policy(
            raw={
                "ranking_mode": "incremental_profit",
                "uplift": {"segment_col": "missing_col"},
            }
        )
        with pytest.raises(ValueError, match="segment column"):
            compute_incremental_profit_scores(
                pd.DataFrame({"x": [1]}), np.array([0.5]), policy
            )

    def test_fallback_to_cost_matrix_values(self):
        policy = _simple_policy(
            raw={
                "ranking_mode": "incremental_profit",
                "cost_matrix": {
                    "tp_value": 80.0,
                    "fp_value": -5.0,
                    "fn_value": -40.0,
                    "tn_value": 0.0,
                },
                "uplift": {},
            }
        )
        scores = compute_incremental_profit_scores(
            pd.DataFrame({"x": [1]}), np.array([0.9]), policy
        )
        assert scores is not None
        assert scores[0] != 0.0
