import numpy as np

from src.policy import decide_actions_from_proba


def test_threshold_only_policy():
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    actions = decide_actions_from_proba(proba, threshold=0.5, max_action_rate=None)
    assert actions.tolist() == [0, 0, 1, 1]


def test_top_k_constraint_applies():
    proba = np.array([0.95, 0.85, 0.7, 0.6, 0.55])
    actions = decide_actions_from_proba(proba, threshold=0.5, max_action_rate=0.4)
    # k=floor(0.4*5)=2
    assert actions.sum() == 2
    assert actions.tolist() == [1, 1, 0, 0, 0]


def test_ranking_scores_override_proba_for_top_k():
    proba = np.array([0.9, 0.8, 0.7, 0.6])
    ranking = np.array([0.1, 0.2, 0.9, 0.8])
    actions = decide_actions_from_proba(
        proba,
        threshold=0.5,
        max_action_rate=0.5,
        ranking_scores=ranking,
    )
    # k=2, highest ranking scores are idx 2 and 3
    assert actions.tolist() == [0, 0, 1, 1]
