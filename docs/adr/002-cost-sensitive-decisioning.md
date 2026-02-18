# ADR-002: Cost-Sensitive Decision Framework

## Status
Accepted — 2026-02-16

## Context
The hotel cancellation prediction model outputs probabilities, but the business needs *actions*
(e.g., send retention offer, pre-authorize payment). The mapping from probability to action
must consider:

1. **Asymmetric costs**: Missing a cancellation (FN) is far more expensive than a false alarm (FP)
2. **Capacity constraints**: Operations can only act on X% of bookings
3. **Segment-specific value**: Contract guests have different value than transient guests

A fixed threshold (e.g., 0.5) ignores all of these.

## Decision

### Cost Matrix Approach
```
             Predicted: Action    Predicted: No Action
Actual: Cancel     TP (+180)          FN (-200)
Actual: No Cancel  FP (-20)           TN (0)
```

Values are configurable via `ExperimentConfig.cost` (not hardcoded).

### Decision Policy Engine
1. For each candidate model+calibration, sweep thresholds
2. At each threshold, compute expected profit = TP×tp_value + FP×fp_value + FN×fn_value
3. Apply action rate constraint (max % of population receiving action)
4. Select threshold+model combination maximizing constrained profit
5. Persist as `decision_policy.json` artifact

### Ranking Modes
- `incremental_profit` (default): Rank by marginal profit contribution
- `proba`: Simple probability ranking with threshold

### Policy Artifact
```json
{
  "policy_version": "1.0.0",
  "selected_model": "challenger_xgboost_calibrated_sigmoid",
  "selected_model_artifact": "models/20260217.../challenger_xgboost_calibrated_sigmoid.joblib",
  "optimal_threshold": 0.42,
  "max_action_rate": 0.15,
  "expected_profit_per_1000": 12450.0
}
```

## Consequences

### Positive
- Business-aligned model selection (profit, not just AUC)
- Action rate constraints prevent operational overload
- Threshold is data-driven, not arbitrary
- Policy is a versioned, auditable artifact

### Negative
- Cost matrix values require business stakeholder alignment
- Segment-specific costs add complexity
- Profit estimates depend on calibration quality
