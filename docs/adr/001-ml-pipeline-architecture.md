# ADR-001: ML Pipeline Architecture

## Status
Accepted — 2026-02-16

## Context
We need a reproducible, auditable ML pipeline for hotel cancellation prediction.
The pipeline must support:
- Multiple model families (baseline + challengers)
- Calibration for reliable probability estimates
- Cost-sensitive evaluation aligned with business metrics
- Batch and real-time inference modes

Key constraints:
- Data team is small (2-3 people), so operational simplicity is critical.
- The model serves a downstream decisioning layer, so calibration quality directly affects business outcomes.
- Regulatory/audit requirements mandate reproducibility.

## Decision

### Pipeline Stages

```
Raw CSV → preprocess → split → train → evaluate → predict → monitor
                                 ↓
                          calibrate (isotonic + sigmoid)
                                 ↓
                          select champion (cost-optimal)
                                 ↓
                          decision policy (JSON artifact)
```

### Technology Choices
- **Orchestration**: DVC pipelines (`dvc.yaml`) for reproducibility + caching
- **Training**: scikit-learn Pipeline + ColumnTransformer for training-serving parity
- **Models**: LogisticRegression (baseline) + XGBoost, LightGBM, CatBoost, HistGradientBoosting (challengers)
- **Calibration**: `CalibratedClassifierCV` with both isotonic and sigmoid methods
- **Artifacts**: joblib serialization, per-run timestamped directories under `models/`
- **Feature Spec**: JSON contract (`feature_spec.json`) with schema versioning

### Separation of Concerns
- `src/preprocess.py` — data cleaning, leakage removal, label normalization
- `src/features.py` — feature engineering (ColumnTransformer)
- `src/train.py` — model training + calibration
- `src/evaluate.py` — cost-matrix evaluation + champion selection
- `src/predict.py` — inference with policy application
- `src/monitoring.py` — PSI drift detection + outcome monitoring

## Consequences

### Positive
- Full reproducibility via DVC
- Training-serving parity guaranteed by sklearn Pipeline
- Multiple calibration methods compared automatically
- Cost-based champion selection aligns with business value

### Negative
- DVC adds operational overhead for small experiments
- joblib artifacts not portable across Python/sklearn versions
- Single-machine training (no distributed training support)

### Risks
- Feature spec drift between training and serving if not enforced
- Model artifact size may grow with ensemble methods
