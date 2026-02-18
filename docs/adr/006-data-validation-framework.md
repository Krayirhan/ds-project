# ADR-006: Data Validation with Pandera

## Status
Accepted — 2026-02-18

## Context
Data quality issues are the #1 cause of silent model degradation:

- Column renames in upstream ETL
- Label encoding changes (`yes/no` → `0/1` → `true/false`)
- Distribution shift (seasonal patterns, new hotel properties)
- Missing value patterns changing
- Numeric overflow or type coercion bugs

The existing `validate.py` provides basic checks (empty dataset, target column exists),
but lacks:
- Schema enforcement (column types, value ranges)
- Distribution assertions
- Composable, declarative validation rules

## Decision

### Framework: Pandera
Chosen over Great Expectations because:
- Pandera is lightweight, pandas-native, and pip-installable
- Schema defined as Python code (type-safe, IDE-friendly)
- Lazy validation (collect all errors, not just first)
- Better fit for our dataclass-based config style

### Validation Layers

```
┌─────────────────────────────────────────┐
│ Layer 1: Raw Schema (build_raw_schema)  │
│ - Column existence + types              │
│ - Value range checks (lead_time ≥ 0)    │
│ - Target label format (0/1/yes/no)      │
│ - Hotel type enum validation            │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Layer 2: Processed Schema               │
│ - Target is int 0/1                     │
│ - Numeric columns are finite            │
│ - No unexpected nulls                   │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Layer 3: Inference Payload              │
│ - Feature spec alignment               │
│ - Coercion-safe types                   │
└─────────────────┬───────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ Layer 4: Distribution Assertions        │
│ - Mean drift (±3σ from reference)       │
│ - Range sanity checks                   │
│ - Missing column detection              │
└─────────────────────────────────────────┘
```

### Integration Points
- **preprocess stage**: `validate_raw_data()` before any transformation
- **post-preprocess**: `validate_processed_data()` before split
- **API inference**: `validate_inference_payload()` in middleware
- **monitoring**: `validate_distributions()` for drift detection

### Reference Stats
`generate_reference_stats()` computes training-set statistics (mean, std, min, max,
median, quantiles) and persists as JSON. Distribution checks compare incoming data
against these references.

## Consequences

### Positive
- Fail-fast on data contract violations (no silent corruption)
- Declarative schemas are self-documenting
- Lazy validation collects all errors in one pass
- Distribution checks complement PSI-based drift monitoring

### Negative
- Schema maintenance burden (must update when adding features)
- Pandera adds a runtime dependency
- Distribution tolerance (`3σ`) may need per-column tuning
