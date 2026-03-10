"""Deprecated compatibility facade for validation utilities.

Official import surface:
    from src.validation import ...

This module remains for backward compatibility and re-exports all validation
symbols from the physically split modules:
    - src.validation.schema
    - src.validation.drift
    - src.validation.anomaly
"""

from __future__ import annotations

import warnings

from .validation import (
    AnomalyReport,
    CardinalityReport,
    CorrelationDriftReport,
    DistributionReport,
    DuplicateReport,
    ImportanceDriftReport,
    LabelDriftReport,
    OutputValidationReport,
    PSIReport,
    SkewReport,
    StalenessReport,
    ValidationProfileReport,
    VolumeReport,
    _js_divergence,
    _psi_score,
    assert_no_nans_after_imputation,
    basic_schema_checks,
    build_inference_schema,
    build_processed_schema,
    build_raw_schema,
    check_data_staleness,
    compute_psi,
    detect_correlation_drift,
    detect_duplicates,
    detect_feature_importance_drift,
    detect_label_drift,
    detect_row_anomalies,
    detect_training_serving_skew,
    detect_unseen_categories,
    generate_reference_categories,
    generate_reference_correlations,
    generate_reference_stats,
    get_schema_fingerprint,
    null_ratio_report,
    run_validation_profile,
    validate_data_volume,
    validate_distributions,
    validate_inference_payload,
    validate_model_output,
    validate_processed_data,
    validate_raw_data,
    validate_row_counts,
    validate_target_labels,
)

warnings.warn(
    "`src.data_validation` is deprecated; use `src.validation` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "build_raw_schema",
    "build_processed_schema",
    "build_inference_schema",
    "validate_raw_data",
    "validate_processed_data",
    "validate_inference_payload",
    "generate_reference_stats",
    "get_schema_fingerprint",
    "basic_schema_checks",
    "validate_target_labels",
    "null_ratio_report",
    "DistributionReport",
    "validate_distributions",
    "LabelDriftReport",
    "detect_label_drift",
    "CorrelationDriftReport",
    "detect_correlation_drift",
    "generate_reference_correlations",
    "SkewReport",
    "detect_training_serving_skew",
    "ImportanceDriftReport",
    "detect_feature_importance_drift",
    "generate_reference_categories",
    "PSIReport",
    "_psi_score",
    "_js_divergence",
    "compute_psi",
    "ValidationProfileReport",
    "run_validation_profile",
    "AnomalyReport",
    "detect_row_anomalies",
    "DuplicateReport",
    "detect_duplicates",
    "assert_no_nans_after_imputation",
    "CardinalityReport",
    "detect_unseen_categories",
    "OutputValidationReport",
    "validate_model_output",
    "VolumeReport",
    "validate_data_volume",
    "StalenessReport",
    "check_data_staleness",
    "validate_row_counts",
]
