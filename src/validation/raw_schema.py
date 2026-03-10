"""Deprecated alias for schema-focused validators.

Use ``src.validation.schema`` (or ``src.validation``) for new imports.
"""

from __future__ import annotations

import warnings

from .schema import (
    basic_schema_checks,
    build_inference_schema,
    build_processed_schema,
    build_raw_schema,
    generate_reference_stats,
    get_schema_fingerprint,
    null_ratio_report,
    validate_inference_payload,
    validate_processed_data,
    validate_raw_data,
    validate_target_labels,
)

warnings.warn(
    "`src.validation.raw_schema` is deprecated; use `src.validation.schema`.",
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
]
