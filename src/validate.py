"""Deprecated alias for lightweight schema checks.

Use ``src.validation`` (official) or ``src.validation.schema`` for new imports.
"""

from __future__ import annotations

import warnings

from .validation import basic_schema_checks, null_ratio_report, validate_target_labels

warnings.warn(
    "`src.validate` is deprecated; use `src.validation` or `src.validation.schema`.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["basic_schema_checks", "validate_target_labels", "null_ratio_report"]
