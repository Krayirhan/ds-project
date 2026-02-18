"""CLI: preprocess command."""

from __future__ import annotations

from datetime import datetime
import json

from ..config import ExperimentConfig, Paths
from ..data_validation import (
    validate_raw_data,
    validate_processed_data,
    generate_reference_stats,
    detect_row_anomalies,
    detect_duplicates,
    check_data_staleness,
    validate_data_volume,
    get_schema_fingerprint,
    generate_reference_categories,
    generate_reference_correlations,
    run_validation_profile,
)
from ..features import infer_feature_spec
from ..io import read_csv, write_parquet
from ..preprocess import preprocess_basic
from ..utils import get_logger, sha256_file
from ..validate import basic_schema_checks, null_ratio_report, validate_target_labels
from ._helpers import json_write

logger = get_logger("cli.preprocess")


def cmd_preprocess(paths: Paths, cfg: ExperimentConfig) -> None:
    raw_path = paths.data_raw / "hotel_bookings.csv"
    df = read_csv(raw_path)

    # ── Katman 0: Veri tazeliği (staleness) ──
    staleness = check_data_staleness(
        str(raw_path), max_age_days=cfg.validation.max_staleness_days
    )
    if staleness.is_stale and cfg.validation.block_on_stale_data:
        raise ValueError(f"Stale data blocked by policy: {staleness.summary}")

    # ── Katman 1: Temel kontroller (hızlı fail) ──
    basic_schema_checks(df, cfg.target_col)
    validate_target_labels(df, cfg.target_col, allowed=set(cfg.label_map.keys()))
    null_ratio_report(df, top_k=8)

    # ── Katman 1b: Duplicate & row anomaly tespiti ──
    dup_report = detect_duplicates(df)
    anomaly_report = detect_row_anomalies(df)
    dup_ratio = dup_report.n_duplicates / max(len(df), 1)
    if dup_ratio > cfg.validation.duplicate_ratio_threshold:
        msg = f"Duplicate ratio {dup_ratio:.2%} > threshold {cfg.validation.duplicate_ratio_threshold:.2%}"
        if cfg.validation.block_on_duplicate:
            raise ValueError(f"Duplicate check blocked by policy: {msg}")
        else:
            logger.warning(f"⚠️  {msg} (warn-only policy)")

    # ── Katman 1c: Schema parmak izi ──
    schema_fp = get_schema_fingerprint(df, include_stats=True)

    # ── Katman 2: Pandera şema doğrulaması (tip, aralık, nullable) ──
    logger.info("Running Pandera raw data schema validation...")
    validate_raw_data(
        df, target_col=cfg.target_col,
        raise_on_error=cfg.validation.block_on_raw_schema_error,
    )

    df = preprocess_basic(
        df=df,
        target_col=cfg.target_col,
        label_map=cfg.label_map,
        drop_cols=list(cfg.leakage_cols),
        extra_blocked_cols=(
            list(cfg.blocked_feature_cols) if cfg.blocked_feature_cols else None
        ),
    )

    # ── Katman 3: İşlenmiş veri doğrulaması ──
    spec = infer_feature_spec(df, cfg.target_col)
    logger.info("Running Pandera processed data schema validation...")
    validate_processed_data(
        df,
        target_col=cfg.target_col,
        numeric_cols=spec.numeric,
        categorical_cols=spec.categorical,
        raise_on_error=cfg.validation.block_on_processed_schema_error,
    )

    # ── Katman 3b: ValidationProfile (policy-aware, tek nokta) ──
    profile = run_validation_profile(
        df,
        target_col=cfg.target_col,
        numeric_cols=spec.numeric,
        categorical_cols=spec.categorical,
        policy=cfg.validation,
        phase="preprocess",
    )
    if not profile.passed:
        raise ValueError(
            f"Validation profile FAILED [preprocess]: blocked_by={profile.blocked_by}"
        )

    # ── Katman 4: Referans istatistikleri üret (drift kontrolü için) ──
    ref_stats = generate_reference_stats(df, numeric_cols=spec.numeric)
    json_write(paths.reports_metrics / "reference_stats.json", ref_stats)
    logger.info(
        f"Reference stats generated for {len(ref_stats)} numeric columns"
    )

    # ── Katman 5: Referans kategoriler (unseen category kontrolü için) ──
    ref_cats = generate_reference_categories(df, categorical_cols=spec.categorical)
    json_write(paths.reports_metrics / "reference_categories.json", ref_cats)

    # ── Katman 6: Referans korelasyonlar (cross-feature drift için) ──
    ref_corr = generate_reference_correlations(
        df, numeric_cols=spec.numeric, target_col=cfg.target_col, top_k=15
    )
    json_write(paths.reports_metrics / "reference_correlations.json", ref_corr)

    # ── Katman 7: Veri hacmi kontrolü ──
    expected_rows = len(df)
    lineage_path = paths.reports_metrics / "data_lineage_preprocess.json"
    if lineage_path.exists():
        try:
            prev = json.loads(lineage_path.read_text(encoding="utf-8"))
            prev_rows = int(prev.get("processed_rows", 0) or 0)
            if prev_rows > 0:
                expected_rows = prev_rows
        except Exception:
            pass
    vol_report = validate_data_volume(df, expected_rows=expected_rows)

    # ── Katman 8: Şema kontratı artefaktı (version + fingerprint) ──
    processed_schema_fp = get_schema_fingerprint(df, include_stats=False)
    schema_contract = {
        "schema_version": cfg.contract.feature_schema_version,
        "raw_schema_fingerprint": schema_fp.get("fingerprint"),
        "processed_schema_fingerprint": processed_schema_fp.get("fingerprint"),
        "raw_columns": int(schema_fp.get("n_columns", 0)),
        "processed_columns": int(processed_schema_fp.get("n_columns", 0)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    json_write(paths.reports_metrics / "schema_contract.json", schema_contract)

    out_path = paths.data_processed / "dataset.parquet"
    write_parquet(df, out_path)
    json_write(
        paths.reports_metrics / "data_lineage_preprocess.json",
        {
            "raw_dataset": str(raw_path),
            "raw_dataset_sha256": (
                sha256_file(str(raw_path)) if raw_path.exists() else None
            ),
            "processed_dataset": str(out_path),
            "processed_rows": int(len(df)),
            "processed_columns": int(df.shape[1]),
            "schema_fingerprint": schema_fp.get("fingerprint"),
            "duplicates": dup_report.n_duplicates,
            "row_anomalies": anomaly_report.n_anomalies,
            "data_staleness_days": staleness.age_days,
            "data_volume_expected_rows": int(expected_rows),
            "data_volume_current_rows": int(vol_report.current_rows),
            "data_volume_expected_range": list(vol_report.expected_range),
            "data_volume_is_anomalous": bool(vol_report.is_anomalous),
            "schema_version": cfg.contract.feature_schema_version,
            "processed_schema_fingerprint": processed_schema_fp.get("fingerprint"),
            "validation_layers": [
                "check_data_staleness",
                "basic_schema_checks",
                "validate_target_labels",
                "null_ratio_report",
                "detect_duplicates",
                "detect_row_anomalies",
                "schema_fingerprint",
                "pandera_raw_schema",
                "pandera_processed_schema",
                "assert_no_nans_after_imputation",
                "reference_stats_generation",
                "reference_categories_generation",
                "reference_correlations_generation",
                "validate_data_volume",
            ],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
