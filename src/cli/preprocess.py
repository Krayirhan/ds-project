"""CLI: preprocess command."""

from __future__ import annotations

from datetime import datetime

from ..config import ExperimentConfig, Paths
from ..io import read_csv, write_parquet
from ..preprocess import preprocess_basic
from ..utils import get_logger, sha256_file
from ..validate import basic_schema_checks, null_ratio_report, validate_target_labels
from ._helpers import json_write

logger = get_logger("cli.preprocess")


def cmd_preprocess(paths: Paths, cfg: ExperimentConfig) -> None:
    raw_path = paths.data_raw / "hotel_bookings.csv"
    df = read_csv(raw_path)

    basic_schema_checks(df, cfg.target_col)
    validate_target_labels(df, cfg.target_col, allowed=set(cfg.label_map.keys()))
    null_ratio_report(df, top_k=8)

    df = preprocess_basic(
        df=df,
        target_col=cfg.target_col,
        label_map=cfg.label_map,
        drop_cols=list(cfg.leakage_cols),
        extra_blocked_cols=(
            list(cfg.blocked_feature_cols) if cfg.blocked_feature_cols else None
        ),
    )

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
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
