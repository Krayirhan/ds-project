"""CLI: split command — persist train/cal/test splits to disk for reproducibility."""

from __future__ import annotations

from datetime import datetime

from ..config import ExperimentConfig, Paths
from ..io import read_parquet, write_parquet
from ..split import stratified_split
from ..utils import get_logger, sha256_file
from ._helpers import json_write

logger = get_logger("cli.split")

# Calibration set fraction (from training split)
_CAL_FRACTION = 0.20


def cmd_split(paths: Paths, cfg: ExperimentConfig) -> None:
    """
    dataset.parquet → train.parquet + cal.parquet + test.parquet

    Bölme stratejisi:
      1) dataset  →  train_full (%80)  +  test (%20)   [stratified]
      2) train_full  →  train (%80)  +  cal (%20)      [stratified]

    Sonuç oranları (dataset'e göre):
      train : %64   — model eğitimi
      cal   : %16   — calibration (Platt / Isotonic)
      test  : %20   — son değerlendirme
    """
    dataset_path = paths.data_processed / "dataset.parquet"
    df = read_parquet(dataset_path)

    # 1) Train+Cal vs Test
    train_full_df, test_df = stratified_split(
        df, cfg.target_col, cfg.test_size, cfg.seed
    )

    # 2) Train vs Calibration
    train_df, cal_df = stratified_split(
        train_full_df, cfg.target_col, test_size=_CAL_FRACTION, seed=cfg.seed
    )

    # Persist all splits
    train_path = paths.data_processed / "train.parquet"
    cal_path = paths.data_processed / "cal.parquet"
    test_path = paths.data_processed / "test.parquet"

    write_parquet(train_df, train_path)
    write_parquet(cal_df, cal_path)
    write_parquet(test_df, test_path)

    # Metadata for audit / lineage
    metadata = {
        "source": str(dataset_path),
        "source_sha256": sha256_file(str(dataset_path)),
        "seed": cfg.seed,
        "test_size": cfg.test_size,
        "cal_size": _CAL_FRACTION,
        "splits": {
            "train": {
                "path": str(train_path),
                "rows": len(train_df),
                "positive_rate": float(train_df[cfg.target_col].mean()),
            },
            "cal": {
                "path": str(cal_path),
                "rows": len(cal_df),
                "positive_rate": float(cal_df[cfg.target_col].mean()),
            },
            "test": {
                "path": str(test_path),
                "rows": len(test_df),
                "positive_rate": float(test_df[cfg.target_col].mean()),
            },
        },
        "total_rows": len(df),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    json_write(paths.reports_metrics / "split_metadata.json", metadata)

    logger.info(
        f"Split done | train={len(train_df)} cal={len(cal_df)} test={len(test_df)} "
        f"(total={len(df)}, seed={cfg.seed})"
    )
