"""Probability calibration utilities (modern sklearn API)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss

from .utils import get_logger

logger = get_logger("calibration")

CalibMethod = Literal["sigmoid", "isotonic"]


@dataclass(frozen=True)
class CalibrationResult:
    """
    Calibration çıktıları tek bir yerde toplansın diye dataclass.

    Neden?
    - Model + metrikler birlikte taşınır.
    - Raporlama/CI karşılaştırması kolaylaşır.
    """
    calibrated_model: CalibratedClassifierCV
    metrics: Dict[str, Any]


def _reliability_table(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> list[Dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[Dict[str, Any]] = []
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == bins - 1:
            mask = (proba >= lo) & (proba <= hi)
        else:
            mask = (proba >= lo) & (proba < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin": i,
                    "left": lo,
                    "right": hi,
                    "count": 0,
                    "avg_pred": None,
                    "empirical_rate": None,
                }
            )
            continue

        rows.append(
            {
                "bin": i,
                "left": lo,
                "right": hi,
                "count": n,
                "avg_pred": float(np.mean(proba[mask])),
                "empirical_rate": float(np.mean(y_true[mask])),
            }
        )
    return rows


def calibrate_frozen_classifier(
    fitted_model,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    method: CalibMethod,
) -> CalibrationResult:
    """
    Fit edilmiş bir sınıflandırıcıyı calibration set ile kalibre eder.

    Parametreler:
    - fitted_model: predict_proba olan ve fit edilmiş model (bizde Pipeline)
    - X_cal, y_cal: calibration set
    - method: "sigmoid" (Platt) veya "isotonic"

    FrozenEstimator ile modern yaklaşım:
    - Ana model yeniden fit edilmez.
    - Calibration layer öğrenilir.
    """
    if method not in ("sigmoid", "isotonic"):
        raise ValueError("method must be 'sigmoid' or 'isotonic'")

    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(fitted_model),
        method=method,
    )

    # Burada fit edilen şey "calibration layer"dır.
    calibrator.fit(X_cal, y_cal)

    # predict_proba ile olasılıkları al
    proba = calibrator.predict_proba(X_cal)[:, 1]

    # Clip neden var?
    # - log_loss, proba tam 0 veya tam 1 olursa (log(0)) numerik sorun yaşayabilir.
    # - Bazı sklearn sürümlerinde eps parametresi yok; bu yüzden clip ile güvenli hale getiriyoruz.
    proba = np.clip(proba, 1e-15, 1 - 1e-15)

    metrics = {
        "calibration_method": method,
        "brier": float(brier_score_loss(y_cal, proba)),
        "log_loss": float(log_loss(y_cal, proba)),
        "n_cal": int(len(y_cal)),
        "positive_rate_cal": float(np.mean(y_cal)),
        "reliability_bins": _reliability_table(y_cal, proba, bins=10),
    }

    logger.info(
        f"Calibration ({method}) done | "
        f"Brier={metrics['brier']:.6f} | LogLoss={metrics['log_loss']:.6f} | "
        f"n={metrics['n_cal']} pos_rate={metrics['positive_rate_cal']:.4f}"
    )

    return CalibrationResult(calibrated_model=calibrator, metrics=metrics)


def calibrate_prefit_classifier(
    fitted_model,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    method: CalibMethod,
) -> CalibrationResult:
    """Backward-compatible wrapper."""
    return calibrate_frozen_classifier(
        fitted_model=fitted_model,
        X_cal=X_cal,
        y_cal=y_cal,
        method=method,
    )
