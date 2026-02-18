"""
cost_matrix.py

Amaç:
- Threshold seçimini metriklerden (F1/Recall/Precision) çıkarıp
  "gerçek iş etkisine" (net kazanç / net maliyet) bağlamak.

Kurumsal gerçek:
- En yüksek profit threshold bazen aşırı düşük çıkar (örn 0.05).
- Bunun nedeni çoğu zaman:
  1) FN maliyeti çok yüksek, FP maliyeti düşük tanımlanmıştır
  2) Aksiyon kapasitesi/bütçesi constraint'i yoktur

Bu dosyada 2 mod var:

1) sweep_thresholds_for_profit:
   - Constraint YOK (sınırsız aksiyon varsayar)

2) sweep_thresholds_for_profit_with_constraint:
   - ✅ Constraint VAR:
     "max_action_rate" ile aksiyon oranını sınırlarız.
   - ✅ Kurumsal iyileştirme:
     Grid ile feasible bulunamazsa quantile tabanlı threshold hesaplarız.
     (Bu, "tam kapasiteyi dolduracak" eşiği direkt bulur.)

Neden quantile?
- Threshold gridini ne kadar inceltsen de bazı dağılımlarda kaçırabilirsin.
- Quantile ile doğrudan "en yüksek p'li %10" seçimini yaparsın.
- Bu, production campaign targeting'de çok yaygın bir yaklaşımdır.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from .utils import get_logger

logger = get_logger("cost_matrix")


@dataclass(frozen=True)
class CostMatrix:
    """
    Confusion matrix hücrelerinin parasal karşılığı.

    Positive class = 1 = "cancel" (iptal)
    Negative class = 0 = "not cancel"

    tp_value: TP başına net değer (kazanım - aksiyon maliyeti)
    fp_value: FP başına değer (genelde negatif: gereksiz aksiyon maliyeti + spam/itibar)
    fn_value: FN başına değer (genelde negatif: kaçırılan iptal maliyeti)
    tn_value: TN başına değer (çoğu senaryoda 0)
    """

    tp_value: float
    fp_value: float
    fn_value: float
    tn_value: float = 0.0


@dataclass(frozen=True)
class ProfitSweepResult:
    """
    Profit sweep çıktıları.
    """

    best_threshold: float
    best_profit: float
    rows: list[Dict[str, Any]]


def compute_profit_from_confusion(
    tn: int, fp: int, fn: int, tp: int, cost: CostMatrix
) -> float:
    """
    Profit = tp*tp_value + fp*fp_value + fn*fn_value + tn*tn_value
    """
    return (
        tp * cost.tp_value
        + fp * cost.fp_value
        + fn * cost.fn_value
        + tn * cost.tn_value
    )


def _default_threshold_grid() -> np.ndarray:
    """
    Default threshold grid (fine).

    Neden ince grid?
    - Constraint'li seçimlerde %10 gibi oranlara denk gelen threshold'ı yakalamak için gerekir.
    - 0.05 adım çok kaba kalıyor.

    0.001..0.999 arası 0.001 adım:
    - 999 threshold -> pratikte hızlı
    """
    return np.arange(0.001, 1.000, 0.001)


def sweep_thresholds_for_profit(
    model,
    df_test: pd.DataFrame,
    target_col: str,
    cost: CostMatrix,
    thresholds: Optional[np.ndarray] = None,
) -> ProfitSweepResult:
    """
    Constraint'siz profit sweep.
    """
    if thresholds is None:
        thresholds = _default_threshold_grid()

    X = df_test.drop(columns=[target_col])
    y = df_test[target_col].astype(int).values

    proba = model.predict_proba(X)[:, 1]

    best_t = None
    best_profit = -float("inf")
    rows: list[Dict[str, Any]] = []

    for t in thresholds:
        pred = (proba >= t).astype(int)

        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        profit = float(compute_profit_from_confusion(tn, fp, fn, tp, cost))
        action_rate = float(pred.mean())

        rows.append(
            {
                "threshold": float(t),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "action_rate": action_rate,
                "profit": profit,
            }
        )

        if profit > best_profit:
            best_profit = profit
            best_t = float(t)

    logger.info(
        f"Best threshold by PROFIT -> t={best_t} profit={best_profit:.2f} "
        f"(tp_value={cost.tp_value}, fp_value={cost.fp_value}, fn_value={cost.fn_value}, tn_value={cost.tn_value})"
    )

    return ProfitSweepResult(best_threshold=best_t, best_profit=best_profit, rows=rows)


def sweep_thresholds_for_profit_with_constraint(
    model,
    df_test: pd.DataFrame,
    target_col: str,
    cost: CostMatrix,
    max_action_rate: float,
    thresholds: Optional[np.ndarray] = None,
    use_quantile_fallback: bool = True,
) -> ProfitSweepResult:
    """
    ✅ Constraint'li profit sweep.

    max_action_rate:
    - "Aksiyon gönderebileceğim maksimum oran"
    - Örn 0.10 -> en fazla %10 müşteriye aksiyon

    thresholds:
    - None ise ince default grid kullanılır.

    use_quantile_fallback:
    - Grid ile feasible threshold bulunamazsa quantile ile threshold hesaplar.
    - Bu kurumsal olarak en stabil yöntemlerden biridir:
      "Top-K targeting" mantığı.
    """
    if not (0.0 < max_action_rate <= 1.0):
        raise ValueError("max_action_rate must be in (0, 1].")

    if thresholds is None:
        thresholds = _default_threshold_grid()

    X = df_test.drop(columns=[target_col])
    y = df_test[target_col].astype(int).values

    proba = model.predict_proba(X)[:, 1]

    best_t = None
    best_profit = -float("inf")
    rows: list[Dict[str, Any]] = []

    feasible_found = False

    for t in thresholds:
        pred = (proba >= t).astype(int)
        action_rate = float(pred.mean())

        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        profit = float(compute_profit_from_confusion(tn, fp, fn, tp, cost))

        rows.append(
            {
                "threshold": float(t),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "action_rate": action_rate,
                "profit": profit,
                "feasible": bool(action_rate <= max_action_rate),
                "selection_strategy": "grid",
            }
        )

        if action_rate <= max_action_rate:
            feasible_found = True
            if profit > best_profit:
                best_profit = profit
                best_t = float(t)

    if feasible_found:
        logger.info(
            f"Best threshold by PROFIT with CONSTRAINT (max_action_rate={max_action_rate:.2f}) -> "
            f"t={best_t} profit={best_profit:.2f}"
        )
        return ProfitSweepResult(
            best_threshold=best_t, best_profit=best_profit, rows=rows
        )

    # ------------------------------------------------------------
    # ✅ Quantile fallback (enterprise-grade)
    # ------------------------------------------------------------
    if use_quantile_fallback:
        # max_action_rate=0.10 ise en yüksek olasılığa sahip %10'u seçmek istiyoruz.
        # Bu, threshold'u proba dağılımının (1 - max_action_rate) quantile'ı olarak verir.
        q = float(1.0 - max_action_rate)
        t_q = float(np.quantile(proba, q))

        pred = (proba >= t_q).astype(int)
        action_rate = float(pred.mean())

        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        profit = float(compute_profit_from_confusion(tn, fp, fn, tp, cost))

        rows.append(
            {
                "threshold": t_q,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "action_rate": action_rate,
                "profit": profit,
                "feasible": bool(action_rate <= max_action_rate + 1e-12),
                "selection_strategy": "quantile_fallback",
                "quantile": q,
            }
        )

        logger.info(
            f"No feasible threshold found on grid for max_action_rate={max_action_rate:.2f}. "
            f"Using quantile fallback -> t={t_q:.6f} action_rate={action_rate:.4f} profit={profit:.2f}"
        )

        # Not:
        # - action_rate, tie'lar yüzünden %10'dan biraz fazla/az olabilir.
        # - Bu normaldir. İstersen tie-breaking ekleyebiliriz (stable top-k).
        return ProfitSweepResult(best_threshold=t_q, best_profit=profit, rows=rows)

    # Fallback kapalıysa: açık uyarı + NaN
    logger.info(
        f"No feasible threshold found for max_action_rate={max_action_rate:.2f}. "
        "Consider using finer thresholds or relaxing the constraint."
    )
    return ProfitSweepResult(
        best_threshold=float(thresholds.max()), best_profit=float("nan"), rows=rows
    )


def default_cost_matrix_example() -> CostMatrix:
    """
    Örnek cost matrix.

    UYARI:
    - Bu sadece başlangıç içindir.
    - Gerçek projede FP maliyeti genelde daha yüksektir (spam/itibar/limit).

    Varsayımsal:
    - TP: +180
    - FP: -20
    - FN: -200
    - TN: 0
    """
    return CostMatrix(tp_value=180.0, fp_value=-20.0, fn_value=-200.0, tn_value=0.0)
