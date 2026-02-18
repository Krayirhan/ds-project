"""
evaluate.py

Test set üzerinde metrik üretir ve JSON olarak kaydeder.

Neden JSON?
- Notebook/rapor bu dosyayı okuyup görselleştirir.
- CI/CD metrik kıyaslaması mümkündür.
- Kurumsal projelerde metrikler "artifact" olarak saklanır.

Bu dosyada 2 ana yaklaşım var:
1) evaluate_binary_classifier:
   - Tek bir threshold ile metrik üretir.
   - Baseline (0.50) ve decision threshold (0.35) gibi senaryolarda kullanılır.

2) sweep_thresholds:
   - Threshold taraması yapar.
   - "best_by_f1" üretir (genel denge).
   - Ayrıca "best_by_rule" üretir:
     Kurumsal pratikte threshold seçimi genelde bir iş kuralına dayanır.
     Örn:
       - "Recall >= 0.80 olsun, bu şart altında precision maksimum olsun"
       - "Precision >= 0.75 olsun, bu şart altında recall maksimum olsun"

Bu sayede threshold seçimi “hisse göre” değil, “politikaya göre” yapılır.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from .utils import get_logger

logger = get_logger("evaluate")


def evaluate_binary_classifier(
    model,
    df_test: pd.DataFrame,
    target_col: str,
    out_path: Path,
    threshold: float = 0.5,
    tag: str = "baseline",
) -> Dict[str, Any]:
    """
    Tek threshold ile metrik üretir.

    threshold neden parametre?
    - AUC threshold bağımsızdır.
    - F1/Precision/Recall threshold'a bağlıdır.
    - Ürün hedefi threshold seçimini belirler (kaçırma vs yanlış alarm).

    tag neden var?
    - Aynı fonksiyonu farklı threshold politikaları için tekrar kullanırız:
      - baseline_0.50
      - decision_0.35
      - vb.
    """
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].astype(int).values

    # predict_proba neden kullanıyoruz?
    # - AUC hesaplamak için olasılık gerekir.
    # - Threshold sweep yapmak için olasılık gerekir.
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "tag": tag,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "threshold": float(threshold),
        "n_test": int(len(y_test)),
        "positive_rate_test": float(np.mean(y_test)),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info(f"Saved metrics ({tag}) -> {out_path}")
    logger.info(
        f"[{tag}] ROC-AUC={metrics['roc_auc']:.4f} | F1={metrics['f1']:.4f} | "
        f"P={metrics['precision']:.4f} | R={metrics['recall']:.4f} | thr={metrics['threshold']:.2f}"
    )
    return metrics


RuleMode = Literal["maximize_precision_given_recall", "maximize_recall_given_precision"]


def sweep_thresholds(
    model,
    df_test: pd.DataFrame,
    target_col: str,
    out_path: Path,
    thresholds: np.ndarray = None,
    # ✅ Kurumsal threshold policy:
    # "minimum precision" veya "minimum recall" şartı koyup diğerini maximize edebiliriz.
    rule_mode: RuleMode = "maximize_precision_given_recall",
    min_recall: Optional[float] = 0.80,
    min_precision: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Threshold sweep (tarama) + kural bazlı seçim.

    Neden kurumsal?
    - Kurumlar threshold'ı genelde "iş kuralı" ile seçer.
      Örn:
        - "iptal kaçmasın" -> recall >= 0.80 olsun
        - bunun altında precision'ı olabildiğince yüksek tut

    rule_mode:
    - maximize_precision_given_recall:
        recall >= min_recall şartı altında precision maksimum olan threshold seçilir
    - maximize_recall_given_precision:
        precision >= min_precision şartı altında recall maksimum olan threshold seçilir

    Bu fonksiyon şunları üretir:
    - best_by_f1: (genel denge) en iyi F1 threshold
    - best_by_rule: (iş kuralı) policy'ye göre en iyi threshold
    - rows: tüm threshold sonuçları
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    if rule_mode == "maximize_precision_given_recall" and min_recall is None:
        raise ValueError(
            "rule_mode='maximize_precision_given_recall' requires min_recall"
        )
    if rule_mode == "maximize_recall_given_precision" and min_precision is None:
        raise ValueError(
            "rule_mode='maximize_recall_given_precision' requires min_precision"
        )

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].astype(int).values

    proba = model.predict_proba(X_test)[:, 1]

    rows = []

    # 1) Best-by-F1: genel denge
    best_f1 = {"threshold": None, "f1": -1.0, "precision": None, "recall": None}

    # 2) Best-by-rule: iş kuralına göre seçim
    # Başlangıç değerleri rule'a göre belirlenir
    best_rule = {"threshold": None, "f1": None, "precision": None, "recall": None}

    for t in thresholds:
        pred = (proba >= t).astype(int)

        f1 = float(f1_score(y_test, pred))
        p = float(precision_score(y_test, pred, zero_division=0))
        r = float(recall_score(y_test, pred, zero_division=0))

        rows.append({"threshold": float(t), "f1": f1, "precision": p, "recall": r})

        # Best by F1
        if f1 > best_f1["f1"]:
            best_f1 = {"threshold": float(t), "f1": f1, "precision": p, "recall": r}

        # Best by rule
        if rule_mode == "maximize_precision_given_recall":
            # Şart: recall >= min_recall
            if r >= float(min_recall):
                # Bu şart altında precision maksimize edilir
                if best_rule["precision"] is None or p > best_rule["precision"]:
                    best_rule = {
                        "threshold": float(t),
                        "f1": f1,
                        "precision": p,
                        "recall": r,
                    }

        elif rule_mode == "maximize_recall_given_precision":
            # Şart: precision >= min_precision
            if p >= float(min_precision):
                # Bu şart altında recall maksimize edilir
                if best_rule["recall"] is None or r > best_rule["recall"]:
                    best_rule = {
                        "threshold": float(t),
                        "f1": f1,
                        "precision": p,
                        "recall": r,
                    }

    result = {
        "best_by_f1": best_f1,
        "best_by_rule": {
            "rule_mode": rule_mode,
            "min_recall": min_recall,
            "min_precision": min_precision,
            "selection": best_rule,
        },
        "rows": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    logger.info(f"Saved threshold sweep -> {out_path}")
    logger.info(
        f"Best threshold by F1 -> t={best_f1['threshold']} f1={best_f1['f1']:.4f} "
        f"p={best_f1['precision']:.4f} r={best_f1['recall']:.4f}"
    )

    # Kural bazlı seçim log'u (kurumsal karar)
    if best_rule["threshold"] is None:
        logger.info(
            "Best threshold by RULE -> No threshold satisfied the rule constraints. "
            "Consider relaxing min_recall/min_precision or using best_by_f1."
        )
    else:
        logger.info(
            f"Best threshold by RULE ({rule_mode}) -> t={best_rule['threshold']} "
            f"f1={best_rule['f1']:.4f} p={best_rule['precision']:.4f} r={best_rule['recall']:.4f}"
        )

    return result
