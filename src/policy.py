"""
policy.py

Amaç:
- evaluate aşamasında üretilen reports/metrics/decision_policy.json dosyasını
  inference tarafında kullanmak.

Kurumsal prensip:
- Model decision threshold / capacity constraint "kodun içinde" olmamalı.
- Policy bir artefact'tır (model artefact + policy artefact birlikte deploy edilir).

Bu dosya 2 işi çözer:
1) policy dosyasını güvenli şekilde okumak
2) model probability'lerini aksiyon kararına çevirmek
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger("policy")


@dataclass(frozen=True)
class DecisionPolicy:
    """
    decision_policy.json içeriğinin typed temsili.

    threshold:
      - "proba >= threshold" olanlara aksiyon verilir (varsayılan kural)

    max_action_rate:
      - capacity/budget constraint
      - eğer None ise sadece threshold ile karar verilir

    selected_model_artifact:
      - hangi model dosyasının kullanılacağı bilgisi (ops için)
    """
    selected_model: str
    selected_model_artifact: Optional[str]
    threshold: float
    max_action_rate: Optional[float]
    expected_net_profit: Optional[float]
    raw: Dict[str, Any]


def load_decision_policy(policy_path: Path) -> DecisionPolicy:
    """
    Policy dosyasını okur ve temel validasyon yapar.

    Neden validasyon?
    - Production ortamında bozuk/eksik policy deploy edilirse
      sessizce yanlış karar vermek yerine fail-fast yapmak daha iyidir.
    """
    if not policy_path.exists():
        raise FileNotFoundError(f"Decision policy not found: {policy_path}")

    payload = json.loads(policy_path.read_text(encoding="utf-8"))

    if payload.get("status") != "ok":
        raise ValueError(f"Policy status is not ok. payload={payload}")

    threshold = float(payload["threshold"])
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"Invalid threshold: {threshold}. Must be in (0,1).")

    mar = payload.get("max_action_rate", None)
    if mar is not None:
        mar = float(mar)
        if not (0.0 < mar <= 1.0):
            raise ValueError(f"Invalid max_action_rate: {mar}. Must be in (0,1].")

    return DecisionPolicy(
        selected_model=str(payload.get("selected_model")),
        selected_model_artifact=payload.get("selected_model_artifact"),
        threshold=threshold,
        max_action_rate=mar,
        expected_net_profit=float(payload["expected_net_profit"]) if "expected_net_profit" in payload else None,
        raw=payload,
    )


def decide_actions_from_proba(
    proba: np.ndarray,
    threshold: float,
    max_action_rate: Optional[float] = None,
    ranking_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Probability -> action(0/1) kararı.

    1) Eğer max_action_rate yoksa:
       - klasik thresholding: proba >= threshold

    2) Eğer max_action_rate varsa:
       - Kurumsal gereksinim: kapasiteyi aşamazsın.
       - "threshold" + "capacity" aynı anda çakışabilir.
         Örn threshold çok düşük -> %60 aksiyon çıkar ama kapasite %30.
       - Bu durumda "en yüksek proba'lı" top-K seçilir.

    Neden top-K?
    - Campaign targeting'de en yaygın operasyonel yöntem.
    - Kapasite sabitken, en riskli müşterilere odaklanırsın.

    Not:
    - Ties (eşit olasılık) durumunda K tam tutmayabilir. Pratikte küçük sapma normaldir.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 1:
        raise ValueError("proba must be 1D array of positive-class probabilities.")

    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0,1).")

    # İlk adım: threshold kuralı
    actions = (proba >= threshold).astype(int)

    if ranking_scores is not None:
        ranking_scores = np.asarray(ranking_scores, dtype=float)
        if ranking_scores.shape != proba.shape:
            raise ValueError("ranking_scores must have same shape as proba")
        # negatif expected incremental profit varsa aksiyon verme
        actions = (actions.astype(bool) & (ranking_scores > 0.0)).astype(int)

    if max_action_rate is None:
        return actions

    if not (0.0 < max_action_rate <= 1.0):
        raise ValueError("max_action_rate must be in (0,1].")

    # Kapasite kontrolü
    current_rate = float(actions.mean())
    if current_rate <= max_action_rate:
        # Threshold zaten kapasiteyi aşmıyor → doğrudan kullan
        return actions

    # Kapasiteyi aşıyorsak: top-K seçime düş
    n = proba.shape[0]
    k = int(np.floor(max_action_rate * n))

    # k=0 olabilir (çok düşük rate + küçük batch)
    if k <= 0:
        return np.zeros_like(actions)

    eligible_idx = np.where(actions == 1)[0]
    if eligible_idx.size == 0:
        return np.zeros_like(actions)

    if eligible_idx.size <= k:
        return actions

    # En yüksek score/proba'lı k örneği seç
    # argpartition O(n) -> büyük batch için daha iyi
    ranking_base = ranking_scores if ranking_scores is not None else proba
    eligible_scores = ranking_base[eligible_idx]
    local_idx = np.argpartition(-eligible_scores, kth=k - 1)[:k]
    idx = eligible_idx[local_idx]

    constrained = np.zeros_like(actions)
    constrained[idx] = 1

    return constrained


def apply_policy_to_proba(
    proba: np.ndarray,
    policy: DecisionPolicy,
    ranking_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    proba + policy -> action kararları

    Bu wrapper:
    - threshold + max_action_rate politikasını tek yerden uygular.
    """
    return decide_actions_from_proba(
        proba=proba,
        threshold=policy.threshold,
        max_action_rate=policy.max_action_rate,
        ranking_scores=ranking_scores,
    )


def apply(
    proba: np.ndarray,
    policy: DecisionPolicy,
    ranking_scores: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Short alias for production usage: policy.apply(...)
    """
    return apply_policy_to_proba(proba=proba, policy=policy, ranking_scores=ranking_scores)


def compute_incremental_profit_scores(
    df_input: pd.DataFrame,
    proba: np.ndarray,
    policy: DecisionPolicy,
) -> Optional[np.ndarray]:
    """
    Policy ranking_mode='incremental_profit' ise müşteri bazlı score üretir.

    score = E[profit | action] - E[profit | no_action]
          = p*tp + (1-p)*fp - (p*fn + (1-p)*tn)
    """
    ranking_mode = str(policy.raw.get("ranking_mode", "proba"))
    if ranking_mode != "incremental_profit":
        return None

    uplift_cfg = policy.raw.get("uplift", {}) or {}
    segment_col = uplift_cfg.get("segment_col")
    default_tp = float(uplift_cfg.get("default_tp_value", policy.raw.get("cost_matrix", {}).get("tp_value", 0.0)))
    fp_value = float(uplift_cfg.get("fp_value", policy.raw.get("cost_matrix", {}).get("fp_value", 0.0)))
    fn_value = float(uplift_cfg.get("fn_value", policy.raw.get("cost_matrix", {}).get("fn_value", 0.0)))
    tn_value = float(uplift_cfg.get("tn_value", policy.raw.get("cost_matrix", {}).get("tn_value", 0.0)))

    tp_values = np.full(shape=proba.shape[0], fill_value=default_tp, dtype=float)

    segment_map = uplift_cfg.get("tp_value_by_segment", {}) or {}
    if segment_col:
        if segment_col not in df_input.columns:
            raise ValueError(
                f"Policy requires segment column '{segment_col}' for uplift scoring, but input is missing it."
            )
        mapped = df_input[segment_col].astype(str).map(segment_map)
        tp_values = mapped.fillna(default_tp).to_numpy(dtype=float)

    action_profit = proba * tp_values + (1.0 - proba) * fp_value
    no_action_profit = proba * fn_value + (1.0 - proba) * tn_value
    return action_profit - no_action_profit
