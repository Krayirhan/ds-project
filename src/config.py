"""
config.py

Bu dosya proje genelinde:
- path'leri
- deney (experiment) parametrelerini
- ✅ iş kararı (decision) parametrelerini (cost matrix, action rates) taşır.

Kurumsal prensip:
- İş kararları (cost values, capacity/budget, model preference) KOD içine gömülmez.
- Config üzerinden yönetilir (feature flag gibi düşün).

Bu sayede:
- Model aynı kalsa bile "threshold policy" iş birimiyle uyumlu şekilde değiştirilebilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Paths:
    """
    Projedeki standart klasör yolları.

    Not:
    - Bu sınıf sadece path üretir.
    - Dosyaların varlığını garanti etmez (mkdir işlemleri pipeline içinde yapılır).
    """
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    data_dir: Path = field(init=False)
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)

    models: Path = field(init=False)

    reports: Path = field(init=False)
    reports_metrics: Path = field(init=False)
    reports_predictions: Path = field(init=False)
    reports_monitoring: Path = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "data_dir", self.project_root / "data")
        object.__setattr__(self, "data_raw", self.data_dir / "raw")
        object.__setattr__(self, "data_processed", self.data_dir / "processed")
        object.__setattr__(self, "models", self.project_root / "models")
        object.__setattr__(self, "reports", self.project_root / "reports")
        object.__setattr__(self, "reports_metrics", self.reports / "metrics")
        object.__setattr__(self, "reports_predictions", self.reports / "predictions")
        object.__setattr__(self, "reports_monitoring", self.reports / "monitoring")


@dataclass(frozen=True)
class DecisionConfig:
    """
    ✅ Karar (Decisioning) konfigürasyonu.

    action_rates:
    - Capacity/budget constraint'leri.
    - Örn 0.30 => en fazla %30 müşteriye aksiyon.

    prefer_models:
    - Net profit eşit/çok yakın olduğunda hangi modeli tercih edeceğiz?
    - Kurumsal bakış: calibration stabilitesi nedeniyle sigmoid genelde öne alınır.
    """
    action_rates: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.30])
    prefer_models: List[str] = field(
        default_factory=lambda: [
            "challenger_xgboost_calibrated_sigmoid",
            "challenger_lightgbm_calibrated_sigmoid",
            "challenger_catboost_calibrated_sigmoid",
            "challenger_histgb_calibrated_sigmoid",
            "baseline_calibrated_sigmoid",
            "challenger_xgboost",
            "challenger_lightgbm",
            "challenger_catboost",
            "challenger_histgb",
            "baseline",
        ]
    )


@dataclass(frozen=True)
class ModelConfig:
    include_challenger: bool = True


@dataclass(frozen=True)
class UpliftConfig:
    ranking_mode: str = "incremental_profit"  # incremental_profit | proba
    segment_col: str = "customer_type"
    tp_value_by_segment: Dict[str, float] = field(
        default_factory=lambda: {
            "Contract": 230.0,
            "Group": 150.0,
            "Transient": 170.0,
            "Transient-Party": 190.0,
        }
    )


@dataclass(frozen=True)
class MonitoringConfig:
    data_drift_psi_threshold: float = 0.20
    prediction_drift_psi_threshold: float = 0.20
    profit_drop_ratio_alert: float = 0.20
    action_rate_tolerance: float = 0.05
    alert_webhook_url: str | None = None


@dataclass(frozen=True)
class ApiConfig:
    api_key_env_var: str = "DS_API_KEY"
    require_api_key: bool = True
    rate_limit_per_minute: int = 120
    rate_limit_backend: str = "memory"  # memory | redis
    redis_url: str | None = None
    redis_key_prefix: str = "ds:rate"
    max_payload_records: int = 5000
    graceful_shutdown_seconds: int = 30


@dataclass(frozen=True)
class ContractConfig:
    feature_schema_version: str = "1.0.0"
    policy_version: str = "1.0.0"


@dataclass(frozen=True)
class CostConfig:
    """
    ✅ Cost/Value parametreleri (iş parametreleri)

    tp_value:
      True Positive (iptal olacak müşteriye aksiyon verdin) başına NET değer.
      (kurtarılan gelir - aksiyon maliyeti gibi)

    fp_value:
      False Positive (iptal olmayacak müşteriye aksiyon verdin) başına NET değer (genelde negatif).
      (gereksiz indirim/iletişim maliyeti + spam etkisi)

    fn_value:
      False Negative (iptal olacak müşteriyi kaçırdın) başına NET değer (genelde negatif).
      (kaçan gelir / boş oda maliyeti / operasyon maliyeti)

    tn_value:
      True Negative çoğu durumda 0 tutulur.
    """
    tp_value: float = 180.0
    fp_value: float = -20.0
    fn_value: float = -200.0
    tn_value: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Model/ML pipeline konfigürasyonu.

    Bu config:
    - data contract (target, label map)
    - split oranları
    - reproducibility
    - baseline model parametreleri
    - ✅ decisioning config (cost + constraint) içerir
    """
    # Target kolonu
    target_col: str = "is_canceled"

    # Dataset'teki label formatı (bu projede "yes/no" idi)
    label_map: Dict[str, int] = field(default_factory=lambda: {"no": 0, "yes": 1})

    # Leakage: hedefi doğrudan ele veren kolonlar
    leakage_cols: Tuple[str, ...] = ("reservation_status", "reservation_status_date")

    # Bazı kolonları "özellik olarak kullanmak istemiyoruz" (policy)
    blocked_feature_cols: Tuple[str, ...] = tuple()

    # Split
    test_size: float = 0.20
    seed: int = 42

    # Training
    cv_folds: int = 5

    # Default “decision threshold” (metric-based). Profit policy ayrı.
    decision_threshold: float = 0.35

    # ✅ Business / decision configs
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    uplift: UpliftConfig = field(default_factory=UpliftConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
