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
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Paths:
    """
    Projedeki standart klasör yolları.

    Not:
    - Bu sınıf sadece path üretir.
    - Dosyaların varlığını garanti etmez (mkdir işlemleri pipeline içinde yapılır).
    """

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )

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

    action_rates: List[float] = field(
        default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.30]
    )
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
class ValidationPolicy:
    """
    Pipeline boyunca hangi validasyon ihlallerinin pipeline'ı durduracağını
    (block) ve hangilerinin yalnızca uyarı (warn) üretip devam edeceğini
    tek noktadan yönetir.

    Tüm eşikler config'ten gelir → kod değişmeden politika güncellenebilir.

    Kullanım profilleri:
    - strict=True  → CI/CD, production deployment gate
    - strict=False → exploratory run, eski veri yeniden işleme
    """
    # ── Blok / warn politikaları ──────────────────────────────────────
    # True → eşik aşılırsa ValueError fırlatır (pipeline durur)
    # False → sadece WARNING log üretir

    # Duplicate
    block_on_duplicate: bool = False          # Warn-only varsayılan
    duplicate_ratio_threshold: float = 0.02   # %2 üzeri → alarm

    # Volume anomalisi
    block_on_volume_anomaly: bool = True       # Ciddi kalite riski → block
    volume_tolerance_ratio: float = 0.50      # ±%50

    # Staleness
    block_on_stale_data: bool = False          # Warn-only; CI ortamında True yapılabilir
    max_staleness_days: float = 180.0

    # Post-imputation NaN
    block_on_nan_after_impute: bool = True     # Daima block — impütasyon hatasına tolerans yok

    # Distribution / drift (eğitim/izleme)
    block_on_distribution_drift: bool = False  # Monitor: warn-only
    distribution_tolerance_sigma: float = 3.0  # |Δmean| / ref_std eşiği

    # PSI drift
    block_on_psi_drift: bool = False
    psi_warn_threshold: float = 0.10           # Orta drift
    psi_block_threshold: float = 0.25          # Şiddetli drift → block

    # Label drift
    block_on_label_drift: bool = False
    label_drift_tolerance: float = 0.10

    # Training-serving skew
    block_on_serving_skew: bool = False
    serving_skew_tolerance_sigma: float = 2.0

    # Raw schema
    block_on_raw_schema_error: bool = True     # Ham veri kontratı kırılırsa → block

    # Processed schema
    block_on_processed_schema_error: bool = True

    # Inference payload (serving — non-blocking by design)
    strict_inference_schema: bool = False      # True → fazla/eksik kolon ValueError
    log_extra_inference_cols: bool = True      # Fazla kolon → her zaman logla

    @classmethod
    def strict(cls) -> "ValidationPolicy":
        """CI / production gate için her şeyi bloke eden profil."""
        return cls(
            block_on_duplicate=True,
            duplicate_ratio_threshold=0.01,
            block_on_volume_anomaly=True,
            block_on_stale_data=True,
            max_staleness_days=30.0,
            block_on_nan_after_impute=True,
            block_on_distribution_drift=True,
            distribution_tolerance_sigma=2.0,
            block_on_psi_drift=True,
            psi_warn_threshold=0.10,
            psi_block_threshold=0.20,
            block_on_label_drift=True,
            label_drift_tolerance=0.05,
            block_on_serving_skew=True,
            serving_skew_tolerance_sigma=1.5,
            block_on_raw_schema_error=True,
            block_on_processed_schema_error=True,
            strict_inference_schema=True,
        )

    @classmethod
    def relaxed(cls) -> "ValidationPolicy":
        """Exploratory / backfill için sadece log üreten profil."""
        return cls(
            block_on_duplicate=False,
            block_on_volume_anomaly=False,
            block_on_stale_data=False,
            block_on_nan_after_impute=False,
            block_on_distribution_drift=False,
            block_on_psi_drift=False,
            block_on_label_drift=False,
            block_on_serving_skew=False,
            block_on_raw_schema_error=False,
            block_on_processed_schema_error=False,
            strict_inference_schema=False,
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
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)


def load_experiment_config(params_path: Optional[Path] = None) -> ExperimentConfig:
    """
    params.yaml dosyasından ExperimentConfig yükler — tek kaynak prensibi.

    DVC pipeline'ı params.yaml'ı versiyon kontrolüne alır.
    Bu fonksiyon sayesinde kod ile yaml arasında değer çiftlenmesi olmaz.

    Dosya bulunamazsa ya da bir anahtar eksikse varsayılan değerler kullanılır
    (güvenli fallback — test ortamı veya minimal kurulum için).

    Kullanım:
        cfg = load_experiment_config()               # otomatik proje kökü
        cfg = load_experiment_config(Path("p.yaml")) # özel yol
    """
    if params_path is None:
        params_path = Path(__file__).resolve().parents[1] / "params.yaml"

    if not params_path.exists():
        return ExperimentConfig()

    try:
        import yaml  # PyYAML — requirements-prod.txt'te mevcut (dvc bağımlılığı)
    except ImportError:
        return ExperimentConfig()

    try:
        with params_path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}
    except Exception:
        return ExperimentConfig()

    exp = raw.get("experiment", {}) or {}
    cost = raw.get("cost_matrix", {}) or {}
    decision = raw.get("decision", {}) or {}

    return ExperimentConfig(
        target_col=str(exp.get("target_col", "is_canceled")),
        test_size=float(exp.get("test_size", 0.20)),
        seed=int(exp.get("seed", 42)),
        cv_folds=int(exp.get("cv_folds", 5)),
        cost=CostConfig(
            tp_value=float(cost.get("tp_value", 180.0)),
            fp_value=float(cost.get("fp_value", -20.0)),
            fn_value=float(cost.get("fn_value", -200.0)),
            tn_value=float(cost.get("tn_value", 0.0)),
        ),
        decision=DecisionConfig(
            action_rates=list(
                decision.get("action_rates", [0.05, 0.10, 0.15, 0.20, 0.30])
            ),
        ),
    )
