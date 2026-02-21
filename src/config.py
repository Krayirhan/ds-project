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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


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


# ── Severity sabitleri ────────────────────────────────────────────────
Severity = Literal["warn", "soft_fail", "hard_fail"]
"""
warn      → Sadece WARNING log; pipeline devam eder, sonuç işaretlenmez.
soft_fail → WARNING log + ValidationProfileReport.soft_failures listesine eklenir;
            caller kendi politikasına göre devam edip etmemeye karar verir.
hard_fail → ValueError fırlatır; pipeline durur.
"""

# ── Per-kolon PSI/JS eşiği (isteğe bağlı override) ───────────────────
ColumnDriftThreshold = Dict[str, float]
"""Örn: {"lead_time": 0.15, "adr": 0.20}  — belirtilmeyenler global eşiği alır."""


@dataclass(frozen=True)
class CheckConfig:
    """
    Tek bir validasyon kontrolünün tam konfigürasyonu.

    severity   : warn | soft_fail | hard_fail
    enabled    : False → kontrol tamamen atlanır (feature flag gibi)
    threshold  : Sayısal eşik (oran, sigma, gün — kontrole göre anlam değişir)
    """

    severity: Severity = "warn"
    enabled: bool = True
    threshold: float = (
        0.0  # anlamsız varsayılan; her kontrol kendi default'unu kullanır
    )


@dataclass(frozen=True)
class ValidationPolicy:
    """
    Pipeline boyunca tüm validasyon kontrollerinin severity'sini ve eşiklerini
    tek noktadan yönetir.

    Severity seviyeleri:
      warn      → log-only, devam et
      soft_fail → log + raporla, caller karar verir
      hard_fail → ValueError, pipeline durur

    Environment profilleri:
      ValidationPolicy.for_env("dev")     → relaxed
      ValidationPolicy.for_env("staging") → moderate
      ValidationPolicy.for_env("prod")    → strict

    Phase profilleri (her environment içinde phase override'ı mümkün):
      ValidationPolicy.for_phase("predict") → inference için özelleştirilmiş
    """

    # ── Duplicate ────────────────────────────────────────────────────
    duplicate: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="warn", enabled=True, threshold=0.02
        )
    )

    # ── Volume anomalisi ─────────────────────────────────────────────
    volume: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="hard_fail", enabled=True, threshold=0.50
        )
    )

    # ── Data staleness ───────────────────────────────────────────────
    staleness: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="warn", enabled=True, threshold=180.0
        )
    )

    # ── Post-imputation NaN ──────────────────────────────────────────
    nan_after_impute: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="hard_fail", enabled=True, threshold=0.0
        )
    )

    # ── Raw schema (Pandera) ─────────────────────────────────────────
    raw_schema: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="hard_fail", enabled=True, threshold=0.0
        )
    )

    # ── Processed schema (Pandera) ───────────────────────────────────
    processed_schema: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="hard_fail", enabled=True, threshold=0.0
        )
    )

    # ── Distribution drift (mean/sigma) ─────────────────────────────
    distribution_drift: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="warn", enabled=True, threshold=3.0
        )
    )

    # ── PSI drift ───────────────────────────────────────────────────
    # threshold = global warn eşiği; hard_fail için psi_block_threshold kullanılır
    psi_drift: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="soft_fail", enabled=True, threshold=0.10
        )
    )
    psi_block_threshold: float = (
        0.25  # Bu PSI değerini aşan kolon → hard_fail gibi davranır
    )
    psi_metric: Literal["psi", "js"] = (
        "psi"  # js → Jensen-Shannon divergence (simetrik, 0-1 arası)
    )
    psi_n_bins: int = 10
    # Per-kolon override: {"lead_time": 0.15} → o kolon için global yerine bu eşik
    column_drift_thresholds: ColumnDriftThreshold = field(default_factory=dict)
    # Kritik kolonlar — bu kolonlarda drift → otomatik hard_fail (column_drift_thresholds'dan bağımsız)
    critical_columns: List[str] = field(default_factory=list)

    # ── Label drift ─────────────────────────────────────────────────
    label_drift: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="soft_fail", enabled=True, threshold=0.10
        )
    )

    # ── Training-serving skew ────────────────────────────────────────
    serving_skew: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="warn", enabled=True, threshold=2.0
        )
    )

    # ── Inference payload ────────────────────────────────────────────
    inference_schema: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="warn", enabled=True, threshold=0.0
        )
    )
    strict_inference_schema: bool = False  # True → beklenmeyen kolon → SchemaError

    # ── Correlation drift ────────────────────────────────────────────
    correlation_drift: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="warn", enabled=True, threshold=0.20
        )
    )

    # ── Row anomaly ──────────────────────────────────────────────────
    row_anomaly: CheckConfig = field(
        default_factory=lambda: CheckConfig(
            severity="warn", enabled=True, threshold=0.0
        )
    )

    # ─────────────────────────────────────────────────────────────────
    # Environment factory methods
    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def for_env(cls, env: str | None = None) -> "ValidationPolicy":
        """
        Environment bazlı hazır profil.

        env parametresi verilmezse DS_ENV ortam değişkenine bakılır.
        Bilinmeyen env → "dev" profili (güvenli fallback).

        Kullanım:
            cfg.validation = ValidationPolicy.for_env()           # DS_ENV'e göre
            cfg.validation = ValidationPolicy.for_env("prod")     # Açık seçim
        """
        resolved = (env or os.getenv("DS_ENV", "dev")).lower().strip()
        if resolved in ("prod", "production"):
            return cls._prod()
        elif resolved in ("staging", "stage"):
            return cls._staging()
        else:
            return cls._dev()

    @classmethod
    def _dev(cls) -> "ValidationPolicy":
        """Dev: Her şey warn-only. Hızlı iterasyon öncelikli."""
        H = CheckConfig
        return cls(
            duplicate=H("warn", True, 0.05),
            volume=H("warn", True, 0.70),
            staleness=H("warn", True, 365.0),
            nan_after_impute=H("soft_fail", True, 0.0),
            raw_schema=H("soft_fail", True, 0.0),
            processed_schema=H("soft_fail", True, 0.0),
            distribution_drift=H("warn", True, 4.0),
            psi_drift=H("warn", True, 0.15),
            psi_block_threshold=0.30,
            psi_metric="psi",
            critical_columns=[],
            label_drift=H("warn", True, 0.15),
            serving_skew=H("warn", True, 3.0),
            inference_schema=H("warn", True, 0.0),
            strict_inference_schema=False,
            correlation_drift=H("warn", True, 0.30),
            row_anomaly=H("warn", True, 0.0),
        )

    @classmethod
    def _staging(cls) -> "ValidationPolicy":
        """Staging: soft_fail → caller loglar ama pipeline devam eder; kritik → hard_fail."""
        H = CheckConfig
        return cls(
            duplicate=H("soft_fail", True, 0.02),
            volume=H("hard_fail", True, 0.50),
            staleness=H("soft_fail", True, 90.0),
            nan_after_impute=H("hard_fail", True, 0.0),
            raw_schema=H("hard_fail", True, 0.0),
            processed_schema=H("hard_fail", True, 0.0),
            distribution_drift=H("soft_fail", True, 3.0),
            psi_drift=H("soft_fail", True, 0.10),
            psi_block_threshold=0.25,
            psi_metric="psi",
            critical_columns=["lead_time", "adr", "is_canceled"],
            label_drift=H("soft_fail", True, 0.08),
            serving_skew=H("soft_fail", True, 2.0),
            inference_schema=H("soft_fail", True, 0.0),
            strict_inference_schema=False,
            correlation_drift=H("soft_fail", True, 0.20),
            row_anomaly=H("warn", True, 0.0),
        )

    @classmethod
    def _prod(cls) -> "ValidationPolicy":
        """Prod: Tüm kritik kontroller hard_fail. PSI/JS tabanlı drift izleme."""
        H = CheckConfig
        return cls(
            duplicate=H("hard_fail", True, 0.01),
            volume=H("hard_fail", True, 0.30),
            staleness=H("hard_fail", True, 30.0),
            nan_after_impute=H("hard_fail", True, 0.0),
            raw_schema=H("hard_fail", True, 0.0),
            processed_schema=H("hard_fail", True, 0.0),
            distribution_drift=H("hard_fail", True, 2.0),
            psi_drift=H("hard_fail", True, 0.10),
            psi_block_threshold=0.20,
            psi_metric="js",  # JS → prod'da daha simetrik
            psi_n_bins=15,
            critical_columns=["lead_time", "adr", "adults", "stays_in_week_nights"],
            column_drift_thresholds={  # Kritik kolonlara daha sıkı eşik
                "lead_time": 0.08,
                "adr": 0.08,
                "adults": 0.12,
            },
            label_drift=H("hard_fail", True, 0.05),
            serving_skew=H("hard_fail", True, 1.5),
            inference_schema=H("hard_fail", True, 0.0),
            strict_inference_schema=True,
            correlation_drift=H("hard_fail", True, 0.15),
            row_anomaly=H("soft_fail", True, 0.0),
        )

    # ── Phase factory methods ─────────────────────────────────────────
    @classmethod
    def for_phase(
        cls,
        phase: Literal["preprocess", "train", "predict", "monitor"],
        env: str | None = None,
    ) -> "ValidationPolicy":
        """
        Hem environment hem phase'e göre özelleştirilmiş profil döner.

        Phase farkları:
          preprocess → raw schema + duplicate + volume kritik
          train      → processed schema + NaN + distribution kritik
          predict    → inference payload + serving skew + strict mode
          monitor    → drift (PSI/JS) + label drift + correlation kritik
        """
        base = cls.for_env(env)
        resolved_env = (env or os.getenv("DS_ENV", "dev")).lower()
        is_prod = resolved_env in ("prod", "production")
        H = CheckConfig

        if phase == "preprocess":
            # Predict ve monitor'de olmayan kontroller burada kritik
            return cls(
                **{k: v for k, v in base.__dict__.items()},
            )

        elif phase == "train":
            # Inference kontrollerini devre dışı bırak (anlamsız bu aşamada)
            d = {k: v for k, v in base.__dict__.items()}
            d["inference_schema"] = H("warn", False, 0.0)  # disabled
            d["serving_skew"] = H("warn", False, 0.0)  # disabled
            return cls(**d)

        elif phase == "predict":
            # Drift ve schema kontrollerini inference odaklı yeniden ağırlıklandır
            d = {k: v for k, v in base.__dict__.items()}
            d["raw_schema"] = H("warn", False, 0.0)  # ham veri yok
            d["processed_schema"] = H("warn", False, 0.0)  # ham veri yok
            d["duplicate"] = H("warn", False, 0.0)  # inference'ta anlamsız
            d["staleness"] = H("warn", False, 0.0)  # inference'ta anlamsız
            # Inference payload → prod'da hard_fail
            d["inference_schema"] = (
                H("hard_fail", True, 0.0) if is_prod else H("soft_fail", True, 0.0)
            )
            d["strict_inference_schema"] = is_prod
            return cls(**d)

        elif phase == "monitor":
            # Drift kontrolleri burada en kritik
            d = {k: v for k, v in base.__dict__.items()}
            d["raw_schema"] = H("warn", False, 0.0)
            d["processed_schema"] = H("warn", False, 0.0)
            d["nan_after_impute"] = H("warn", False, 0.0)
            # PSI + label drift → prod'da hard_fail
            d["psi_drift"] = (
                H("hard_fail", True, base.psi_drift.threshold)
                if is_prod
                else H("soft_fail", True, base.psi_drift.threshold)
            )
            d["label_drift"] = (
                H("hard_fail", True, base.label_drift.threshold)
                if is_prod
                else H("soft_fail", True, base.label_drift.threshold)
            )
            return cls(**d)

        return base


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
