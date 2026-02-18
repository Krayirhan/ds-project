"""
api_shared.py — Shared models, state accessors, and utilities for versioned API routers.

Extracted to avoid circular imports between api.py ↔ api_v1.py / api_v2.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from .config import ExperimentConfig, Paths
from .policy import load_decision_policy
from .predict import load_feature_spec
from .utils import get_logger, sha256_file

import joblib

logger = get_logger("api_shared")


# ─── Serving State ─────────────────────────────────────────────────────
@dataclass
class ServingState:
    model: Any
    policy_path: Path
    feature_spec: Dict[str, Any]
    policy: Any


# ─── Request / Response Models ─────────────────────────────────────────
class RecordsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[Dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str


class ReadyResponse(BaseModel):
    status: Literal["ok"]
    service: Literal["ready"]
    model: str
    policy_path: str


class SchemaReportResponse(BaseModel):
    missing_columns: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)
    feature_count_expected: int
    feature_count_input: int
    feature_count_used: int


class PredictProbaResponse(BaseModel):
    n: int
    proba: List[float]
    schema_report: SchemaReportResponse


class DecideResultItem(BaseModel):
    proba: float
    action: int
    threshold_used: float
    max_action_rate_used: float | None = None
    model_used: str


class DecideReportResponse(BaseModel):
    missing_columns: List[str] = Field(default_factory=list)
    extra_columns: List[str] = Field(default_factory=list)
    feature_count_expected: int
    feature_count_input: int
    feature_count_used: int
    n_rows: int
    predicted_action_rate: float
    threshold_used: float
    max_action_rate_used: float | None = None
    model_used: str
    ranking_mode: str


class DecideResponse(BaseModel):
    n: int
    results: List[DecideResultItem]
    report: DecideReportResponse


class ReloadResponse(BaseModel):
    status: Literal["ok"]
    message: str
    model: str
    policy_path: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str | None = None


# ─── State Loading ─────────────────────────────────────────────────────
def load_serving_state() -> ServingState:
    cfg = ExperimentConfig()
    paths = Paths()
    active_slot_path = paths.reports_metrics / "active_slot.json"
    if active_slot_path.exists():
        slot_payload = json.loads(active_slot_path.read_text(encoding="utf-8"))
        slot = str(slot_payload.get("active_slot", "default"))
        if slot in {"blue", "green"}:
            policy_path = paths.reports_metrics / f"decision_policy.{slot}.json"
        else:
            policy_path = paths.reports_metrics / "decision_policy.json"
    else:
        policy_path = paths.reports_metrics / "decision_policy.json"
    policy = load_decision_policy(policy_path)
    if policy.raw.get("policy_version") != cfg.contract.policy_version:
        raise RuntimeError("Policy contract version mismatch")

    model_artifact = policy.selected_model_artifact
    if not model_artifact:
        raise RuntimeError("Policy does not contain selected_model_artifact")

    model_path = paths.project_root / model_artifact
    if not model_path.exists():
        raise RuntimeError(f"Model artifact not found: {model_path}")

    expected_sha = policy.raw.get("selected_model_sha256")
    if expected_sha:
        actual_sha = sha256_file(str(model_path))
        if actual_sha != expected_sha:
            raise RuntimeError("Model checksum mismatch")

    model = joblib.load(model_path)

    run_id = str(policy.raw.get("run_id", ""))
    run_feature_spec = paths.reports_metrics / run_id / "feature_spec.json"
    global_feature_spec = paths.reports / "feature_spec.json"
    feature_spec_path = run_feature_spec if run_feature_spec.exists() else global_feature_spec
    feature_spec = load_feature_spec(feature_spec_path)
    if feature_spec.get("schema_version") != cfg.contract.feature_schema_version:
        raise RuntimeError("Feature schema contract version mismatch")

    return ServingState(
        model=model,
        policy_path=policy_path,
        feature_spec=feature_spec,
        policy=policy,
    )


def error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    request_id: str | None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error_code": error_code,
            "message": message,
            "request_id": request_id,
        },
    )
