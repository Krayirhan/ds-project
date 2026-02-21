from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.api_shared as api_shared


def _cfg(max_rows: int = 100):
    return SimpleNamespace(
        api=SimpleNamespace(max_payload_records=max_rows),
        contract=SimpleNamespace(
            policy_version="policy-v1",
            feature_schema_version="schema-v1",
        ),
    )


def _paths(tmp_path: Path):
    reports = tmp_path / "reports"
    reports_metrics = reports / "metrics"
    reports.mkdir(parents=True, exist_ok=True)
    reports_metrics.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_root=tmp_path,
        reports=reports,
        reports_metrics=reports_metrics,
    )


class _DummyPolicy:
    def __init__(
        self,
        *,
        policy_version: str = "policy-v1",
        schema_version: str = "schema-v1",
        run_id: str = "run-1",
        artifact: str = "models/model.joblib",
        selected_model: str = "xgb",
        sha256: str | None = "ok-sha",
    ):
        self.selected_model = selected_model
        self.selected_model_artifact = artifact
        self.raw = {
            "policy_version": policy_version,
            "feature_schema_version": schema_version,
            "run_id": run_id,
            "selected_model_sha256": sha256,
        }


class _Counter:
    def __init__(self):
        self.labels_args = None
        self.inc_value = None

    def labels(self, **kwargs):
        self.labels_args = kwargs
        return self

    def inc(self, value):
        self.inc_value = value


@contextmanager
def _noop_trace(*args, **kwargs):
    yield None


def test_load_serving_state_success_with_optional_artifacts(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    (paths.reports_metrics / "active_slot.json").write_text(
        json.dumps({"active_slot": "blue"}),
        encoding="utf-8",
    )
    run_dir = paths.reports_metrics / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_spec.json").write_text("{}", encoding="utf-8")
    (run_dir / "schema_contract.json").write_text(
        json.dumps({"schema_version": "schema-v1"}),
        encoding="utf-8",
    )
    (paths.reports_metrics / "reference_stats.json").write_text(
        json.dumps({"x": {"mean": 1.0}}),
        encoding="utf-8",
    )
    (paths.reports_metrics / "reference_categories.json").write_text(
        json.dumps({"country": ["TR", "US"]}),
        encoding="utf-8",
    )
    (paths.reports_metrics / "data_lineage_preprocess.json").write_text(
        json.dumps({"processed_rows": 123}),
        encoding="utf-8",
    )

    model_path = tmp_path / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("bin", encoding="utf-8")

    called = {}

    def _load_policy(policy_path):
        called["policy_path"] = policy_path
        return _DummyPolicy()

    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg())
    monkeypatch.setattr(api_shared, "Paths", lambda: paths)
    monkeypatch.setattr(api_shared, "load_decision_policy", _load_policy)
    monkeypatch.setattr(api_shared, "sha256_file", lambda _: "ok-sha")
    monkeypatch.setattr(api_shared.joblib, "load", lambda _: "MODEL")
    monkeypatch.setattr(
        api_shared,
        "load_feature_spec",
        lambda p: {"schema_version": "schema-v1", "loaded_from": str(p)},
    )

    state = api_shared.load_serving_state()
    assert str(called["policy_path"]).endswith("decision_policy.blue.json")
    assert state.model == "MODEL"
    assert state.feature_spec["_schema_contract"]["schema_version"] == "schema-v1"
    assert state.feature_spec["_reference_stats"]["x"]["mean"] == 1.0
    assert state.feature_spec["_reference_categories"]["country"] == ["TR", "US"]
    assert state.feature_spec["_reference_volume_rows"] == 123


def test_load_serving_state_ignores_invalid_lineage_json(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    run_dir = paths.reports_metrics / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "feature_spec.json").write_text("{}", encoding="utf-8")
    (paths.reports_metrics / "data_lineage_preprocess.json").write_text(
        "{invalid-json",
        encoding="utf-8",
    )
    model_path = tmp_path / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("bin", encoding="utf-8")

    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg())
    monkeypatch.setattr(api_shared, "Paths", lambda: paths)
    monkeypatch.setattr(api_shared, "load_decision_policy", lambda _: _DummyPolicy())
    monkeypatch.setattr(api_shared, "sha256_file", lambda _: "ok-sha")
    monkeypatch.setattr(api_shared.joblib, "load", lambda _: "MODEL")
    monkeypatch.setattr(api_shared, "load_feature_spec", lambda _: {"schema_version": "schema-v1"})

    state = api_shared.load_serving_state()
    assert "_reference_volume_rows" not in state.feature_spec


def test_load_serving_state_policy_contract_mismatch(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg())
    monkeypatch.setattr(api_shared, "Paths", lambda: paths)
    monkeypatch.setattr(
        api_shared,
        "load_decision_policy",
        lambda _: _DummyPolicy(policy_version="old"),
    )
    with pytest.raises(RuntimeError, match="Policy contract version mismatch"):
        api_shared.load_serving_state()


def test_load_serving_state_missing_model_artifact(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg())
    monkeypatch.setattr(api_shared, "Paths", lambda: paths)
    monkeypatch.setattr(
        api_shared,
        "load_decision_policy",
        lambda _: _DummyPolicy(artifact=""),
    )
    with pytest.raises(RuntimeError, match="selected_model_artifact"):
        api_shared.load_serving_state()


def test_load_serving_state_missing_model_file(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg())
    monkeypatch.setattr(api_shared, "Paths", lambda: paths)
    monkeypatch.setattr(api_shared, "load_decision_policy", lambda _: _DummyPolicy())
    with pytest.raises(RuntimeError, match="Model artifact not found"):
        api_shared.load_serving_state()


def test_load_serving_state_checksum_mismatch(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    model_path = tmp_path / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("bin", encoding="utf-8")

    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg())
    monkeypatch.setattr(api_shared, "Paths", lambda: paths)
    monkeypatch.setattr(api_shared, "load_decision_policy", lambda _: _DummyPolicy(sha256="expected"))
    monkeypatch.setattr(api_shared, "sha256_file", lambda _: "actual")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        api_shared.load_serving_state()


def test_load_serving_state_feature_schema_mismatch(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    model_path = tmp_path / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("bin", encoding="utf-8")

    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg())
    monkeypatch.setattr(api_shared, "Paths", lambda: paths)
    monkeypatch.setattr(api_shared, "load_decision_policy", lambda _: _DummyPolicy(sha256=None))
    monkeypatch.setattr(api_shared.joblib, "load", lambda _: "MODEL")
    monkeypatch.setattr(api_shared, "load_feature_spec", lambda _: {"schema_version": "bad"})

    with pytest.raises(RuntimeError, match="Feature schema contract version mismatch"):
        api_shared.load_serving_state()


def test_load_serving_state_schema_contract_mismatch(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    run_dir = paths.reports_metrics / "run-1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "schema_contract.json").write_text(
        json.dumps({"schema_version": "bad"}),
        encoding="utf-8",
    )
    model_path = tmp_path / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("bin", encoding="utf-8")

    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg())
    monkeypatch.setattr(api_shared, "Paths", lambda: paths)
    monkeypatch.setattr(api_shared, "load_decision_policy", lambda _: _DummyPolicy(sha256=None))
    monkeypatch.setattr(api_shared.joblib, "load", lambda _: "MODEL")
    monkeypatch.setattr(api_shared, "load_feature_spec", lambda _: {"schema_version": "schema-v1"})

    with pytest.raises(RuntimeError, match="Schema contract artifact version mismatch"):
        api_shared.load_serving_state()


def test_exec_predict_proba_success_and_size_guard(monkeypatch):
    import src.metrics as metrics
    import src.predict as predict
    import src.tracing as tracing

    counter = _Counter()
    monkeypatch.setattr(metrics, "INFERENCE_ROWS", counter)
    monkeypatch.setattr(tracing, "trace_inference", _noop_trace)
    monkeypatch.setattr(tracing, "set_span_attribute", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        predict,
        "validate_and_prepare_features",
        lambda df, spec, fail_on_missing: (df, {"feature_count_used": len(df.columns)}),
    )
    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg(max_rows=5))

    class _Model:
        def predict_proba(self, X):
            return np.asarray([[0.1, 0.9], [0.2, 0.8]])

    serving = SimpleNamespace(
        model=_Model(),
        feature_spec={"x": 1},
        policy=SimpleNamespace(selected_model="xgb"),
    )
    payload = api_shared.RecordsPayload(records=[{"a": 1}, {"a": 2}])
    proba, schema_report, model_name = api_shared.exec_predict_proba(
        payload,
        serving,
        "predict_proba",
    )
    assert proba == [0.9, 0.8]
    assert schema_report["feature_count_used"] == 1
    assert model_name == "xgb"
    assert counter.labels_args["endpoint"] == "predict_proba"
    assert counter.inc_value == 2

    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg(max_rows=1))
    with pytest.raises(ValueError, match="Payload too large"):
        api_shared.exec_predict_proba(payload, serving, "predict_proba")


def test_exec_decide_success_and_size_guard(monkeypatch):
    import src.metrics as metrics
    import src.predict as predict
    import src.tracing as tracing

    counter = _Counter()
    monkeypatch.setattr(metrics, "INFERENCE_ROWS", counter)
    monkeypatch.setattr(tracing, "trace_inference", _noop_trace)
    monkeypatch.setattr(tracing, "set_span_attribute", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        predict,
        "predict_with_policy",
        lambda **kwargs: (
            pd.DataFrame([{"proba": 0.9, "action": 1}]),
            {"n_rows": 1},
        ),
    )
    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg(max_rows=5))

    serving = SimpleNamespace(
        model=object(),
        feature_spec={"x": 1},
        policy=SimpleNamespace(selected_model_artifact="models/xgb.joblib"),
    )
    payload = api_shared.RecordsPayload(records=[{"a": 1}])
    actions_df, report, model_name = api_shared.exec_decide(payload, serving, "decide")
    assert len(actions_df) == 1
    assert report["n_rows"] == 1
    assert model_name == "models/xgb.joblib"
    assert counter.labels_args["endpoint"] == "decide"
    assert counter.inc_value == 1

    monkeypatch.setattr(api_shared, "ExperimentConfig", lambda: _cfg(max_rows=0))
    with pytest.raises(ValueError, match="Payload too large"):
        api_shared.exec_decide(payload, serving, "decide")


def test_error_response_shape():
    resp = api_shared.error_response(
        status_code=401,
        error_code="unauthorized",
        message="Unauthorized",
        request_id="rid-1",
    )
    assert resp.status_code == 401
    body = json.loads(resp.body.decode("utf-8"))
    assert body["error_code"] == "unauthorized"
    assert body["request_id"] == "rid-1"
