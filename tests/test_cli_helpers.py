from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

import src.cli._helpers as h


def test_run_id_and_latest_helpers(tmp_path: Path):
    rid = h.new_run_id()
    assert len(rid) == 15
    assert "_" in rid

    base = tmp_path / "runs"
    h.mark_latest(base, "run-1", extra={"tag": "blue"})
    payload = h.json_read(base / "latest.json")
    assert payload["run_id"] == "run-1"
    assert payload["tag"] == "blue"

    resolved = h.resolve_latest_run_id(base / "latest.json")
    assert resolved == "run-1"


def test_resolve_latest_run_id_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        h.resolve_latest_run_id(tmp_path / "missing-1.json", tmp_path / "missing-2.json")


def test_copy_safe_load_and_read_input_dataset(tmp_path: Path, monkeypatch):
    src = tmp_path / "src.txt"
    src.write_text("content", encoding="utf-8")
    dst = tmp_path / "nested" / "dst.txt"
    h.copy_to_latest(src, dst)
    assert dst.read_text(encoding="utf-8") == "content"

    model_path = tmp_path / "m.joblib"
    joblib.dump({"a": 1}, model_path)
    assert h.safe_load(model_path) == {"a": 1}
    assert h.safe_load(tmp_path / "missing.joblib") is None

    monkeypatch.setattr(h, "read_parquet", lambda p: {"kind": "parquet", "path": str(p)})
    monkeypatch.setattr(h, "read_csv", lambda p: {"kind": "csv", "path": str(p)})

    assert h.read_input_dataset(Path("data.parquet"))["kind"] == "parquet"
    assert h.read_input_dataset(Path("data.csv"))["kind"] == "csv"
    with pytest.raises(ValueError, match="Unsupported input format"):
        h.read_input_dataset(Path("data.json"))


def test_append_dead_letter_writes_jsonl(tmp_path: Path):
    dlq = tmp_path / "dlq" / "events.jsonl"
    h.append_dead_letter(dlq, {"event": "failed"})
    rows = dlq.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    obj = json.loads(rows[0])
    assert obj["payload"]["event"] == "failed"


def test_notify_webhook_no_url_is_noop(monkeypatch):
    called = {"n": 0}

    def _boom(*args, **kwargs):
        called["n"] += 1
        raise RuntimeError("must not be called")

    monkeypatch.setattr(h.urllib.request, "urlopen", _boom)
    h.notify_webhook(None, {"x": 1})
    assert called["n"] == 0


def test_notify_webhook_success(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        def read(self):
            return b"ok"

    def _ok(req, timeout):
        calls["n"] += 1
        assert timeout == 5
        assert req.method == "POST"
        return _Resp()

    monkeypatch.setattr(h.urllib.request, "urlopen", _ok)
    h.notify_webhook("https://example.com/hook", {"status": "ok"})
    assert calls["n"] == 1


def test_notify_webhook_retries_and_writes_dlq(tmp_path: Path, monkeypatch):
    calls = {"n": 0}

    def _fail(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("down")

    monkeypatch.setattr(h.urllib.request, "urlopen", _fail)
    monkeypatch.setattr(h.time, "sleep", lambda _: None)

    dlq = tmp_path / "deadletter.jsonl"
    h.notify_webhook("https://example.com/hook", {"id": 42}, dlq_path=dlq)

    assert calls["n"] == 3
    line = dlq.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["payload"]["id"] == 42


def test_pick_best_policy_no_valid_candidate():
    summary = {
        "models": {
            "xgb": [
                {
                    "max_action_rate": 0.2,
                    "best_threshold": 0.5,
                    "best_profit": float("nan"),
                }
            ]
        }
    }
    out = h.pick_best_policy(summary, prefer_models=["xgb"])
    assert out["status"] == "no_valid_candidate"


def test_pick_best_policy_tie_breaks_with_preference():
    summary = {
        "models": {
            "xgb": [
                {
                    "max_action_rate": 0.2,
                    "best_threshold": 0.5,
                    "best_profit": 10.0,
                }
            ],
            "lgbm": [
                {
                    "max_action_rate": 0.3,
                    "best_threshold": 0.6,
                    "best_profit": 10.0,
                }
            ],
        }
    }
    out = h.pick_best_policy(summary, prefer_models=["lgbm", "xgb"])
    assert out["status"] == "ok"
    assert out["selected"]["model"] == "lgbm"
    assert out["num_top_ties"] == 2


def test_build_policy_payload():
    selected = {
        "model": "xgb",
        "max_action_rate": 0.3,
        "best_threshold": 0.55,
        "best_profit": 123.4,
    }
    payload = h.build_policy_payload(
        run_id="run-9",
        selected=selected,
        cost=SimpleNamespace(tp_value=50, fp_value=-10, fn_value=-80, tn_value=0),
        prefer_models=["xgb"],
        model_registry={"xgb": "models/xgb.joblib"},
        model_checksums={"xgb": "sha256-1"},
        debug={"grid": 10},
        uplift_cfg=SimpleNamespace(
            ranking_mode="threshold",
            segment_col="market_segment",
            tp_value_by_segment={"OTA": 55},
        ),
        contract_cfg=SimpleNamespace(
            policy_version="1.0.0",
            feature_schema_version="2.0.0",
        ),
    )
    assert payload["status"] == "ok"
    assert payload["selected_model"] == "xgb"
    assert payload["selected_model_artifact"] == "models/xgb.joblib"
    assert payload["selected_model_sha256"] == "sha256-1"
    assert payload["uplift"]["segment_col"] == "market_segment"
