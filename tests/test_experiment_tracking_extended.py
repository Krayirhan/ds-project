from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from src.experiment_tracking import ExperimentTracker


def test_tracker_init_importerror_branch(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setitem(__import__("sys").modules, "mlflow", None)
    monkeypatch.setitem(__import__("sys").modules, "mlflow.sklearn", None)
    tracker = ExperimentTracker()
    assert tracker.active is False


def _active_tracker():
    mock_mlflow = MagicMock()
    mock_sklearn = MagicMock()
    with (
        patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}),
        patch.dict("sys.modules", {"mlflow": mock_mlflow, "mlflow.sklearn": mock_sklearn}),
    ):
        tracker = ExperimentTracker(tracking_uri="http://localhost:5000")
    return tracker, mock_mlflow, mock_sklearn


def test_active_tracker_swallow_exceptions_for_all_loggers():
    tracker, mock_mlflow, mock_sklearn = _active_tracker()
    assert tracker.active is True

    mock_mlflow.log_param.side_effect = RuntimeError("p")
    mock_mlflow.log_metrics.side_effect = RuntimeError("ms")
    mock_mlflow.log_artifact.side_effect = RuntimeError("a")
    mock_mlflow.set_tag.side_effect = RuntimeError("t")
    tracker._sklearn_mod.log_model.side_effect = RuntimeError("m")

    # all should swallow and not raise
    tracker.log_param("k", "v")
    tracker.log_metrics({"x": 1.0})
    tracker.log_artifact("path.json")
    tracker.log_model(object(), "model")
    tracker.set_tag("k", "v")
