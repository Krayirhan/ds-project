"""Tests for src/experiment_tracking.py — ExperimentTracker."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestExperimentTrackerInactive:
    """ExperimentTracker with no MLFLOW_TRACKING_URI — all ops are no-ops."""

    def test_tracker_inactive_by_default(self):
        """Without MLFLOW_TRACKING_URI, tracker should be inactive."""
        from src.experiment_tracking import ExperimentTracker

        env = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
        with patch.dict(os.environ, env, clear=True):
            tracker = ExperimentTracker()
        assert tracker.active is False

    def test_start_run_noop_when_inactive(self):
        """start_run context manager works as no-op when inactive."""
        from src.experiment_tracking import ExperimentTracker

        with patch.dict(os.environ, {}, clear=True):
            tracker = ExperimentTracker()

        entered = False
        with tracker.start_run("test-run"):
            entered = True

        assert entered  # context manager still yields

    def test_log_param_noop_when_inactive(self):
        from src.experiment_tracking import ExperimentTracker

        with patch.dict(os.environ, {}, clear=True):
            tracker = ExperimentTracker()

        # Should not raise
        tracker.log_param("key", "value")

    def test_log_params_noop_when_inactive(self):
        from src.experiment_tracking import ExperimentTracker

        with patch.dict(os.environ, {}, clear=True):
            tracker = ExperimentTracker()

        tracker.log_params({"seed": 42, "cv_folds": 5})

    def test_log_metric_noop_when_inactive(self):
        from src.experiment_tracking import ExperimentTracker

        with patch.dict(os.environ, {}, clear=True):
            tracker = ExperimentTracker()

        tracker.log_metric("roc_auc", 0.85)
        tracker.log_metric("accuracy", 0.90, step=1)

    def test_log_metrics_noop_when_inactive(self):
        from src.experiment_tracking import ExperimentTracker

        with patch.dict(os.environ, {}, clear=True):
            tracker = ExperimentTracker()

        tracker.log_metrics({"roc_auc": 0.85, "f1": 0.82})

    def test_log_artifact_noop_when_inactive(self):
        from src.experiment_tracking import ExperimentTracker

        with patch.dict(os.environ, {}, clear=True):
            tracker = ExperimentTracker()

        tracker.log_artifact("reports/metrics/cv_summary.json")
        tracker.log_artifact(Path("reports/metrics/model.json"))

    def test_log_model_noop_when_inactive(self):
        from src.experiment_tracking import ExperimentTracker

        with patch.dict(os.environ, {}, clear=True):
            tracker = ExperimentTracker()

        tracker.log_model(MagicMock(), "model")

    def test_set_tag_noop_when_inactive(self):
        from src.experiment_tracking import ExperimentTracker

        with patch.dict(os.environ, {}, clear=True):
            tracker = ExperimentTracker()

        tracker.set_tag("git_sha", "abc123")


class TestExperimentTrackerActive:
    """ExperimentTracker with mocked MLflow and tracking URI set."""

    def _make_tracker(self, mock_mlflow, mock_sklearn):
        """Create an active tracker with mocked mlflow."""
        from src.experiment_tracking import ExperimentTracker

        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}),
            patch.dict(
                "sys.modules",
                {
                    "mlflow": mock_mlflow,
                    "mlflow.sklearn": mock_sklearn,
                },
            ),
        ):
            tracker = ExperimentTracker(tracking_uri="http://localhost:5000")
        return tracker

    def test_tracker_active_with_uri(self):
        mock_mlflow = MagicMock()
        mock_sklearn = MagicMock()
        tracker = self._make_tracker(mock_mlflow, mock_sklearn)
        assert tracker.active is True

    def test_log_params_called_when_active(self):
        mock_mlflow = MagicMock()
        mock_sklearn = MagicMock()
        tracker = self._make_tracker(mock_mlflow, mock_sklearn)

        tracker.log_params({"seed": 42, "cv_folds": 5})
        mock_mlflow.log_params.assert_called_once()

    def test_log_metric_called_when_active(self):
        mock_mlflow = MagicMock()
        mock_sklearn = MagicMock()
        tracker = self._make_tracker(mock_mlflow, mock_sklearn)

        tracker.log_metric("roc_auc", 0.85)
        mock_mlflow.log_metric.assert_called_once_with("roc_auc", 0.85, step=None)

    def test_log_metrics_called_when_active(self):
        mock_mlflow = MagicMock()
        mock_sklearn = MagicMock()
        tracker = self._make_tracker(mock_mlflow, mock_sklearn)

        tracker.log_metrics({"roc_auc": 0.85})
        mock_mlflow.log_metrics.assert_called_once()

    def test_log_artifact_called_when_active(self):
        mock_mlflow = MagicMock()
        mock_sklearn = MagicMock()
        tracker = self._make_tracker(mock_mlflow, mock_sklearn)

        tracker.log_artifact("some/path.json")
        mock_mlflow.log_artifact.assert_called_once_with("some/path.json")

    def test_log_model_called_when_active(self):
        mock_mlflow = MagicMock()
        mock_sklearn = MagicMock()
        tracker = self._make_tracker(mock_mlflow, mock_sklearn)

        dummy_model = MagicMock()
        tracker.log_model(dummy_model, "my_model")
        # _sklearn_mod is the actual mlflow.sklearn attribute used internally
        tracker._sklearn_mod.log_model.assert_called_once_with(dummy_model, "my_model")

    def test_set_tag_called_when_active(self):
        mock_mlflow = MagicMock()
        mock_sklearn = MagicMock()
        tracker = self._make_tracker(mock_mlflow, mock_sklearn)

        tracker.set_tag("env", "production")
        mock_mlflow.set_tag.assert_called_once_with("env", "production")

    def test_log_params_exception_does_not_raise(self):
        """Exceptions in MLflow calls should be swallowed gracefully."""
        mock_mlflow = MagicMock()
        mock_sklearn = MagicMock()
        mock_mlflow.log_params.side_effect = RuntimeError("MLflow server down")
        tracker = self._make_tracker(mock_mlflow, mock_sklearn)

        # Should NOT raise
        tracker.log_params({"seed": 42})

    def test_log_metric_exception_does_not_raise(self):
        mock_mlflow = MagicMock()
        mock_sklearn = MagicMock()
        mock_mlflow.log_metric.side_effect = RuntimeError("connection failed")
        tracker = self._make_tracker(mock_mlflow, mock_sklearn)

        tracker.log_metric("roc_auc", 0.85)  # should not raise

    def test_start_run_uses_mlflow_start_run(self):
        mock_mlflow = MagicMock()
        mock_sklearn = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        tracker = self._make_tracker(mock_mlflow, mock_sklearn)

        with tracker.start_run("my-run"):
            pass

        mock_mlflow.start_run.assert_called()

    def test_params_truncated_to_250_chars(self):
        """log_params should truncate values to 250 chars."""
        mock_mlflow = MagicMock()
        mock_sklearn = MagicMock()
        tracker = self._make_tracker(mock_mlflow, mock_sklearn)

        long_val = "x" * 500
        tracker.log_params({"key": long_val})

        call_args = mock_mlflow.log_params.call_args[0][0]
        assert len(call_args["key"]) <= 250
