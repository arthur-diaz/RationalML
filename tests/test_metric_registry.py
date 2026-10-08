import numpy as np
import pandas as pd
import pytest

from rationalml import MetricRegistry, MetricSpec, TaskType
from rationalml.evaluation import evaluate_binary


def test_binary_metric_metadata():
    assert set(MetricRegistry.available(TaskType.BINARY)) == {
        "roc_auc", "average_precision", "accuracy", "precision", "recall", "f1",
        "balanced_accuracy", "log_loss",
    }
    assert MetricRegistry.available(TaskType.REGRESSION) == []
    assert MetricRegistry.get("log_loss").direction == "minimize"
    assert MetricRegistry.get("roc_auc").prediction_type == "proba"
    assert MetricRegistry.get("accuracy").prediction_type == "predict"
    with pytest.raises(KeyError, match="Unknown metric"):
        MetricRegistry.get("missing")


def test_metrics_dispatch_predict_or_proba_and_evaluate_once():
    class Estimator:
        predictions = 0
        probabilities = 0

        def predict(self, X):
            self.predictions += 1
            return np.array([0, 0, 1, 1])

        def predict_proba(self, X):
            self.probabilities += 1
            return np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])

    X, y = pd.DataFrame({"x": range(4)}), pd.Series([0, 0, 1, 1])
    estimator = Estimator()
    assert MetricRegistry.get("roc_auc").evaluate(estimator, X, y) == 1.0
    assert (estimator.predictions, estimator.probabilities) == (0, 1)
    assert MetricRegistry.get("accuracy").evaluate(estimator, X, y) == 1.0
    estimator.predictions = estimator.probabilities = 0
    metrics = evaluate_binary(estimator, X, y)
    assert metrics["f1"] == 1.0
    assert metrics["log_loss"] > 0
    assert (estimator.predictions, estimator.probabilities) == (1, 1)


def test_precision_with_no_positive_predictions():
    assert MetricRegistry.get("precision").score(pd.Series([0, 1]), np.array([0, 0])) == 0.0


def test_registry_extension_without_engine_changes(monkeypatch):
    monkeypatch.setattr(MetricRegistry, "_specs", MetricRegistry._specs.copy())
    spec = MetricSpec("custom", lambda y, p: 0.25, "minimize", "predict", frozenset({TaskType.BINARY}))
    MetricRegistry.register(spec)
    assert MetricRegistry.get("custom") is spec
    with pytest.raises(ValueError, match="already registered"):
        MetricRegistry.register(spec)


def test_nonfinite_metric_is_not_hidden():
    spec = MetricSpec("broken", lambda y, p: float("nan"), "maximize", "predict", frozenset({TaskType.BINARY}))
    with pytest.raises(ValueError, match="non-finite"):
        spec.score(pd.Series([0, 1]), np.array([0, 1]))
