from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from rationalml import AutoML, AutoMLConfig, MetricRegistry, ModelRegistry, TaskType
from rationalml.data import MulticlassLabelEncoder
from rationalml.evaluation import evaluate_metrics
from rationalml.exceptions import DataValidationError, UnsupportedTaskError
from rationalml.tasks import resolve_task


@pytest.mark.parametrize("task,values,expected", [
    ("auto", [0, 1], TaskType.BINARY), ("auto", ["A", "B", "C"], TaskType.MULTICLASS),
    ("multiclass", [0, 1, 2], TaskType.MULTICLASS), ("regression", [.1, .2, .3], TaskType.REGRESSION),
    ("regression", [1, 1, 1], TaskType.REGRESSION), ("multiclass", ["A", "B", "C"], TaskType.MULTICLASS),
])
def test_conservative_task_resolution(task, values, expected):
    assert resolve_task(task, pd.Series(values)) is expected


@pytest.mark.parametrize("values", [[0, 1, 2], [.1, .2, .3], [100, -3, 99]])
@pytest.mark.parametrize("dtype", [None, object, "category"])
def test_auto_numeric_targets_require_explicit_task(values, dtype):
    with pytest.raises(UnsupportedTaskError, match="Numeric targets.*ambiguous.*task='multiclass'.*task='regression'"):
        resolve_task("auto", pd.Series(values, dtype=dtype))


@pytest.mark.parametrize("values", [["a", "b", "c"], [1., np.inf, 3.], [1., -np.inf, 3.], [1., np.nan, 3.], [1j, 2j, 3j]])
def test_regression_requires_finite_numeric_target(values):
    with pytest.raises(DataValidationError, match="numeric|missing"):
        resolve_task("regression", pd.Series(values))


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_missing_target_is_rejected_for_all_tasks(task):
    with pytest.raises(DataValidationError, match="missing"):
        resolve_task(task, pd.Series([0, 1, 2, None]))


@pytest.mark.parametrize("task,metric", [("binary", "roc_auc"), ("multiclass", "f1_macro"), ("regression", "rmse")])
def test_default_metric_is_resolved_after_task(request, task, metric):
    config = AutoMLConfig(target="target")
    assert config.metric == "auto" and config.preprocessing is None
    df = request.getfixturevalue(f"{task}_df")
    result = AutoML(target="target", task=task, models="ridge" if task == "regression" else "logistic_regression",
                    n_trials=1, cv=2, verbose=0).fit(df)
    assert result.primary_metric == result.config.metric == metric


@pytest.mark.parametrize("task,metric", [("multiclass", "roc_auc"), ("multiclass", "precision"),
                                         ("multiclass", "rmse"), ("regression", "log_loss"),
                                         ("regression", "accuracy"), ("binary", "f1_macro"), ("binary", "mae")])
def test_incompatible_metric_is_rejected_before_optimization(request, monkeypatch, task, metric):
    import rationalml.automl as facade

    def unexpected(*args):
        pytest.fail("Invalid metrics must fail before optimization")

    monkeypatch.setattr(facade, "optimize_model", unexpected)
    with pytest.raises(UnsupportedTaskError, match=f"Metric.*does not support {task}"):
        AutoML(target="target", task=task, metric=metric).fit(request.getfixturevalue(f"{task}_df"))


@pytest.mark.parametrize("task,model", [("regression", "logistic_regression"), ("multiclass", "ridge"), ("binary", "ridge")])
def test_incompatible_models_are_explicit(request, task, model):
    with pytest.raises(UnsupportedTaskError, match=f"Model.*does not support {task}"):
        AutoML(target="target", task=task, models=model).fit(request.getfixturevalue(f"{task}_df"))


def test_new_task_metric_metadata_and_single_evaluation_calls():
    assert set(MetricRegistry.available("multiclass")) == {"accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "log_loss"}
    assert set(MetricRegistry.available("regression")) == {"rmse", "mae", "r2"}

    class Estimator:
        calls = []

        def predict(self, X):
            self.calls.append("predict")
            return np.array([0, 1, 2])

        def predict_proba(self, X):
            self.calls.append("proba")
            return np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])

    estimator = Estimator()
    X, y = pd.DataFrame({"x": [1, 2, 3]}), pd.Series([0, 1, 2])
    metrics = evaluate_metrics(estimator, X, y, TaskType.MULTICLASS)
    assert estimator.calls == ["predict", "proba"]
    assert metrics["log_loss"] == pytest.approx(-np.log(.8))
    estimator.calls.clear()
    assert evaluate_metrics(estimator, X, y, TaskType.REGRESSION) == {"rmse": 0., "mae": 0., "r2": 1.}
    assert estimator.calls == ["predict"]
    for name in MetricRegistry.available("regression"):
        spec = MetricRegistry.get(name)
        assert spec.direction == ("maximize" if name == "r2" else "minimize")
        assert spec.prediction_type == "predict"
    assert MetricRegistry.get("log_loss").score(pd.Series([0, 1]), np.array([[.8, .1, .1], [.1, .8, .1]])) == pytest.approx(-np.log(.8))


@pytest.mark.parametrize("name,module,objective", [("lightgbm", "lightgbm", "multiclass"), ("xgboost", "xgboost", "multi:softprob")])
def test_multiclass_objective_is_explicit(name, module, objective):
    pytest.importorskip(module)
    spec = ModelRegistry.get(name)
    assert spec.parameters({}, 42, 1, TaskType.MULTICLASS)["objective"] == objective
    assert spec.parameters({}, 42, 1)["objective"] in {"binary", "binary:logistic"}


def test_model_registry_filters_regular_and_optional_specs(monkeypatch):
    import rationalml.models.registry as registry

    monkeypatch.setattr(registry, "find_spec", lambda module: object())
    assert set(ModelRegistry.available("binary")) == {"logistic_regression", "lightgbm", "xgboost"}
    assert set(ModelRegistry.available("multiclass")) == {"logistic_regression", "lightgbm", "xgboost"}
    assert set(ModelRegistry.available("regression")) == {"ridge", "lightgbm_regressor", "xgboost_regressor"}
    monkeypatch.setattr(registry, "find_spec", lambda module: None)
    assert ModelRegistry.available("regression") == ["ridge"]
    assert ModelRegistry.available("multiclass") == ["logistic_regression"]
    monkeypatch.setattr(ModelRegistry, "_specs", ModelRegistry._specs.copy())
    ModelRegistry.register(replace(ModelRegistry.get("ridge"), name="custom_regression"))
    assert "custom_regression" in ModelRegistry.available("regression")
    assert "custom_regression" not in ModelRegistry.available("multiclass")


def test_multiclass_encoder_rejects_unknown_labels_and_restores_originals():
    labels = pd.Series(["C", "A", "B", "A"])
    encoder = MulticlassLabelEncoder.from_target(labels)
    np.testing.assert_array_equal(encoder.classes_, ["A", "B", "C"])
    np.testing.assert_array_equal(encoder.transform(labels), [2, 0, 1, 0])
    np.testing.assert_array_equal(encoder.inverse_transform(encoder.transform(labels)), labels)
    with pytest.raises(DataValidationError, match="outside.*fitted"):
        encoder.transform(pd.Series(["unknown"]))
    with pytest.raises(DataValidationError, match="internal class"):
        encoder.inverse_transform(np.array([3]))
    with pytest.raises(DataValidationError, match="uniformly numeric or strings"):
        MulticlassLabelEncoder.from_target(pd.Series([1, "B", "C"]))


@pytest.mark.parametrize("task", ["multiclass", "regression"])
@pytest.mark.parametrize("mode", [None, "custom"])
def test_requires_scaling_has_no_effect_outside_basic(request, monkeypatch, task, mode):
    def unexpected(*args, **kwargs):
        pytest.fail("No automatic StandardScaler outside basic")

    monkeypatch.setattr(StandardScaler, "fit", unexpected)
    result = AutoML(target="target", task=task, models="ridge" if task == "regression" else "logistic_regression",
                    preprocessing=SimpleImputer() if mode == "custom" else None,
                    cv=2, n_trials=1, verbose=0).fit(request.getfixturevalue(f"{task}_df"))
    assert result.feature_schema is None


@pytest.mark.parametrize("model,module,scaled", [("ridge", "sklearn", True), ("lightgbm_regressor", "lightgbm", False),
                                                ("xgboost_regressor", "xgboost", False)])
def test_basic_scaling_uses_model_metadata_for_regression(regression_df, model, module, scaled):
    pytest.importorskip(module)
    result = AutoML(target="target", task="regression", models=model, preprocessing="basic",
                    n_trials=1, cv=2, verbose=0).fit(regression_df)
    steps = result.best_model.named_steps["preprocessing"].named_transformers_["numeric"].named_steps
    assert ("scaler" in steps) == scaled


def test_insufficient_regression_samples_are_explicit(regression_df):
    with pytest.raises(DataValidationError, match="two rows per validation fold"):
        AutoML(target="target", task="regression", cv=5).fit(regression_df.iloc[:8])


@pytest.mark.parametrize("task", ["multiclass", "regression"])
def test_unknown_custom_feature_names_never_break_new_tasks(request, task):
    class UnnamedImputer(SimpleImputer):
        get_feature_names_out = None

    with pytest.warns(UserWarning, match="feature_importance is unavailable"):
        result = AutoML(target="target", task=task, models="ridge" if task == "regression" else "logistic_regression",
                        preprocessing=UnnamedImputer(), n_trials=1, cv=2, verbose=0).fit(request.getfixturevalue(f"{task}_df"))
    assert result.transformed_feature_names is None and result.feature_importance.empty
    assert result.predict(request.getfixturevalue(f"{task}_df").drop(columns="target")).ndim == 1

