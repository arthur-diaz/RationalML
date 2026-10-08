from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from rationalml import AutoML, AutoMLConfig, AutoMLResult, MetricRegistry, ModelRegistry, TaskType
from rationalml.exceptions import ConfigurationError, DataValidationError, UnsupportedTaskError


def small_automl(**options):
    return AutoML(**({
        "target": "target", "models": ["logistic_regression"],
        "cv": 2, "n_trials": 2, "random_state": 17, "verbose": 0,
    } | options))


def test_fit_result_and_predictions_preserve_original_labels_and_data(binary_df):
    binary_df["target"] = binary_df["target"].map({0: "retained", 1: "churn"})
    original = binary_df.copy(deep=True)
    result = small_automl().fit(binary_df)
    assert isinstance(result, AutoMLResult)
    assert result.task is TaskType.BINARY
    assert not result.leaderboard.empty
    assert result.best_model is not None
    assert result.best_model_name == "logistic_regression"
    assert isinstance(result.cv_results, pd.DataFrame)
    assert isinstance(result.feature_importance, pd.DataFrame)
    assert pd.api.types.is_numeric_dtype(result.feature_importance["importance"])
    assert set(result.test_metrics) == set(MetricRegistry.available(TaskType.BINARY))
    assert all(np.isfinite(value) for value in result.test_metrics.values())
    features = binary_df.drop(columns="target")
    predictions, probabilities = result.predict(features), result.predict_proba(features)
    assert predictions.shape == (len(binary_df),)
    assert set(predictions) <= {"retained", "churn"}
    assert probabilities.shape == (len(binary_df), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_array_equal(result.classes_, ["churn", "retained"])
    np.testing.assert_array_equal(result.predict(features.to_numpy()), predictions)
    pd.testing.assert_frame_equal(binary_df, original)
    assert set(result.train_indices).isdisjoint(result.test_indices)
    assert len(result.train_indices) + len(result.test_indices) == len(binary_df)


def test_holdout_never_reaches_optimization_fit_or_cv_and_is_evaluated_once(binary_df, monkeypatch):
    import rationalml.automl as facade

    fit_events, prediction_events, optimizer_inputs = [], [], []

    class RecordingLogistic(LogisticRegression):
        def fit(self, X, y, **fit_options):
            fit_events.append((set(X.index), fit_options))
            return super().fit(X, y, **fit_options)

        def predict(self, X):
            prediction_events.append(("predict", set(X.index)))
            return super().predict(X)

        def predict_proba(self, X):
            prediction_events.append(("proba", set(X.index)))
            return super().predict_proba(X)

    class RecordingRegistry(ModelRegistry):
        _optional = {}
        _specs = {"recording": replace(
            ModelRegistry.get("logistic_regression"), name="recording", estimator_class=RecordingLogistic,
        )}

    optimize = facade.optimize_model

    def record_optimizer(spec, X_train, y_train, metric, config):
        optimizer_inputs.append((set(X_train.index), set(y_train.index)))
        return optimize(spec, X_train, y_train, metric, config)

    monkeypatch.setattr(facade, "optimize_model", record_optimizer)
    result = small_automl(models=["recording"], model_registry=RecordingRegistry).fit(binary_df)
    train, holdout = set(result.train_indices), set(result.test_indices)
    assert optimizer_inputs == [(train, train)]
    assert len(fit_events) == 2 * 2 + 1  # trials * folds + final fit
    assert fit_events[-1][0] == train
    assert all(rows <= train and not options for rows, options in fit_events)
    assert all(rows.isdisjoint(holdout) for rows, _ in fit_events)
    holdout_events = [(kind, rows) for kind, rows in prediction_events if rows & holdout]
    assert holdout_events == [("predict", holdout), ("proba", holdout)]
    assert all(rows <= train for _, rows in prediction_events[:-2])


def test_holdout_features_cannot_change_hyperparameters_or_model_selection(binary_df):
    class ComparisonRegistry(ModelRegistry):
        _optional = {}
        _specs = {name: replace(
            ModelRegistry.get("logistic_regression"), name=name,
            default_params={"solver": "lbfgs", "fit_intercept": intercept},
        ) for name, intercept in [("with_intercept", True), ("without_intercept", False)]}

    options = {"models": list(ComparisonRegistry._specs), "model_registry": ComparisonRegistry}
    first = small_automl(**options).fit(binary_df)
    changed = binary_df.copy(deep=True)
    changed.loc[list(first.test_indices), changed.columns != "target"] *= -1000
    second = small_automl(**options).fit(changed)
    assert first.test_indices == second.test_indices
    assert first.best_model_name == second.best_model_name
    assert first.best_params == second.best_params
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    assert first.test_metrics != second.test_metrics


@pytest.mark.parametrize("model, module", [
    ("logistic_regression", "sklearn"), ("lightgbm", "lightgbm"), ("xgboost", "xgboost"),
])
def test_models_fit_and_are_reproducible(binary_df, model, module):
    pytest.importorskip(module)
    first = small_automl(models=model).fit(binary_df)
    second = small_automl(models=model).fit(binary_df)
    assert first.best_params == second.best_params
    assert first.test_metrics == second.test_metrics
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    np.testing.assert_allclose(
        first.predict_proba(binary_df.drop(columns="target")),
        second.predict_proba(binary_df.drop(columns="target")),
    )


def test_all_models_and_minimizing_metric(binary_df):
    result = small_automl(models="auto", metric="log_loss").fit(binary_df)
    scores = result.leaderboard["cv_score"]
    assert scores.is_monotonic_increasing
    assert result.best_model_name == result.leaderboard.index[0]
    assert len(result.leaderboard) == len(ModelRegistry.available(TaskType.BINARY))
    assert result.best_params["random_state"] == 17
    for name, group in result.cv_results.groupby("model"):
        assert result.leaderboard.loc[name, "cv_score"] == group["value"].min()


def test_split_matches_config_and_uses_positional_rows_with_duplicate_index(binary_df):
    expected_train, expected_test = train_test_split(
        np.arange(len(binary_df)), test_size=0.3, stratify=binary_df["target"], random_state=23,
    )
    binary_df.index = ["same"] * len(binary_df)
    config = AutoMLConfig(target="target", models="logistic_regression", cv=2, n_trials=1,
                          test_size=0.3, random_state=23, verbose=0)
    result = AutoML(config=config).fit(binary_df)
    assert result.train_indices == tuple(expected_train)
    assert result.test_indices == tuple(expected_test)


@pytest.mark.parametrize("corruption", ["missing_target", "nan_feature", "inf_feature", "categorical",
                                         "missing_target_value", "duplicate_column", "no_features"])
def test_bad_data_is_rejected_without_mutation(binary_df, corruption):
    if corruption == "missing_target":
        binary_df = binary_df.drop(columns="target")
    elif corruption == "nan_feature":
        binary_df.loc[0, "feature_0"] = np.nan
    elif corruption == "inf_feature":
        binary_df.loc[0, "feature_0"] = np.inf
    elif corruption == "categorical":
        binary_df["feature_0"] = "category"
    elif corruption == "missing_target_value":
        binary_df["target"] = binary_df["target"].astype(float)
        binary_df.loc[0, "target"] = np.nan
    elif corruption == "duplicate_column":
        binary_df.columns = ["feature_0", "feature_0", *binary_df.columns[2:]]
    else:
        binary_df = binary_df[["target"]]
    original = binary_df.copy(deep=True)
    with pytest.raises(DataValidationError):
        small_automl(preprocessing=None).fit(binary_df)
    pd.testing.assert_frame_equal(binary_df, original)


def test_invalid_input_and_too_small_classes(binary_df):
    with pytest.raises(DataValidationError):
        small_automl().fit([[0, 1]])
    binary_df["target"] = 0
    binary_df.loc[:1, "target"] = 1
    with pytest.raises(DataValidationError, match="cv"):
        small_automl().fit(binary_df)
    with pytest.raises(UnsupportedTaskError):
        small_automl(task="multiclass").fit(binary_df)


def test_prediction_schema_is_explicit(binary_df):
    result = small_automl(n_trials=1).fit(binary_df)
    features = binary_df.drop(columns="target")
    np.testing.assert_array_equal(result.predict(features[features.columns[::-1]]), result.predict(features))
    with pytest.raises(DataValidationError, match="missing"):
        result.predict(features.drop(columns=features.columns[0]))
    with pytest.raises(DataValidationError, match="extra"):
        result.predict(features.assign(extra=1))
    with pytest.raises(DataValidationError, match="2D"):
        result.predict(np.ones(6))


def test_errors_during_training_are_not_hidden(binary_df):
    class BrokenModel(LogisticRegression):
        def fit(self, X, y):
            raise RuntimeError("intentional training failure")

    class BrokenRegistry(ModelRegistry):
        _optional = {}
        _specs = {"broken": replace(
            ModelRegistry.get("logistic_regression"), name="broken", estimator_class=BrokenModel,
        )}

    with pytest.raises(RuntimeError, match="intentional training failure"):
        small_automl(models="broken", model_registry=BrokenRegistry).fit(binary_df)


def test_models_without_probabilities_are_rejected(binary_df):
    class Registry(ModelRegistry):
        _optional = {}
        _specs = {"no_proba": replace(
            ModelRegistry.get("logistic_regression"), name="no_proba", supports_proba=False,
        )}

    with pytest.raises(ConfigurationError, match="predict_proba"):
        small_automl(models="no_proba", model_registry=Registry).fit(binary_df)
