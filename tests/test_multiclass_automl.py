import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rationalml import AutoML, ModelRegistry, TaskType
from rationalml.exceptions import ConfigurationError, UnsupportedTaskError


def fit_multiclass(df, **options):
    return AutoML(**({"target": "target", "task": "multiclass", "models": "logistic_regression",
                     "cv": 3, "n_trials": 1, "verbose": 0, "random_state": 17} | options)).fit(df)


@pytest.mark.parametrize("model,module", [
    ("logistic_regression", "sklearn"), ("lightgbm", "lightgbm"), ("xgboost", "xgboost"),
])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
def test_multiclass_models_preprocessing_predictions_and_metrics(multiclass_df, model, module, mode):
    pytest.importorskip(module)
    numeric = list(multiclass_df.columns[:-1])
    preprocessing = mode
    if mode is not None:
        multiclass_df["country"] = np.resize(["France", "Spain", "Germany"], len(multiclass_df))
        multiclass_df.loc[::11, numeric[0]] = np.nan
        multiclass_df.loc[::13, "country"] = np.nan
    if mode == "custom":
        preprocessing = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("country", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["country"]),
        ])
    original = multiclass_df.copy(deep=True)
    result = fit_multiclass(multiclass_df, models=model, preprocessing=preprocessing)
    assert result.task is TaskType.MULTICLASS
    assert result.primary_metric == result.config.metric == "f1_macro"
    assert result.config.task is TaskType.MULTICLASS
    np.testing.assert_array_equal(result.classes_, ["bronze", "gold", "silver"])
    assert isinstance(result.best_model, Pipeline)
    if mode is None:
        assert list(result.best_model.named_steps) == ["estimator"]
        assert result.feature_schema is None
    if mode == "custom":
        assert not hasattr(preprocessing, "transformers_")
        assert result.feature_schema is None
    X = multiclass_df.drop(columns="target")
    probabilities = result.predict_proba(X)
    assert probabilities.shape == (len(X), 3)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
    np.testing.assert_array_equal(result.predict(X), result.classes_[probabilities.argmax(axis=1)])
    np.testing.assert_allclose(result.predict_proba(X[X.columns[::-1]]), probabilities)
    test_X = X.iloc[list(result.test_indices)]
    y = result.label_encoder.transform(multiclass_df.target.iloc[list(result.test_indices)])
    predicted = result.best_model.predict(test_X)
    test_probabilities = result.best_model.predict_proba(test_X)
    expected = {"accuracy": accuracy_score(y, predicted), "balanced_accuracy": balanced_accuracy_score(y, predicted),
                "f1_macro": f1_score(y, predicted, average="macro"),
                "f1_weighted": f1_score(y, predicted, average="weighted"),
                "log_loss": log_loss(y, test_probabilities, labels=[0, 1, 2])}
    assert result.test_metrics == pytest.approx(expected)
    assert result.metrics == result.test_metrics
    assert pd.api.types.is_numeric_dtype(result.feature_importance.importance)
    assert set(result.feature_importance.feature) == set(result.transformed_feature_names)
    if mode is not None:
        new = X.iloc[:3].copy(deep=True)
        new["country"] = ["unseen", np.nan, "France"]
        new[numeric[0]] = np.nan
        before = new.copy(deep=True)
        assert result.predict(new).shape == (3,)
        np.testing.assert_allclose(result.predict_proba(new).sum(axis=1), 1, atol=1e-7)
        pd.testing.assert_frame_equal(new, before)
    pd.testing.assert_frame_equal(multiclass_df, original)
    restored = pickle.loads(pickle.dumps(result))
    np.testing.assert_allclose(restored.predict_proba(X), probabilities)


@pytest.mark.parametrize("labels", ["text", "numeric", "categorical"])
def test_multiclass_explicit_labels_and_auto_text(multiclass_df, labels):
    if labels == "numeric":
        multiclass_df.target = multiclass_df.target.map({"bronze": -3, "gold": 99, "silver": 10})
    if labels == "categorical":
        multiclass_df.target = pd.Categorical(multiclass_df.target, categories=["silver", "gold", "bronze", "unused"])
    result = fit_multiclass(multiclass_df, task="multiclass" if labels == "numeric" else "auto")
    assert set(result.classes_) == set(multiclass_df.target)
    assert set(result.predict(multiclass_df.drop(columns="target"))) <= set(multiclass_df.target)


@pytest.mark.parametrize("metric", ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "log_loss"])
def test_multiclass_metric_selection(multiclass_df, metric):
    result = fit_multiclass(multiclass_df, metric=metric)
    assert result.primary_metric == metric
    assert np.isfinite(result.leaderboard.cv_score).all()


def test_multiclass_signed_coefficients_for_every_class_and_feature(multiclass_df):
    result = fit_multiclass(multiclass_df)
    importance = result.feature_importance
    assert list(importance.columns) == ["feature", "class", "importance", "source_feature"]
    assert len(importance) == len(result.classes_) * len(result.feature_names)
    coefficients = result.best_model.named_steps["estimator"].coef_
    for row, label in enumerate(result.classes_):
        actual = importance.loc[importance["class"] == label].set_index("feature").loc[list(result.feature_names)]
        np.testing.assert_allclose(actual.importance, coefficients[row])
        assert list(actual.source_feature) == list(result.feature_names)


def test_multiclass_binary_only_interfaces_and_positive_class(multiclass_df):
    with pytest.raises(ConfigurationError, match="positive_class.*binary"):
        fit_multiclass(multiclass_df, task="auto", positive_class="gold")
    result = fit_multiclass(multiclass_df)
    X = multiclass_df.drop(columns="target")
    for operation in [lambda: result.positive_class, lambda: result.negative_class,
                      lambda: result.predict_positive_proba(X)]:
        with pytest.raises(UnsupportedTaskError, match="only available for binary"):
            operation()


@pytest.mark.parametrize("model,module", [
    ("logistic_regression", "sklearn"), ("lightgbm", "lightgbm"), ("xgboost", "xgboost"),
])
def test_multiclass_reproducibility(multiclass_df, model, module):
    pytest.importorskip(module)
    first = fit_multiclass(multiclass_df, models=model, n_trials=2)
    second = fit_multiclass(multiclass_df, models=model, n_trials=2)
    assert first.train_indices == second.train_indices and first.test_indices == second.test_indices
    assert first.best_model_name == second.best_model_name and first.best_params == second.best_params
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    X = multiclass_df.drop(columns="target")
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))


def test_multiclass_auto_models_are_task_filtered(multiclass_df):
    result = fit_multiclass(multiclass_df, models="auto")
    assert set(result.leaderboard.index) == set(ModelRegistry.available(TaskType.MULTICLASS))
    assert set(result.leaderboard.index) <= {"logistic_regression", "lightgbm", "xgboost"}

