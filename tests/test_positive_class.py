from dataclasses import replace
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from rationalml import AutoML, AutoMLConfig, ModelRegistry, ModelSpec, TaskType
from rationalml.data import BinaryLabelEncoder
from rationalml.exceptions import DataValidationError


def fit_binary(df, **options):
    return AutoML(**({
        "target": "target", "models": "logistic_regression", "cv": 2,
        "n_trials": 1, "random_state": 17, "verbose": 0,
    } | options)).fit(df)


@pytest.fixture
def churn_df(binary_df):
    df = binary_df.copy(deep=True)
    df["target"] = df["target"].map({0: "retained", 1: "churn"})
    return df


@pytest.mark.parametrize("model, module", [
    ("logistic_regression", "sklearn"), ("lightgbm", "lightgbm"), ("xgboost", "xgboost"),
])
def test_churn_is_internal_one_and_probability_columns_are_explicit(churn_df, model, module, monkeypatch):
    pytest.importorskip(module)
    import rationalml.automl as facade

    original = churn_df.copy(deep=True)
    optimizer = facade.optimize_model
    seen_train = []

    def inspect_train(spec, X_train, y_train, metric, config):
        expected = churn_df.loc[X_train.index, "target"].eq("churn").astype(int)
        np.testing.assert_array_equal(y_train, expected)
        seen_train.append(set(X_train.index))
        return optimizer(spec, X_train, y_train, metric, config)

    monkeypatch.setattr(facade, "optimize_model", inspect_train)
    result = fit_binary(churn_df, models=model, positive_class="churn")
    assert result.positive_class == "churn"
    assert result.negative_class == "retained"
    assert result.config.positive_class == "churn"
    np.testing.assert_array_equal(result.classes_, ["retained", "churn"])
    np.testing.assert_array_equal(result.best_model.classes_, [0, 1])
    assert seen_train == [set(result.train_indices)]
    assert seen_train[0].isdisjoint(result.test_indices)
    np.testing.assert_array_equal(result.label_encoder.transform(pd.Series(["retained", "churn"])), [0, 1])
    features = churn_df.drop(columns="target")
    probabilities = result.predict_proba(features)
    positive_probability = result.predict_positive_proba(features)
    assert positive_probability.shape == (len(churn_df),)
    np.testing.assert_allclose(positive_probability, probabilities[:, 1])
    np.testing.assert_allclose(probabilities[:, 0], 1 - positive_probability)
    np.testing.assert_allclose(positive_probability, result.best_model.predict_proba(features)[:, 1])
    np.testing.assert_allclose(result.predict_positive_proba(features.to_numpy()), positive_probability)
    np.testing.assert_array_equal(
        result.predict(features),
        np.where(result.best_model.predict(features) == 1, "churn", "retained"),
    )
    # Callers cannot change the label contract by modifying a returned array.
    exposed_classes = result.classes_
    exposed_classes[1] = "unexpected"
    np.testing.assert_array_equal(result.classes_, ["retained", "churn"])
    pd.testing.assert_frame_equal(churn_df, original)


class AlwaysPositiveRegistry(ModelRegistry):
    _optional = {}
    _specs = {"always_positive": ModelSpec(
        name="always_positive", estimator_class=DummyClassifier,
        tasks=frozenset({TaskType.BINARY}), search_space=lambda trial: {},
        default_params={"strategy": "constant", "constant": 1}, n_jobs_parameter=None,
    )}


@pytest.mark.parametrize("metric_name", ["precision", "recall", "f1", "roc_auc", "average_precision", "log_loss"])
def test_cv_and_test_metrics_use_churn_as_positive(churn_df, metric_name):
    result = fit_binary(
        churn_df, positive_class="churn", metric=metric_name,
        models="always_positive", model_registry=AlwaysPositiveRegistry,
    )
    features = churn_df.drop(columns="target")
    holdout = features.iloc[list(result.test_indices)]
    original_test = churn_df["target"].iloc[list(result.test_indices)]
    predictions = result.predict(holdout)
    for name, scorer in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]:
        expected = scorer(original_test, predictions, pos_label="churn", zero_division=0)
        assert result.test_metrics[name] == pytest.approx(expected)
        assert expected > scorer(original_test, predictions, pos_label="retained", zero_division=0)
    encoded_test = original_test.eq("churn").astype(int)
    positive_probability = result.predict_positive_proba(holdout)
    assert result.test_metrics["roc_auc"] == pytest.approx(roc_auc_score(encoded_test, positive_probability))
    assert result.test_metrics["average_precision"] == pytest.approx(average_precision_score(encoded_test, positive_probability))
    assert result.test_metrics["log_loss"] == pytest.approx(log_loss(encoded_test, positive_probability, labels=[0, 1]))

    original_train = churn_df["target"].iloc[list(result.train_indices)]
    folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=17)
    scorers = {
        "precision": precision_score, "recall": recall_score, "f1": f1_score,
        "roc_auc": roc_auc_score, "average_precision": average_precision_score, "log_loss": log_loss,
    }
    expected_fold_scores = []
    for _, validation_rows in folds.split(features.iloc[list(result.train_indices)], original_train):
        positive_labels = original_train.iloc[validation_rows].eq("churn").astype(int)
        params = {"zero_division": 0} if metric_name in {"precision", "recall", "f1"} else {}
        if metric_name == "log_loss":
            params["labels"] = [0, 1]
        expected_fold_scores.append(scorers[metric_name](positive_labels, np.ones(len(positive_labels)), **params))
    for fold, expected in enumerate(expected_fold_scores):
        assert result.cv_results.loc[0, f"fold_{fold}"] == pytest.approx(expected)


@pytest.mark.parametrize("positive", ["missing", np.nan, pd.NA, ["churn", "retained"]])
def test_invalid_positive_class_fails_before_optimization(churn_df, positive, monkeypatch):
    import rationalml.automl as facade

    def unexpected_optimization(*args):
        pytest.fail("An invalid positive_class must fail before any optimization.")

    monkeypatch.setattr(facade, "optimize_model", unexpected_optimization)
    original = churn_df.copy(deep=True)
    with pytest.raises(DataValidationError, match="positive_class"):
        fit_binary(churn_df, positive_class=positive)
    pd.testing.assert_frame_equal(churn_df, original)


def test_standard_zero_one_defaults_do_not_warn(binary_df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = fit_binary(binary_df)
    assert not caught
    assert result.positive_class == 1
    assert result.negative_class == 0
    np.testing.assert_array_equal(result.classes_, [0, 1])


def test_boolean_defaults_use_true_without_warning(binary_df):
    binary_df["target"] = binary_df["target"].astype(bool)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = fit_binary(binary_df)
    assert not caught
    assert result.positive_class is True
    assert result.negative_class is False
    np.testing.assert_array_equal(result.classes_, [False, True])
    predictions = result.predict(binary_df.drop(columns="target"))
    assert predictions.dtype == np.dtype(bool)


@pytest.mark.parametrize("labels, positive, negative", [
    ([0, 1], 0, 1), ([False, True], False, True),
    ([-3, 9], -3, 9), (["", "churn"], "", "churn"),
])
def test_explicit_negative_sorted_label_can_be_positive(binary_df, labels, positive, negative):
    binary_df["target"] = binary_df["target"].map({0: labels[0], 1: labels[1]})
    result = fit_binary(binary_df, positive_class=positive)
    assert result.positive_class == positive
    assert result.negative_class == negative
    np.testing.assert_array_equal(result.classes_, [negative, positive])
    np.testing.assert_array_equal(
        result.label_encoder.transform(binary_df["target"]), binary_df["target"].eq(positive).astype(int),
    )


def test_string_defaults_warn_and_are_deterministic(churn_df):
    with pytest.warns(UserWarning, match="selected 'retained'.*Set positive_class explicitly"):
        first = fit_binary(churn_df)
    with pytest.warns(UserWarning, match="selected 'retained'.*Set positive_class explicitly"):
        second = fit_binary(churn_df.iloc[::-1])
    assert first.positive_class == second.positive_class == "retained"
    assert first.negative_class == second.negative_class == "churn"
    np.testing.assert_array_equal(first.classes_, ["churn", "retained"])


def test_nonstandard_numeric_defaults_warn(binary_df):
    binary_df["target"] = binary_df["target"].map({0: -3, 1: 9})
    with pytest.warns(UserWarning, match="selected 9.*positive_class explicitly"):
        result = fit_binary(binary_df)
    assert result.positive_class == 9
    assert result.negative_class == -3


def test_config_positive_class_is_data_independent(churn_df):
    config = AutoMLConfig(target="target", positive_class="churn", models="logistic_regression",
                          cv=2, n_trials=1, verbose=0)
    assert AutoMLConfig(target="target").positive_class is None
    assert replace(config).positive_class == "churn"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = AutoML(config).fit(churn_df)
    assert not caught
    assert result.positive_class == "churn"
    assert config.positive_class == "churn"


def test_binary_encoder_rejects_unknown_labels_and_invalid_predictions():
    encoder = BinaryLabelEncoder.from_target(pd.Series(["retained", "churn"]), "churn")
    with pytest.raises(DataValidationError, match="outside"):
        encoder.transform(pd.Series(["other"]))
    with pytest.raises(DataValidationError, match="0/1"):
        encoder.inverse_transform(np.array([0, 2]))
    np.testing.assert_array_equal(encoder.inverse_transform(np.array([0, 1])), ["retained", "churn"])
