import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rationalml import AutoML, ModelRegistry, TaskType
from rationalml.exceptions import ConfigurationError, UnsupportedTaskError


def fit_regression(df, **options):
    return AutoML(**({"target": "target", "task": "regression", "models": "ridge",
                     "cv": 3, "n_trials": 1, "verbose": 0, "random_state": 17} | options)).fit(df)


@pytest.mark.parametrize("model,module", [
    ("ridge", "sklearn"), ("lightgbm_regressor", "lightgbm"), ("xgboost_regressor", "xgboost"),
])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
def test_regression_models_preprocessing_predictions_and_metrics(regression_df, model, module, mode):
    pytest.importorskip(module)
    numeric = list(regression_df.columns[:-1])
    preprocessing = mode
    if mode is not None:
        regression_df["country"] = np.resize(["France", "Spain", "Germany"], len(regression_df))
        regression_df.loc[::11, numeric[0]] = np.nan
        regression_df.loc[::13, "country"] = np.nan
    if mode == "custom":
        preprocessing = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("country", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["country"]),
        ])
    original = regression_df.copy(deep=True)
    result = fit_regression(regression_df, models=model, preprocessing=preprocessing)
    assert result.task is TaskType.REGRESSION
    assert result.primary_metric == result.config.metric == "rmse"
    assert result.label_encoder is None
    assert isinstance(result.best_model, Pipeline)
    if mode is None:
        assert list(result.best_model.named_steps) == ["estimator"]
    if mode == "custom":
        assert not hasattr(preprocessing, "transformers_")
    X = regression_df.drop(columns="target")
    predictions = result.predict(X)
    assert predictions.shape == (len(X),)
    assert np.issubdtype(predictions.dtype, np.number) and np.isfinite(predictions).all()
    np.testing.assert_allclose(result.predict(X[X.columns[::-1]]), predictions)
    test_X, y = X.iloc[list(result.test_indices)], regression_df.target.iloc[list(result.test_indices)]
    predicted = result.best_model.predict(test_X)
    assert result.test_metrics == pytest.approx({
        "rmse": np.sqrt(mean_squared_error(y, predicted)),
        "mae": mean_absolute_error(y, predicted), "r2": r2_score(y, predicted),
    })
    assert np.isfinite(list(result.test_metrics.values())).all()
    assert pd.api.types.is_numeric_dtype(result.feature_importance.importance)
    assert set(result.feature_importance.feature) == set(result.transformed_feature_names)
    if mode is not None:
        new = X.iloc[:3].copy(deep=True)
        new["country"] = ["unseen", np.nan, "France"]
        new[numeric[0]] = np.nan
        before = new.copy(deep=True)
        assert np.isfinite(result.predict(new)).all()
        pd.testing.assert_frame_equal(new, before)
    pd.testing.assert_frame_equal(regression_df, original)
    restored = pickle.loads(pickle.dumps(result))
    np.testing.assert_allclose(restored.predict(X), predictions)


@pytest.mark.parametrize("metric,ascending", [("rmse", True), ("mae", True), ("r2", False)])
def test_regression_leaderboard_direction(regression_df, metric, ascending):
    result = fit_regression(regression_df, metric=metric, models="auto")
    assert result.primary_metric == metric
    scores = result.leaderboard.cv_score
    assert scores.is_monotonic_increasing if ascending else scores.is_monotonic_decreasing
    assert result.best_model_name == scores.index[0]
    assert set(scores.index) == set(ModelRegistry.available(TaskType.REGRESSION))
    assert set(scores.index) <= {"ridge", "lightgbm_regressor", "xgboost_regressor"}


def test_ridge_signed_coefficients(regression_df):
    result = fit_regression(regression_df)
    importance = result.feature_importance.set_index("feature").loc[list(result.feature_names)]
    assert list(importance.columns) == ["importance", "source_feature"]
    np.testing.assert_allclose(importance.importance, result.best_model.named_steps["estimator"].coef_)


def test_regression_classification_only_interfaces(regression_df):
    with pytest.raises(ConfigurationError, match="positive_class.*binary"):
        fit_regression(regression_df, positive_class=1)
    result = fit_regression(regression_df)
    X = regression_df.drop(columns="target")
    with pytest.raises(UnsupportedTaskError, match="predict_proba is only available for classification tasks"):
        result.predict_proba(X)
    with pytest.raises(UnsupportedTaskError, match="classes_ is only available for classification"):
        _ = result.classes_
    for operation in [lambda: result.positive_class, lambda: result.negative_class,
                      lambda: result.predict_positive_proba(X)]:
        with pytest.raises(UnsupportedTaskError, match="only available for binary"):
            operation()


@pytest.mark.parametrize("model,module", [
    ("ridge", "sklearn"), ("lightgbm_regressor", "lightgbm"), ("xgboost_regressor", "xgboost"),
])
def test_regression_reproducibility(regression_df, model, module):
    pytest.importorskip(module)
    first = fit_regression(regression_df, models=model, n_trials=2)
    second = fit_regression(regression_df, models=model, n_trials=2)
    assert first.train_indices == second.train_indices and first.test_indices == second.test_indices
    assert first.best_model_name == second.best_model_name and first.best_params == second.best_params
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    np.testing.assert_allclose(first.predict(regression_df.drop(columns="target")),
                               second.predict(regression_df.drop(columns="target")))

