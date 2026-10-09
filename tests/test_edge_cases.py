import numpy as np
import pandas as pd
import pytest

from rationalml import AutoML, AutoMLConfig
from rationalml.exceptions import AutoMLError, ConfigurationError, DataValidationError


@pytest.mark.parametrize("names, duplicates", [
    (["logistic_regression", "logistic_regression"], ["logistic_regression"]),
    ([" XGBoost ", "xgboost", "lightgbm", " LIGHTGBM"], ["xgboost", "lightgbm"]),
])
def test_duplicate_model_error_identifies_all_normalized_names(names, duplicates):
    with pytest.raises(ConfigurationError, match="unique model names; duplicate") as raised:
        AutoMLConfig(target="target", models=names)
    for name in duplicates:
        assert repr(name) in str(raised.value)


@pytest.mark.parametrize("option", ["cv", "n_trials", "random_state", "n_jobs", "verbose", "timeout", "test_size"])
@pytest.mark.parametrize("value", [True, False])
def test_numeric_configuration_does_not_accept_booleans(option, value):
    with pytest.raises(ConfigurationError, match=option):
        AutoMLConfig(target="target", **{option: value})


@pytest.mark.parametrize("problem, message", [
    ("absent_target", "target"), ("duplicate_target", "unique"), ("duplicate_feature", "unique"),
    ("missing_target", "missing"), ("one_class", "two target classes"),
    ("binary_three_classes", "two target classes"), ("multiclass_two_classes", "three target classes"),
    ("regression_text", "numeric, finite target"), ("regression_inf", "infinite"),
    ("feature_nan", "missing"), ("feature_inf", "infinite"), ("feature_minus_inf", "infinite"),
    ("feature_text", "basic"),
])
def test_invalid_data_raise_explicit_errors_before_optuna(binary_df, monkeypatch, problem, message):
    import optuna

    df = binary_df.copy(deep=True)
    task, model = "binary", "logistic_regression"
    if problem == "absent_target":
        df = df.drop(columns="target")
    elif problem.startswith("duplicate_"):
        df = pd.concat([df, df[["target" if problem == "duplicate_target" else "feature_0"]]], axis=1)
    elif problem == "missing_target":
        df["target"] = np.nan
    elif problem == "one_class":
        df["target"] = 1
    elif problem == "binary_three_classes":
        df["target"] = np.arange(len(df)) % 3
    elif problem == "multiclass_two_classes":
        task = "multiclass"
    elif problem.startswith("regression_"):
        task, model = "regression", "ridge"
        df["target"] = "text" if problem.endswith("text") else np.inf
    elif problem == "feature_text":
        df["feature_0"] = "unprepared"
    else:
        df.loc[0, "feature_0"] = {"feature_nan": np.nan, "feature_inf": np.inf, "feature_minus_inf": -np.inf}[problem]
    monkeypatch.setattr(optuna, "create_study", lambda **kwargs: pytest.fail("Invalid data reached Optuna"))
    with pytest.raises(AutoMLError, match=message):
        AutoML(target="target", task=task, models=model, cv=2, n_trials=1, verbose=0).fit(df)


@pytest.mark.parametrize("task, values, cv, test_size", [
    ("binary", [0, 1], 2, .2),
    ("binary", [0] * 9 + [1], 2, .2),
    ("binary", [0, 1] * 6, 10, .25),
    ("multiclass", ["a", "b", "c"] * 3, 2, .1),
    ("regression", list(range(6)), 3, .3),
    ("regression", list(range(10)), 2, .1),
])
def test_insufficient_data_fail_before_optuna(monkeypatch, task, values, cv, test_size):
    import optuna

    df = pd.DataFrame({"feature": np.arange(len(values), dtype=float), "target": values})
    monkeypatch.setattr(optuna, "create_study", lambda **kwargs: pytest.fail("Small data reached Optuna"))
    with pytest.raises(DataValidationError, match="split|cv|rows"):
        AutoML(target="target", task=task, models="ridge" if task == "regression" else "logistic_regression",
               cv=cv, test_size=test_size, n_trials=1, verbose=0).fit(df)


def test_prediction_duplicate_columns_raise_library_error(binary_df):
    result = AutoML(target="target", models="logistic_regression", cv=2, n_trials=1, verbose=0).fit(binary_df)
    X = binary_df.drop(columns="target")
    duplicated = pd.concat([X, X[["feature_0"]]], axis=1)
    for method in (result.predict, result.predict_proba, result.predict_positive_proba):
        with pytest.raises(DataValidationError, match="unique"):
            method(duplicated)
