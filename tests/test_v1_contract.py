"""Golden V1 table contracts, independent of exact numerical model scores."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rationalml import AutoML, AutoMLConfig, ModelRegistry, TaskType
from rationalml.optimization.optimizer import OptimizedModel
from rationalml.exceptions import (
    AutoMLError, ConfigurationError, DataValidationError, UnsupportedTaskError,
    MissingDependencyError, OptimizationError, LegacyAPIError,
)

LEADERBOARD = ["rank", "cv_score", "cv_std", "cv_min", "cv_max",
               "improvement_vs_baseline", "n_trials_completed"]
CLASSIFICATION_RANKING = ["segment", "count", "score_min", "score_max", "score_mean",
                          "positives", "positive_rate", "population_share", "positive_capture",
                          "cumulative_positive_capture", "lift", "cumulative_lift"]
REGRESSION_RANKING = ["segment", "count", "pred_min", "pred_max", "pred_mean",
                      "actual_mean", "bias", "mae", "rmse"]
CALIBRATION = ["bin", "count", "proba_min", "proba_max", "proba_mean", "observed_rate",
               "calibration_gap", "absolute_calibration_gap", "population_share"]
SUMMARY = ["brier_score", "expected_calibration_error", "max_calibration_error"]


@pytest.fixture(params=[(task, mode, family)
                        for task in ("binary", "multiclass", "regression")
                        for mode in (None, "basic", "custom")
                        for family in ("linear", "lightgbm", "xgboost")])
def result_contract(request):
    task, mode, family = request.param
    if family != "linear":
        pytest.importorskip(family)
    df = request.getfixturevalue(task + "_df").copy(deep=True)
    # An unusual index proves that prediction artifacts retain row labels,
    # rather than silently substituting positions.
    df.index = pd.Index([f"customer-{i * 7}" for i in range(len(df))], name="customer")
    preprocessing = StandardScaler() if mode == "custom" else mode
    if mode == "basic":
        df["country"] = np.array(["France", "Germany", "Spain"])[np.arange(len(df)) % 3]
        df.iloc[::9, 0] = np.nan
    model = ("ridge" if task == "regression" else "logistic_regression") if family == "linear" else (
        family + ("_regressor" if task == "regression" else ""))
    original = df.copy(deep=True)
    result = AutoML(target="target", task=task, models=[model], preprocessing=preprocessing,
                    cv=2, n_trials=1, verbose=0, random_state=17).fit(df)
    pd.testing.assert_frame_equal(df, original)
    return result, df, task, family


def test_v1_table_schemas_and_result_fields(result_contract):
    result, df, task, family = result_contract
    assert result.leaderboard.columns.tolist() == LEADERBOARD
    assert result.leaderboard.index.name == "model"
    assert result.leaderboard.index.tolist() == [result.best_model_name]
    assert result.baseline_name not in result.leaderboard.index
    assert result.cv_results.columns.tolist() == ["model", "trial", "value", "fold_0", "fold_1"]
    assert isinstance(result.best_model, Pipeline)
    assert result.primary_metric == {"binary": "roc_auc", "multiclass": "f1_macro", "regression": "rmse"}[task]
    assert result.config.metric == result.primary_metric
    assert result.task is TaskType(task)
    for name in ("task", "target", "leaderboard", "best_model", "best_model_name", "best_params",
                 "model_best_params", "metrics", "cv_results", "test_metrics", "feature_importance",
                 "config", "feature_names", "feature_schema", "transformed_feature_names", "train_indices",
                 "test_indices", "baseline_name", "baseline_score", "baseline_cv_std", "baseline_fold_scores"):
        assert hasattr(result, name)

    expected_predictions = {
        "binary": ["y_true", "y_pred", "score"],
        "multiclass": ["y_true", "y_pred", "proba_0", "proba_1", "proba_2"],
        "regression": ["y_true", "y_pred", "residual"],
    }[task]
    stored = result.test_predictions
    assert stored.columns.tolist() == expected_predictions
    pd.testing.assert_index_equal(stored.index, df.iloc[list(result.test_indices)].index)
    assert not set(result.feature_names).intersection(stored.columns)
    stored.iloc[0, 0] = "changed" if task == "multiclass" else -999
    assert result.test_predictions.iloc[0, 0] != stored.iloc[0, 0]

    importance_columns = ["feature", "importance", "source_feature"]
    if task == "multiclass" and family == "linear":
        importance_columns.insert(1, "class")
    assert result.feature_importance.columns.tolist() == importance_columns
    assert pd.api.types.is_numeric_dtype(result.feature_importance["importance"])
    assert set(result.feature_importance["feature"]) <= set(result.transformed_feature_names)
    options = {"class_label": result.classes_[0]} if task == "multiclass" else {}
    assert result.ranking_table(**options).columns.tolist() == (
        REGRESSION_RANKING if task == "regression" else CLASSIFICATION_RANKING)
    assert result.top_segment(**options).columns.tolist() == expected_predictions
    if task != "regression":
        assert result.calibration_table(**options).columns.tolist() == CALIBRATION
        assert list(result.calibration_summary(**options)) == SUMMARY


def test_v1_documented_quickstart_runs_without_optional_dependencies(capsys):
    path = Path(__file__).resolve().parents[1] / "examples/quickstart.py"
    spec = importlib.util.spec_from_file_location("rationalml_quickstart", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run()
    assert result.best_model_name == "logistic_regression"
    assert "logistic_regression" in capsys.readouterr().out
    assert AutoMLConfig(target="target").preprocessing is None


@pytest.mark.parametrize("exception", [ConfigurationError, DataValidationError, UnsupportedTaskError,
                                      MissingDependencyError, OptimizationError, LegacyAPIError])
def test_v1_exception_hierarchy(exception):
    assert issubclass(exception, AutoMLError)


@pytest.mark.parametrize("change, exception, fragment", [
    ({"models": ["logistic_regression", "logistic_regression"]}, ConfigurationError, "duplicate"),
    ({"cv": 1}, ConfigurationError, "cv"),
    ({"preprocessing": "auto"}, ConfigurationError, "basic"),
    ({"target": "absent"}, DataValidationError, "missing"),
    ({"positive_class": "absent"}, DataValidationError, "positive_class"),
])
def test_v1_error_type_and_useful_message(binary_df, change, exception, fragment):
    options = {"target": "target", "models": "logistic_regression", "cv": 2,
               "n_trials": 1, "verbose": 0} | change
    with pytest.raises(exception, match=fragment):
        AutoML(**options).fit(binary_df)


def test_v1_task_auto_numeric_ambiguity_and_classification_defaults(binary_df, multiclass_df, regression_df):
    for df, expected in [(binary_df, TaskType.BINARY), (multiclass_df, TaskType.MULTICLASS)]:
        result = AutoML(target="target", models="logistic_regression", cv=2, n_trials=1, verbose=0).fit(df)
        assert result.task is expected
        assert result.primary_metric == ("roc_auc" if expected is TaskType.BINARY else "f1_macro")
    with pytest.raises(UnsupportedTaskError, match="ambiguous"):
        AutoML(target="target", models="ridge", cv=2, n_trials=1, verbose=0).fit(regression_df)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("all_tied", [False, True])
def test_v1_leaderboard_cv_sort_ties_and_requested_cv_row_order(request, monkeypatch, task, all_tied):
    import rationalml.automl as facade

    requested = ["third", "first", "second"]
    base = ModelRegistry.get("ridge" if task == "regression" else "logistic_regression")

    class Registry(ModelRegistry):
        _optional = {}
        _specs = {name: replace(base, name=name) for name in requested}

    def controlled_cv(spec, X, y, metric, config, *, folds):
        score = .5 if all_tied else ({"third": 3., "first": 1., "second": 1.} if task == "regression"
                                    else {"third": .7, "first": .9, "second": .9})[spec.name]
        params = spec.parameters({}, config.random_state, config.n_jobs, task=TaskType(task))
        rows = pd.DataFrame([{"model": spec.name, "trial": trial, "value": score,
                              "fold_0": score, "fold_1": score} for trial in range(2)])
        return OptimizedModel(spec, params, score, (score, score), rows)

    monkeypatch.setattr(facade, "optimize_model", controlled_cv)
    result = AutoML(target="target", task=task, models=requested, model_registry=Registry,
                    cv=2, n_trials=2, verbose=0).fit(request.getfixturevalue(task + "_df"))
    expected = requested if all_tied else ["first", "second", "third"]
    assert result.leaderboard.index.tolist() == expected
    assert result.best_model_name == expected[0]
    assert result.leaderboard["rank"].tolist() == [1, 2, 3]
    assert result.cv_results["model"].tolist() == [name for name in requested for _ in range(2)]
    assert result.cv_results["trial"].tolist() == [0, 1] * 3
    assert result.baseline_name not in result.leaderboard.index
