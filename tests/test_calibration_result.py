from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from rationalml import AutoML
from rationalml.exceptions import ConfigurationError, UnsupportedTaskError


METHODS = ["calibration_table", "calibration_summary"]


def fit_result(df, task="binary", **options):
    return AutoML(**({"target": "target", "task": task, "models": "ridge" if task == "regression" else "logistic_regression",
                     "n_trials": 1, "cv": 2, "random_state": 17, "verbose": 0} | options)).fit(df)


@pytest.mark.parametrize("labels,positive", [([0, 1], 1), ([False, True], True),
                                            (["retained", "churn"], "churn"), ([0, 1], 0)])
def test_binary_uses_original_explicit_positive_class(binary_df, labels, positive):
    binary_df.target = binary_df.target.map(dict(enumerate(labels)))
    result = fit_result(binary_df, positive_class=positive)
    y = np.array([labels[0], labels[0], labels[1], labels[1]])
    artifact = pd.DataFrame({"y_true": y, "score": [.1, .2, .7, .8]})
    result = replace(result, _test_predictions=artifact)
    positives = (y == positive).astype(float)
    assert result.positive_class == positive
    table = result.calibration_table(2)
    np.testing.assert_allclose(table.observed_rate, [positives[:2].mean(), positives[2:].mean()])
    assert result.calibration_summary(2)["brier_score"] == pytest.approx(np.mean((positives - artifact.score.to_numpy()) ** 2))


@pytest.mark.parametrize("class_label", ["bronze", "gold", 0])
def test_multiclass_one_vs_rest_uses_matching_probability_position(multiclass_df, class_label):
    if class_label == 0:
        multiclass_df.target = multiclass_df.target.map({"bronze": 0, "gold": 1, "silver": 2})
        labels = [0, 1, 2]
    else:
        labels = ["bronze", "gold", "silver"]
    result = fit_result(multiclass_df, task="multiclass")
    targets = [labels[1], labels[0], labels[2], labels[1]]
    probabilities = {labels[0]: [.1, .7, .2, .1], labels[1]: [.8, .2, .1, .7], labels[2]: [.1, .1, .7, .2]}
    artifact = pd.DataFrame({"y_true": targets,
                            **{f"proba_{j}": probabilities[label] for j, label in enumerate(result.classes_)}})
    result = replace(result, _test_predictions=artifact)
    p = np.asarray(probabilities[class_label])
    y = np.asarray(targets) == class_label
    table = result.calibration_table(2, class_label=class_label)
    order = np.argsort(p, kind="stable")
    np.testing.assert_allclose(table.proba_mean, [p[order[:2]].mean(), p[order[2:]].mean()])
    np.testing.assert_allclose(table.observed_rate, [y[order[:2]].mean(), y[order[2:]].mean()])
    assert result.calibration_summary(2, class_label=class_label)["brier_score"] == pytest.approx(np.mean((y - p) ** 2))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("class_label", [None, "unknown"])
def test_multiclass_requires_known_class_label(multiclass_df, method, class_label):
    result = fit_result(multiclass_df, task="multiclass")
    message = "^Multiclass calibration requires class_label\\.$" if class_label is None else "available classes:.*bronze.*gold.*silver"
    with pytest.raises(ConfigurationError, match=message):
        getattr(result, method)(class_label=class_label)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("class_label", [0, 1, "churn"])
def test_binary_rejects_class_label(binary_df, method, class_label):
    result = fit_result(binary_df)
    with pytest.raises(ConfigurationError, match="class_label.*multiclass calibration"):
        getattr(result, method)(class_label=class_label)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("options", [{}, {"class_label": 0}, {"n_bins": True}])
def test_regression_always_explicitly_unsupported(regression_df, method, options):
    result = fit_result(regression_df, task="regression")
    with pytest.raises(UnsupportedTaskError, match="^Probability calibration diagnostics are only available for classification tasks\\.$"):
        getattr(result, method)(**options)


@pytest.mark.parametrize("task", ["binary", "multiclass"])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
def test_diagnostics_never_call_ml_and_preserve_the_entire_result(request, monkeypatch, task, mode):
    import optuna
    import rationalml.automl as facade
    import rationalml.evaluation.metrics as metrics
    import rationalml.evaluation.predictions as predictions
    import rationalml.optimization.optimizer as optimizer
    import rationalml.preprocessing.builder as builder

    df = request.getfixturevalue(f"{task}_df").copy()
    if task == "binary":
        df.target = df.target.map({0: "retained", 1: "churn"})
    if mode is not None:
        df.loc[::11, "feature_0"] = np.nan
    preprocessing = SimpleImputer().set_output(transform="pandas") if mode == "custom" else mode
    result = fit_result(df, task, preprocessing=preprocessing, positive_class="churn" if task == "binary" else None)
    artifact = result.test_predictions
    assert set(artifact.index) == set(result.test_indices)
    assert set(artifact.index).isdisjoint(result.train_indices)
    frames = {name: getattr(result, name).copy(deep=True) for name in ["leaderboard", "cv_results", "feature_importance", "test_predictions"]}
    values = {name: deepcopy(getattr(result, name)) for name in ["best_model_name", "best_params", "model_best_params",
        "test_metrics", "metrics", "baseline_name", "baseline_score", "baseline_cv_std", "baseline_fold_scores"]}
    model, config = result.best_model, result.config
    label = "gold" if task == "multiclass" else None

    def forbidden(*args, **kwargs):
        pytest.fail("Calibration must only read the stored holdout artifact")

    for owner in [result, model, *model.named_steps.values()]:
        for name in ["fit", "predict", "predict_proba", "transform", "fit_transform"]:
            if callable(getattr(owner, name, None)):
                monkeypatch.setattr(owner, name, forbidden)
    for owner, names in [(facade, ["optimize_model", "evaluate_holdout", "evaluate_baseline", "make_cv_splits", "build_model_pipeline"]),
        (optimizer, ["optimize_model"]), (metrics, ["evaluate_metrics"]), (predictions, ["evaluate_holdout"]),
        (builder, ["build_preprocessor", "build_model_pipeline"]), (optuna, ["create_study"]), (optuna.study.Study, ["optimize"])]:
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    df.drop(df.index, inplace=True)
    table = result.calibration_table(3, class_label=label)
    summary = result.calibration_summary(3, class_label=label)
    assert table["count"].sum() == len(result.test_indices)
    # The estimator itself can be absent without affecting the diagnostics.
    monkeypatch.setattr(result, "best_model", None)
    pd.testing.assert_frame_equal(result.calibration_table(3, class_label=label), table)
    assert result.calibration_summary(3, class_label=label) == summary
    monkeypatch.setattr(result, "best_model", model)
    table.loc[0, "calibration_gap"] = 999.
    summary["brier_score"] = 999.
    assert result.calibration_summary(3, class_label=label)["brier_score"] != 999.
    assert result.calibration_table(3, class_label=label).loc[0, "calibration_gap"] != 999.
    for name, saved in frames.items():
        pd.testing.assert_frame_equal(getattr(result, name), saved)
    for name, saved in values.items():
        assert getattr(result, name) == saved
    assert result.best_model is model and result.config is config


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_changing_only_holdout_targets_cannot_change_selection(request, monkeypatch, task):
    import rationalml.automl as facade

    df = request.getfixturevalue(f"{task}_df")
    first = fit_result(df, task)
    changed = df.copy(deep=True)
    holdout = list(first.test_indices)
    if task == "binary":
        changed.loc[holdout, "target"] = 1 - changed.loc[holdout, "target"]
    else:
        changed.loc[holdout, "target"] = changed.loc[holdout, "target"].map({"bronze": "gold", "gold": "silver", "silver": "bronze"})

    def fixed_split(rows, **kwargs):
        return np.asarray(first.train_indices), np.asarray(first.test_indices)

    monkeypatch.setattr(facade, "train_test_split", fixed_split)
    second = fit_result(changed, task)
    label = "gold" if task == "multiclass" else None
    assert not first.calibration_table(class_label=label).equals(second.calibration_table(class_label=label))
    assert first.best_model_name == second.best_model_name and first.best_params == second.best_params
    assert first.model_best_params == second.model_best_params
    assert first.baseline_score == second.baseline_score and first.baseline_fold_scores == second.baseline_fold_scores
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    pd.testing.assert_frame_equal(first.feature_importance, second.feature_importance)
    np.testing.assert_allclose(first.predict_proba(df.drop(columns="target")), second.predict_proba(df.drop(columns="target")))
