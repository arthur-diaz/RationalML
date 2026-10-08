from dataclasses import replace

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import KFold, StratifiedKFold

from rationalml import AutoML, MetricRegistry, ModelRegistry, TaskType


def fit_task(df, task, **options):
    return AutoML(**({
        "target": "target", "task": task, "models": "auto", "cv": 3,
        "n_trials": 3, "random_state": 17, "verbose": 0,
    } | options)).fit(df)


@pytest.mark.parametrize("task,metric", [
    ("binary", "roc_auc"), ("binary", "log_loss"),
    ("multiclass", "f1_macro"), ("multiclass", "accuracy"), ("multiclass", "log_loss"),
    ("regression", "rmse"), ("regression", "mae"), ("regression", "r2"),
])
def test_baseline_matches_sklearn_and_leaderboard_uses_best_trial_folds(request, task, metric):
    df = request.getfixturevalue(f"{task}_df")
    original = df.copy(deep=True)
    result = fit_task(df, task, metric=metric)
    assert result.baseline_name == ("dummy_regressor" if task == "regression" else "dummy_classifier")
    assert isinstance(result.baseline_fold_scores, tuple) and len(result.baseline_fold_scores) == 3
    assert isinstance(result.baseline_score, float) and isinstance(result.baseline_cv_std, float)
    assert np.isfinite(result.baseline_fold_scores).all()
    X = df.iloc[list(result.train_indices)].drop(columns="target")
    y = df.target.iloc[list(result.train_indices)]
    if task != "regression":
        y = pd.Series(result.label_encoder.transform(y), index=y.index)
    splitter = KFold if task == "regression" else StratifiedKFold
    spec = MetricRegistry.get(metric)
    expected = []
    for training_rows, validation_rows in splitter(n_splits=3, shuffle=True, random_state=17).split(X, y):
        dummy = DummyRegressor(strategy="mean") if task == "regression" else DummyClassifier(strategy="prior")
        dummy.fit(X.iloc[training_rows], y.iloc[training_rows])
        expected.append(spec.evaluate(dummy, X.iloc[validation_rows], y.iloc[validation_rows], task=TaskType(task)))
    np.testing.assert_allclose(result.baseline_fold_scores, expected)
    assert result.baseline_score == pytest.approx(np.mean(expected))
    assert result.baseline_cv_std == pytest.approx(np.std(expected, ddof=0))
    if metric == "roc_auc":
        assert result.baseline_score == pytest.approx(.5)
    assert list(result.leaderboard.columns) == [
        "rank", "cv_score", "cv_std", "cv_min", "cv_max", "improvement_vs_baseline", "n_trials_completed",
    ]
    assert list(result.cv_results.columns) == ["model", "trial", "value", "fold_0", "fold_1", "fold_2"]
    assert pd.api.types.is_integer_dtype(result.leaderboard.n_trials_completed)
    for name, row in result.leaderboard.iterrows():
        trials = result.cv_results.loc[result.cv_results.model == name]
        best_trial = trials.sort_values("value", ascending=spec.direction == "minimize", kind="stable").iloc[0]
        scores = best_trial[["fold_0", "fold_1", "fold_2"]].to_numpy(dtype=float)
        assert row.cv_score == pytest.approx(scores.mean())
        assert row.cv_std == pytest.approx(np.std(scores, ddof=0))
        assert row.cv_min == pytest.approx(min(scores))
        assert row.cv_max == pytest.approx(max(scores))
        assert row.cv_min - 1e-12 <= row.cv_score <= row.cv_max + 1e-12
        delta = row.cv_score - result.baseline_score if spec.direction == "maximize" else result.baseline_score - row.cv_score
        assert row.improvement_vs_baseline == pytest.approx(delta)
        assert row.n_trials_completed == 3
    assert result.best_model_name == result.leaderboard.index[0]
    assert set(result.leaderboard.index) == set(ModelRegistry.available(task))
    assert result.baseline_name not in result.leaderboard.index
    assert not isinstance(result.best_model.named_steps["estimator"], (DummyClassifier, DummyRegressor))
    assert not any("baseline" in name for name in result.test_metrics)
    pd.testing.assert_frame_equal(df, original)


def test_model_losing_to_baseline_is_still_the_requested_winner(binary_df):
    binary_df.target = np.resize([0, 0, 0, 0, 1], len(binary_df))

    class Registry(ModelRegistry):
        _optional = {}
        _specs = {"bad_model": replace(
            ModelRegistry.get("logistic_regression"), name="bad_model", estimator_class=DummyClassifier,
            search_space=lambda trial: {}, default_params={"strategy": "constant", "constant": 1},
        )}

    result = fit_task(binary_df, "binary", metric="accuracy", models="bad_model", model_registry=Registry)
    assert result.best_model_name == "bad_model"
    assert list(result.leaderboard.index) == ["bad_model"]
    assert result.baseline_score > result.leaderboard.loc["bad_model", "cv_score"]
    assert result.leaderboard.loc["bad_model", "improvement_vs_baseline"] < 0
    np.testing.assert_array_equal(result.predict(binary_df.drop(columns="target")), np.ones(len(binary_df)))


@pytest.mark.parametrize("metric", ["rmse", "mae", "r2"])
def test_equivalent_constant_regression_has_zero_improvement(regression_df, metric):
    regression_df.target = 1.
    result = fit_task(regression_df, "regression", metric=metric, models="ridge", n_trials=1)
    assert result.leaderboard.loc["ridge", "improvement_vs_baseline"] == pytest.approx(0.)
    assert result.baseline_cv_std == pytest.approx(0.)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_baseline_and_comparison_are_reproducible(request, task):
    df = request.getfixturevalue(f"{task}_df")
    first, second = fit_task(df, task, n_trials=2), fit_task(df, task, n_trials=2)
    assert first.train_indices == second.train_indices and first.test_indices == second.test_indices
    assert first.baseline_name == second.baseline_name
    assert first.baseline_score == second.baseline_score
    assert first.baseline_cv_std == second.baseline_cv_std
    assert first.baseline_fold_scores == second.baseline_fold_scores
    assert first.best_model_name == second.best_model_name and first.best_params == second.best_params
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    X = df.drop(columns="target")
    if task == "regression":
        np.testing.assert_allclose(first.predict(X), second.predict(X))
    else:
        np.testing.assert_array_equal(first.predict(X), second.predict(X))
        np.testing.assert_array_equal(first.classes_, second.classes_)
        np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))


@pytest.mark.parametrize("task", ["binary", "regression"])
def test_completed_trial_count_excludes_failed_and_pruned_without_real_timeout(request, monkeypatch, task):
    df = request.getfixturevalue(f"{task}_df")
    name = "ridge" if task == "regression" else "logistic_regression"
    metric = MetricRegistry.resolve("auto", TaskType(task))
    params = ModelRegistry.get(name).parameters({}, 17, 1, TaskType(task))
    study = optuna.create_study(direction=metric.direction)
    for scores in ([.7, .8, .6], [.4, .5, .3]):
        study.add_trial(optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE, value=float(np.mean(scores)),
            user_attrs={"fold_scores": list(scores), "estimator_params": params.copy()},
        ))
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED))
    calls = []

    def simulate_finished_budget(objective, **options):
        # A pre-populated study simulates a shortened run, without wall-clock assumptions.
        calls.append(options)

    def create_study(**options):
        assert isinstance(options["pruner"], optuna.pruners.NopPruner)
        return study

    monkeypatch.setattr(study, "optimize", simulate_finished_budget)
    monkeypatch.setattr(optuna, "create_study", create_study)
    result = fit_task(df, task, models=name, n_trials=7, timeout=1)
    assert calls == [{"n_trials": 7, "timeout": 1, "n_jobs": 1}]
    assert result.leaderboard.loc[name, "n_trials_completed"] == 2
    assert len(result.cv_results) == 2
    assert list(result.cv_results.trial) == [0, 1]

