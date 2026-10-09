from dataclasses import replace
from io import StringIO
import logging
import pickle
import warnings

import numpy as np
import optuna
import pandas as pd
import pytest

from rationalml import AutoML, AutoMLConfig, ModelRegistry
from rationalml.exceptions import ConfigurationError
from rationalml.runtime import TrialProgress
from _robustness import assert_training_equal


def fit(df, verbose=0, **options):
    return AutoML(**(dict(target="target", models="logistic_regression", cv=2,
                         n_trials=2, random_state=17, verbose=verbose) | options)).fit(df)


@pytest.mark.parametrize("verbose", [0, 1, 2, 3, 10, 10**30])
def test_verbose_accepts_nonnegative_integers_without_upper_bound(verbose):
    assert AutoMLConfig(target="target", verbose=verbose).verbose == verbose


@pytest.mark.parametrize("verbose", [-1, True, False, 1.5])
def test_verbose_rejects_negative_values_booleans_and_nonintegers(verbose):
    with pytest.raises(ConfigurationError, match="verbose"):
        AutoMLConfig(target="target", verbose=verbose)


@pytest.mark.parametrize("model, module", [("logistic_regression", "sklearn"),
                                           ("lightgbm", "lightgbm"), ("xgboost", "xgboost")])
def test_silent_fit_has_no_normal_output(binary_df, capfd, model, module):
    pytest.importorskip(module)
    fit(binary_df, models=model)
    output = capfd.readouterr()
    assert output.out == output.err == ""


@pytest.mark.parametrize("verbose", [1, 2, 3, 10])
def test_verbose_run_presentation_trials_and_final_summary(binary_df, capfd, verbose):
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("optuna")
    logger.addHandler(handler)
    try:
        result = fit(binary_df, verbose)
    finally:
        logger.removeHandler(handler)
    output = capfd.readouterr()
    assert "RationalML - binary classification" in output.out
    assert "CV folds      2" in output.out
    assert result.summary() in output.out
    if verbose == 1:
        assert "Optimizing models" in output.out and "2/2" in output.out
        assert "Trial 0 finished" not in output.err
    else:
        assert "Optimizing models" not in output.out
        assert "Trial 0 finished" in stream.getvalue() and "Trial 1 finished" in stream.getvalue()


@pytest.mark.parametrize("verbose", [0, 1, 2, 3, 10])
@pytest.mark.parametrize("failure", [None, "study", "objective", "holdout"])
def test_optuna_logger_user_state_restored_even_on_errors(binary_df, monkeypatch, verbose, failure):
    import rationalml.automl as facade
    from sklearn.linear_model import LogisticRegression

    optuna.logging.get_verbosity()
    root = logging.getLogger("optuna")
    child = logging.getLogger("optuna.study.study")
    original = [(logger, logger.level, logger.propagate, logger.handlers[:], logger.filters[:], logger.disabled)
                for logger in (root, child)]
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter("user: %(message)s")
    handler.setFormatter(formatter)
    marker = logging.Filter()
    handler.addFilter(marker)
    root.addHandler(handler)
    root.setLevel(logging.NOTSET)
    child.setLevel(logging.DEBUG)
    saved = [(logger.level, logger.propagate, logger.handlers[:], logger.filters[:], logger.disabled)
             for logger in (root, child)]

    def fail(*args, **kwargs):
        raise RuntimeError("intentional failure")

    if failure == "study":
        monkeypatch.setattr(optuna, "create_study", fail)
    elif failure == "objective":
        monkeypatch.setattr(LogisticRegression, "fit", fail)
    elif failure == "holdout":
        monkeypatch.setattr(facade, "evaluate_holdout", fail)
    try:
        if failure:
            with pytest.raises(RuntimeError, match="intentional"):
                fit(binary_df, verbose)
        else:
            fit(binary_df, verbose)
            assert ("Trial 0 finished" in stream.getvalue()) == (verbose >= 2)
        assert [(logger.level, logger.propagate, logger.handlers[:], logger.filters[:], logger.disabled)
                for logger in (root, child)] == saved
        assert handler.formatter is formatter and handler.filters == [marker]
    finally:
        for logger, level, propagation, handlers, filters, disabled in original:
            logger.setLevel(level)
            logger.propagate, logger.handlers, logger.filters, logger.disabled = propagation, handlers, filters, disabled


@pytest.mark.parametrize("verbose", [0, 1, 2, 3, 10])
def test_python_warnings_keep_user_filters(binary_df, verbose):
    df = binary_df.copy()
    df.target = df.target.map({0: "retained", 1: "churn"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filters = warnings.filters[:]
        fit(df, verbose)
        assert warnings.filters == filters
    assert any("positive_class=None" in str(w.message) for w in caught)


@pytest.mark.parametrize("task, model, module", [
    ("binary", "logistic_regression", "sklearn"), ("multiclass", "logistic_regression", "sklearn"),
    ("regression", "ridge", "sklearn"),
    ("binary", "lightgbm", "lightgbm"), ("multiclass", "lightgbm", "lightgbm"),
    ("regression", "lightgbm_regressor", "lightgbm"),
    ("binary", "xgboost", "xgboost"), ("multiclass", "xgboost", "xgboost"),
    ("regression", "xgboost_regressor", "xgboost"),
])
def test_verbose_does_not_change_training_predictions_or_user_data(request, task, model, module):
    pytest.importorskip(module)
    df = request.getfixturevalue(task + "_df")
    original = df.copy(deep=True)
    X = df.drop(columns="target")
    results = [fit(df, verbose, task=task, models=model, preprocessing="basic") for verbose in (0, 1, 2, 3, 10)]
    for other in results[1:]:
        assert_training_equal(results[0], other)
        pd.testing.assert_frame_equal(results[0].test_predictions, other.test_predictions)
        assert results[0].test_metrics == other.test_metrics
        np.testing.assert_array_equal(results[0].predict(X), other.predict(X))
        if task != "regression":
            np.testing.assert_allclose(results[0].predict_proba(X), other.predict_proba(X))
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("task, model", [("binary", "logistic_regression"),
                                        ("multiclass", "logistic_regression"), ("regression", "ridge")])
def test_summary_is_a_pure_string_from_stored_results(request, monkeypatch, capsys, task, model):
    result = fit(request.getfixturevalue(task + "_df"), task=task, models=model)
    saved_pipeline = pickle.dumps(result.best_model)
    attributes = set(vars(result))
    saved_frames = {name: getattr(result, name).copy(deep=True) for name in
                    ("leaderboard", "cv_results", "feature_importance", "test_predictions")}
    saved_metrics = result.test_metrics.copy()

    def forbidden(*args, **kwargs):
        pytest.fail("summary must not train, predict or evaluate")

    with monkeypatch.context() as patch:
        for name in ("fit", "predict", "predict_proba"):
            if hasattr(type(result.best_model), name):
                patch.setattr(type(result.best_model), name, forbidden)
        patch.setattr(type(result), "ranking_table", forbidden)
        patch.setattr(type(result), "calibration_summary", forbidden)
        text = result.summary()
        assert isinstance(text, str) and text == result.summary()
        assert result.best_model_name in text
        assert f"Test {result.primary_metric}" in text
        assert "Elapsed time" in text
    assert pickle.dumps(result.best_model) == saved_pipeline
    assert set(vars(result)) == attributes and result.test_metrics == saved_metrics
    for name, saved in saved_frames.items():
        pd.testing.assert_frame_equal(getattr(result, name), saved)
    assert capsys.readouterr().out == ""


def test_timing_boundaries_use_perf_counter_even_in_silent_mode(binary_df, monkeypatch):
    import rationalml.automl as facade

    ticks = iter([10., 12., 15., 17., 19., 23.])
    monkeypatch.setattr(facade, "perf_counter", lambda: next(ticks))
    result = fit(binary_df)
    assert result.fit_time == 13.
    assert result.model_fit_times == {"logistic_regression": 5.}  # CV 3s + final fit 2s; excludes holdout.


def test_model_times_cover_each_model_without_changing_leaderboard_schema(binary_df):
    base = ModelRegistry.get("logistic_regression")

    class Registry(ModelRegistry):
        _optional = {}
        _specs = {name: replace(base, name=name) for name in ("first", "second")}

    result = fit(binary_df, models="auto", model_registry=Registry)
    assert list(result.model_fit_times) == ["first", "second"]
    assert all(isinstance(t, float) and t > 0 for t in result.model_fit_times.values())
    assert result.fit_time >= sum(result.model_fit_times.values())
    assert "fit_time" not in result.leaderboard and "fit_time" not in result.cv_results


def test_progress_counts_complete_failed_pruned_once_and_ignores_running(capsys):
    from optuna.trial import TrialState, create_trial

    progress = TrialProgress(6)
    study = optuna.create_study()
    for number, state in enumerate((TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED, TrialState.RUNNING)):
        trial = create_trial(state=state, value=1. if state == TrialState.COMPLETE else None)
        trial.number = number
        progress.update(study, trial)
        progress.update(study, trial)
    progress.close(interrupted=True)
    assert (progress.finished, progress.failed, progress.pruned) == (3, 1, 1)
    output = capsys.readouterr().out
    assert "3/6" in output and "6/6" not in output
    assert "failed=1, pruned=1" in output and "interrupted" in output


def test_early_stop_keeps_actual_trial_count_and_incomplete_bar(binary_df, monkeypatch, capsys):
    optimize = optuna.study.Study.optimize

    def stop_after_one(self, objective, **options):
        callbacks = list(options.get("callbacks") or [])
        options["callbacks"] = callbacks + [lambda study, trial: study.stop()]
        return optimize(self, objective, **options)

    monkeypatch.setattr(optuna.study.Study, "optimize", stop_after_one)
    result = fit(binary_df, 1, n_trials=5)
    assert len(result.cv_results) == result.leaderboard.iloc[0].n_trials_completed == 1
    output = capsys.readouterr().out
    assert "1/5" in output and "5/5" not in output and "budget incomplete" in output


def test_timeout_keeps_partial_progress(binary_df, monkeypatch, capsys):
    from time import sleep

    optimize = optuna.study.Study.optimize

    def slow_objective(self, objective, **options):
        def slow(trial):
            value = objective(trial)
            sleep(1.05)  # Optuna checks its wall-clock timeout between trials.
            return value
        return optimize(self, slow, **options)

    monkeypatch.setattr(optuna.study.Study, "optimize", slow_objective)
    result = fit(binary_df, 1, n_trials=5, timeout=1)
    assert len(result.cv_results) == 1
    output = capsys.readouterr().out
    assert "1/5" in output and "5/5" not in output and "budget incomplete" in output


def test_pruned_trials_are_counted_across_multiple_models(binary_df, capsys):
    base = ModelRegistry.get("logistic_regression")

    def prune_first(trial):
        if trial.number == 0:
            raise optuna.TrialPruned("intentional prune")
        return base.search_space(trial)

    class Registry(ModelRegistry):
        _optional = {}
        _specs = {name: replace(base, name=name, search_space=prune_first) for name in ("first", "second")}

    result = fit(binary_df, 1, models="auto", model_registry=Registry)
    assert len(result.cv_results) == 2 and result.leaderboard.n_trials_completed.tolist() == [1, 1]
    output = capsys.readouterr().out
    assert "4/4" in output and "pruned=2" in output


def test_uncaught_failed_trial_is_counted_without_hiding_error(binary_df, monkeypatch, capsys):
    from sklearn.linear_model import LogisticRegression

    def fail(*args, **kwargs):
        raise RuntimeError("intentional failure")

    monkeypatch.setattr(LogisticRegression, "fit", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        fit(binary_df, 1, n_trials=5)
    output = capsys.readouterr().out
    assert "1/5" in output and "failed=1" in output and "interrupted" in output
