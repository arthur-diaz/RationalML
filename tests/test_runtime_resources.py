from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_info

from rationalml import AutoML, AutoMLConfig, ModelRegistry
from rationalml.exceptions import ConfigurationError


@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_n_jobs_supported_values(n_jobs):
    assert AutoMLConfig(target="target", n_jobs=n_jobs).n_jobs == n_jobs


@pytest.mark.parametrize("n_jobs", [0, -2, -10, True, False])
def test_n_jobs_invalid_values(n_jobs):
    with pytest.raises(ConfigurationError, match="n_jobs"):
        AutoMLConfig(target="target", n_jobs=n_jobs)


@pytest.mark.parametrize("model", ["logistic_regression", "ridge", "lightgbm", "xgboost",
                                   "lightgbm_regressor", "xgboost_regressor"])
@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_estimator_cpu_parameters_and_immutable_defaults(model, n_jobs):
    if model.startswith(("lightgbm", "xgboost")):
        pytest.importorskip(model.split("_")[0])
    spec = ModelRegistry.get(model)
    before = spec.default_params.copy()
    task = next(iter(spec.tasks))
    params = spec.parameters({}, 42, n_jobs, task=task)
    estimator = spec.estimator_class(**params)
    if spec.n_jobs_parameter is None:
        assert "n_jobs" not in params
    else:
        assert estimator.get_params()[spec.n_jobs_parameter] == n_jobs
    assert spec.default_params == before


def pools():
    return {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()}


@pytest.mark.parametrize("task, base", [("binary", LogisticRegression), ("regression", Ridge)])
@pytest.mark.parametrize("n_jobs", [1, 2, -1])
def test_native_pools_fit_inference_restore_and_sequential_trials(request, monkeypatch, task, base, n_jobs):
    import optuna

    seen_fit, seen_predict, workers = [], [], []
    before = pools()

    class Recording(base):
        def fit(self, X, y, **options):
            seen_fit.append((set(X.index), pools()))
            return super().fit(X, y, **options)

        def predict(self, X):
            seen_predict.append(pools())
            return super().predict(X)

    name = "ridge" if task == "regression" else "logistic_regression"

    class Registry(ModelRegistry):
        _optional = {}
        _specs = {name: replace(ModelRegistry.get(name), estimator_class=Recording)}

    optimize = optuna.study.Study.optimize

    def record_workers(self, objective, **options):
        workers.append(options["n_jobs"])
        return optimize(self, objective, **options)

    monkeypatch.setattr(optuna.study.Study, "optimize", record_workers)
    df = request.getfixturevalue(task + "_df")
    original = df.copy(deep=True)
    config = AutoMLConfig(target="target", task=task, models=name, cv=2, n_trials=1,
                          preprocessing="basic", n_jobs=n_jobs, verbose=0)
    saved = asdict(config)
    result = AutoML(config, model_registry=Registry).fit(df)
    result.predict(df.drop(columns="target").iloc[:5])
    assert workers == [1]
    assert len(seen_fit) == 3  # Two sequential folds, then one train-only refit.
    assert seen_fit[-1][0] == set(result.train_indices)
    assert all(rows.isdisjoint(result.test_indices) for rows, _ in seen_fit)
    observed = [pool for _, pool in seen_fit] + seen_predict
    if n_jobs > 0:
        assert all(count <= n_jobs for pool in observed for count in pool.values())
    else:
        assert all(pool == before for pool in observed)
    assert pools() == before
    assert asdict(config) == saved
    pd.testing.assert_frame_equal(df, original)


def test_native_pools_restore_when_fit_or_predict_raises(binary_df, monkeypatch):
    before = pools()
    result = AutoML(target="target", models="logistic_regression", cv=2, n_trials=1, verbose=0).fit(binary_df)

    def fail(*args, **kwargs):
        assert all(count <= 1 for count in pools().values())
        raise RuntimeError("intentional failure")

    monkeypatch.setattr(result.best_model, "predict", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        result.predict(binary_df.drop(columns="target"))
    assert pools() == before
    monkeypatch.setattr(LogisticRegression, "fit", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        AutoML(target="target", models="logistic_regression", cv=2, n_trials=1, verbose=0).fit(binary_df)
    assert pools() == before
