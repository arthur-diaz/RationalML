from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from rationalml import AutoML, ModelRegistry
from _robustness import assert_training_equal, case


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
@pytest.mark.parametrize("changed_part", ["features", "target"])
def test_fixed_holdout_counterfactual_does_not_change_learning(request, monkeypatch, task, mode, changed_part):
    import rationalml.automl as facade

    df, options = case(request, task, mode)
    base = ModelRegistry.get(options["models"])

    class Registry(ModelRegistry):
        _optional = {}
        _specs = {name: replace(base, name=name, default_params={**base.default_params, "fit_intercept": intercept})
                  for name, intercept in [("with_intercept", True), ("without_intercept", False)]}

    options.update(models=["with_intercept", "without_intercept"], model_registry=Registry)
    first = AutoML(**options).fit(df)
    changed = df.copy(deep=True)
    test = list(first.test_indices)
    split_calls = []

    def fixed_split(rows, **kwargs):
        np.testing.assert_array_equal(rows, np.arange(len(df)))
        split_calls.append(True)
        return np.array(first.train_indices), np.array(first.test_indices)

    monkeypatch.setattr(facade, "train_test_split", fixed_split)
    if changed_part == "features":
        numeric = [name for name in changed if name.startswith("feature_")]
        changed.loc[test, numeric] = changed.loc[test, numeric] * -10000 + 1e6
        if mode == "basic":
            changed.loc[test, "country"] = "holdout_only"
        if mode is not None:
            changed.loc[test[:3], "feature_0"] = np.nan
    elif task == "binary":
        changed.loc[test, "target"] = changed.loc[test, "target"].map({"churn": "retained", "retained": "churn"})
    elif task == "multiclass":
        changed.loc[test, "target"] = changed.loc[test, "target"].map({"gold": "silver", "silver": "bronze", "bronze": "gold"})
    else:
        changed.loc[test, "target"] += 1e6
    pd.testing.assert_frame_equal(df.iloc[list(first.train_indices)], changed.iloc[list(first.train_indices)])
    second = AutoML(**options).fit(changed)
    assert split_calls == [True]
    assert_training_equal(first, second)
    X = df.drop(columns="target")
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    if task != "regression":
        np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))
    assert not first.test_predictions.equals(second.test_predictions)


def test_basic_fitted_statistics_ignore_extreme_test_values_and_categories(binary_df, monkeypatch):
    import rationalml.automl as facade

    train, test = np.arange(80), np.arange(80, 100)
    df = binary_df.copy(deep=True)
    df["country"] = np.where(np.arange(len(df)) % 2, "France", "Germany")
    df.loc[test, "country"] = "test_only"
    df.loc[train[::7], "feature_0"] = np.nan
    df.loc[test, "feature_0"] = 1e15
    df.loc[test[:2], "feature_0"] = np.nan
    monkeypatch.setattr(facade, "train_test_split", lambda *args, **kwargs: (train, test))
    result = AutoML(target="target", preprocessing="basic", models="logistic_regression",
                    cv=2, n_trials=1, verbose=0).fit(df)
    fitted = result.best_model.named_steps["preprocessing"]
    numeric = fitted.named_transformers_["numeric"]
    names = list(result.feature_schema.numeric)
    raw_train = df.iloc[train][names].to_numpy()
    medians = np.nanmedian(raw_train, axis=0)
    np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, medians)
    imputed_train = np.where(np.isnan(raw_train), medians, raw_train)
    np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, imputed_train.mean(axis=0))
    np.testing.assert_allclose(numeric.named_steps["scaler"].var_, imputed_train.var(axis=0))
    encoder = fitted.named_transformers_["categorical"].named_steps["encoder"]
    assert encoder.categories_[0].tolist() == ["France", "Germany"]
    assert "test_only" not in encoder.categories_[0]
    assert np.isfinite(result.predict_proba(df.iloc[test].drop(columns="target"))).all()


@pytest.mark.parametrize("names", [["zeta", "alpha", "middle"], ["middle", "zeta", "alpha"]])
def test_three_models_and_baseline_use_exact_shared_folds_and_stable_ties(binary_df, monkeypatch, names):
    import rationalml.automl as facade
    import rationalml.optimization.optimizer as optimizer

    base = ModelRegistry.get("logistic_regression")

    class Registry(ModelRegistry):
        _optional = {}
        _specs = {name: replace(base, name=name, search_space=lambda trial: {}) for name in names}

    baseline_fits, estimator_fits, passed_folds = [], [], []
    real_dummy_fit, real_fit = DummyClassifier.fit, LogisticRegression.fit
    real_baseline, real_optimize = facade.evaluate_baseline, facade.optimize_model

    def dummy_fit(self, X, y, **kwargs):
        baseline_fits.append(tuple(X.index))
        return real_dummy_fit(self, X, y, **kwargs)

    def estimator_fit(self, X, y, **kwargs):
        estimator_fits.append(tuple(X.index))
        return real_fit(self, X, y, **kwargs)

    def baseline(y, metric, task, folds):
        passed_folds.append(folds)
        return real_baseline(y, metric, task, folds)

    def optimize(*args, folds):
        passed_folds.append(folds)
        return real_optimize(*args, folds=folds)

    monkeypatch.setattr(DummyClassifier, "fit", dummy_fit)
    monkeypatch.setattr(LogisticRegression, "fit", estimator_fit)
    monkeypatch.setattr(facade, "evaluate_baseline", baseline)
    monkeypatch.setattr(facade, "optimize_model", optimize)
    monkeypatch.setattr(optimizer, "make_cv_splits", lambda *args: pytest.fail("CV must be shared"))
    result = AutoML(target="target", models=names, model_registry=Registry,
                    cv=3, n_trials=1, random_state=17, verbose=0).fit(binary_df)
    X_train = binary_df.iloc[list(result.train_indices)]
    expected = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=17).split(X_train, X_train.target))
    expected_rows = [tuple(X_train.iloc[train].index) for train, _ in expected]
    assert baseline_fits == expected_rows
    assert estimator_fits == expected_rows * 3 + [tuple(X_train.index)]
    assert len(passed_folds) == 4 and all(folds is passed_folds[0] for folds in passed_folds)
    for actual, wanted in zip(passed_folds[0], expected):
        np.testing.assert_array_equal(actual[0], wanted[0])
        np.testing.assert_array_equal(actual[1], wanted[1])
    assert result.leaderboard.cv_score.nunique() == 1
    assert result.leaderboard.index.tolist() == names
    assert result.best_model_name == names[0]
