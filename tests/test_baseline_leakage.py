from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold

from rationalml import AutoML, ModelRegistry, TaskType


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("preprocessing", [None, "basic", "custom"])
def test_baseline_sees_only_fold_train_and_shares_exact_folds_with_models(request, monkeypatch, task, preprocessing):
    import rationalml.automl as facade
    import rationalml.optimization.optimizer as optimizer

    df = request.getfixturevalue(f"{task}_df")
    if task == "binary":
        df.target = df.target.map({0: "retained", 1: "churn"})
    if preprocessing is not None:
        df.loc[::11, "feature_0"] = np.nan
    original = df.copy(deep=True)
    dummy_class = DummyRegressor if task == "regression" else DummyClassifier
    dummy_fits, dummy_predictions, imputer_fits, baseline_inputs, model_inputs, splitter_inputs = [], [], [], [], [], []
    real_fit = dummy_class.fit

    def record_dummy_fit(self, X, y, **options):
        # Only an API placeholder reaches the dummy, regardless of preprocessing mode.
        assert list(X.columns) == ["_baseline"]
        assert (X.to_numpy() == 0).all()
        assert self.strategy == ("mean" if task == "regression" else "prior")
        fitted = real_fit(self, X, y, **options)
        statistics = self.constant_.copy() if task == "regression" else self.class_prior_.copy()
        dummy_fits.append((self, tuple(X.index), y.copy(), statistics))
        return fitted

    prediction_method = "predict_proba" if task == "binary" else "predict"
    real_predict = getattr(dummy_class, prediction_method)

    def record_dummy_prediction(self, X):
        dummy_predictions.append((self, set(X.index)))
        return real_predict(self, X)

    monkeypatch.setattr(dummy_class, "fit", record_dummy_fit)
    monkeypatch.setattr(dummy_class, prediction_method, record_dummy_prediction)
    real_imputer_fit = SimpleImputer.fit

    def record_imputer_fit(self, X, y=None):
        imputer_fits.append((self, set(X.index)))
        return real_imputer_fit(self, X, y)

    monkeypatch.setattr(SimpleImputer, "fit", record_imputer_fit)
    real_baseline, real_optimize, real_splits = facade.evaluate_baseline, facade.optimize_model, facade.make_cv_splits

    def record_baseline(y, metric, resolved, folds):
        baseline_inputs.append((y.copy(), folds))
        return real_baseline(y, metric, resolved, folds)

    def record_optimizer(spec, X, y, metric, config, *, folds):
        model_inputs.append((spec.name, set(X.index), set(y.index), folds))
        return real_optimize(spec, X, y, metric, config, folds=folds)

    def record_splits(X, y, resolved, config):
        splitter_inputs.append((set(X.index), set(y.index), resolved))
        return real_splits(X, y, resolved, config)

    def unexpected_splits(*args):
        pytest.fail("The optimizer must reuse the supplied folds")

    monkeypatch.setattr(facade, "evaluate_baseline", record_baseline)
    monkeypatch.setattr(facade, "optimize_model", record_optimizer)
    monkeypatch.setattr(facade, "make_cv_splits", record_splits)
    monkeypatch.setattr(optimizer, "make_cv_splits", unexpected_splits)
    base = ModelRegistry.get("ridge" if task == "regression" else "logistic_regression")

    class Registry(ModelRegistry):
        _optional = {}
        _specs = {name: replace(base, name=name, default_params={**base.default_params, "fit_intercept": intercept})
                  for name, intercept in [("first", True), ("second", False)]}

    user = SimpleImputer(strategy="median").set_output(transform="pandas") if preprocessing == "custom" else preprocessing
    result = AutoML(target="target", task=task, preprocessing=user, models="auto", model_registry=Registry,
                    positive_class="churn" if task == "binary" else None,
                    cv=3, n_trials=2, random_state=17, verbose=0).fit(df)
    train, holdout = set(result.train_indices), set(result.test_indices)
    assert splitter_inputs == [(train, train, TaskType(task))]
    assert len(baseline_inputs) == 1
    encoded_y, shared_folds = baseline_inputs[0]
    assert set(encoded_y.index) == train
    assert len(model_inputs) == 2
    assert all(X_rows == y_rows == train and folds is shared_folds for _, X_rows, y_rows, folds in model_inputs)
    expected_y = df.target.iloc[list(result.train_indices)]
    if task != "regression":
        expected_y = pd.Series(result.label_encoder.transform(expected_y), index=expected_y.index)
    pd.testing.assert_series_equal(encoded_y, expected_y)
    X_train = df.iloc[list(result.train_indices)].drop(columns="target")
    splitter = KFold if task == "regression" else StratifiedKFold
    expected_folds = list(splitter(n_splits=3, shuffle=True, random_state=17).split(X_train, encoded_y))
    assert len(dummy_fits) == len(dummy_predictions) == 3
    assert len({id(instance) for instance, _, _, _ in dummy_fits}) == 3
    for fold, ((instance, rows, seen_y, statistics), (training, validation)) in enumerate(zip(dummy_fits, expected_folds)):
        np.testing.assert_array_equal(shared_folds[fold][0], training)
        np.testing.assert_array_equal(shared_folds[fold][1], validation)
        fold_train, fold_validation = set(X_train.iloc[training].index), set(X_train.iloc[validation].index)
        assert set(rows) == fold_train
        assert set(rows).isdisjoint(holdout | fold_validation)
        pd.testing.assert_series_equal(seen_y, encoded_y.iloc[training])
        expected_statistics = ([seen_y.mean()] if task == "regression"
                               else [seen_y.eq(label).mean() for label in instance.classes_])
        np.testing.assert_allclose(statistics.ravel(), expected_statistics)
        assert [(dummy, seen) for dummy, seen in dummy_predictions if dummy is instance] == [(instance, fold_validation)]
    assert all(rows.isdisjoint(holdout) for _, rows in dummy_predictions)
    assert not any(set(rows) == train for _, rows, _, _ in dummy_fits)  # No final baseline fit.
    assert len(imputer_fits) == (0 if preprocessing is None else 2 * 2 * 3 + 1)
    assert all(rows <= train and rows.isdisjoint(holdout) for _, rows in imputer_fits)
    if preprocessing == "custom":
        assert not hasattr(user, "statistics_")
    assert result.best_model_name in {"first", "second"}
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_baseline_scores_cannot_influence_trials_or_winner(request, monkeypatch, task):
    import rationalml.automl as facade

    df = request.getfixturevalue(f"{task}_df")
    options = dict(target="target", task=task, models="auto", cv=3, n_trials=2, random_state=17, verbose=0)
    first = AutoML(**options).fit(df)
    real_baseline = facade.evaluate_baseline

    def changed_baseline(y, metric, resolved, folds):
        name, scores = real_baseline(y, metric, resolved, folds)
        return name, tuple(score + .1 for score in scores)

    monkeypatch.setattr(facade, "evaluate_baseline", changed_baseline)
    second = AutoML(**options).fit(df)
    assert second.baseline_score == pytest.approx(first.baseline_score + .1)
    assert first.best_model_name == second.best_model_name and first.best_params == second.best_params
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    columns = [column for column in first.leaderboard if column != "improvement_vs_baseline"]
    pd.testing.assert_frame_equal(first.leaderboard[columns], second.leaderboard[columns])
    np.testing.assert_array_equal(first.predict(df.drop(columns="target")), second.predict(df.drop(columns="target")))


def test_regression_holdout_targets_cannot_influence_baseline(regression_df):
    options = dict(target="target", task="regression", models="ridge", cv=3, n_trials=2, random_state=17, verbose=0)
    first = AutoML(**options).fit(regression_df)
    changed = regression_df.copy(deep=True)
    changed.loc[list(first.test_indices), "target"] += 1e9
    second = AutoML(**options).fit(changed)
    assert first.train_indices == second.train_indices and first.test_indices == second.test_indices
    assert first.baseline_fold_scores == second.baseline_fold_scores
    assert first.baseline_score == second.baseline_score
    assert first.best_params == second.best_params
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    assert first.test_metrics != second.test_metrics
