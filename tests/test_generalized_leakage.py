from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from rationalml import AutoML, ModelRegistry
from rationalml.data import MulticlassLabelEncoder


class RecordingTransformer(TransformerMixin, BaseEstimator):
    fit_events = []
    transform_events = []

    def fit(self, X, y=None):
        self.columns_ = np.asarray(X.columns, dtype=object)
        self.medians_ = X.median()
        self.fit_events.append((self, tuple(X.index), self.medians_.copy()))
        return self

    def transform(self, X):
        self.transform_events.append((self, set(X.index)))
        return X.fillna(self.medians_).copy(deep=True)

    def get_feature_names_out(self, input_features=None):
        return self.columns_.copy()


@pytest.mark.parametrize("task", ["multiclass", "regression"])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
def test_new_tasks_fit_only_fold_train_then_full_train(request, monkeypatch, task, mode):
    import rationalml.automl as facade

    df = request.getfixturevalue(f"{task}_df")
    stratify = df.target if task == "multiclass" else None
    train_rows, test_rows = train_test_split(np.arange(len(df)), test_size=.2, random_state=17, stratify=stratify)
    splitter = StratifiedKFold if task == "multiclass" else KFold
    folds = list(splitter(n_splits=3, shuffle=True, random_state=17).split(df.iloc[train_rows], df.target.iloc[train_rows]))
    if mode is not None:
        for number, (_, validation) in enumerate(folds):
            rows = train_rows[validation]
            df.loc[rows, "feature_0"] = [1., 11., 101.][number]
            df.loc[rows[0], "feature_0"] = np.nan
        df.loc[test_rows, "feature_0"] = 1e6
    original = df.copy(deep=True)
    base = ModelRegistry.get("logistic_regression" if task == "multiclass" else "ridge")
    estimator_fits, predictions, optimizer_inputs, encoding_inputs = [], [], [], []

    class RecordingEstimator(base.estimator_class):
        def fit(self, X, y, **options):
            estimator_fits.append((self, tuple(X.index), y.copy(), options))
            return super().fit(X, y, **options)

        def predict(self, X):
            predictions.append((self, "predict", set(X.index)))
            return super().predict(X)

        def predict_proba(self, X):
            predictions.append((self, "proba", set(X.index)))
            return super().predict_proba(X)

    class RecordingRegistry(ModelRegistry):
        _optional = {}
        _specs = {"recording": replace(base, name="recording", estimator_class=RecordingEstimator)}

    real_optimize = facade.optimize_model

    def record_optimizer(spec, X, y, metric, config):
        optimizer_inputs.append((set(X.index), set(y.index)))
        return real_optimize(spec, X, y, metric, config)

    monkeypatch.setattr(facade, "optimize_model", record_optimizer)
    real_encoding = MulticlassLabelEncoder.from_target

    def record_encoding(cls, y):
        encoding_inputs.append(set(y.index))
        return real_encoding(y)

    monkeypatch.setattr(MulticlassLabelEncoder, "from_target", classmethod(record_encoding))
    monkeypatch.setattr(RecordingTransformer, "fit_events", [])
    monkeypatch.setattr(RecordingTransformer, "transform_events", [])
    preprocessor_fits, preprocessor_transforms = [], []
    real_fit_transform, real_transform = ColumnTransformer.fit_transform, ColumnTransformer.transform

    def record_basic_fit(self, X, y=None, **options):
        result = real_fit_transform(self, X, y, **options)
        preprocessor_fits.append((self, tuple(X.index), self.named_transformers_["numeric"].named_steps["imputer"].statistics_.copy()))
        return result

    def record_basic_transform(self, X, **options):
        preprocessor_transforms.append((self, set(X.index)))
        return real_transform(self, X, **options)

    monkeypatch.setattr(ColumnTransformer, "fit_transform", record_basic_fit)
    monkeypatch.setattr(ColumnTransformer, "transform", record_basic_transform)
    user = RecordingTransformer() if mode == "custom" else mode
    result = AutoML(target="target", task=task, models="recording", model_registry=RecordingRegistry,
                    preprocessing=user, cv=3, n_trials=2, random_state=17, verbose=0).fit(df)
    train, holdout = set(train_rows), set(test_rows)
    assert result.train_indices == tuple(train_rows) and result.test_indices == tuple(test_rows)
    assert optimizer_inputs == [(train, train)]
    assert encoding_inputs == ([train] if task == "multiclass" else [])
    assert len(estimator_fits) == 2 * 3 + 1
    assert len({id(instance) for instance, _, _, _ in estimator_fits}) == len(estimator_fits)
    for number, (_, rows, y, options) in enumerate(estimator_fits[:-1]):
        expected_train, validation = folds[number % 3]
        assert set(rows) == set(train_rows[expected_train])
        assert set(rows).isdisjoint(holdout | set(train_rows[validation]))
        assert not options
        if task == "regression":
            pd.testing.assert_series_equal(y, df.target.loc[list(rows)])
    final_estimator, rows, _, _ = estimator_fits[-1]
    assert set(rows) == train and set(rows).isdisjoint(holdout)
    expected_calls = ["predict", "proba"] if task == "multiclass" else ["predict"]
    assert [(kind, seen) for instance, kind, seen in predictions if seen & holdout] == [(kind, holdout) for kind in expected_calls]
    assert all(instance is final_estimator for instance, _, seen in predictions if seen & holdout)
    if mode is None:
        assert preprocessor_fits == [] and RecordingTransformer.fit_events == []
        assert list(result.best_model.named_steps) == ["estimator"]
    else:
        fits = RecordingTransformer.fit_events if mode == "custom" else preprocessor_fits
        transforms = RecordingTransformer.transform_events if mode == "custom" else preprocessor_transforms
        assert len(fits) == 7 and len({id(instance) for instance, _, _ in fits}) == 7
        full_median = df.iloc[train_rows].feature_0.median()
        for number, (instance, rows, statistics) in enumerate(fits[:-1]):
            fold_train, validation = folds[number % 3]
            assert set(rows) == set(train_rows[fold_train])
            assert set(rows).isdisjoint(holdout | set(train_rows[validation]))
            actual = statistics["feature_0"] if mode == "custom" else statistics[0]
            assert actual == pytest.approx(df.iloc[train_rows[fold_train]].feature_0.median())
            assert actual != full_median
            expected_transforms = [set(train_rows[validation])]
            if mode == "custom":
                expected_transforms.insert(0, set(rows))
                assert instance is not user
            assert [seen for fitted, seen in transforms if fitted is instance] == expected_transforms
        final, rows, statistics = fits[-1]
        assert set(rows) == train
        assert final is result.best_model.named_steps["preprocessing"]
        assert (statistics["feature_0"] if mode == "custom" else statistics[0]) == pytest.approx(full_median)
        expected_transforms = [holdout] * len(expected_calls)
        if mode == "custom":
            expected_transforms.insert(0, train)
            assert not hasattr(user, "medians_")
        assert [seen for instance, seen in transforms if instance is final] == expected_transforms
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("task", ["multiclass", "regression"])
def test_holdout_feature_changes_cannot_change_cv_selection(request, task):
    df = request.getfixturevalue(f"{task}_df")
    options = dict(target="target", task=task, models="auto", preprocessing="basic",
                   cv=3, n_trials=2, random_state=17, verbose=0)
    first = AutoML(**options).fit(df)
    changed = df.copy(deep=True)
    changed.loc[list(first.test_indices), "feature_0"] = 1e9
    second = AutoML(**options).fit(changed)
    assert first.train_indices == second.train_indices and first.test_indices == second.test_indices
    assert first.best_model_name == second.best_model_name and first.best_params == second.best_params
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    pd.testing.assert_frame_equal(first.feature_importance, second.feature_importance)
    if task == "multiclass":
        np.testing.assert_array_equal(first.predict(df.drop(columns="target")), second.predict(df.drop(columns="target")))
        np.testing.assert_allclose(first.predict_proba(df.drop(columns="target")), second.predict_proba(df.drop(columns="target")))
    else:
        np.testing.assert_allclose(first.predict(df.drop(columns="target")), second.predict(df.drop(columns="target")))

