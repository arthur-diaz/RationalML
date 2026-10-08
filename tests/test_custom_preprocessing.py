import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rationalml import AutoML, AutoMLConfig, ModelRegistry
from rationalml.exceptions import ConfigurationError, DataValidationError


class RecordingImputer(TransformerMixin, BaseEstimator):
    fit_events = []
    transform_events = []

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.medians_ = X.median()
        self.fit_events.append((self, tuple(X.index), self.medians_.copy()))
        return self

    def transform(self, X):
        self.transform_events.append((self, set(X.index)))
        return X.fillna(self.medians_).copy(deep=True)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_.copy()


class OpaqueImputer(RecordingImputer):
    get_feature_names_out = None


class BrokenNamesImputer(RecordingImputer):
    def get_feature_names_out(self, input_features=None):
        raise RuntimeError("names not implemented")


class WrongCountImputer(RecordingImputer):
    def get_feature_names_out(self, input_features=None):
        return np.asarray(["wrong_count"])


class DuplicateNamesImputer(RecordingImputer):
    def get_feature_names_out(self, input_features=None):
        return np.asarray(["duplicate"] * len(self.feature_names_in_))


class NoArgumentNamesImputer(RecordingImputer):
    def get_feature_names_out(self):
        return self.feature_names_in_.copy()


class MutatingImputer(RecordingImputer):
    def transform(self, X):
        # Even a transformer writing into its input must not alter the caller's DataFrame.
        for name in X.columns:
            X[name] = X[name].fillna(self.medians_[name])
        return X


class SelfCloningImputer(RecordingImputer):
    def __sklearn_clone__(self):
        return self


def fit_custom(df, transformer, **options):
    return AutoML(**({
        "target": "target", "preprocessing": transformer, "models": "logistic_regression",
        "cv": 3, "n_trials": 1, "random_state": 17, "verbose": 0,
    } | options)).fit(df)


def test_custom_transformer_is_cloned_for_every_fold_and_final_fit(binary_df, monkeypatch):
    import rationalml.preprocessing.builder as builder

    binary_df.loc[::9, "feature_0"] = np.nan
    original = binary_df.copy(deep=True)
    monkeypatch.setattr(RecordingImputer, "fit_events", [])
    monkeypatch.setattr(RecordingImputer, "transform_events", [])

    def unexpected(*args):
        pytest.fail("Custom preprocessing must not trigger basic schema inference")

    monkeypatch.setattr(builder, "infer_schema", unexpected)
    transformer = RecordingImputer()
    result = fit_custom(binary_df, transformer, n_trials=2)
    train, holdout = set(result.train_indices), set(result.test_indices)
    X_train = binary_df.iloc[list(result.train_indices)].drop(columns="target")
    y_train = binary_df.target.iloc[list(result.train_indices)]
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=17).split(X_train, y_train))
    fits = RecordingImputer.fit_events
    assert len(fits) == 2 * 3 + 1
    assert len({id(instance) for instance, _, _ in fits}) == len(fits)
    assert all(instance is not transformer for instance, _, _ in fits)
    assert not hasattr(transformer, "medians_")
    assert result.feature_schema is None
    for number, (instance, rows, statistics) in enumerate(fits[:-1]):
        fold_train, fold_validation = folds[number % 3]
        expected_train = set(X_train.iloc[fold_train].index)
        expected_validation = set(X_train.iloc[fold_validation].index)
        assert set(rows) == expected_train
        assert set(rows).isdisjoint(expected_validation | holdout)
        pd.testing.assert_series_equal(statistics, X_train.iloc[fold_train].median())
        assert [seen for fitted, seen in RecordingImputer.transform_events if fitted is instance] == [
            expected_train, expected_validation,
        ]
    final, rows, statistics = fits[-1]
    assert set(rows) == train and set(rows).isdisjoint(holdout)
    pd.testing.assert_series_equal(statistics, X_train.median())
    assert result.best_model.named_steps["preprocessing"] is final
    assert [seen for instance, seen in RecordingImputer.transform_events if instance is final] == [train, holdout, holdout]
    pd.testing.assert_frame_equal(binary_df, original)


def test_prefitted_user_transformer_is_cloned_without_its_learned_state(binary_df, monkeypatch):
    X = binary_df.drop(columns="target")
    user = SimpleImputer(strategy="median").fit(X)
    old_statistics = user.statistics_.copy()
    fits = []
    real_fit = SimpleImputer.fit

    def record_fit(self, X, y=None):
        assert self is not user
        assert not hasattr(self, "statistics_")
        fits.append((self, set(X.index)))
        return real_fit(self, X, y)

    monkeypatch.setattr(SimpleImputer, "fit", record_fit)
    result = fit_custom(binary_df, user)
    assert len(fits) == 4
    assert all(rows.isdisjoint(result.test_indices) for _, rows in fits)
    assert fits[-1][1] == set(result.train_indices)
    np.testing.assert_array_equal(user.statistics_, old_statistics)
    np.testing.assert_allclose(result.best_model.named_steps["preprocessing"].statistics_,
                               X.iloc[list(result.train_indices)].median())


def test_custom_pipeline_is_authoritative_and_receives_no_automatic_scaler(binary_df, monkeypatch):
    import rationalml.preprocessing.builder as builder

    def unexpected(*args, **kwargs):
        pytest.fail("Custom preprocessing must not add basic preprocessing or a scaler")

    monkeypatch.setattr(builder, "build_preprocessor", unexpected)
    monkeypatch.setattr(StandardScaler, "fit", unexpected)
    transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    assert ModelRegistry.get("logistic_regression").requires_scaling
    result = fit_custom(binary_df, transformer)
    fitted = result.best_model.named_steps["preprocessing"]
    assert fitted is not transformer
    assert tuple(fitted.named_steps) == ("imputer",)
    assert not hasattr(transformer.named_steps["imputer"], "statistics_")
    assert result.feature_schema is None
    assert result.transformed_feature_names == result.feature_names
    assert result.feature_importance.source_feature.isna().all()


@pytest.mark.parametrize("model, module", [
    ("logistic_regression", "sklearn"), ("lightgbm", "lightgbm"), ("xgboost", "xgboost"),
])
def test_user_column_transformer_fit_inference_and_serialization(binary_df, model, module):
    pytest.importorskip(module)
    binary_df["country"] = pd.Series(["France", "Spain"] * 50, dtype=object)
    binary_df.loc[::9, "country"] = np.nan
    binary_df.loc[::7, "feature_0"] = np.nan
    binary_df.target = binary_df.target.map({0: "retained", 1: "churn"})
    original = binary_df.copy(deep=True)
    numeric_columns = [name for name in binary_df if name.startswith("feature_")]
    user = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())]), numeric_columns),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["country"]),
    ]).set_output(transform="pandas")
    result = fit_custom(binary_df, user, models=model, positive_class="churn")
    assert not hasattr(user, "transformers_")
    assert result.positive_class == "churn"
    assert result.negative_class == "retained"
    fitted = result.best_model.named_steps["preprocessing"]
    assert result.transformed_feature_names == tuple(fitted.get_feature_names_out())
    assert set(result.feature_importance.feature) == set(result.transformed_feature_names)
    assert pd.api.types.is_numeric_dtype(result.feature_importance.importance)
    assert result.feature_importance.source_feature.isna().all()
    X = binary_df.drop(columns="target").iloc[:3].copy(deep=True)
    X["country"] = ["unseen", np.nan, "France"]
    X["feature_0"] = np.nan
    new_original = X.copy(deep=True)
    probabilities = result.predict_proba(X)
    assert probabilities.shape == (3, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(result.predict_positive_proba(X), probabilities[:, 1])
    np.testing.assert_allclose(result.predict_proba(X[X.columns[::-1]]), probabilities)
    assert set(result.predict(X)) <= {"retained", "churn"}
    restored = pickle.loads(pickle.dumps(result))
    np.testing.assert_allclose(restored.predict_proba(X), probabilities)
    pd.testing.assert_frame_equal(binary_df, original)
    pd.testing.assert_frame_equal(X, new_original)


def test_custom_transformer_owns_dtype_rules(binary_df):
    columns = list(binary_df.drop(columns="target"))
    binary_df["date"] = pd.date_range("2026-01-01", periods=len(binary_df))
    binary_df["nested"] = [[number] for number in range(len(binary_df))]
    original = binary_df.copy(deep=True)
    user = ColumnTransformer([("prepared", "passthrough", columns)], remainder="drop")
    result = fit_custom(binary_df, user)
    assert result.feature_schema is None
    assert result.predict(binary_df.drop(columns="target")).shape == (len(binary_df),)
    pd.testing.assert_frame_equal(binary_df, original)


@pytest.mark.parametrize("transformer_class", [OpaqueImputer, BrokenNamesImputer, WrongCountImputer, DuplicateNamesImputer])
def test_unreliable_custom_feature_names_only_disable_importance(binary_df, transformer_class):
    with pytest.warns(UserWarning, match="Custom preprocessing.*feature_importance is unavailable"):
        result = fit_custom(binary_df, transformer_class())
    assert result.transformed_feature_names is None
    assert result.feature_importance.empty
    assert list(result.feature_importance.columns) == ["feature", "importance", "source_feature"]
    assert pd.api.types.is_numeric_dtype(result.feature_importance.importance)
    X = binary_df.drop(columns="target")
    assert result.predict(X).shape == (len(binary_df),)
    np.testing.assert_allclose(result.predict_proba(X).sum(axis=1), 1)


def test_custom_get_feature_names_out_without_argument_is_supported(binary_df):
    result = fit_custom(binary_df, NoArgumentNamesImputer())
    assert result.transformed_feature_names == result.feature_names
    assert not result.feature_importance.empty


def test_user_dataframe_is_protected_from_transformer_in_place_writes(binary_df):
    binary_df.loc[::9, "feature_0"] = np.nan
    original = binary_df.copy(deep=True)
    result = fit_custom(binary_df, MutatingImputer())
    X = binary_df.drop(columns="target").iloc[:3].copy(deep=True)
    X.iloc[0, 0] = np.nan
    original_X = X.copy(deep=True)
    result.predict(X)
    result.predict_proba(X)
    pd.testing.assert_frame_equal(binary_df, original)
    pd.testing.assert_frame_equal(X, original_X)


def test_custom_config_is_copied_without_fitting_user_transformer():
    user = SimpleImputer(strategy="median")
    config = AutoMLConfig(target="target", preprocessing=user)
    engine = AutoML(config)
    assert engine.config.preprocessing is not user
    assert engine.config.preprocessing.get_params() == user.get_params()
    assert not hasattr(user, "statistics_")
    assert not hasattr(engine.config.preprocessing, "statistics_")


def test_custom_transformer_cannot_share_itself_between_folds(binary_df):
    with pytest.raises(ConfigurationError, match="distinct unfitted instance"):
        fit_custom(binary_df, SelfCloningImputer())


@pytest.mark.parametrize("error", ["missing", "extra", "duplicate"])
def test_custom_inference_still_requires_exact_raw_columns(binary_df, error):
    result = fit_custom(binary_df, SimpleImputer())
    X = binary_df.drop(columns="target")
    if error == "missing":
        X = X.drop(columns=X.columns[0])
    elif error == "extra":
        X = X.assign(extra=0)
    else:
        X = X.copy()
        X.columns = [X.columns[1], *X.columns[1:]]
    with pytest.raises(DataValidationError, match="missing|extra|unique"):
        result.predict(X)


def test_custom_preprocessing_is_reproducible(binary_df):
    binary_df.loc[::9, "feature_0"] = np.nan
    first = fit_custom(binary_df, SimpleImputer(strategy="median"), n_trials=2)
    second = fit_custom(binary_df, SimpleImputer(strategy="median"), n_trials=2)
    assert first.train_indices == second.train_indices
    assert first.test_indices == second.test_indices
    assert first.best_model_name == second.best_model_name
    assert first.best_params == second.best_params
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    np.testing.assert_allclose(first.predict_proba(binary_df.drop(columns="target")),
                               second.predict_proba(binary_df.drop(columns="target")))
