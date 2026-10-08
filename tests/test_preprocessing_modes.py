from dataclasses import replace
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rationalml import AutoML, AutoMLConfig, ModelRegistry
from rationalml.exceptions import ConfigurationError, DataValidationError


def test_default_none_preserves_raw_features_and_does_not_infer_schema(binary_df, monkeypatch):
    import rationalml.preprocessing.builder as builder

    fits = []

    class RecordingLogistic(LogisticRegression):
        def fit(self, X, y):
            fits.append(X.copy(deep=True))
            return super().fit(X, y)

    class Registry(ModelRegistry):
        _optional = {}
        _specs = {"recording": replace(ModelRegistry.get("logistic_regression"),
                                      name="recording", estimator_class=RecordingLogistic)}

    def unexpected(*args, **kwargs):
        pytest.fail("preprocessing=None must not infer a schema or fit a scaler")

    monkeypatch.setattr(builder, "infer_schema", unexpected)
    monkeypatch.setattr(StandardScaler, "fit", unexpected)
    binary_df["constant"] = 7.0
    original = binary_df.copy(deep=True)
    assert AutoMLConfig(target="target").preprocessing is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        result = AutoML(target="target", models="recording", model_registry=Registry,
                        cv=2, n_trials=1, verbose=0).fit(binary_df)
    assert not caught
    assert Registry.get("recording").requires_scaling
    assert result.config.preprocessing is None
    assert isinstance(result.best_model, Pipeline)
    assert tuple(result.best_model.named_steps) == ("estimator",)
    assert result.feature_schema is None
    assert result.transformed_feature_names == result.feature_names
    assert len(fits) == 3
    for X in fits:
        pd.testing.assert_frame_equal(X, binary_df.loc[X.index].drop(columns="target"))
    X = binary_df.drop(columns="target")
    np.testing.assert_allclose(result.predict_proba(X[X.columns[::-1]]), result.predict_proba(X))
    importance = result.feature_importance.set_index("feature").loc[list(X.columns)]
    np.testing.assert_allclose(importance.importance, result.best_model.named_steps["estimator"].coef_[0])
    assert list(importance.source_feature) == list(X.columns)
    pd.testing.assert_frame_equal(binary_df, original)


@pytest.mark.parametrize("problem", ["categorical", "nan", "nullable_boolean"])
def test_default_none_rejects_unprepared_data_with_migration_guidance(binary_df, problem, monkeypatch):
    import rationalml.automl as facade

    if problem == "categorical":
        binary_df["country"] = ["France", "Spain"] * 50
    elif problem == "nan":
        binary_df.loc[0, "feature_0"] = np.nan
    else:
        binary_df["flag"] = pd.Series([True, False] * 50, dtype="boolean")
        binary_df.loc[0, "flag"] = pd.NA
    original = binary_df.copy(deep=True)

    def unexpected(*args):
        pytest.fail("Unprepared data must be rejected before Optuna")

    monkeypatch.setattr(facade, "optimize_model", unexpected)
    with pytest.raises(DataValidationError, match="Prepare.*preprocessing='basic'.*custom sklearn transformer"):
        AutoML(target="target", models="logistic_regression", cv=2, n_trials=1).fit(binary_df)
    pd.testing.assert_frame_equal(binary_df, original)


def test_default_none_rejects_missing_values_at_inference(binary_df):
    result = AutoML(target="target", models="logistic_regression", cv=2, n_trials=1, verbose=0).fit(binary_df)
    X = binary_df.drop(columns="target").iloc[:3].copy()
    X.iloc[0, 0] = np.nan
    with pytest.raises(DataValidationError, match="preprocessing=None.*missing"):
        result.predict(X)


def test_auto_preprocessing_is_removed_but_auto_models_remain_available():
    with pytest.raises(ConfigurationError, match="Use 'basic'.*removed.*'auto'"):
        AutoMLConfig(target="target", preprocessing="auto")
    assert AutoMLConfig(target="target", preprocessing="basic", models="auto").models == "auto"


def test_estimator_without_transform_is_not_a_preprocessor():
    with pytest.raises(ConfigurationError, match="fit/transform"):
        AutoMLConfig(target="target", preprocessing=LogisticRegression())
