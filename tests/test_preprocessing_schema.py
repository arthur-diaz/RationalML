import numpy as np
import pandas as pd
import pytest

from rationalml import AutoML, AutoMLConfig, FeatureSchema, PreprocessingConfig, infer_schema
from rationalml.data import validate_features
from rationalml.exceptions import ConfigurationError, DataValidationError


def test_schema_preserves_raw_names_order_and_types_without_mutation(mixed_df):
    X = mixed_df.drop(columns="target")
    original = X.copy(deep=True)
    schema = infer_schema(X)
    assert isinstance(schema, FeatureSchema)
    assert schema.feature_names == tuple(X.columns)
    assert schema.numeric == ("numeric_float", "numeric_int", "nullable_int")
    assert schema.boolean == ("boolean",)
    assert schema.categorical == ("categorical", "category_dtype", "string_dtype")
    assert schema.constant_columns == ()
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("dtype", ["float64", "object", "string", "category", "boolean", "Int64"])
def test_entirely_missing_training_columns_fail_explicitly(dtype):
    X = pd.DataFrame({"empty": pd.Series([None, None, None], dtype=dtype)})
    with pytest.raises(DataValidationError, match="entirely missing.*empty"):
        infer_schema(X)


@pytest.mark.parametrize("values", [[1, 1, None], ["only", "only", None], [True, True, None]])
def test_constants_are_retained_and_warned(values):
    X = pd.DataFrame({"constant": values, "variable": [1, 2, 3]})
    with pytest.warns(UserWarning, match="Constant.*retained.*constant"):
        schema = infer_schema(X)
    assert schema.constant_columns == ("constant",)
    assert schema.feature_names == ("constant", "variable")


@pytest.mark.parametrize("values", [
    pd.date_range("2026-01-01", periods=3),
    pd.date_range("2026-01-01", periods=3, tz="UTC"),
    pd.to_timedelta([1, 2, 3], unit="D"),
])
def test_datetime_and_timedelta_are_rejected_explicitly(values):
    with pytest.raises(DataValidationError, match="Datetime/timedelta.*not supported"):
        infer_schema(pd.DataFrame({"date": values}))


@pytest.mark.parametrize("values, message", [
    ([1.0, np.inf, 3.0], "infinite"),
    ([1.0, -np.inf, None], "infinite"),
    (pd.Series([1.0, np.inf, None], dtype=object), "infinite"),
    ([[1], [2], [3]], "nested"),
    ([{"a": 1}, {"a": 2}, None], "nested"),
    ([np.array([1]), None, "a"], "nested"),
    (["a", 1, None], "mixes"),
    ([1 + 2j, 2 + 3j, 3 + 4j], "Unsupported dtype"),
])
def test_unsupported_values_fail_without_mutation(values, message):
    X = pd.DataFrame({"bad": values})
    original = X.copy(deep=True)
    with pytest.raises(DataValidationError, match=message):
        validate_features(X, preprocessing="basic")
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("columns", [["a", "a"], ["", "b"], [" ", "b"], [0, "b"]])
def test_invalid_or_duplicate_feature_names_fail(columns):
    with pytest.raises(DataValidationError, match="names"):
        infer_schema(pd.DataFrame([[1, 2], [3, 4]], columns=columns))


@pytest.mark.parametrize("options", [
    {"numeric_imputation": "magic"}, {"categorical_imputation": "median"},
    {"categorical_encoding": "ordinal"}, {"scale_numeric": "always"},
    {"scale_numeric": 1}, {"handle_unknown": "unknown"},
])
def test_preprocessing_config_rejects_unsupported_options(options):
    with pytest.raises(ConfigurationError):
        PreprocessingConfig(**options)


@pytest.mark.parametrize("mode", [False, True, "off", "auto", {}, []])
def test_automl_rejects_invalid_preprocessing_modes(mode):
    with pytest.raises(ConfigurationError, match="preprocessing"):
        AutoMLConfig(target="target", preprocessing=mode)


def test_default_and_explicit_config_are_data_independent_and_copied():
    assert AutoMLConfig(target="target").preprocessing is None
    preprocessing = PreprocessingConfig(scale_numeric=False)
    config = AutoMLConfig(target="target", preprocessing=preprocessing)
    engine = AutoML(config)
    assert config.preprocessing == preprocessing
    assert engine.config.preprocessing == preprocessing
    assert engine.config.preprocessing is not config.preprocessing


@pytest.mark.parametrize("column, values, message", [
    ("empty", [np.nan] * 100, "entirely missing"),
    ("date", pd.date_range("2026-01-01", periods=100), "Datetime"),
])
def test_automl_reports_unsupported_training_columns(binary_df, column, values, message):
    binary_df[column] = values
    with pytest.raises(DataValidationError, match=message):
        AutoML(target="target", models="logistic_regression", preprocessing="basic",
               cv=2, n_trials=1, verbose=0).fit(binary_df)
