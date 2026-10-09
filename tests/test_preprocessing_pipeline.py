from dataclasses import replace
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from rationalml import AutoML, ModelRegistry, PreprocessingConfig
from rationalml.preprocessing import build_preprocessor, infer_schema
from rationalml.exceptions import DataValidationError


def fit_mixed(df, **options):
    return AutoML(**({
        "target": "target", "positive_class": "churn", "models": "logistic_regression",
        "cv": 3, "n_trials": 1, "random_state": 17, "verbose": 0, "preprocessing": "basic",
    } | options)).fit(df)


@pytest.mark.parametrize("model, module", [
    ("logistic_regression", "sklearn"), ("lightgbm", "lightgbm"), ("xgboost", "xgboost"),
])
def test_mixed_data_fit_inference_positive_class_and_importance(mixed_df, model, module):
    pytest.importorskip(module)
    original = mixed_df.copy(deep=True)
    result = fit_mixed(mixed_df, models=model)
    X = mixed_df.drop(columns="target")
    assert isinstance(result.best_model, Pipeline)
    assert tuple(result.best_model.named_steps) == ("preprocessing", "estimator")
    assert result.positive_class == "churn"
    assert result.negative_class == "retained"
    np.testing.assert_array_equal(result.classes_, ["retained", "churn"])
    np.testing.assert_array_equal(result.best_model.classes_, [0, 1])
    np.testing.assert_array_equal(result.label_encoder.transform(pd.Series(["retained", "churn"])), [0, 1])
    assert set(result.predict(X)) <= {"retained", "churn"}
    probabilities = result.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(result.predict_positive_proba(X), probabilities[:, 1])
    np.testing.assert_allclose(result.best_model.predict_proba(X), probabilities)
    np.testing.assert_allclose(result.predict_proba(X[X.columns[::-1]]), probabilities)
    holdout = X.iloc[list(result.test_indices)]
    target = mixed_df["target"].iloc[list(result.test_indices)]
    for name, scorer in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]:
        assert result.test_metrics[name] == pytest.approx(
            scorer(target, result.predict(holdout), pos_label="churn", zero_division=0),
        )
    preprocessor = result.best_model.named_steps["preprocessing"]
    assert result.transformed_feature_names == tuple(preprocessor.get_feature_names_out())
    assert result.feature_schema.feature_names == tuple(X.columns)
    numeric = preprocessor.named_transformers_["numeric"]
    assert ("scaler" in numeric.named_steps) == (model == "logistic_regression")
    assert "scaler" not in preprocessor.named_transformers_["boolean"].named_steps
    assert isinstance(numeric.named_steps["imputer"], SimpleImputer)
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    assert encoder.handle_unknown == "ignore" and not encoder.sparse_output
    assert "unused" not in encoder.categories_[1]
    importance = result.feature_importance
    assert list(importance.columns) == ["feature", "importance", "source_feature"]
    assert pd.api.types.is_numeric_dtype(importance["importance"])
    assert np.isfinite(importance["importance"]).all()
    assert set(importance["feature"]) == set(result.transformed_feature_names)
    assert set(importance["source_feature"]) == set(X.columns)
    assert set(importance.loc[importance.source_feature == "categorical", "feature"]) == {
        "categorical__categorical_France", "categorical__categorical_Germany", "categorical__categorical_Spain",
    }
    estimator = result.best_model.named_steps["estimator"]
    expected = estimator.coef_[0] if model == "logistic_regression" else estimator.feature_importances_
    indexed = importance.set_index("feature").loc[list(result.transformed_feature_names)]
    np.testing.assert_allclose(indexed["importance"], expected)
    # Missing values in every column are legitimate for inference, using fitted statistics.
    new = X.iloc[:3].copy(deep=True)
    new["categorical"] = pd.Series(["never_seen", None, "France"], index=new.index, dtype=object)
    for name in ["numeric_float", "numeric_int", "nullable_int", "boolean", "category_dtype", "string_dtype"]:
        new[name] = np.nan
    before = new.copy(deep=True)
    assert result.predict(new).shape == (3,)
    new_probability = result.predict_proba(new)
    assert np.isfinite(new_probability).all()
    np.testing.assert_allclose(new_probability.sum(axis=1), 1)
    assert result.predict_positive_proba(new).shape == (3,)
    pd.testing.assert_frame_equal(new, before)
    pd.testing.assert_frame_equal(mixed_df, original)


def test_preprocessor_is_unfitted_and_transforms_missing_and_unknown_categories():
    X = pd.DataFrame({
        "amount": [1.0, 3.0, np.nan, 9.0],
        "flag": pd.Series([True, False, pd.NA, True], dtype="boolean"),
        "country": pd.Series(["France", None, "France", "Spain"], dtype=object),
    })
    preprocessor = build_preprocessor(infer_schema(X), requires_scaling=False)
    assert not hasattr(preprocessor, "transformers_")
    fitted = preprocessor.fit_transform(X)
    assert isinstance(fitted, pd.DataFrame)
    np.testing.assert_allclose(fitted["numeric__amount"], [1, 3, 3, 9])
    np.testing.assert_allclose(fitted["boolean__flag"], [1, 0, 1, 1])
    new = pd.DataFrame({"amount": [np.nan], "flag": [None], "country": ["unknown"]})
    transformed = preprocessor.transform(new)
    assert transformed.loc[0, "numeric__amount"] == 3
    assert transformed.loc[0, "boolean__flag"] == 1
    assert transformed.filter(like="categorical__").to_numpy().sum() == 0


@pytest.mark.parametrize("required, scale, expected", [
    (True, "auto", True), (True, True, True), (True, False, False),
    (False, "auto", False), (False, True, True), (False, False, False),
])
def test_scaling_is_declared_by_custom_model_metadata(mixed_df, required, scale, expected):
    class CustomRegistry(ModelRegistry):
        _optional = {}
        _specs = {}

    CustomRegistry.register(replace(ModelRegistry.get("logistic_regression"), name="custom", requires_scaling=required))
    result = fit_mixed(mixed_df, models="custom", model_registry=CustomRegistry,
                       preprocessing=PreprocessingConfig(scale_numeric=scale))
    numeric = result.best_model.named_steps["preprocessing"].named_transformers_["numeric"]
    assert ("scaler" in numeric.named_steps) == expected


@pytest.mark.parametrize("strategy", ["mean", "most_frequent"])
def test_explicit_numeric_imputation_strategy(strategy):
    X = pd.DataFrame({"number": [1.0, 1.0, 8.0, np.nan]})
    preprocessor = build_preprocessor(infer_schema(X), PreprocessingConfig(numeric_imputation=strategy))
    transformed = preprocessor.fit_transform(X)
    expected = 10 / 3 if strategy == "mean" else 1
    assert transformed.iloc[-1, 0] == pytest.approx(expected)


@pytest.mark.parametrize("branch", ["numeric", "categorical", "boolean"])
def test_single_type_datasets_work(mixed_df, branch):
    columns = {"numeric": ["numeric_float", "nullable_int"], "categorical": ["categorical", "string_dtype"],
               "boolean": ["boolean"]}[branch]
    df = mixed_df[columns + ["target"]]
    result = fit_mixed(df)
    assert result.predict_positive_proba(df.drop(columns="target")).shape == (len(df),)


def test_constant_columns_are_kept_in_final_pipeline(mixed_df):
    mixed_df["constant"] = "same"
    with pytest.warns(UserWarning, match="Constant.*retained.*constant"):
        result = fit_mixed(mixed_df)
    assert "constant" in result.feature_schema.constant_columns
    assert "constant" in set(result.feature_importance.source_feature)
    assert "categorical__constant_same" in result.transformed_feature_names


def test_disabled_preprocessing_keeps_numeric_contract(binary_df, mixed_df):
    result = AutoML(target="target", models="logistic_regression", preprocessing=None,
                    cv=2, n_trials=1, verbose=0).fit(binary_df)
    assert isinstance(result.best_model, Pipeline)
    assert tuple(result.best_model.named_steps) == ("estimator",)
    assert result.feature_schema is None
    assert result.transformed_feature_names == result.feature_names
    X = binary_df.drop(columns="target")
    np.testing.assert_allclose(result.predict_proba(X.to_numpy()), result.predict_proba(X))
    with pytest.raises(DataValidationError):
        fit_mixed(mixed_df, preprocessing=None)


@pytest.mark.parametrize("corruption, message", [
    ("missing", "missing"), ("extra", "extra"), ("duplicate", "unique"),
    ("numeric_strings", "numeric"), ("bad_bool", "booleans"),
    ("category_numbers", "incompatible"), ("infinity", "infinite"), ("nested", "nested"),
])
def test_prediction_schema_errors_are_explicit(mixed_df, corruption, message):
    result = fit_mixed(mixed_df)
    X = mixed_df.drop(columns="target").iloc[:3].copy()
    if corruption == "missing":
        X = X.drop(columns="numeric_float")
    elif corruption == "extra":
        X["extra"] = 0
    elif corruption == "duplicate":
        X.columns = ["numeric_int", *X.columns[1:]]
    elif corruption == "numeric_strings":
        X["numeric_float"] = "bad"
    elif corruption == "bad_bool":
        X["boolean"] = 2
    elif corruption == "category_numbers":
        X["categorical"] = [1, 2, 3]
    elif corruption == "infinity":
        X["numeric_float"] = np.inf
    else:
        X["categorical"] = [[1], [2], [3]]
    with pytest.raises(DataValidationError, match=message):
        result.predict(X)


def test_categorical_numpy_inference_recommends_dataframe(mixed_df):
    result = fit_mixed(mixed_df)
    with pytest.raises(DataValidationError, match="DataFrame"):
        result.predict(mixed_df.drop(columns="target").to_numpy())


def test_numeric_boolean_numpy_and_compatible_dtypes(mixed_df):
    df = mixed_df[["numeric_float", "numeric_int", "boolean", "target"]]
    result = fit_mixed(df)
    X = df.drop(columns="target")
    expected = result.predict_proba(X)
    # A numeric matrix represents booleans as 0/1 and missing values as NaN.
    np.testing.assert_allclose(result.predict_proba(X.to_numpy(dtype=float, na_value=np.nan)), expected)
    compatible = X.copy()
    compatible["numeric_int"] = compatible.numeric_int.astype(float)
    compatible["boolean"] = compatible.boolean.astype("Float64")
    np.testing.assert_allclose(result.predict_proba(compatible), expected)


def test_prediction_accepts_booleans_with_none_in_object_dtype(mixed_df):
    result = fit_mixed(mixed_df)
    X = mixed_df.drop(columns="target").iloc[:3].copy()
    X["boolean"] = pd.Series([True, None, False], dtype=object)
    expected = X.copy()
    expected["boolean"] = expected.boolean.astype("boolean")
    np.testing.assert_allclose(result.predict_proba(X), result.predict_proba(expected))


def test_model_and_result_are_serializable(mixed_df):
    result = fit_mixed(mixed_df)
    restored = pickle.loads(pickle.dumps(result))
    X = mixed_df.drop(columns="target")
    np.testing.assert_allclose(restored.predict_proba(X), result.predict_proba(X))
    assert restored.feature_schema == result.feature_schema
    assert restored.transformed_feature_names == result.transformed_feature_names


@pytest.mark.parametrize("model, module", [
    ("logistic_regression", "sklearn"), ("lightgbm", "lightgbm"), ("xgboost", "xgboost"),
])
def test_mixed_preprocessing_is_reproducible(mixed_df, model, module):
    pytest.importorskip(module)
    first = fit_mixed(mixed_df, models=model, n_trials=2)
    second = fit_mixed(mixed_df, models=model, n_trials=2)
    assert first.train_indices == second.train_indices
    assert first.test_indices == second.test_indices
    assert first.best_model_name == second.best_model_name
    assert first.best_params == second.best_params
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    pd.testing.assert_frame_equal(first.feature_importance, second.feature_importance)
    X = mixed_df.drop(columns="target")
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X), rtol=1e-12, atol=1e-12)


def test_transformed_names_and_sources_handle_special_category_names(mixed_df):
    pytest.importorskip("xgboost")
    mixed_df = mixed_df.rename(columns={"categorical": "country[raw]"})
    mixed_df["country[raw]"] = mixed_df["country[raw]"].map({
        "France": "a_b", "Germany": "a[b]", "Spain": "<c>",
    })
    result = fit_mixed(mixed_df, models="xgboost")
    assert len(set(result.transformed_feature_names)) == len(result.transformed_feature_names)
    assert all(not any(char in name for char in "[]<") for name in result.transformed_feature_names)
    importance = result.feature_importance
    assert len(importance.loc[importance.source_feature == "country[raw]"]) == 3
    assert result.predict(mixed_df.drop(columns="target")).shape == (len(mixed_df),)
