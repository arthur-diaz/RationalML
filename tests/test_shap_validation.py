from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from rationalml import AutoML, ModelRegistry
from rationalml.exceptions import ConfigurationError, DataValidationError
from test_shap_explainability import block_fitting, fit_result, mixed_features, raw_output

shap = pytest.importorskip("shap")


@pytest.mark.parametrize("mode", [None, "basic", "custom"])
@pytest.mark.parametrize("argument", ["X", "background"])
@pytest.mark.parametrize("invalid", ["missing", "extra", "duplicate", "empty", "array_width", "array_1d", "bad_dtype"])
def test_X_and_background_share_prediction_validation(binary_df, mode, argument, invalid):
    df, preprocessing = mixed_features(binary_df, mode)
    result = fit_result(df, preprocessing=preprocessing)
    good = df.drop(columns="target").iloc[:4].copy()
    bad = good.copy(deep=True)
    if invalid == "missing":
        bad = bad.drop(columns=bad.columns[0])
    elif invalid == "extra":
        bad["unexpected"] = 1
    elif invalid == "duplicate":
        bad.columns = [bad.columns[0]] * len(bad.columns)
    elif invalid == "empty":
        bad = bad.iloc[:0]
    elif invalid == "array_width":
        bad = np.zeros((4, len(bad.columns) - 1))
    elif invalid == "array_1d":
        bad = np.zeros(len(bad.columns))
    else:
        bad[bad.columns[0]] = "not_numeric"
    options = {"X": good, "background": good}
    options[argument] = bad
    # A custom transform owns its dtype validation, just as it does for predict.
    expected = (DataValidationError, ValueError) if mode == "custom" and invalid == "bad_dtype" else DataValidationError
    with pytest.raises(expected):
        result.explain(**options)


@pytest.mark.parametrize("mode", [None, "basic", "custom"])
def test_column_reordering_is_explicit_for_both_inputs(binary_df, mode):
    df, preprocessing = mixed_features(binary_df, mode)
    result = fit_result(df, preprocessing=preprocessing)
    X = df.drop(columns="target").iloc[:4]
    background = df.drop(columns="target").iloc[10:30]
    expected = result.explain(X, background=background)
    reversed_X, reversed_background = X.iloc[:, ::-1].copy(), background.iloc[:, ::-1].copy()
    actual = result.explain(reversed_X, background=reversed_background)
    np.testing.assert_allclose(actual.values, expected.values)
    np.testing.assert_allclose(actual.base_values, expected.base_values)
    np.testing.assert_allclose(actual.data, expected.data)
    assert actual.feature_names == expected.feature_names
    assert list(reversed_X.columns) == list(X.columns)[::-1]


@pytest.mark.parametrize("mode", [None, "basic", "custom"])
@pytest.mark.parametrize("index", [pd.Index([5, 5, 1, 1]), pd.Index(["a", "b", "a", "b"]),
    pd.date_range("2025-01-01", periods=4), pd.MultiIndex.from_tuples([("a", 1), ("a", 1), ("b", 2), ("b", 2)])])
def test_indices_do_not_become_features(binary_df, mode, index):
    df, preprocessing = mixed_features(binary_df, mode)
    result = fit_result(df, preprocessing=preprocessing)
    X = df.drop(columns="target").iloc[:4].copy()
    expected = result.explain(X, background=X)
    X.index = index
    e = result.explain(X, background=X.iloc[::-1])
    np.testing.assert_allclose(e.values, expected.values)
    np.testing.assert_allclose(e.base_values, expected.base_values)
    assert e.feature_names == expected.feature_names
    pd.testing.assert_index_equal(X.index, index)


@pytest.mark.parametrize("mode", [None, "basic", "custom"])
def test_numeric_ndarrays_when_supported_by_predict(binary_df, mode):
    result = fit_result(binary_df, preprocessing=mode if mode != "custom" else FunctionTransformer(feature_names_out="one-to-one"))
    X = binary_df.drop(columns="target").iloc[:4]
    expected = result.explain(X, background=X)
    values = X.to_numpy(copy=True)
    e = result.explain(values, background=values)
    np.testing.assert_allclose(e.values, expected.values)
    np.testing.assert_allclose(e.data, expected.data)
    np.testing.assert_allclose(values, X.to_numpy())


@pytest.mark.parametrize("task,label", [("binary", 0), ("binary", 1), ("regression", 0),
                                       ("multiclass", None), ("multiclass", "unknown")])
def test_class_selection_errors_are_explicit(request, task, label):
    result = fit_result(request.getfixturevalue(f"{task}_df"), task)
    X = request.getfixturevalue(f"{task}_df").drop(columns="target").iloc[:3]
    message = "available classes" if label == "unknown" else "class_label"
    with pytest.raises(ConfigurationError, match=message):
        result.explain(X, background=X, class_label=label)


def two_columns(X):
    return np.column_stack([X.iloc[:, 0] + X.iloc[:, 1], X.iloc[:, 0] - X.iloc[:, 1]])


def test_custom_unknown_names_use_warned_positions_without_breaking_explanations(binary_df):
    with pytest.warns(UserWarning, match="transformed feature names are unavailable"):
        result = fit_result(binary_df, preprocessing=FunctionTransformer(two_columns))
    assert result.transformed_feature_names is None and result.feature_importance.empty
    X = binary_df.drop(columns="target").iloc[:3]
    with pytest.warns(UserWarning, match="^Transformed feature names are unavailable; SHAP uses positional feature names\\.$"):
        e = result.explain(X, background=X)
    assert e.feature_names == ["feature_0", "feature_1"] and e.values.shape == (3, 2)
    np.testing.assert_allclose(e.data, two_columns(X))
    np.testing.assert_allclose(e.base_values + e.values.sum(axis=1), raw_output(result, two_columns(X)))
    assert result.feature_importance.empty and result.transformed_feature_names is None


def mutate_input(X):
    X.loc[:, :] = X.to_numpy() * 2
    return X


def test_custom_transform_writing_to_input_does_not_mutate_user_data(binary_df):
    result = fit_result(binary_df, preprocessing=FunctionTransformer(mutate_input, feature_names_out="one-to-one"))
    X, reference = binary_df.drop(columns="target").iloc[:4].copy(), binary_df.drop(columns="target").iloc[10:20].copy()
    saved_X, saved_reference = X.copy(deep=True), reference.copy(deep=True)
    e = result.explain(X, background=reference)
    np.testing.assert_allclose(e.data, saved_X.to_numpy() * 2)
    pd.testing.assert_frame_equal(X, saved_X)
    pd.testing.assert_frame_equal(reference, saved_reference)


def dates_as_numbers(X):
    return np.column_stack([X.drop(columns="visit_date").to_numpy(), X.visit_date.astype("int64").to_numpy() / 1e18])


def test_custom_preprocessing_controls_dtypes_without_basic_schema(binary_df):
    binary_df["visit_date"] = pd.date_range("2025-01-01", periods=len(binary_df))
    transformer = FunctionTransformer(dates_as_numbers, feature_names_out="one-to-one")
    result = fit_result(binary_df, preprocessing=transformer)
    X = binary_df.drop(columns="target").iloc[:3]
    e = result.explain(X, background=X)
    np.testing.assert_allclose(e.data, dates_as_numbers(X))
    assert "visit_date" in e.feature_names


def test_unregistered_explainer_family_has_no_agnostic_fallback(binary_df, monkeypatch):
    class Registry(ModelRegistry):
        _optional = {}
        _specs = {"custom": replace(ModelRegistry.get("logistic_regression"), name="custom",
                                   estimator_class=DecisionTreeClassifier, default_params={}, search_space=lambda trial: {})}

    result = AutoML(target="target", models="custom", model_registry=Registry, n_trials=1, cv=2, verbose=0).fit(binary_df)
    X = binary_df.drop(columns="target").iloc[:3]
    with monkeypatch.context() as patches:
        block_fitting(patches, result)
        with pytest.raises(ConfigurationError, match="no supported explainer.*DecisionTreeClassifier"):
            result.explain(X, background=X)


def test_registered_name_does_not_decide_explainer_family(binary_df):
    class Registry(ModelRegistry):
        _optional = {}
        _specs = {"business_model": replace(ModelRegistry.get("logistic_regression"), name="business_model")}

    result = AutoML(target="target", models="business_model", model_registry=Registry, n_trials=1, cv=2, verbose=0).fit(binary_df)
    X = binary_df.drop(columns="target").iloc[:3]
    e = result.explain(X, background=X)
    np.testing.assert_allclose(e.base_values + e.values.sum(axis=1), raw_output(result, X))


def to_csr(X):
    return sparse.csr_matrix(X.to_numpy(dtype=float))


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_linear_sparse_stays_sparse_and_preserves_raw_additivity(request, monkeypatch, task):
    df = request.getfixturevalue(f"{task}_df")
    result = fit_result(df, task, preprocessing=FunctionTransformer(to_csr, feature_names_out="one-to-one"))
    X = df.drop(columns="target").iloc[:3]
    reference = df.drop(columns="target").iloc[10:30]
    label = "gold" if task == "multiclass" else None
    expected = raw_output(result, to_csr(X), label)
    for method in ["toarray", "todense"]:
        original = getattr(sparse.csr_matrix, method)

        def only_single_row(self, *args, _original=original, **kwargs):
            assert self.shape[0] == 1, "Whole sparse input/background must never be densified"
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(sparse.csr_matrix, method, only_single_row)
    e = result.explain(X, background=reference, class_label=label)
    assert sparse.isspmatrix_csr(e.data) and e.data.shape == X.shape
    assert (e.data != to_csr(X)).nnz == 0
    np.testing.assert_allclose(e.base_values + e.values.sum(axis=1), expected)


@pytest.mark.parametrize("family", ["lightgbm", "xgboost"])
def test_tree_sparse_is_explicitly_rejected_without_densification(binary_df, family, monkeypatch):
    result = fit_result(binary_df, family=family, preprocessing=FunctionTransformer(to_csr, feature_names_out="one-to-one"))
    X = binary_df.drop(columns="target").iloc[:3]

    def forbidden(*args, **kwargs):
        pytest.fail("Sparse tree inputs must not be densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(sparse.csr_matrix, "todense", forbidden)
    with pytest.raises(ConfigurationError, match="TreeExplainer requires dense.*not densified"):
        result.explain(X, background=X)
