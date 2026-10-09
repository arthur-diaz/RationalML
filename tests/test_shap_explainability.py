import json
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rationalml import AutoML
from rationalml.exceptions import ConfigurationError

shap = pytest.importorskip("shap")


def fit_result(df, task="binary", family="linear", **options):
    if family != "linear":
        pytest.importorskip(family)
    model = ("ridge" if task == "regression" else "logistic_regression") if family == "linear" else (
        family + ("_regressor" if task == "regression" else ""))
    return AutoML(**({"target": "target", "task": task, "models": model, "n_trials": 1, "cv": 2,
                     "random_state": 17, "verbose": 0} | options)).fit(df)


def mixed_features(df, mode):
    df = df.copy(deep=True)
    df["is_customer"] = np.arange(len(df)) % 2 == 0
    if mode is None:
        return df, None
    numeric = [name for name in df.columns if name not in {"target", "is_customer"}]
    df["country"] = pd.Series(np.array(["Paris", "Rennes", "Lyon"])[np.arange(len(df)) % 3], index=df.index, dtype=object)
    df.loc[::9, numeric[0]] = np.nan
    df.loc[::11, "country"] = np.nan
    if mode == "basic":
        return df, "basic"
    custom = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]), numeric),
        ("category", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["country"]),
        ("boolean", "passthrough", ["is_customer"]),
    ], verbose_feature_names_out=False)
    return df, custom


def transformed(result, X):
    preprocessor = result.best_model.named_steps.get("preprocessing")
    return X if preprocessor is None else preprocessor.transform(X)


def raw_output(result, values, label=None):
    estimator = result.best_model.named_steps["estimator"]
    family = type(estimator).__module__.split(".")[0]
    if family == "lightgbm":
        raw = estimator.predict(values, raw_score=True)
    elif family == "xgboost":
        raw = estimator.predict(values, output_margin=True)
    elif result.task.value == "regression":
        raw = estimator.predict(values)
    else:
        raw = estimator.decision_function(values)
    if result.task.value == "multiclass":
        return np.asarray(raw)[:, result.classes_.tolist().index(label)]
    return np.asarray(raw)


def vector_intercept(result):
    estimator = result.best_model.named_steps["estimator"]
    if type(estimator).__module__.split(".")[0] != "xgboost":
        return False
    base = json.loads(estimator.get_booster().save_config())["learner"]["learner_model_param"]["base_score"]
    return isinstance(base, list) or str(base).startswith("[")


def block_fitting(monkeypatch, result):
    import optuna
    import rationalml.automl as facade
    import rationalml.preprocessing.builder as builder

    def forbidden(*args, **kwargs):
        pytest.fail("SHAP must never fit, clone, optimize or rebuild a pipeline")

    def visit(owner):
        for name in ["fit", "fit_transform"]:
            if callable(getattr(owner, name, None)):
                # Patch classes so undo does not add inherited methods to the
                # fitted instance's __dict__ and invalidate the state snapshot.
                monkeypatch.setattr(type(owner), name, forbidden)
        if isinstance(owner, Pipeline):
            for step in owner.named_steps.values():
                visit(step)
        elif isinstance(owner, ColumnTransformer):
            for _, child, _ in owner.transformers_:
                if not isinstance(child, str):
                    visit(child)

    visit(result.best_model)
    for name in ["to_excel", "log_mlflow"]:
        monkeypatch.setattr(type(result), name, forbidden)
    for owner, names in [(facade, ["optimize_model", "evaluate_baseline", "evaluate_holdout", "make_cv_splits", "build_model_pipeline"]),
                         (builder, ["build_model_pipeline", "build_preprocessor", "clone"]),
                         (optuna, ["create_study"]), (optuna.study.Study, ["optimize"]),
                         (shap, ["KernelExplainer", "PermutationExplainer", "DeepExplainer"])]:
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("family", ["linear", "lightgbm", "xgboost"])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
@pytest.mark.parametrize("n_rows", [1, 5])
def test_fitted_model_raw_additivity_features_no_refit_or_mutation(request, monkeypatch, task, family, mode, n_rows):
    df, preprocessing = mixed_features(request.getfixturevalue(f"{task}_df"), mode)
    if task == "binary":
        df.target = df.target.map({0: "retained", 1: "churn"})
    result = fit_result(df, task, family, preprocessing=preprocessing, positive_class="churn" if task == "binary" else None)
    X = df.drop(columns="target").iloc[:n_rows].copy()
    background = df.drop(columns="target").iloc[20:37].copy()
    if mode is not None:
        X.loc[X.index[0], "country"] = "never_seen"
    original_X, original_background = X.copy(deep=True), background.copy(deep=True)
    label = "gold" if task == "multiclass" else None
    expected_data = transformed(result, X)
    expected_raw = raw_output(result, expected_data, label)
    saved_model = pickle.dumps(result.best_model)
    model = result.best_model
    keys = set(vars(result))
    frames = {name: getattr(result, name).copy(deep=True) for name in ["leaderboard", "cv_results", "feature_importance", "test_predictions"]}
    saved = {name: deepcopy(getattr(result, name)) for name in ["metrics", "test_metrics", "best_params", "model_best_params"]}
    with monkeypatch.context() as patches:
        block_fitting(patches, result)
        if family == "xgboost" and vector_intercept(result):
            # Same test runs the full additivity path in the isolated XGBoost 3.0.5 matrix.
            with pytest.raises(ConfigurationError, match="SHAP 0.49.x.*XGBoost vector base_score.*<3.1"):
                result.explain(X, background=background, class_label=label)
        else:
            explanation = result.explain(X, background=background, class_label=label)
            assert isinstance(explanation, shap.Explanation)
            assert explanation.values.shape == expected_data.shape
            assert explanation.values.ndim == 2 and explanation.base_values.shape == (n_rows,)
            np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1), expected_raw, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(explanation.data, np.asarray(expected_data, dtype=float))
            assert explanation.feature_names == list(result.transformed_feature_names)
            assert explanation[0].values.shape == (expected_data.shape[1],)
            assert explanation[0].data.shape == (expected_data.shape[1],)
            if task == "binary":
                assert explanation.output_names == "churn"
            elif task == "multiclass":
                assert explanation.output_names == label
            if mode is not None:
                assert any("country_" in name for name in explanation.feature_names)
    assert result.best_model is model and pickle.dumps(model) == saved_model
    assert set(vars(result)) == keys
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(getattr(result, name), frame)
    for name, value in saved.items():
        assert getattr(result, name) == value
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_frame_equal(background, original_background)


@pytest.mark.parametrize("family", ["linear", "lightgbm", "xgboost"])
@pytest.mark.parametrize("mapping,positive", [({0: 0, 1: 1}, 1), ({0: 0, 1: 1}, 0),
    ({0: False, 1: True}, True), ({0: "retained", 1: "churn"}, "churn")])
def test_binary_explains_internal_class_one_for_every_original_label(binary_df, family, mapping, positive):
    binary_df.target = binary_df.target.map(mapping)
    result = fit_result(binary_df, family=family, positive_class=positive)
    X = binary_df.drop(columns="target").iloc[:4]
    if family == "xgboost" and vector_intercept(result):
        with pytest.raises(ConfigurationError, match="XGBoost vector base_score"):
            result.explain(X, background=X)
    else:
        e = result.explain(X, background=X)
        assert e.output_names == str(positive)
        np.testing.assert_allclose(e.base_values + e.values.sum(axis=1), raw_output(result, X), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("family", ["linear", "lightgbm", "xgboost"])
@pytest.mark.parametrize("label", ["bronze", "gold", 0])
def test_multiclass_selects_business_class_including_zero(multiclass_df, family, label):
    if label == 0:
        multiclass_df.target = multiclass_df.target.map({"bronze": 0, "gold": 1, "silver": 2})
    result = fit_result(multiclass_df, "multiclass", family)
    X = multiclass_df.drop(columns="target").iloc[:4]
    if family == "xgboost" and vector_intercept(result):
        with pytest.raises(ConfigurationError, match="XGBoost vector base_score"):
            result.explain(X, background=X, class_label=label)
    else:
        e = result.explain(X, background=X, class_label=label)
        assert e.values.shape == X.shape and e.output_names == str(label)
        np.testing.assert_allclose(e.base_values + e.values.sum(axis=1), raw_output(result, X, label), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("family", ["linear", "lightgbm", "xgboost"])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
def test_full_explicit_background_is_preserved_beyond_shap_default_limit(request, monkeypatch, family, mode):
    df, preprocessing = mixed_features(request.getfixturevalue("binary_df"), mode)
    result = fit_result(df, family=family, preprocessing=preprocessing)
    X = df.drop(columns="target").iloc[:2]
    background = df.drop(columns="target").iloc[np.arange(137) % len(df)].copy()
    expected_background = transformed(result, background)
    observed = []
    original = shap.LinearExplainer if family == "linear" else shap.TreeExplainer

    def factory(estimator, *args, **kwargs):
        assert estimator is result.best_model.named_steps["estimator"]
        masker = args[0] if family == "linear" else kwargs["data"]
        observed.append(masker.data)
        assert masker.max_samples == 137
        np.testing.assert_allclose(masker.data, np.asarray(expected_background, dtype=float))
        if family != "linear":
            assert kwargs["model_output"] == "raw" and kwargs["feature_perturbation"] == "interventional"
        return original(estimator, *args, **kwargs)

    def no_sampling(*args, **kwargs):
        pytest.fail("RationalML must preserve the user's whole background")

    monkeypatch.setattr(shap, "LinearExplainer" if family == "linear" else "TreeExplainer", factory)
    monkeypatch.setattr(shap.utils, "sample", no_sampling)
    if vector_intercept(result):
        with pytest.raises(ConfigurationError, match="XGBoost vector base_score"):
            result.explain(X, background=background)
        assert observed == []
        return
    e = result.explain(X, background=background)
    assert len(observed) == 1 and observed[0].shape[0] == 137
    np.testing.assert_allclose(e.base_values, raw_output(result, expected_background).mean(), rtol=2e-5, atol=2e-5)
