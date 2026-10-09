import json

import numpy as np
import pytest

from rationalml import AutoML
from rationalml.exceptions import ConfigurationError, DataValidationError, MissingDependencyError

shap = pytest.importorskip("shap")


@pytest.mark.parametrize("error_type", [ConfigurationError, DataValidationError, MissingDependencyError])
@pytest.mark.parametrize("stage", ["construction", "explanation"])
def test_existing_rationalml_errors_pass_through_shap_unchanged(binary_df, monkeypatch, error_type, stage):
    result = AutoML(target="target", models="logistic_regression", cv=2, n_trials=1, verbose=0).fit(binary_df)
    error = error_type("explicit RationalML diagnosis")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(shap, "LinearExplainer", fail if stage == "construction" else lambda *args, **kwargs: fail)
    X = binary_df.drop(columns="target")
    with pytest.raises(error_type) as raised:
        result.explain(X.iloc[:3], background=X.iloc[list(result.train_indices)[:10]])
    assert raised.value is error
    assert raised.value.__cause__ is None


def test_foreign_shap_error_is_wrapped_with_its_original_cause(binary_df, monkeypatch):
    result = AutoML(target="target", models="logistic_regression", cv=2, n_trials=1, verbose=0).fit(binary_df)
    error = RuntimeError("foreign SHAP failure")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(shap, "LinearExplainer", fail)
    X = binary_df.drop(columns="target")
    with pytest.raises(ConfigurationError, match="foreign SHAP failure") as raised:
        result.explain(X.iloc[:3], background=X.iloc[list(result.train_indices)[:10]])
    assert raised.value.__cause__ is error


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_xgboost_shap_compatibility_guard_never_blocks_fit_or_inference(request, task):
    pytest.importorskip("xgboost")
    df = request.getfixturevalue(f"{task}_df")
    result = AutoML(target="target", task=task, models="xgboost", cv=2, n_trials=1, verbose=0).fit(df)
    X = df.drop(columns="target")
    before = result.predict(X)
    probabilities = result.predict_proba(X)
    assert before.shape == (len(df),)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    base = json.loads(result.best_model.named_steps["estimator"].get_booster().save_config())
    score = base["learner"]["learner_model_param"]["base_score"]
    label = "gold" if task == "multiclass" else None
    if isinstance(score, list) or str(score).lstrip().startswith("["):
        with pytest.raises(ConfigurationError, match="vector base_score.*training and prediction remain supported"):
            result.explain(X.iloc[:3], background=X.iloc[list(result.train_indices)[:10]], class_label=label)
    else:
        assert result.explain(X.iloc[:3], background=X.iloc[list(result.train_indices)[:10]], class_label=label).values.shape == (3, X.shape[1])
    np.testing.assert_array_equal(result.predict(X), before)
    np.testing.assert_allclose(result.predict_proba(X), probabilities)
