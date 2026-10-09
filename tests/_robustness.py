"""Small datasets and assertions shared by the robustness tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def case(request, task, mode):
    df = request.getfixturevalue(f"{task}_df").copy(deep=True)
    if task == "binary":
        df["target"] = df.target.map({0: "retained", 1: "churn"})
    preprocessing = mode
    if mode is not None:
        df.loc[::11, "feature_0"] = np.nan
    if mode == "basic":
        df["country"] = np.array(["France", "Germany", "Spain"])[np.arange(len(df)) % 3]
        df["is_customer"] = np.arange(len(df)) % 2 == 0
    elif mode == "custom":
        preprocessing = Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
        ]).set_output(transform="pandas")
    return df, dict(target="target", task=task, preprocessing=preprocessing,
                    positive_class="churn" if task == "binary" else None,
                    models="ridge" if task == "regression" else "logistic_regression",
                    cv=2, n_trials=1, random_state=17, verbose=0)


def assert_training_equal(first, second):
    assert first.train_indices == second.train_indices
    assert first.test_indices == second.test_indices
    assert first.best_model_name == second.best_model_name
    assert first.best_params == second.best_params
    assert first.model_best_params == second.model_best_params
    assert first.baseline_name == second.baseline_name
    assert first.baseline_score == pytest.approx(second.baseline_score, rel=1e-10, abs=1e-12)
    np.testing.assert_allclose(first.baseline_fold_scores, second.baseline_fold_scores, rtol=1e-10, atol=1e-12)
    for name in ("leaderboard", "cv_results", "feature_importance"):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name), rtol=1e-10, atol=1e-12)
    left, right = (result.best_model.named_steps["estimator"] for result in (first, second))
    assert left.get_params(deep=True) == right.get_params(deep=True)
    for name in ("coef_", "intercept_", "feature_importances_"):
        if hasattr(left, name):
            np.testing.assert_allclose(getattr(left, name), getattr(right, name), rtol=1e-10, atol=1e-12)


def class_label(task):
    return "gold" if task == "multiclass" else None
