from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from rationalml import AutoML, ModelRegistry


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
def test_ranking_is_holdout_only_and_never_calls_model_or_optimizer(request, monkeypatch, task, mode):
    import rationalml.automl as facade

    df = request.getfixturevalue(f"{task}_df")
    if task == "binary":
        df.target = df.target.map({0: "retained", 1: "churn"})
    if mode is not None:
        df.loc[::11, "feature_0"] = np.nan
    base = ModelRegistry.get("ridge" if task == "regression" else "logistic_regression")
    fits, predictions = [], []

    class RecordingEstimator(base.estimator_class):
        def fit(self, X, y, **options):
            fits.append((self, set(X.index)))
            return super().fit(X, y, **options)

        def predict(self, X):
            predictions.append(("predict", set(X.index)))
            return super().predict(X)

        def predict_proba(self, X):
            predictions.append(("proba", set(X.index)))
            return super().predict_proba(X)

    class Registry(ModelRegistry):
        _optional = {}
        _specs = {"recording": replace(base, name="recording", estimator_class=RecordingEstimator)}

    preprocessing = SimpleImputer().set_output(transform="pandas") if mode == "custom" else mode
    result = AutoML(target="target", task=task, models="recording", model_registry=Registry,
                    positive_class="churn" if task == "binary" else None, preprocessing=preprocessing,
                    cv=3, n_trials=2, random_state=17, verbose=0).fit(df)
    train, holdout = set(result.train_indices), set(result.test_indices)
    assert len(fits) == 7
    assert fits[-1][1] == train
    assert all(rows <= train and rows.isdisjoint(holdout) for _, rows in fits)
    expected_calls = ["predict"] if task == "regression" else ["predict", "proba"]
    assert [(kind, rows) for kind, rows in predictions if rows & holdout] == [(kind, holdout) for kind in expected_calls]
    artifact = result.test_predictions
    assert set(artifact.index) == holdout and len(artifact) == len(holdout)

    def unexpected(*args, **kwargs):
        pytest.fail("Post-model ranking must only read stored holdout predictions")

    monkeypatch.setattr(result.best_model, "predict", unexpected)
    if task != "regression":
        monkeypatch.setattr(result.best_model, "predict_proba", unexpected)
    monkeypatch.setattr(result.best_model, "fit", unexpected)
    monkeypatch.setattr(facade, "optimize_model", unexpected)
    monkeypatch.setattr(facade, "evaluate_baseline", unexpected)
    label = "gold" if task == "multiclass" else None
    ranking = result.ranking_table(n_bins=5, class_label=label)
    top = result.top_segment(.1, class_label=label)
    assert ranking["count"].sum() == len(holdout)
    assert set(top.index) <= holdout
    # The result owns its artifact and has no dependence on the caller's train data.
    df.loc[list(train), df.columns != "target"] = 12345.
    if task == "regression":
        df.loc[list(train), "target"] += 1e6
    else:
        df.loc[list(train), "target"] = "changed_train"
    pd.testing.assert_frame_equal(result.ranking_table(n_bins=5, class_label=label), ranking)
    pd.testing.assert_frame_equal(result.top_segment(.1, class_label=label), top)
    pd.testing.assert_frame_equal(result.test_predictions, artifact)
    ranking.loc[0, "count"] = 999
    top.iloc[0, 0] = -999. if task == "regression" else "changed_copy"
    assert result.ranking_table(n_bins=5, class_label=label)["count"].sum() == len(holdout)
    pd.testing.assert_frame_equal(result.test_predictions, artifact)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("changed_part", ["features", "targets"])
def test_holdout_changes_only_affect_post_model_results(request, monkeypatch, task, changed_part):
    import rationalml.automl as facade

    df = request.getfixturevalue(f"{task}_df")
    options = dict(target="target", task=task, models="auto", preprocessing="basic",
                   cv=3, n_trials=2, random_state=17, verbose=0)
    first = AutoML(**options).fit(df)
    changed = df.copy(deep=True)
    heldout = list(first.test_indices)
    if changed_part == "features":
        changed.loc[heldout, changed.columns != "target"] *= -1000.
    else:
        # Change labels after fixing the existing holdout boundary. Otherwise a
        # stratified split could legitimately move rows when full-data labels change.
        def existing_split(rows, **options):
            np.testing.assert_array_equal(rows, np.arange(len(df)))
            return np.asarray(first.train_indices), np.asarray(first.test_indices)

        monkeypatch.setattr(facade, "train_test_split", existing_split)
        if task == "binary":
            changed.loc[heldout, "target"] = 1 - changed.loc[heldout, "target"]
        elif task == "multiclass":
            changed.loc[heldout, "target"] = changed.loc[heldout, "target"].map({"bronze": "gold", "gold": "silver", "silver": "bronze"})
        else:
            changed.loc[heldout, "target"] += 1000.
    second = AutoML(**options).fit(changed)
    assert first.train_indices == second.train_indices and first.test_indices == second.test_indices
    assert first.best_model_name == second.best_model_name and first.best_params == second.best_params
    assert first.baseline_score == second.baseline_score and first.baseline_fold_scores == second.baseline_fold_scores
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    pd.testing.assert_frame_equal(first.cv_results, second.cv_results)
    pd.testing.assert_frame_equal(first.feature_importance, second.feature_importance)
    assert not first.test_predictions.equals(second.test_predictions)
    label = "gold" if task == "multiclass" else None
    assert not first.ranking_table(class_label=label).equals(second.ranking_table(class_label=label))
    X = df.drop(columns="target")
    if task == "regression":
        np.testing.assert_allclose(first.predict(X), second.predict(X))
    else:
        np.testing.assert_array_equal(first.predict(X), second.predict(X))
        np.testing.assert_allclose(first.predict_proba(X), second.predict_proba(X))
