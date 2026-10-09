import pickle

import numpy as np
import pandas as pd
import pytest

from rationalml import AutoML
from _robustness import assert_training_equal, case, class_label


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
@pytest.mark.parametrize("family", ["linear", "lightgbm", "xgboost"])
def test_independent_runs_reproduce_training_predictions_and_diagnostics(request, task, mode, family):
    if family != "linear":
        pytest.importorskip(family)
    df, options = case(request, task, mode)
    if family != "linear":
        options["models"] = family + ("_regressor" if task == "regression" else "")
    first, second = AutoML(**options).fit(df), AutoML(**options).fit(df)
    assert_training_equal(first, second)
    pd.testing.assert_frame_equal(first.test_predictions, second.test_predictions, rtol=1e-10, atol=1e-12)
    assert first.test_metrics == pytest.approx(second.test_metrics, rel=1e-10, abs=1e-12)
    label = class_label(task)
    pd.testing.assert_frame_equal(first.ranking_table(class_label=label), second.ranking_table(class_label=label),
                                  rtol=1e-10, atol=1e-12)
    if task != "regression":
        pd.testing.assert_frame_equal(first.calibration_table(class_label=label), second.calibration_table(class_label=label),
                                      rtol=1e-10, atol=1e-12)
    # Intra-version roundtrip only: no cross-version persistence promise.
    restored = pickle.loads(pickle.dumps(first.best_model))
    X = df.drop(columns="target")
    np.testing.assert_allclose(restored.predict(X), first.best_model.predict(X), rtol=1e-10, atol=1e-12)
    if task != "regression":
        np.testing.assert_allclose(restored.predict_proba(X), first.predict_proba(X), rtol=1e-10, atol=1e-12)


def test_different_seeds_can_change_split_without_requiring_another_winner(binary_df):
    options = dict(target="target", models="logistic_regression", cv=2, n_trials=1, verbose=0)
    first = AutoML(**options, random_state=17).fit(binary_df)
    second = AutoML(**options, random_state=19).fit(binary_df)
    assert set(first.test_indices) != set(second.test_indices)
