import numpy as np
import pandas as pd
import pytest

from rationalml.exceptions import LegacyAPIError, UnsupportedTaskError
from rationalml.optimization.spaces import legacy_search_domains
from strategy import local_optimizer
from sco_mod import sco_mod
from second_step import second_step
from scoring import opti
from report import prediction
from metrics import Metrics


def legacy_context(df, **options):
    with pytest.warns(DeprecationWarning):
        return local_optimizer(df, "target", **({
            "algo": "skl_rl", "nb_cv": 2, "nb_iter": 1, "seed": 29, "verbose": 0,
        } | options))


def test_legacy_data_and_search_domains_remain_independent(binary_df):
    context = legacy_context(binary_df)
    original = binary_df.copy(deep=True)
    context._pandas_sets()
    context.discretize(binary_df)
    pd.testing.assert_frame_equal(binary_df, original)
    assert context.h_param == legacy_search_domains()


@pytest.mark.parametrize("algorithm", ["skl_rl", "skl_rs", "xgb_gb", "xgb_rf", "xgb_mx", "lgb_gb", "lgb_rf"])
def test_legacy_algorithms_use_safe_engine(binary_df, algorithm):
    if algorithm.startswith("xgb"):
        pytest.importorskip("xgboost")
    elif algorithm.startswith("lgb"):
        pytest.importorskip("lightgbm")
    context = legacy_context(binary_df, algo=algorithm, n_estimators=5)
    with pytest.warns((DeprecationWarning, UserWarning)):
        table = sco_mod(context)
    assert isinstance(table, pd.DataFrame)
    assert not table.empty
    assert context.result_.best_model_name == algorithm
    assert context.result_.best_params["random_state"] == 29
    assert set(context.result_.train_indices).isdisjoint(context.result_.test_indices)


def test_old_full_data_optimization_and_reports_cannot_run(binary_df):
    context = legacy_context(binary_df)
    X, y = context._pandas_sets()
    with pytest.raises(LegacyAPIError, match="training_only"):
        opti(X, y, context, 0, "skl_rl")
    with pytest.raises(LegacyAPIError, match="retired"):
        prediction(None, X, y, "Sklearn", "LR", context, 0, "", {}, "roc_auc")
    with pytest.warns(DeprecationWarning):
        params, scores = opti(X, y, context, 0, "skl_rl", training_only=True)
    assert params["random_state"] == 29
    assert np.isfinite(scores["cv_std"])


@pytest.mark.parametrize("options", [{"excel_report": True}, {"model_save": "models"}, {"class_weight": "auto"}])
def test_legacy_unimplemented_options_fail_explicitly(binary_df, options):
    context = legacy_context(binary_df, **options)
    with pytest.warns(DeprecationWarning), pytest.raises(LegacyAPIError):
        sco_mod(context)


def test_segment_errors_are_not_swallowed(binary_df):
    binary_df["segment"] = "large"
    binary_df.loc[:1, "segment"] = "small"
    context = legacy_context(binary_df)
    with pytest.warns((DeprecationWarning, UserWarning)), pytest.raises(UnsupportedTaskError):
        second_step(context, "segment")


def test_legacy_threshold_handles_named_targets_and_extreme_rates():
    y = pd.Series([0, 0, 1, 1], name="target")
    prediction, threshold = Metrics.apply_threshold("FP", np.array([0.1, 0.2, 0.8, 0.9]), y, 1, "target")
    assert len(prediction) == len(y)
    assert np.isfinite(threshold)
