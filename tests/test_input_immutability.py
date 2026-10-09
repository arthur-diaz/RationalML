from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from rationalml import AutoML, AutoMLConfig
from _robustness import case


@pytest.mark.parametrize("mode", [None, "basic", "custom"])
@pytest.mark.parametrize("index_kind", ["string", "duplicate", "datetime", "unsorted"])
def test_fit_and_result_calls_preserve_caller_data_config_and_transformer(request, mode, index_kind):
    df, options = case(request, "binary", mode)
    if index_kind == "string":
        df.index = pd.Index([f"row_{i}" for i in range(len(df))], name="client")
    elif index_kind == "duplicate":
        df.index = pd.Index([f"row_{i // 2}" for i in range(len(df))], name="client")
    elif index_kind == "datetime":
        df.index = pd.date_range("2024-01-01", periods=len(df), name="date")
    else:
        df.index = pd.Index(np.random.default_rng(4).permutation(len(df)), name="position")
    original = df.copy(deep=True)
    options.update(task="auto", metric="auto", models=["logistic_regression"])
    config = AutoMLConfig(**options)
    original_config = deepcopy(config)
    transformer = config.preprocessing
    transformer_state = deepcopy(vars(transformer)) if mode == "custom" else None
    result = AutoML(config).fit(df)
    X = df.drop(columns="target").iloc[:10]
    original_X = X.copy(deep=True)
    predictions = result.predict(X)
    probabilities = result.predict_proba(X)
    np.testing.assert_array_equal(predictions, result.predict(X[X.columns[::-1]]))
    np.testing.assert_allclose(probabilities, result.predict_proba(X[X.columns[::-1]]))
    np.testing.assert_allclose(result.predict_positive_proba(X), probabilities[:, 1])
    expected_index = df.index.take(result.test_indices)
    pd.testing.assert_index_equal(result.test_predictions.index, expected_index)
    assert result.feature_names == tuple(df.columns.drop("target"))
    assert all(type(position) is int for position in result.train_indices + result.test_indices)
    assert set(result.train_indices).isdisjoint(result.test_indices)
    assert sorted(result.train_indices + result.test_indices) == list(range(len(df)))
    assert config.task == "auto" and config.metric == "auto"
    for name, value in vars(original_config).items():
        if name != "preprocessing":
            assert getattr(config, name) == value
    assert result.config is not config
    assert result.config.models is not config.models
    assert config.preprocessing is transformer
    if mode == "custom":
        assert vars(transformer).keys() == transformer_state.keys()
        assert repr(transformer) == repr(original_config.preprocessing)
        for step in transformer.named_steps.values():
            with pytest.raises(NotFittedError):
                check_is_fitted(step)
        assert result.best_model.named_steps["preprocessing"] is not transformer
    stored, ranking, calibration = result.test_predictions, result.ranking_table(), result.calibration_table()
    for name in ("test_predictions", "ranking_table", "calibration_table"):
        copy = getattr(result, name)
        copy = copy() if callable(copy) else copy
        numeric = copy.select_dtypes(include="number").columns[0]
        copy.loc[:, numeric] = -12345
    pd.testing.assert_frame_equal(result.test_predictions, stored)
    pd.testing.assert_frame_equal(result.ranking_table(), ranking)
    pd.testing.assert_frame_equal(result.calibration_table(), calibration)
    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(X, original_X)
