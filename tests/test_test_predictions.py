from dataclasses import replace
import pickle

import numpy as np
import pandas as pd
import pytest

from rationalml import AutoML
from rationalml.exceptions import ConfigurationError


def fit_task(df, task, **options):
    return AutoML(**({"target": "target", "task": task, "cv": 2, "n_trials": 1,
                     "random_state": 17, "verbose": 0} | options)).fit(df)


@pytest.mark.parametrize("task,model,module", [
    ("binary", "logistic_regression", "sklearn"), ("binary", "lightgbm", "lightgbm"), ("binary", "xgboost", "xgboost"),
    ("multiclass", "logistic_regression", "sklearn"), ("multiclass", "lightgbm", "lightgbm"), ("multiclass", "xgboost", "xgboost"),
    ("regression", "ridge", "sklearn"), ("regression", "lightgbm_regressor", "lightgbm"), ("regression", "xgboost_regressor", "xgboost"),
])
def test_holdout_artifact_contract_labels_probability_order_and_index(request, task, model, module):
    pytest.importorskip(module)
    df = request.getfixturevalue(f"{task}_df")
    if task == "binary":
        df.target = df.target.map({0: "retained", 1: "churn"})
    elif task == "multiclass":
        df.target = df.target.map({"bronze": "proba_0", "gold": "y_pred", "silver": "a/b_ c"})
    df.index = pd.Index([f"customer_{number % 7}" for number in range(len(df))], name="customer")
    original = df.copy(deep=True)
    result = fit_task(df, task, models=model, positive_class="churn" if task == "binary" else None)
    table = result.test_predictions
    positions = list(result.test_indices)
    pd.testing.assert_index_equal(table.index, df.index.take(positions))
    np.testing.assert_array_equal(table.y_true, df.target.iloc[positions])
    test_X = df.drop(columns="target").iloc[positions]
    np.testing.assert_array_equal(table.y_pred, result.predict(test_X))
    if task == "regression":
        assert list(table.columns) == ["y_true", "y_pred", "residual"]
        np.testing.assert_allclose(table.residual, table.y_true - table.y_pred)
    elif task == "binary":
        assert list(table.columns) == ["y_true", "y_pred", "score"]
        assert set(table.y_true) == {"churn", "retained"}
        np.testing.assert_allclose(table.score, result.predict_positive_proba(test_X))
    else:
        assert list(table.columns) == ["y_true", "y_pred", "proba_0", "proba_1", "proba_2"]
        probability_columns = [f"proba_{position}" for position in range(len(result.classes_))]
        probabilities = table[probability_columns].to_numpy()
        np.testing.assert_allclose(probabilities, result.predict_proba(test_X))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
        np.testing.assert_array_equal(table.y_pred, result.classes_[probabilities.argmax(axis=1)])
    labels = [result.classes_[0], result.classes_[1]] if task == "multiclass" else [None]
    for label in labels:
        ranking = result.ranking_table(class_label=label)
        assert ranking["count"].sum() == len(table)
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in ranking.dtypes)
        top = result.top_segment(.10, class_label=label)
        assert len(top) == int(np.ceil(len(table) * .10))
    copied = result.test_predictions
    copied.iloc[0, 0] = 0. if task == "regression" else "edited"
    copied.index = range(len(copied))
    pd.testing.assert_frame_equal(result.test_predictions, table)
    result.predict(df.drop(columns="target").iloc[:3])
    if task != "regression":
        result.predict_proba(df.drop(columns="target").iloc[:3])
    pd.testing.assert_frame_equal(result.test_predictions, table)
    restored = pickle.loads(pickle.dumps(result))
    pd.testing.assert_frame_equal(restored.test_predictions, table)
    pd.testing.assert_frame_equal(restored.ranking_table(class_label=labels[0]), result.ranking_table(class_label=labels[0]))
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("positive", [0, 1, False, True, "churn"])
def test_binary_artifact_always_scores_the_explicit_positive_class(binary_df, positive):
    if isinstance(positive, bool):
        binary_df.target = binary_df.target.astype(bool)
    elif isinstance(positive, str):
        binary_df.target = binary_df.target.map({0: "retained", 1: "churn"})
    result = fit_task(binary_df, "binary", models="logistic_regression", positive_class=positive)
    table = result.test_predictions
    X = binary_df.drop(columns="target").iloc[list(result.test_indices)]
    np.testing.assert_allclose(table.score, result.predict_proba(X)[:, 1])
    assert result.ranking_table().positives.sum() == table.y_true.eq(positive).sum()


@pytest.fixture
def multiclass_result(multiclass_df):
    return fit_task(multiclass_df, "multiclass", models="logistic_regression")


@pytest.mark.parametrize("label,positives,positions", [("bronze", [2, 1, 0], [1, 4, 7]), ("gold", [2, 0, 1], [0, 6, 3])])
def test_multiclass_ranking_and_top_use_selected_class(multiclass_result, label, positives, positions):
    bronze = np.array([.1, .9, .2, .2, .7, .1, .1, .6, .2])
    gold = np.array([.8, .05, .2, .6, .1, .2, .7, .3, .1])
    table = pd.DataFrame({
        "y_true": ["gold", "bronze", "silver", "bronze", "gold", "silver", "gold", "bronze", "silver"],
        "y_pred": ["gold"] * 9, "proba_0": bronze, "proba_1": gold, "proba_2": 1 - bronze - gold,
    }, index=pd.Index([f"row_{number}" for number in range(9)], name="customer"))
    result = replace(multiclass_result, _test_predictions=table)
    ranking = result.ranking_table(n_bins=3, class_label=label)
    assert list(ranking.positives) == positives
    assert ranking.positives.sum() == table.y_true.eq(label).sum()
    pd.testing.assert_frame_equal(result.top_segment(1 / 3, class_label=label), table.iloc[positions])
    table.iloc[0, 0] = "changed"
    assert result.test_predictions.y_true.iloc[0] == "gold"  # Owned constructor copy.


@pytest.mark.parametrize("operation", ["ranking_table", "top_segment"])
@pytest.mark.parametrize("label", [None, "absent", np.nan])
def test_multiclass_class_label_is_required_and_checked(multiclass_result, operation, label):
    message = "Multiclass ranking requires class_label" if label is None else "available classes"
    with pytest.raises(ConfigurationError, match=message):
        getattr(multiclass_result, operation)(class_label=label)


@pytest.mark.parametrize("task", ["binary", "regression"])
def test_class_label_is_reserved_for_multiclass(request, task):
    result = fit_task(request.getfixturevalue(f"{task}_df"), task, models="ridge" if task == "regression" else "logistic_regression")
    with pytest.raises(ConfigurationError, match="only used for multiclass"):
        result.ranking_table(class_label="other")
    with pytest.raises(ConfigurationError, match="only used for multiclass"):
        result.top_segment(class_label="other")


def test_numeric_multiclass_zero_label_is_explicitly_supported(multiclass_df):
    multiclass_df.target = multiclass_df.target.map({"bronze": 0, "gold": 1, "silver": 2})
    result = fit_task(multiclass_df, "multiclass", models="logistic_regression")
    table = result.test_predictions
    assert result.ranking_table(class_label=0).positives.sum() == table.y_true.eq(0).sum()
    expected = table.sort_values("proba_0", ascending=False, kind="stable").iloc[:int(np.ceil(len(table) * .1))]
    pd.testing.assert_frame_equal(result.top_segment(class_label=0), expected)
