import numpy as np
import pandas as pd
import pytest

from rationalml.evaluation.ranking import binary_ranking_table, regression_ranking_table, top_segment
from rationalml.exceptions import ConfigurationError, DataValidationError


@pytest.fixture
def binary_predictions():
    return pd.DataFrame({
        "y_true": ["churn", "retained", "churn", "churn", "retained", "retained", "churn", "retained"],
        "y_pred": ["churn"] * 5 + ["retained"] * 3,
        "score": [.95, .90, .80, .70, .60, .40, .30, .10],
    }, index=pd.Index([42, 17, 100, 29, 8, 60, 51, 73], name="customer_id"))


def test_binary_ranking_exact_counts_capture_and_lift(binary_predictions):
    original = binary_predictions.copy(deep=True)
    expected = pd.DataFrame({
        "segment": [1, 2, 3, 4], "count": [2, 2, 2, 2],
        "score_min": [.90, .70, .40, .10], "score_max": [.95, .80, .60, .30],
        "score_mean": [.925, .75, .50, .20], "positives": [1, 2, 0, 1],
        "positive_rate": [.5, 1., 0., .5], "population_share": [.25] * 4,
        "positive_capture": [.25, .5, 0., .25], "cumulative_positive_capture": [.25, .75, .75, 1.],
        "lift": [1., 2., 0., 1.], "cumulative_lift": [1., 1.5, 1., 1.],
    })
    table = binary_ranking_table(binary_predictions, positive_class="churn", n_bins=4)
    pd.testing.assert_frame_equal(table, expected)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in table.dtypes)
    pd.testing.assert_frame_equal(binary_predictions, original)


def test_regression_ranking_exact_signed_bias_and_errors():
    predictions = pd.DataFrame({"y_true": [9., 11., 3., 5.], "y_pred": [10., 8., 6., 4.]},
                               index=pd.Index(["a", "a", "b", "b"], name="customer"))
    original = predictions.copy(deep=True)
    expected = pd.DataFrame({
        "segment": [1, 2], "count": [2, 2], "pred_min": [8., 4.], "pred_max": [10., 6.],
        "pred_mean": [9., 5.], "actual_mean": [10., 4.], "bias": [-1., 1.],
        "mae": [2., 2.], "rmse": [np.sqrt(5)] * 2,
    })
    pd.testing.assert_frame_equal(regression_ranking_table(predictions, n_bins=2), expected)
    pd.testing.assert_frame_equal(predictions, original)


@pytest.mark.parametrize("n_rows,n_bins", [(1, 10), (3, 10), (11, 5), (11, 10), (20, 5), (20, 10), (23, 20)])
@pytest.mark.parametrize("task", ["binary", "regression"])
def test_equal_scores_have_balanced_nonempty_deterministic_segments(n_rows, n_bins, task, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Equal scores must not require qcut")

    monkeypatch.setattr(pd, "qcut", unexpected)
    predictions = pd.DataFrame({"y_true": np.arange(n_rows) % 2, "y_pred": np.ones(n_rows), "score": .5},
                               index=pd.Index(["duplicate"] * n_rows, name="id"))
    if n_rows == 1:
        predictions.y_true = 1
    ranking = (binary_ranking_table(predictions, positive_class=1, n_bins=n_bins) if task == "binary"
               else regression_ranking_table(predictions, n_bins=n_bins))
    repeated = (binary_ranking_table(predictions, positive_class=1, n_bins=n_bins) if task == "binary"
                else regression_ranking_table(predictions, n_bins=n_bins))
    pd.testing.assert_frame_equal(ranking, repeated)
    assert list(ranking.segment) == list(range(1, min(n_rows, n_bins) + 1))
    assert ranking["count"].sum() == n_rows
    assert ranking["count"].min() >= 1
    assert ranking["count"].max() - ranking["count"].min() <= 1
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in ranking.dtypes)
    if task == "binary":
        assert ranking.population_share.sum() == pytest.approx(1.)
        assert ranking.positive_capture.sum() == pytest.approx(1.)
        assert ranking.cumulative_positive_capture.iloc[-1] == 1.
        assert ranking.cumulative_lift.iloc[-1] == pytest.approx(1.)


@pytest.mark.parametrize("task", ["binary", "regression"])
def test_segments_sort_predictions_descending(task):
    predictions = pd.DataFrame({"y_true": [0, 1, 1, 0], "y_pred": [2., 4., 1., 3.], "score": [.2, .9, .1, .8]})
    if task == "binary":
        ranking = binary_ranking_table(predictions, positive_class=1, n_bins=2)
        assert list(ranking.score_min) == [.8, .1]
        assert list(ranking.score_max) == [.9, .2]
    else:
        ranking = regression_ranking_table(predictions, n_bins=2)
        assert list(ranking.pred_min) == [3., 1.]
        assert list(ranking.pred_max) == [4., 2.]


@pytest.mark.parametrize("n_bins", [True, False, 0, 1, -1, 2., 2.5, "10", None, np.bool_(True), np.nan, np.inf])
@pytest.mark.parametrize("task", ["binary", "regression"])
def test_invalid_bin_counts_are_explicit(binary_predictions, n_bins, task):
    with pytest.raises(ConfigurationError, match="n_bins must be an integer >= 2"):
        if task == "binary":
            binary_ranking_table(binary_predictions, positive_class="churn", n_bins=n_bins)
        else:
            regression_ranking_table(binary_predictions, n_bins=n_bins)


@pytest.mark.parametrize("fraction,count", [(.01, 1), (.10, 1), (.25, 2), (.26, 3), (1., 8)])
def test_top_segment_uses_exact_ceil_size_and_preserves_index(binary_predictions, fraction, count):
    original = binary_predictions.copy(deep=True)
    top = top_segment(binary_predictions, fraction=fraction)
    pd.testing.assert_frame_equal(top, binary_predictions.iloc[:count])
    assert top.index.name == "customer_id"
    top.iloc[0, top.columns.get_loc("score")] = 0
    pd.testing.assert_frame_equal(binary_predictions, original)


def test_top_segment_ties_keep_original_order_without_expanding_boundary():
    predictions = pd.DataFrame({"score": [.5, .9, .5, .5, .9], "y_true": range(5)},
                               index=pd.Index(["z", "b", "z", "a", "c"], name="id"))
    top = top_segment(predictions, fraction=.6)
    pd.testing.assert_frame_equal(top, predictions.iloc[[1, 4, 0]])
    assert len(top) == 3


@pytest.mark.parametrize("fraction", [True, False, 0, -.1, 1.1, "0.1", None, np.nan, np.inf, -np.inf, 1j])
def test_invalid_top_fraction_is_explicit(binary_predictions, fraction):
    with pytest.raises(ConfigurationError, match="fraction.*0 < fraction <= 1"):
        top_segment(binary_predictions, fraction=fraction)


def test_zero_positive_count_is_an_error(binary_predictions):
    binary_predictions.y_true = "retained"
    with pytest.raises(DataValidationError, match="at least one positive"):
        binary_ranking_table(binary_predictions, positive_class="churn")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "0.5", 1j, -.1, 1.1])
def test_classification_ranking_rejects_invalid_probabilities(value):
    predictions = pd.DataFrame({"score": [value, .5], "y_true": [1, 0]})
    with pytest.raises(DataValidationError, match="numeric|between 0 and 1"):
        binary_ranking_table(predictions, positive_class=1)


@pytest.mark.parametrize("corruption", ["empty", "missing", "duplicate", "missing_target"])
def test_invalid_ranking_input_is_explicit(binary_predictions, corruption):
    if corruption == "empty":
        table = binary_predictions.iloc[:0]
    elif corruption == "missing":
        table = binary_predictions.drop(columns="score")
    elif corruption == "duplicate":
        table = pd.concat([binary_predictions, binary_predictions[["score"]]], axis=1)
    else:
        table = binary_predictions.copy()
        table.iloc[0, 0] = None
    with pytest.raises(DataValidationError):
        binary_ranking_table(table, positive_class="churn")


@pytest.mark.parametrize("column", ["y_true", "y_pred"])
def test_regression_ranking_requires_finite_numeric_values(column):
    predictions = pd.DataFrame({"y_true": [1., 2.], "y_pred": [2., 1.]})
    predictions.loc[0, column] = np.inf
    with pytest.raises(DataValidationError, match="finite real numeric"):
        regression_ranking_table(predictions)
