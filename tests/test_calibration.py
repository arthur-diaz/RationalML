import numpy as np
import pandas as pd
import pytest

from rationalml.evaluation.calibration import calibration_summary, calibration_table
from rationalml.exceptions import ConfigurationError, DataValidationError


FUNCTIONS = [calibration_table, calibration_summary]
COLUMNS = ["bin", "count", "proba_min", "proba_max", "proba_mean", "observed_rate",
           "calibration_gap", "absolute_calibration_gap", "population_share"]


def predictions():
    return pd.DataFrame({"y_true": [0, 0, 1, 1], "score": [.10, .20, .70, .80]})


def test_manual_table_and_summary():
    artifact = predictions()
    expected = pd.DataFrame({
        "bin": [1, 2], "count": [2, 2], "proba_min": [.1, .7], "proba_max": [.2, .8],
        "proba_mean": [.15, .75], "observed_rate": [0., 1.], "calibration_gap": [-.15, .25],
        "absolute_calibration_gap": [.15, .25], "population_share": [.5, .5],
    })
    pd.testing.assert_frame_equal(calibration_table(artifact, positive_class=1, n_bins=2), expected)
    assert calibration_summary(artifact, positive_class=1, n_bins=2) == pytest.approx({
        "brier_score": (.01 + .04 + .09 + .04) / 4,
        "expected_calibration_error": .20, "max_calibration_error": .25,
    })


@pytest.mark.parametrize("index", [pd.Index([9, 9, 2, 2]), pd.Index(["b", "a", "b", "a"]),
    pd.date_range("2025-01-01", periods=4), pd.DatetimeIndex(["2025-01-01"] * 4),
    pd.MultiIndex.from_tuples([("a", 1), ("a", 1), ("b", 2), ("b", 2)])])
def test_indices_are_positional_and_input_is_immutable(index):
    artifact = predictions().iloc[[3, 0, 2, 1]].copy()
    artifact.index = index
    original = artifact.copy(deep=True)
    expected = calibration_table(predictions(), positive_class=1, n_bins=2)
    table = calibration_table(artifact, positive_class=1, n_bins=2)
    pd.testing.assert_frame_equal(table, expected)
    assert list(table.columns) == COLUMNS
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in table.dtypes)
    table.iloc[0, 1] = 999
    summary = calibration_summary(artifact, positive_class=1, n_bins=2)
    summary["brier_score"] = 999.
    pd.testing.assert_frame_equal(artifact, original)
    pd.testing.assert_frame_equal(calibration_table(artifact, positive_class=1, n_bins=2), expected)
    assert calibration_summary(artifact, positive_class=1, n_bins=2)["brier_score"] == pytest.approx(.045)


def test_identical_probabilities_keep_holdout_order_deterministically():
    artifact = pd.DataFrame({"y_true": [0, 0, 1, 1, 0, 1, 0], "score": [.4] * 7}, index=[5] * 7)
    table = calibration_table(artifact, positive_class=1, n_bins=3)
    np.testing.assert_array_equal(table["count"], [3, 2, 2])
    np.testing.assert_allclose(table["observed_rate"], [1 / 3, 1 / 2, 1 / 2])
    np.testing.assert_allclose(table[["proba_min", "proba_max", "proba_mean"]], .4)
    pd.testing.assert_frame_equal(table, calibration_table(artifact, positive_class=1, n_bins=3))


@pytest.mark.parametrize("n_rows,n_bins,counts", [(7, 3, [3, 2, 2]), (5, 2, [3, 2]),
                                                (3, 10, [1, 1, 1]), (1, 10, [1])])
def test_balanced_nonempty_bins_and_inclusive_probability_bounds(n_rows, n_bins, counts):
    artifact = pd.DataFrame({"y_true": np.arange(n_rows) % 2, "score": np.linspace(1, 0, n_rows)})
    table = calibration_table(artifact, positive_class=1, n_bins=n_bins)
    np.testing.assert_array_equal(table["count"], counts)
    np.testing.assert_array_equal(table["bin"], np.arange(1, len(counts) + 1))
    assert table["count"].sum() == n_rows and table["population_share"].sum() == pytest.approx(1.)
    assert table["proba_mean"].is_monotonic_increasing


@pytest.mark.parametrize("positive", [0, 1])
def test_all_targets_same_is_a_defined_probability_diagnostic(positive):
    artifact = pd.DataFrame({"y_true": [positive] * 4, "score": [0., .2, .7, 1.]})
    table = calibration_table(artifact, positive_class=1)
    assert (table["observed_rate"] == positive).all()
    assert calibration_summary(artifact, positive_class=1)["brier_score"] == pytest.approx(
        np.mean((positive - artifact.score.to_numpy()) ** 2))


@pytest.mark.parametrize("n_bins", [2, 3, 4, 10, np.int64(3)])
def test_summary_uses_exact_table_bins_and_brier_is_bin_independent(n_bins):
    artifact = pd.DataFrame({"y_true": [0, 1, 0, 1], "score": [.1, .2, .7, .8]})
    table = calibration_table(artifact, positive_class=1, n_bins=n_bins)
    summary = calibration_summary(artifact, positive_class=1, n_bins=n_bins)
    assert summary == pytest.approx({
        "brier_score": .295,
        "expected_calibration_error": sum(table.population_share * table.absolute_calibration_gap),
        "max_calibration_error": max(table.absolute_calibration_gap),
    })
    assert all(type(value) is float and 0 <= value <= 1 for value in summary.values())
    coarse = calibration_summary(artifact, positive_class=1, n_bins=2)
    fine = calibration_summary(artifact, positive_class=1, n_bins=4)
    assert coarse["expected_calibration_error"] != fine["expected_calibration_error"]
    assert coarse["max_calibration_error"] != fine["max_calibration_error"]


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("n_bins", [True, False, np.bool_(True), 0, 1, -2, 2., "2", None])
def test_invalid_bin_count(function, n_bins):
    with pytest.raises(ConfigurationError, match="n_bins.*integer.*2.*bool"):
        function(predictions(), positive_class=1, n_bins=n_bins)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("scores", [[-.001, .2, .7, .8], [.1, .2, .7, 1.001],
    [np.nan, .2, .7, .8], [np.inf, .2, .7, .8], [-np.inf, .2, .7, .8],
    [".1", ".2", ".7", ".8"], [.1 + 0j, .2, .7, .8],
    pd.Series([pd.NA, .2, .7, .8], dtype="Float64"), pd.Series([.1, .2, .7, .8], dtype=object)])
def test_invalid_probabilities_are_rejected_without_clipping(function, scores):
    artifact = predictions().assign(score=scores)
    original = artifact.copy(deep=True)
    with pytest.raises(DataValidationError, match="Calibration probabilities"):
        function(artifact, positive_class=1)
    pd.testing.assert_frame_equal(artifact, original)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("artifact", [None, [], pd.DataFrame(), pd.DataFrame({"y_true": [0]}),
    pd.DataFrame({"score": [.2]}), pd.DataFrame([[0, .2, .3]], columns=["y_true", "score", "score"]),
    pd.DataFrame({"y_true": [None], "score": [.2]})])
def test_invalid_prediction_artifact_is_explicit(function, artifact):
    with pytest.raises(DataValidationError, match="Calibration"):
        function(artifact, positive_class=1)


def test_nullable_numeric_probability_dtype():
    artifact = predictions().astype({"score": "Float64", "y_true": "Int64"})
    assert calibration_summary(artifact, positive_class=1, n_bins=2)["brier_score"] == pytest.approx(.045)
