"""Pure numerical ranking diagnostics from stored holdout predictions."""

from math import ceil, isfinite
from numbers import Real

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_numeric_dtype

from ..exceptions import ConfigurationError, DataValidationError


def _validate_bins(n_bins: int) -> None:
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ConfigurationError("n_bins must be an integer >= 2, excluding bool.")


def _numeric_values(table: pd.DataFrame, column: str) -> NDArray[np.float64]:
    if not is_numeric_dtype(table[column].dtype) or np.iscomplexobj(table[column].to_numpy()):
        raise DataValidationError(f"Ranking column {column!r} must contain finite real numeric values.")
    values = table[column].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise DataValidationError(f"Ranking column {column!r} must contain finite real numeric values.")
    return values


def _sorted_predictions(table: pd.DataFrame, score_column: str, required: tuple[str, ...] = ()) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise DataValidationError("Ranking requires a nonempty prediction DataFrame.")
    if not table.columns.is_unique:
        raise DataValidationError("Ranking prediction columns must be unique.")
    missing = [name for name in (score_column, *required) if name not in table.columns]
    if missing:
        raise DataValidationError(f"Ranking prediction columns are missing: {missing}.")
    _numeric_values(table, score_column)
    if required and table[list(required)].isna().to_numpy().any():
        raise DataValidationError("Ranking targets must contain no missing values.")
    return table.sort_values(score_column, ascending=False, kind="stable").copy(deep=True)


def _segments(n_rows: int, n_bins: int) -> NDArray[np.int64]:
    return np.arange(n_rows, dtype=np.int64) * min(n_bins, n_rows) // n_rows + 1


def binary_ranking_table(
    predictions: pd.DataFrame, *, positive_class: object, n_bins: int = 10, score_column: str = "score",
) -> pd.DataFrame:
    """Binary or one-vs-rest capture/lift, ordered by descending probability."""
    _validate_bins(n_bins)
    ordered = _sorted_predictions(predictions, score_column, ("y_true",))
    scores = _numeric_values(ordered, score_column)
    if ((scores < 0) | (scores > 1)).any():
        raise DataValidationError("Classification ranking probabilities must be between 0 and 1.")
    positives = ordered["y_true"].eq(positive_class).to_numpy(dtype=np.int64)
    total_positive = int(positives.sum())
    if total_positive == 0:
        raise DataValidationError(f"Ranking requires at least one positive observation for class {positive_class!r}.")
    working = pd.DataFrame({"segment": _segments(len(ordered), n_bins), "score": scores, "positive": positives})
    table = working.groupby("segment", sort=True).agg(
        count=("score", "size"), score_min=("score", "min"), score_max=("score", "max"),
        score_mean=("score", "mean"), positives=("positive", "sum"),
    ).reset_index()
    global_rate = total_positive / len(ordered)
    table["positive_rate"] = table["positives"] / table["count"]
    table["population_share"] = table["count"] / len(ordered)
    table["positive_capture"] = table["positives"] / total_positive
    table["cumulative_positive_capture"] = table["positives"].cumsum() / total_positive
    table["lift"] = table["positive_rate"] / global_rate
    table["cumulative_lift"] = (table["positives"].cumsum() / table["count"].cumsum()) / global_rate
    return table


def regression_ranking_table(predictions: pd.DataFrame, *, n_bins: int = 10) -> pd.DataFrame:
    """Prediction-level segments; bias is predicted mean minus actual mean."""
    _validate_bins(n_bins)
    ordered = _sorted_predictions(predictions, "y_pred", ("y_true",))
    predicted, actual = _numeric_values(ordered, "y_pred"), _numeric_values(ordered, "y_true")
    errors = actual - predicted
    working = pd.DataFrame({
        "segment": _segments(len(ordered), n_bins), "predicted": predicted, "actual": actual,
        "absolute_error": np.abs(errors), "squared_error": errors ** 2,
    })
    table = working.groupby("segment", sort=True).agg(
        count=("predicted", "size"), pred_min=("predicted", "min"), pred_max=("predicted", "max"),
        pred_mean=("predicted", "mean"), actual_mean=("actual", "mean"),
        mae=("absolute_error", "mean"), squared_error=("squared_error", "mean"),
    ).reset_index()
    table["bias"] = table["pred_mean"] - table["actual_mean"]
    table["rmse"] = np.sqrt(table.pop("squared_error"))
    return table[["segment", "count", "pred_min", "pred_max", "pred_mean", "actual_mean", "bias", "mae", "rmse"]]


def top_segment(predictions: pd.DataFrame, *, fraction: float = .10, score_column: str = "score") -> pd.DataFrame:
    """Return exactly ceil(n*fraction) rows, preserving order within score ties."""
    if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real)
            or not isfinite(fraction) or not 0 < fraction <= 1):
        raise ConfigurationError("fraction must be a finite number satisfying 0 < fraction <= 1.")
    ordered = _sorted_predictions(predictions, score_column)
    return ordered.iloc[:max(1, ceil(len(ordered) * fraction))].copy(deep=True)
