"""Pure probability diagnostics from stored holdout predictions; never recalibrate."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_numeric_dtype

from ..exceptions import ConfigurationError, DataValidationError


def _values(
    predictions: pd.DataFrame, positive_class: object, score_column: str, n_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ConfigurationError("n_bins must be an integer >= 2, excluding bool.")
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise DataValidationError("Calibration requires a nonempty prediction DataFrame.")
    if not predictions.columns.is_unique:
        raise DataValidationError("Calibration prediction columns must be unique.")
    missing = [name for name in (score_column, "y_true") if name not in predictions.columns]
    if missing:
        raise DataValidationError(f"Calibration prediction columns are missing: {missing}.")
    scores = predictions[score_column]
    if not is_numeric_dtype(scores.dtype) or np.iscomplexobj(scores.to_numpy()):
        raise DataValidationError("Calibration probabilities must contain finite real numeric values.")
    probabilities = scores.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(probabilities).all():
        raise DataValidationError("Calibration probabilities must contain finite real numeric values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise DataValidationError("Calibration probabilities must be between 0 and 1 inclusive.")
    if predictions["y_true"].isna().any():
        raise DataValidationError("Calibration targets must contain no missing values.")
    try:
        positives = predictions["y_true"].eq(positive_class).to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise DataValidationError("Calibration targets must support comparison with the positive class.") from error
    order = np.argsort(probabilities, kind="stable")
    return probabilities[order], positives[order]


def _bin_table(probabilities: NDArray[np.float64], positives: NDArray[np.float64], n_bins: int) -> pd.DataFrame:
    n_rows = len(probabilities)
    # Positional arrays deliberately avoid alignment on arbitrary/duplicate indices.
    working = pd.DataFrame({
        "bin": np.arange(n_rows, dtype=np.int64) * min(n_bins, n_rows) // n_rows + 1,
        "probability": probabilities, "positive": positives,
    })
    table = working.groupby("bin", sort=True).agg(
        count=("probability", "size"), proba_min=("probability", "min"),
        proba_max=("probability", "max"), proba_mean=("probability", "mean"),
        observed_rate=("positive", "mean"),
    ).reset_index()
    table["calibration_gap"] = table["observed_rate"] - table["proba_mean"]
    table["absolute_calibration_gap"] = table["calibration_gap"].abs()
    table["population_share"] = table["count"] / n_rows
    return table


def calibration_table(
    predictions: pd.DataFrame, *, positive_class: object, n_bins: int = 10, score_column: str = "score",
) -> pd.DataFrame:
    """Balanced bins in ascending stable probability order; gap = observed - predicted.

    Bin 1 contains the lowest probabilities. Equal scores retain holdout order.
    Binary and multiclass one-vs-rest both compare original labels to positive_class.
    """
    probabilities, positives = _values(predictions, positive_class, score_column, n_bins)
    return _bin_table(probabilities, positives, n_bins)


def calibration_summary(
    predictions: pd.DataFrame, *, positive_class: object, n_bins: int = 10, score_column: str = "score",
) -> dict[str, float]:
    """Return Brier (probability quality), ECE and MCE from the same ascending bins.

    ECE and MCE depend on n_bins; Brier is independent of binning and is not a
    pure calibration measure. All calculations use stored predictions only.
    """
    probabilities, positives = _values(predictions, positive_class, score_column, n_bins)
    table = _bin_table(probabilities, positives, n_bins)
    gaps = table["absolute_calibration_gap"]
    return {
        "brier_score": float(np.mean((positives - probabilities) ** 2)),
        "expected_calibration_error": float((table["population_share"] * gaps).sum()),
        "max_calibration_error": float(gaps.max()),
    }
