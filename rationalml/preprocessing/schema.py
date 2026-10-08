"""Raw feature contracts; training diagnostics are inferred on fold train only."""

from dataclasses import dataclass
from numbers import Real
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_datetime64_any_dtype, is_float_dtype, is_integer_dtype,
    is_object_dtype, is_scalar, is_string_dtype, is_timedelta64_dtype,
)

from ..exceptions import DataValidationError


def feature_kind(series: pd.Series) -> str:
    dtype = series.dtype
    if is_datetime64_any_dtype(dtype) or is_timedelta64_dtype(dtype):
        raise DataValidationError(f"Datetime/timedelta feature {series.name!r} is not supported in V0.2.")
    if is_bool_dtype(dtype):
        return "boolean"
    if is_integer_dtype(dtype) or is_float_dtype(dtype):
        return "numeric"
    if isinstance(dtype, pd.CategoricalDtype) or is_object_dtype(dtype) or is_string_dtype(dtype):
        return "categorical"
    raise DataValidationError(f"Unsupported dtype {dtype!s} for feature {series.name!r}.")


def categorical_family(series: pd.Series) -> str:
    """Reject nested and mixed scalar families before sklearn sees them."""
    families = set()
    for value in series:
        if not is_scalar(value):
            raise DataValidationError(f"Feature {series.name!r} contains unsupported nested values.")
        if pd.isna(value):
            continue
        if isinstance(value, str):
            families.add("string")
        elif isinstance(value, (Real, np.integer, np.floating, bool, np.bool_)):
            if not np.isfinite(value):
                raise DataValidationError(f"Feature {series.name!r} contains infinite values.")
            families.add("numeric")
        else:
            raise DataValidationError(f"Feature {series.name!r} contains unsupported categorical values.")
    if len(families) > 1:
        raise DataValidationError(f"Feature {series.name!r} mixes string and numeric categories.")
    return next(iter(families), "empty")


def validate_feature_structure(X: pd.DataFrame) -> None:
    """Check raw columns without interpreting the user's feature values."""
    if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
        raise DataValidationError("Features must be a nonempty DataFrame with unique column names.")
    if any(not isinstance(name, str) or not name.strip() for name in X.columns):
        raise DataValidationError("Feature names must be nonempty strings.")


def validate_raw_features(X: pd.DataFrame) -> None:
    """Validate basic preprocessing inputs, without learning statistics."""
    validate_feature_structure(X)
    for name in X.columns:
        column = X[name]
        if feature_kind(column) == "categorical":
            categorical_family(column)
        elif np.isinf(column.to_numpy(dtype=float, na_value=np.nan)).any():
            raise DataValidationError(f"Feature {name!r} contains infinite values.")


@dataclass(frozen=True)
class FeatureSchema:
    """Names preserve input order; groups preserve their relative input order."""

    feature_names: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    boolean: tuple[str, ...]
    constant_columns: tuple[str, ...] = ()
    categorical_families: tuple[tuple[str, str], ...] = ()

    def validate_prediction(self, X: pd.DataFrame) -> None:
        """Check compatible raw types, without inspecting cardinality or fitting."""
        validate_raw_features(X)
        families = dict(self.categorical_families)
        for name in self.feature_names:
            column = X[name]
            if column.isna().all():
                continue  # Fitted imputers handle an entirely missing inference batch.
            kind = feature_kind(column)
            if name in self.numeric and kind != "numeric":
                raise DataValidationError(f"Feature {name!r} must remain numeric at prediction.")
            if name in self.boolean and not (
                kind == "boolean" or (
                    kind in ("numeric", "categorical") and column.dropna().isin([0, 1]).all()
                )
            ):
                raise DataValidationError(f"Feature {name!r} must contain booleans or numeric 0/1 at prediction.")
            if name in self.categorical:
                family = categorical_family(column)
                if family != families[name]:
                    raise DataValidationError(f"Feature {name!r} has incompatible categorical types at prediction.")


def infer_schema(X: pd.DataFrame) -> FeatureSchema:
    """Infer from the supplied training rows only; retain and warn on constants.

    An entirely missing training column is an error, including inside a CV fold.
    This avoids silent removal and inventing statistics for unseen values.
    """
    validate_raw_features(X)
    empty = tuple(name for name in X.columns if X[name].isna().all())
    if empty:
        raise DataValidationError(f"Training features are entirely missing: {list(empty)}.")
    constants = tuple(name for name in X.columns if X[name].nunique(dropna=True) == 1)
    if constants:
        warnings.warn(
            f"Constant training features are retained by preprocessing: {list(constants)}.",
            UserWarning, stacklevel=2,
        )
    groups = {kind: [] for kind in ("numeric", "categorical", "boolean")}
    for name in X.columns:
        groups[feature_kind(X[name])].append(name)
    return FeatureSchema(
        feature_names=tuple(X.columns),
        numeric=tuple(groups["numeric"]), categorical=tuple(groups["categorical"]),
        boolean=tuple(groups["boolean"]), constant_columns=constants,
        categorical_families=tuple((name, categorical_family(X[name])) for name in groups["categorical"]),
    )
