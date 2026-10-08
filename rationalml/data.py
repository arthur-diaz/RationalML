"""Input validation and train-fitted classification labels, without mutation."""

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import is_bool_dtype, is_float_dtype, is_integer_dtype
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from .exceptions import DataValidationError
from .preprocessing.config import PreprocessingConfig
from .preprocessing.schema import validate_feature_structure, validate_raw_features


@dataclass(frozen=True)
class BinaryLabelEncoder:
    """Map negative/positive labels to 0/1 without relying on LabelEncoder."""

    negative_class: object
    positive_class: object

    @classmethod
    def from_target(cls, y: pd.Series, positive_class: object | None = None) -> "BinaryLabelEncoder":
        """Choose labels from train only; warn about nonstandard defaults."""
        labels = y.drop_duplicates().tolist()
        if y.isna().any() or len(labels) != 2:
            raise DataValidationError("Binary encoding requires exactly two non-missing target classes.")
        if positive_class is None:
            if all(isinstance(label, (bool, np.bool_)) for label in labels):
                positive_class = True
            elif set(labels) == {0, 1}:
                positive_class = 1
            else:
                try:
                    positive_class = sorted(labels)[-1]
                except TypeError as error:
                    raise DataValidationError(
                        "Cannot choose a deterministic positive class for these labels; set positive_class explicitly."
                    ) from error
                warnings.warn(
                    f"positive_class=None: selected {positive_class!r} as the positive class. "
                    "Set positive_class explicitly to confirm the intended positive label.",
                    UserWarning, stacklevel=3,
                )
        try:
            matches = [label == positive_class for label in labels]
            positive_index = matches.index(True)
        except (TypeError, ValueError) as error:
            raise DataValidationError(
                f"positive_class={positive_class!r} must be one of the target classes {labels!r}."
            ) from error
        # Keep the original label value/type, including False and zero.
        return cls(negative_class=labels[1 - positive_index], positive_class=labels[positive_index])

    @property
    def classes_(self) -> NDArray[Any]:
        """Original labels in the fixed order [negative_class, positive_class]."""
        return pd.Series([self.negative_class, self.positive_class]).to_numpy(copy=True)

    def transform(self, y: pd.Series) -> NDArray[np.int64]:
        """Encode only known labels; class 1 always means positive_class."""
        if not y.isin([self.negative_class, self.positive_class]).all():
            raise DataValidationError("The target contains labels outside the fitted binary classes.")
        return y.eq(self.positive_class).to_numpy(dtype=np.int64)

    def inverse_transform(self, y: NDArray[Any]) -> NDArray[Any]:
        """Restore original labels from a one-dimensional 0/1 vector."""
        encoded = np.asarray(y)
        if encoded.ndim != 1 or not np.isin(encoded, [0, 1]).all():
            raise DataValidationError("Binary predictions must be a one-dimensional 0/1 vector.")
        return self.classes_[encoded.astype(np.int64)]


class MulticlassLabelEncoder:
    """Encode multiclass labels using a LabelEncoder fitted on train only."""

    def __init__(self, encoder: LabelEncoder) -> None:
        self._encoder = encoder

    @classmethod
    def from_target(cls, y: pd.Series) -> "MulticlassLabelEncoder":
        if y.isna().any() or y.nunique() < 3:
            raise DataValidationError("Multiclass encoding requires at least three non-missing target classes.")
        try:
            return cls(LabelEncoder().fit(y))
        except (TypeError, ValueError) as error:
            raise DataValidationError("Multiclass labels must be uniformly numeric or strings.") from error

    @property
    def classes_(self) -> NDArray[Any]:
        return self._encoder.classes_.copy()

    def transform(self, y: pd.Series) -> NDArray[np.int64]:
        if not y.isin(self.classes_).all():
            raise DataValidationError("The target contains labels outside the fitted multiclass classes.")
        return self._encoder.transform(y)

    def inverse_transform(self, y: NDArray[Any]) -> NDArray[Any]:
        try:
            return self._encoder.inverse_transform(y)
        except (ValueError, TypeError) as error:
            raise DataValidationError("Multiclass predictions must contain fitted internal class indices.") from error


def validate_features(
    X: pd.DataFrame, *, preprocessing: str | PreprocessingConfig | BaseEstimator | None = None,
) -> None:
    """Validate prepared data, basic inputs or just custom-transformer columns."""
    validate_feature_structure(X)
    if isinstance(preprocessing, PreprocessingConfig) or (
        isinstance(preprocessing, str) and preprocessing == "basic"
    ):
        validate_raw_features(X)
        return
    if preprocessing is not None:
        return
    guidance = (
        "Prepare the data before RationalML, use preprocessing='basic', "
        "or supply a custom sklearn transformer."
    )
    invalid = [name for name, dtype in X.dtypes.items() if not (
        is_bool_dtype(dtype) or is_integer_dtype(dtype) or is_float_dtype(dtype)
    )]
    if invalid:
        raise DataValidationError(
            f"preprocessing=None requires numeric or boolean features; incompatible columns: {invalid}. {guidance}"
        )
    if X.isna().to_numpy().any():
        raise DataValidationError(f"preprocessing=None does not accept missing values. {guidance}")
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise DataValidationError("preprocessing=None does not accept infinite values; prepare the data before RationalML.")


def validate_dataframe(
    df: pd.DataFrame, target: str, *, preprocessing: str | PreprocessingConfig | BaseEstimator | None = None,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("fit expects a pandas DataFrame.")
    if df.empty or not df.columns.is_unique:
        raise DataValidationError("The DataFrame must be nonempty and have unique columns.")
    if target not in df.columns:
        raise DataValidationError(f"Target column {target!r} is missing.")
    validate_features(df.drop(columns=[target]), preprocessing=preprocessing)
