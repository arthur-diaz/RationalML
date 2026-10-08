"""Raw Python results and prediction using the fitted winning estimator."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import AutoMLConfig
from .data import BinaryLabelEncoder, validate_features
from .exceptions import DataValidationError
from .tasks import TaskType


@dataclass
class AutoMLResult:
    task: TaskType
    target: str
    leaderboard: pd.DataFrame
    best_model: Any
    best_model_name: str
    best_params: dict[str, Any]
    metrics: dict[str, float]
    cv_results: pd.DataFrame
    test_metrics: dict[str, float]
    feature_importance: pd.DataFrame
    config: AutoMLConfig
    feature_names: tuple[str, ...]
    label_encoder: BinaryLabelEncoder
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    @property
    def positive_class(self) -> object:
        """Original target label encoded as internal class 1."""
        return self.label_encoder.positive_class

    @property
    def negative_class(self) -> object:
        """Original target label encoded as internal class 0."""
        return self.label_encoder.negative_class

    @property
    def classes_(self) -> NDArray[Any]:
        """Original labels in probability column order: [negative, positive]."""
        return self.label_encoder.classes_.copy()

    def _features(self, X: pd.DataFrame | NDArray[Any]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            if tuple(X.columns) != self.feature_names:
                raise DataValidationError("Prediction columns and their order must match the training features.")
            features = X.copy(deep=True)
        else:
            values = np.asarray(X)
            if values.ndim != 2 or values.shape[1] != len(self.feature_names):
                raise DataValidationError("Prediction input must be a 2D array with the training feature count.")
            features = pd.DataFrame(values.copy(), columns=self.feature_names)
        validate_features(features)
        return features

    def predict(self, X: pd.DataFrame | NDArray[Any]) -> NDArray[Any]:
        """Predict original target labels, reversing the internal binary encoding."""
        encoded = np.asarray(self.best_model.predict(self._features(X)))
        return self.label_encoder.inverse_transform(encoded)

    def predict_proba(self, X: pd.DataFrame | NDArray[Any]) -> NDArray[np.float64]:
        """Return [P(negative_class), P(positive_class)] for each row."""
        if not callable(getattr(self.best_model, "predict_proba", None)):
            raise TypeError(f"Model {self.best_model_name!r} does not support predict_proba.")
        return np.asarray(self.best_model.predict_proba(self._features(X)), dtype=float)

    def predict_positive_proba(self, X: pd.DataFrame | NDArray[Any]) -> NDArray[np.float64]:
        """Return a 1D vector of P(y = positive_class)."""
        return self.predict_proba(X)[:, 1]
