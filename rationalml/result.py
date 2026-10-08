"""Raw Python results and prediction using the fitted winning pipeline."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from .config import AutoMLConfig
from .data import BinaryLabelEncoder, validate_features
from .exceptions import DataValidationError
from .preprocessing.schema import FeatureSchema
from .tasks import TaskType


@dataclass
class AutoMLResult:
    task: TaskType
    target: str
    leaderboard: pd.DataFrame
    best_model: Pipeline
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
    feature_schema: FeatureSchema | None = None
    transformed_feature_names: tuple[str, ...] | None = None

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
            if not X.columns.is_unique:
                raise DataValidationError("Prediction columns must have unique names.")
            missing = [name for name in self.feature_names if name not in X.columns]
            extra = [name for name in X.columns if name not in self.feature_names]
            if missing or extra:
                raise DataValidationError(f"Prediction columns differ from training: missing={missing}, extra={extra}.")
            features = X.loc[:, list(self.feature_names)].copy(deep=True)
        else:
            if self.feature_schema is not None and self.feature_schema.categorical:
                raise DataValidationError("Categorical prediction features require a pandas DataFrame with named columns.")
            values = np.asarray(X)
            if values.ndim != 2 or values.shape[1] != len(self.feature_names):
                raise DataValidationError("Prediction input must be a 2D array with the training feature count.")
            features = pd.DataFrame(values.copy(), columns=self.feature_names)
            features = features.infer_objects()
        if self.feature_schema is not None:
            self.feature_schema.validate_prediction(features)
        else:
            validate_features(features, preprocessing=self.config.preprocessing)
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
