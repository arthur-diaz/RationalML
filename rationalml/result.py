"""Raw Python results and prediction using the fitted winning pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from .config import AutoMLConfig
from .data import BinaryLabelEncoder, MulticlassLabelEncoder, validate_features
from .exceptions import ConfigurationError, DataValidationError, UnsupportedTaskError
from .evaluation.calibration import calibration_summary, calibration_table
from .evaluation.ranking import binary_ranking_table, regression_ranking_table, top_segment
from .preprocessing.schema import FeatureSchema
from .reporting.config import ExcelReportConfig
from .tasks import TaskType
from .tracking.config import MLflowConfig


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
    label_encoder: BinaryLabelEncoder | MulticlassLabelEncoder | None
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    baseline_name: str
    baseline_score: float
    baseline_cv_std: float
    baseline_fold_scores: tuple[float, ...]
    _test_predictions: pd.DataFrame = field(repr=False)
    feature_schema: FeatureSchema | None = None
    transformed_feature_names: tuple[str, ...] | None = None
    model_best_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._test_predictions = self._test_predictions.copy(deep=True)

    @property
    def test_predictions(self) -> pd.DataFrame:
        """Return a copy of initial holdout predictions, with original row index."""
        return self._test_predictions.copy(deep=True)

    def _ranking_target(self, class_label: object | None) -> tuple[str, object | None]:
        if self.task is TaskType.MULTICLASS:
            if class_label is None:
                raise ConfigurationError("Multiclass ranking requires class_label.")
            labels = self.classes_.tolist()
            try:
                position = labels.index(class_label)
            except (ValueError, TypeError) as error:
                raise ConfigurationError(f"Unknown class_label={class_label!r}; available classes: {labels!r}.") from error
            return f"proba_{position}", labels[position]
        if class_label is not None:
            raise ConfigurationError("class_label is only used for multiclass ranking.")
        return ("score", self.positive_class) if self.task is TaskType.BINARY else ("y_pred", None)

    def ranking_table(self, n_bins: int = 10, *, class_label: object | None = None) -> pd.DataFrame:
        """Segment stored holdout predictions; multiclass uses explicit one-vs-rest."""
        score_column, positive = self._ranking_target(class_label)
        if self.task is TaskType.REGRESSION:
            return regression_ranking_table(self._test_predictions, n_bins=n_bins)
        return binary_ranking_table(self._test_predictions, positive_class=positive, n_bins=n_bins, score_column=score_column)

    def top_segment(self, fraction: float = .10, *, class_label: object | None = None) -> pd.DataFrame:
        """Select the highest stored holdout scores/predictions without model calls."""
        score_column, _ = self._ranking_target(class_label)
        return top_segment(self._test_predictions, fraction=fraction, score_column=score_column)

    def _calibration_target(self, class_label: object | None) -> tuple[str, object]:
        if self.task is TaskType.REGRESSION:
            raise UnsupportedTaskError("Probability calibration diagnostics are only available for classification tasks.")
        if self.task is TaskType.MULTICLASS and class_label is None:
            raise ConfigurationError("Multiclass calibration requires class_label.")
        if self.task is TaskType.BINARY and class_label is not None:
            raise ConfigurationError("class_label is only used for multiclass calibration.")
        return self._ranking_target(class_label)

    def calibration_table(self, n_bins: int = 10, *, class_label: object | None = None) -> pd.DataFrame:
        """Diagnose stored holdout probabilities; bin 1 is lowest, gap = observed - predicted.

        Multiclass requires class_label (one-vs-rest); regression is unsupported.
        No model calls or probability changes are performed.
        """
        score_column, positive = self._calibration_target(class_label)
        return calibration_table(self._test_predictions, positive_class=positive, n_bins=n_bins, score_column=score_column)

    def calibration_summary(self, n_bins: int = 10, *, class_label: object | None = None) -> dict[str, float]:
        """Stored-holdout Brier, ECE and MCE; ECE/MCE depend on the selected n_bins.

        Brier measures probability quality, not calibration alone. Multiclass
        requires class_label; regression is unsupported. Never recalibrates.
        """
        score_column, positive = self._calibration_target(class_label)
        return calibration_summary(self._test_predictions, positive_class=positive, n_bins=n_bins, score_column=score_column)

    def to_excel(
        self, path: str | Path, *, config: ExcelReportConfig | None = None,
        class_label: object | None = None,
    ) -> Path:
        """Export stored results to .xlsx; overwrite files, require an existing parent."""
        from .reporting.excel import export_excel

        return export_excel(self, path, config=config, class_label=class_label)

    def log_mlflow(
        self, *, config: MLflowConfig | None = None, class_label: object | None = None,
    ) -> str:
        """Track this stored result in a new RationalML run and return its run_id."""
        from .tracking.mlflow import log_result

        return log_result(self, config=config, class_label=class_label)

    @property
    def positive_class(self) -> object:
        """Original target label encoded as internal class 1."""
        self._require_binary()
        return self.label_encoder.positive_class

    @property
    def negative_class(self) -> object:
        """Original target label encoded as internal class 0."""
        self._require_binary()
        return self.label_encoder.negative_class

    @property
    def classes_(self) -> NDArray[Any]:
        """Original classification labels, ordered exactly as probability columns."""
        if self.task is TaskType.REGRESSION:
            raise UnsupportedTaskError("classes_ is only available for classification tasks.")
        return self.label_encoder.classes_.copy()

    @property
    def primary_metric(self) -> str:
        """Effective metric used for CV selection, after resolving metric='auto'."""
        return self.config.metric

    def _require_binary(self) -> None:
        if self.task is not TaskType.BINARY:
            raise UnsupportedTaskError("positive_class, negative_class and predict_positive_proba are only available for binary classification.")

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
        """Predict original classification labels or numeric regression values."""
        encoded = np.asarray(self.best_model.predict(self._features(X)))
        return encoded if self.task is TaskType.REGRESSION else self.label_encoder.inverse_transform(encoded)

    def predict_proba(self, X: pd.DataFrame | NDArray[Any]) -> NDArray[np.float64]:
        """Return P(y=classes_[j]) in column j; binary order is [negative, positive]."""
        if self.task is TaskType.REGRESSION:
            raise UnsupportedTaskError("predict_proba is only available for classification tasks.")
        if not callable(getattr(self.best_model, "predict_proba", None)):
            raise TypeError(f"Model {self.best_model_name!r} does not support predict_proba.")
        return np.asarray(self.best_model.predict_proba(self._features(X)), dtype=float)

    def predict_positive_proba(self, X: pd.DataFrame | NDArray[Any]) -> NDArray[np.float64]:
        """Return a 1D vector of P(y = positive_class)."""
        self._require_binary()
        return self.predict_proba(X)[:, 1]
