"""Metric metadata and evaluation, independent of orchestration."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..tasks import TaskType


@dataclass(frozen=True)
class MetricSpec:
    name: str
    scorer: Callable[..., float]
    direction: Literal["maximize", "minimize"]
    prediction_type: Literal["predict", "proba"]
    tasks: frozenset[TaskType]

    def __post_init__(self) -> None:
        if not self.name or not callable(self.scorer):
            raise ValueError("A metric needs a name and a callable scorer.")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("Metric direction must be maximize or minimize.")
        if self.prediction_type not in {"predict", "proba"}:
            raise ValueError("Metric prediction_type must be predict or proba.")
        if not self.tasks or any(not isinstance(task, TaskType) for task in self.tasks):
            raise ValueError("A metric must declare supported TaskType values.")

    def score(self, y: pd.Series, predictions: NDArray[Any]) -> float:
        value = float(self.scorer(y, predictions))
        if not np.isfinite(value):
            raise ValueError(f"Metric {self.name!r} returned a non-finite value.")
        return value

    def evaluate(
        self, estimator: Any, X: pd.DataFrame, y: pd.Series, task: TaskType = TaskType.BINARY,
    ) -> float:
        predictions = getattr(estimator, "predict_proba" if self.prediction_type == "proba" else "predict")(X)
        if self.prediction_type == "proba" and task is TaskType.BINARY:
            predictions = predictions[:, 1]
        return self.score(y, predictions)


def evaluate_metrics(
    estimator: Any, X: pd.DataFrame, y: pd.Series, task: TaskType,
    *, predictions: dict[str, NDArray[Any]] | None = None,
) -> dict[str, float]:
    """Compute metrics using full cached predictions, or one call of each type."""
    from .registry import MetricRegistry

    specs = [MetricRegistry.get(name) for name in MetricRegistry.available(task)]
    if predictions is None:
        predictions = {}
        for prediction_type in ("predict", "proba"):
            if not any(spec.prediction_type == prediction_type for spec in specs):
                continue
            predictions[prediction_type] = np.asarray(
                getattr(estimator, "predict_proba" if prediction_type == "proba" else "predict")(X)
            )
    return {
        spec.name: spec.score(y, predictions["proba"][:, 1] if spec.prediction_type == "proba"
                             and task is TaskType.BINARY else predictions[spec.prediction_type])
        for spec in specs
    }


def evaluate_binary(estimator: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Retain the binary evaluation entry point and positive-class contract."""
    return evaluate_metrics(estimator, X, y, TaskType.BINARY)
