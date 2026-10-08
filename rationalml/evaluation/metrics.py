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

    def evaluate(self, estimator: Any, X: pd.DataFrame, y: pd.Series) -> float:
        predictions = (
            estimator.predict_proba(X)[:, 1]
            if self.prediction_type == "proba"
            else estimator.predict(X)
        )
        return self.score(y, predictions)


def evaluate_binary(estimator: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute all binary metrics using one prediction call of each type."""
    from .registry import MetricRegistry

    predictions = estimator.predict(X)
    probabilities = estimator.predict_proba(X)[:, 1]
    return {
        name: spec.score(y, probabilities if spec.prediction_type == "proba" else predictions)
        for name in MetricRegistry.available(TaskType.BINARY)
        for spec in [MetricRegistry.get(name)]
    }
