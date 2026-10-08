"""Model metadata and estimator construction."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from optuna import Trial
from sklearn.base import BaseEstimator

from ..tasks import TaskType


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_class: type[BaseEstimator]
    tasks: frozenset[TaskType]
    search_space: Callable[[Trial], dict[str, Any]]
    default_params: dict[str, Any] = field(default_factory=dict)
    supports_proba: bool = True
    supports_early_stopping: bool = False
    n_jobs_parameter: str | None = "n_jobs"

    def __post_init__(self) -> None:
        if not self.name or not callable(self.estimator_class) or not callable(self.search_space):
            raise ValueError("A model needs a name, estimator class and search space.")
        if not self.tasks or any(not isinstance(task, TaskType) for task in self.tasks):
            raise ValueError("A model must declare supported TaskType values.")

    def parameters(self, suggested: dict[str, Any], random_state: int, n_jobs: int) -> dict[str, Any]:
        params = {**self.default_params, **suggested, "random_state": random_state}
        if self.n_jobs_parameter is not None:
            params[self.n_jobs_parameter] = n_jobs
        return params


def feature_importance(estimator: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    """Return raw importances or signed linear coefficients, never a Styler."""
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_[0], dtype=float)
    else:
        return pd.DataFrame({"importance": pd.Series(dtype=float)}).rename_axis("feature")
    result = pd.DataFrame({"importance": values}, index=pd.Index(feature_names, name="feature"))
    return result.loc[result["importance"].abs().sort_values(ascending=False, kind="stable").index]
