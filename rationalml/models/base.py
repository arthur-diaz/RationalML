"""Model metadata and estimator construction."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
import warnings

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
    requires_scaling: bool = False

    def __post_init__(self) -> None:
        if not self.name or not callable(self.estimator_class) or not callable(self.search_space):
            raise ValueError("A model needs a name, estimator class and search space.")
        if not self.tasks or any(not isinstance(task, TaskType) for task in self.tasks):
            raise ValueError("A model must declare supported TaskType values.")
        if not isinstance(self.requires_scaling, bool):
            raise ValueError("requires_scaling must be a boolean.")

    def parameters(self, suggested: dict[str, Any], random_state: int, n_jobs: int) -> dict[str, Any]:
        params = {**self.default_params, **suggested, "random_state": random_state}
        if self.n_jobs_parameter is not None:
            params[self.n_jobs_parameter] = n_jobs
        return params


def feature_importance(
    estimator: Any, feature_names: Sequence[str] | None, source_features: Sequence[str | None] | None = None,
) -> pd.DataFrame:
    """Return raw importances or signed linear coefficients, never a Styler."""
    empty = pd.DataFrame({
        "feature": pd.Series(dtype=str), "importance": pd.Series(dtype=float),
        "source_feature": pd.Series(dtype=str),
    })
    if feature_names is None:
        return empty
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_[0], dtype=float)
    else:
        return empty
    if values.ndim != 1 or len(values) != len(feature_names):
        warnings.warn("feature_importance is unavailable: native importances do not match feature names.",
                      UserWarning, stacklevel=2)
        return empty
    result = pd.DataFrame({
        "feature": list(feature_names), "importance": values,
        "source_feature": list(feature_names if source_features is None else source_features),
    })
    return result.loc[result["importance"].abs().sort_values(ascending=False, kind="stable").index].reset_index(drop=True)
