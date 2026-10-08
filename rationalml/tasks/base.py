"""Explicit task names and conservative target-based resolution."""

from enum import Enum

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from ..exceptions import DataValidationError, UnsupportedTaskError


class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


def normalize_task(task: str | TaskType) -> TaskType:
    """Normalize an explicit task name; auto requires a target."""
    if isinstance(task, TaskType):
        return task
    if not isinstance(task, str):
        raise UnsupportedTaskError("task must be a string or TaskType.")
    try:
        return TaskType(task.strip().lower())
    except ValueError as error:
        raise UnsupportedTaskError(f"Unknown task {task!r}.") from error


def resolve_task(task: str | TaskType, y: pd.Series) -> TaskType:
    """Infer binary/text classification, requiring an explicit numeric task."""
    if y.empty or y.isna().any():
        raise DataValidationError("The target must be nonempty and contain no missing values.")
    is_auto = isinstance(task, str) and task.strip().lower() == "auto"
    count = y.nunique()
    numeric = is_numeric_dtype(y.dtype) or all(
        isinstance(value, (int, float, np.number, bool)) for value in y
    )
    if is_auto:
        if count == 2:
            return TaskType.BINARY
        if count > 2 and numeric:
            raise UnsupportedTaskError(
                "Numeric targets with more than two distinct values are ambiguous. "
                "Set task='multiclass' or task='regression' explicitly."
            )
        resolved = TaskType.MULTICLASS if count > 2 else TaskType.BINARY
    else:
        resolved = normalize_task(task)
    if resolved is TaskType.BINARY and count != 2:
        raise UnsupportedTaskError(
            f"Binary classification requires exactly two target classes; found {count}."
        )
    if resolved is TaskType.MULTICLASS and count < 3:
        raise UnsupportedTaskError(f"Multiclass classification requires at least three target classes; found {count}.")
    if resolved is TaskType.REGRESSION:
        if not is_numeric_dtype(y.dtype) or np.iscomplexobj(y.to_numpy()):
            raise DataValidationError("Regression requires a numeric, finite target.")
        if not np.isfinite(y.to_numpy(dtype=float)).all():
            raise DataValidationError("Regression requires a numeric, finite target; infinite values are not supported.")
    return resolved
