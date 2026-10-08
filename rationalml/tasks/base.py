"""Task names and deliberately binary-only task resolution."""

from enum import Enum

import pandas as pd

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
    """Accept exactly two non-missing classes, with no task guessing."""
    if y.empty or y.isna().any():
        raise DataValidationError("The target must be nonempty and contain no missing values.")
    is_auto = isinstance(task, str) and task.strip().lower() == "auto"
    resolved = TaskType.BINARY if is_auto else normalize_task(task)
    if resolved is not TaskType.BINARY:
        raise UnsupportedTaskError(f"Task {resolved.value!r} is not implemented in V0.2.")
    if y.nunique() != 2:
        raise UnsupportedTaskError(
            f"V0.2 requires exactly two target classes; found {y.nunique()}. "
            "Multiclass and regression are not implemented."
        )
    return resolved
