import pandas as pd
import pytest

from rationalml import TaskType, normalize_task
from rationalml.exceptions import DataValidationError, UnsupportedTaskError
from rationalml.tasks import resolve_task


@pytest.mark.parametrize("name, expected", [
    ("binary", TaskType.BINARY), (" MULTICLASS ", TaskType.MULTICLASS),
    ("regression", TaskType.REGRESSION), (TaskType.BINARY, TaskType.BINARY),
])
def test_normalization(name, expected):
    assert normalize_task(name) is expected


@pytest.mark.parametrize("labels", [[0, 1], ["no", "yes"], [-3, 9], [False, True], [0.1, 0.2]])
def test_auto_detects_exactly_two_classes(labels):
    assert resolve_task("auto", pd.Series(labels * 2)) is TaskType.BINARY


@pytest.mark.parametrize("labels", [[0, 0], [0, 1, 2], [0.1, 0.2, 0.3]])
def test_auto_rejects_ambiguous_targets(labels):
    with pytest.raises(UnsupportedTaskError, match="exactly two"):
        resolve_task("auto", pd.Series(labels))


@pytest.mark.parametrize("task", [TaskType.MULTICLASS, "regression", "unknown", "auto?"])
def test_unimplemented_and_unknown_tasks(task):
    with pytest.raises(UnsupportedTaskError):
        resolve_task(task, pd.Series([0, 1]))


@pytest.mark.parametrize("labels", [[], [0, 1, None]])
def test_missing_and_empty_targets(labels):
    with pytest.raises(DataValidationError):
        resolve_task("auto", pd.Series(labels))
