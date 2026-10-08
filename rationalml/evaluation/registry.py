"""Extensible metric registry; names are the only orchestration contract."""

from functools import partial

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    r2_score,
)

from ..exceptions import UnsupportedTaskError
from ..tasks import TaskType, normalize_task
from .metrics import MetricSpec


class MetricRegistry:
    _specs: dict[str, MetricSpec] = {}

    @classmethod
    def register(cls, spec: MetricSpec) -> None:
        if spec.name in cls._specs:
            raise ValueError(f"Metric {spec.name!r} is already registered.")
        cls._specs[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> MetricSpec:
        try:
            return cls._specs[name.strip().lower()]
        except KeyError as error:
            raise KeyError(f"Unknown metric {name!r}; available: {cls.available()}.") from error

    @classmethod
    def available(cls, task: str | TaskType | None = None) -> list[str]:
        resolved = normalize_task(task) if task is not None else None
        return [name for name, spec in cls._specs.items() if resolved is None or resolved in spec.tasks]

    @classmethod
    def resolve(cls, name: str, task: TaskType) -> MetricSpec:
        defaults = {TaskType.BINARY: "roc_auc", TaskType.MULTICLASS: "f1_macro", TaskType.REGRESSION: "rmse"}
        spec = cls.get(defaults[task] if name == "auto" else name)
        if task not in spec.tasks:
            raise UnsupportedTaskError(f"Metric {spec.name!r} does not support {task.value}.")
        return spec


def _classification_log_loss(y, probabilities) -> float:
    """Encoded labels follow probability columns, including absent labels."""
    classes = [0, 1] if np.ndim(probabilities) == 1 else np.arange(np.shape(probabilities)[1])
    return log_loss(y, probabilities, labels=classes)


for _name, _scorer, _direction, _prediction_type in (
    ("roc_auc", roc_auc_score, "maximize", "proba"),
    ("average_precision", average_precision_score, "maximize", "proba"),
    ("accuracy", accuracy_score, "maximize", "predict"),
    ("precision", partial(precision_score, zero_division=0), "maximize", "predict"),
    ("recall", partial(recall_score, zero_division=0), "maximize", "predict"),
    ("f1", partial(f1_score, zero_division=0), "maximize", "predict"),
    ("balanced_accuracy", balanced_accuracy_score, "maximize", "predict"),
    ("log_loss", _classification_log_loss, "minimize", "proba"),
    ("f1_macro", partial(f1_score, average="macro", zero_division=0), "maximize", "predict"),
    ("f1_weighted", partial(f1_score, average="weighted", zero_division=0), "maximize", "predict"),
    ("rmse", root_mean_squared_error, "minimize", "predict"),
    ("mae", mean_absolute_error, "minimize", "predict"),
    ("r2", r2_score, "maximize", "predict"),
):
    _tasks = (
        {TaskType.REGRESSION} if _name in {"rmse", "mae", "r2"}
        else {TaskType.MULTICLASS} if _name in {"f1_macro", "f1_weighted"}
        else {TaskType.BINARY, TaskType.MULTICLASS} if _name in {"accuracy", "balanced_accuracy", "log_loss"}
        else {TaskType.BINARY}
    )
    MetricRegistry.register(MetricSpec(
        name=_name,
        scorer=_scorer,
        direction=_direction,
        prediction_type=_prediction_type,
        tasks=frozenset(_tasks),
    ))
