"""Extensible metric registry; names are the only orchestration contract."""

from functools import partial

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


for _name, _scorer, _direction, _prediction_type in (
    ("roc_auc", roc_auc_score, "maximize", "proba"),
    ("average_precision", average_precision_score, "maximize", "proba"),
    ("accuracy", accuracy_score, "maximize", "predict"),
    ("precision", partial(precision_score, zero_division=0), "maximize", "predict"),
    ("recall", partial(recall_score, zero_division=0), "maximize", "predict"),
    ("f1", partial(f1_score, zero_division=0), "maximize", "predict"),
    ("balanced_accuracy", balanced_accuracy_score, "maximize", "predict"),
    ("log_loss", partial(log_loss, labels=[0, 1]), "minimize", "proba"),
):
    MetricRegistry.register(MetricSpec(
        name=_name,
        scorer=_scorer,
        direction=_direction,
        prediction_type=_prediction_type,
        tasks=frozenset({TaskType.BINARY}),
    ))
