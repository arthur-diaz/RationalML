"""Configuration contains options only, never training data."""

from dataclasses import dataclass
from math import isfinite

from .exceptions import ConfigurationError
from .tasks import TaskType, normalize_task


@dataclass
class AutoMLConfig:
    target: str
    task: str | TaskType = "auto"
    metric: str = "roc_auc"
    models: list[str] | str = "auto"
    test_size: float = 0.20
    cv: int = 5
    n_trials: int = 30
    random_state: int = 42
    n_jobs: int = 1
    timeout: int | None = None
    verbose: int = 1
    positive_class: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target.strip():
            raise ConfigurationError("target must be a nonempty column name.")
        if isinstance(self.task, str) and self.task.strip().lower() == "auto":
            self.task = "auto"
        else:
            self.task = normalize_task(self.task)
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ConfigurationError("metric must be a nonempty name.")
        self.metric = self.metric.strip().lower()
        if isinstance(self.models, str):
            if not self.models.strip():
                raise ConfigurationError("models must be 'auto', a model name or a nonempty list.")
            self.models = self.models.strip().lower()
        elif isinstance(self.models, list) and self.models:
            if any(not isinstance(name, str) or not name.strip() for name in self.models):
                raise ConfigurationError("Every model name must be a nonempty string.")
            self.models = [name.strip().lower() for name in self.models]
            if len(set(self.models)) != len(self.models) or "auto" in self.models:
                raise ConfigurationError("models must contain distinct explicit names.")
        else:
            raise ConfigurationError("models must be 'auto', a model name or a nonempty list.")
        if (
            isinstance(self.test_size, bool)
            or not isinstance(self.test_size, (int, float))
            or not isfinite(self.test_size)
            or not 0 < self.test_size < 1
        ):
            raise ConfigurationError("test_size must satisfy 0 < test_size < 1.")
        for name, minimum in (("cv", 2), ("n_trials", 1), ("verbose", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ConfigurationError(f"{name} must be an integer >= {minimum}.")
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or not 0 <= self.random_state < 2**32
        ):
            raise ConfigurationError("random_state must be an integer in [0, 2**32).")
        if (
            isinstance(self.n_jobs, bool)
            or not isinstance(self.n_jobs, int)
            or (self.n_jobs != -1 and self.n_jobs < 1)
        ):
            raise ConfigurationError("n_jobs must be -1 or a positive integer.")
        if self.timeout is not None and (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, int)
            or self.timeout < 1
        ):
            raise ConfigurationError("timeout must be None or a positive integer in seconds.")
