"""RationalML V0.1.1 public API."""

from .automl import AutoML
from .config import AutoMLConfig
from .evaluation import MetricRegistry, MetricSpec
from .models import ModelRegistry, ModelSpec
from .result import AutoMLResult
from .tasks import TaskType, normalize_task

__all__ = [
    "AutoML", "AutoMLConfig", "AutoMLResult", "TaskType", "normalize_task",
    "MetricRegistry", "MetricSpec", "ModelRegistry", "ModelSpec",
]
__version__ = "0.1.1"
