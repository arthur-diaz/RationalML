"""RationalML V0.2.1 public API."""

from .automl import AutoML
from .config import AutoMLConfig
from .evaluation import MetricRegistry, MetricSpec
from .models import ModelRegistry, ModelSpec
from .preprocessing import PreprocessingConfig, FeatureSchema, infer_schema, build_preprocessor
from .result import AutoMLResult
from .tasks import TaskType, normalize_task

__all__ = [
    "AutoML", "AutoMLConfig", "AutoMLResult", "TaskType", "normalize_task",
    "MetricRegistry", "MetricSpec", "ModelRegistry", "ModelSpec",
    "PreprocessingConfig", "FeatureSchema", "infer_schema", "build_preprocessor",
]
__version__ = "0.2.1"
