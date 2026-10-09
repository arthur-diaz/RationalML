"""RationalML public API."""

from importlib.metadata import version as _distribution_version

from .automl import AutoML
from .config import AutoMLConfig
from .evaluation import MetricRegistry, MetricSpec
from .models import ModelRegistry, ModelSpec
from .preprocessing import PreprocessingConfig
from .result import AutoMLResult
from .reporting import ExcelReportConfig
from .tasks import TaskType
from .tracking import MLflowConfig

__all__ = [
    "AutoML", "AutoMLConfig", "AutoMLResult", "TaskType",
    "MetricRegistry", "MetricSpec", "ModelRegistry", "ModelSpec",
    "PreprocessingConfig",
    "ExcelReportConfig",
    "MLflowConfig",
]
__version__ = _distribution_version("RationalML")
