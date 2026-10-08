"""Model registry with lazy loading for optional boosting dependencies."""

from importlib import import_module
from importlib.util import find_spec
from typing import Any

from sklearn.linear_model import LogisticRegression

from ..exceptions import MissingDependencyError
from ..optimization.spaces import suggest_lightgbm, suggest_logistic_regression, suggest_xgboost
from ..tasks import TaskType, normalize_task
from .base import ModelSpec


class ModelRegistry:
    _specs: dict[str, ModelSpec] = {}
    _optional: dict[str, dict[str, Any]] = {
        "lightgbm": {
            "module": "lightgbm", "class_name": "LGBMClassifier",
            "search_space": suggest_lightgbm,
            "default_params": {
                "objective": "binary", "boosting_type": "gbdt", "n_estimators": 100,
                "verbosity": -1, "deterministic": True, "force_col_wise": True,
            },
        },
        "xgboost": {
            "module": "xgboost", "class_name": "XGBClassifier",
            "search_space": suggest_xgboost,
            "default_params": {
                "objective": "binary:logistic", "booster": "gbtree", "tree_method": "hist",
                "eval_metric": "logloss", "n_estimators": 100, "verbosity": 0,
            },
        },
    }

    @classmethod
    def register(cls, spec: ModelSpec) -> None:
        if spec.name in cls._specs or spec.name in cls._optional:
            raise ValueError(f"Model {spec.name!r} is already registered.")
        cls._specs[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> ModelSpec:
        key = name.strip().lower()
        if key in cls._specs:
            return cls._specs[key]
        if key not in cls._optional:
            raise KeyError(f"Unknown model {name!r}; available: {cls.available()}.")
        metadata = cls._optional[key]
        try:
            module = import_module(metadata["module"])
        except ModuleNotFoundError as error:
            if error.name != metadata["module"]:
                raise
            raise MissingDependencyError(
                f"Model {key!r} requires {metadata['module']}; install RationalML[boosting]."
            ) from error
        return ModelSpec(
            name=key, estimator_class=getattr(module, metadata["class_name"]),
            tasks=frozenset({TaskType.BINARY}), search_space=metadata["search_space"],
            default_params=metadata["default_params"].copy(), supports_early_stopping=True,
        )

    @classmethod
    def available(cls, task: str | TaskType | None = None) -> list[str]:
        """List registered models whose optional dependencies are installed."""
        resolved = normalize_task(task) if task is not None else None
        names = [name for name, spec in cls._specs.items() if resolved is None or resolved in spec.tasks]
        if resolved is None or resolved is TaskType.BINARY:
            names.extend(name for name, info in cls._optional.items() if find_spec(info["module"]) is not None)
        return names


ModelRegistry.register(ModelSpec(
    name="logistic_regression", estimator_class=LogisticRegression,
    tasks=frozenset({TaskType.BINARY}), search_space=suggest_logistic_regression,
    default_params={"solver": "lbfgs", "max_iter": 1000}, n_jobs_parameter=None, requires_scaling=True,
))
