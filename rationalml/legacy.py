"""Small compatibility bridge for the seven historical algorithm names."""

from dataclasses import replace
from functools import partial
from typing import Any

from .models import ModelRegistry
from .optimization.spaces import suggest_lightgbm, suggest_logistic_regression, suggest_xgboost
from .tasks import TaskType

_ALGORITHMS = {
    "xgb_gb": ("xgboost", suggest_xgboost, {}),
    "lgb_gb": ("lightgbm", suggest_lightgbm, {}),
    "xgb_rf": ("xgboost", suggest_xgboost, {}),
    "lgb_rf": ("lightgbm", suggest_lightgbm, {"boosting_type": "rf"}),
    "xgb_mx": ("xgboost", suggest_xgboost, {}),
    "skl_rl": ("logistic_regression", suggest_logistic_regression, {"solver": "lbfgs"}),
    "skl_rs": ("logistic_regression", suggest_logistic_regression, {"solver": "saga"}),
}


def legacy_model_registry(context: Any) -> type[ModelRegistry]:
    """Isolate customized historical spaces without mutating the global registry."""
    class LegacyRegistry(ModelRegistry):
        _specs = {}
        _optional = {}

    for name in context.algo:
        if name not in _ALGORITHMS:
            raise ValueError(f"Unknown legacy algorithm {name!r}.")
        public_name, space, overrides = _ALGORITHMS[name]
        spec = ModelRegistry.get(public_name)
        params = {**spec.default_params, **overrides}
        if public_name == "logistic_regression":
            params["class_weight"] = {0: 1, 1: context.class_weight}
        else:
            params.update(n_estimators=context.n_estimators, scale_pos_weight=context.class_weight)
        if name == "xgb_rf":
            # Equivalent to the retired XGBRFClassifier: one boosting round
            # containing n_estimators parallel trees.
            params.update(n_estimators=1, num_parallel_tree=context.n_estimators)
        LegacyRegistry.register(replace(
            spec, name=name, default_params=params, tasks=frozenset({TaskType.BINARY}),
            search_space=partial(space, bounds=context.h_param[name]),
        ))
    return LegacyRegistry
