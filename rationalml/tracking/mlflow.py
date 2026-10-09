"""Post-fit tracking of stored results; no training data or ML computations."""

import json
import os
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
import pandas as pd

from ..exceptions import ConfigurationError, MissingDependencyError
from ..tasks import TaskType
from .config import MLflowConfig

if TYPE_CHECKING:
    from ..result import AutoMLResult


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        converted = {(key if isinstance(key, str) else json.dumps(_json_value(key), sort_keys=True)):
                     _json_value(item) for key, item in value.items()}
        if len(converted) != len(value):
            raise ConfigurationError("MLflow parameter dictionary keys collide after conversion to strings.")
        return converted
    label = f"<{type(value).__module__}.{type(value).__qualname__}>"
    warnings.warn(f"MLflow parameter is not JSON serializable; logging its type {label} instead of its value.",
                  UserWarning, stacklevel=3)
    return label


def _params(parameters: dict[str, Any]) -> dict[str, str]:
    values = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or not key.strip():
            raise ConfigurationError("MLflow parameter names must be nonempty strings.")
        value = _json_value(value)
        values[key] = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return values


def _parent_params(result: "AutoMLResult", class_label: object | None) -> dict[str, str]:
    values = {
        "task": result.task.value, "target": result.target, "primary_metric": result.primary_metric,
        "best_model": result.best_model_name, "baseline_name": result.baseline_name,
        "cv_folds": result.config.cv, "n_trials_requested": result.config.n_trials,
        "random_state": result.config.random_state, "train_rows": len(result.train_indices),
        "test_rows": len(result.test_indices), "n_features": len(result.feature_names),
    }
    if result.task is TaskType.BINARY:
        values.update(positive_class=result.positive_class, negative_class=result.negative_class)
    elif result.task is TaskType.MULTICLASS:
        values["n_classes"] = len(result.classes_)
        if class_label is not None:
            values["ranking_class"] = class_label
    return _params(values)


def _prediction_table(result: "AutoMLResult") -> pd.DataFrame:
    table = result.test_predictions
    name = table.index.name if isinstance(table.index.name, str) else "index"
    base, suffix = name, 0
    while name in table.columns:
        suffix += 1
        name = f"{base}_{suffix}"
    table.insert(0, name, table.index.to_numpy(copy=True))
    return table


def log_result(
    result: "AutoMLResult", *, config: MLflowConfig | None = None, class_label: object | None = None,
) -> str:
    """Create one result run and one candidate run per leaderboard row."""
    try:
        import mlflow
    except ImportError as error:
        raise MissingDependencyError(
            'MLflow tracking requires MLflow. Install it with: pip install "rationalml[mlflow]".'
        ) from error
    if config is None:
        config = MLflowConfig()
    elif not isinstance(config, MLflowConfig):
        raise ConfigurationError("config must be an MLflowConfig or None.")
    # Revalidate copied tags in case the mapping itself was edited after construction.
    config = MLflowConfig(**vars(config))
    active = mlflow.active_run()
    if active is not None and not config.nested:
        raise ConfigurationError("An MLflow run is already active. Set MLflowConfig(nested=True) to log RationalML inside it.")
    tables = {
        "leaderboard.json": result.leaderboard.rename_axis("model").reset_index(),
        "cv_results.json": result.cv_results.copy(deep=True),
    }
    if not result.feature_importance.empty:
        tables["feature_importance.json"] = result.feature_importance.copy(deep=True)
    if result.task is not TaskType.MULTICLASS or class_label is not None:
        # Validate class selection and ranking before creating any backend run.
        tables["ranking.json"] = result.ranking_table(class_label=class_label)
    calibration_metrics = {}
    if result.task is TaskType.BINARY or (result.task is TaskType.MULTICLASS and class_label is not None):
        tables["calibration.json"] = result.calibration_table(n_bins=10, class_label=class_label)
        summary = result.calibration_summary(n_bins=10, class_label=class_label)
        calibration_metrics = {"calibration_brier_score": summary["brier_score"],
                               "calibration_expected_error": summary["expected_calibration_error"],
                               "calibration_max_error": summary["max_calibration_error"]}
    if config.log_predictions:
        tables["predictions.json"] = _prediction_table(result)
    parent_params = _parent_params(result, class_label)
    candidates = []
    model_params = getattr(result, "model_best_params", {})
    for name, row in result.leaderboard.iterrows():
        params = model_params.get(name)
        if params is None:
            if name == result.best_model_name:
                params = result.best_params
            else:
                warnings.warn(f"Best parameters unavailable for candidate {name!r}; logging its CV diagnostics only.",
                              UserWarning, stacklevel=2)
                params = {}
        candidates.append((name, row, _params(params)))
    best = result.leaderboard.loc[result.best_model_name]
    metrics = {"baseline_score": result.baseline_score, "best_cv_score": best["cv_score"],
               "best_cv_std": best["cv_std"], "improvement_vs_baseline": best["improvement_vs_baseline"],
               **{f"test_{name}": value for name, value in result.test_metrics.items()}, **calibration_metrics}
    from .. import __version__

    tags = {**(config.tags or {}), "rationalml.version": __version__, "rationalml.task": result.task.value,
            "rationalml.best_model": result.best_model_name}
    original_uri = mlflow.get_tracking_uri()
    # start_run can otherwise resume an unrelated run via this environment variable.
    original_environment = {name: os.environ.get(name) for name in ("MLFLOW_RUN_ID", "MLFLOW_TRACKING_URI")}
    os.environ.pop("MLFLOW_RUN_ID", None)
    try:
        if config.tracking_uri is not None:
            mlflow.set_tracking_uri(config.tracking_uri)
            if active is not None and mlflow.get_tracking_uri() != original_uri:
                raise ConfigurationError("Nested RationalML tracking must use the active run's tracking URI.")
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(config.experiment_name)
        experiment_id = (experiment.experiment_id if experiment is not None
                         else client.create_experiment(config.experiment_name))
        with mlflow.start_run(experiment_id=experiment_id, run_name=config.run_name, nested=config.nested, tags=tags) as parent:
            mlflow.log_params(parent_params)
            mlflow.log_metrics(metrics)
            for name, table in tables.items():
                # MLflow's object-column inspection predates pandas 3 string dtypes.
                text_types = {column: object for column, dtype in table.dtypes.items() if isinstance(dtype, pd.StringDtype)}
                mlflow.log_table(data=table.astype(text_types), artifact_file=name)
            for name, row, params in candidates:
                with mlflow.start_run(experiment_id=experiment_id, run_name=name, nested=True,
                                      tags={"rationalml.model_name": name,
                                            "rationalml.selected": str(name == result.best_model_name).lower()}):
                    mlflow.log_params(params)
                    mlflow.log_metrics({key: float(row[key]) for key in
                                        ("cv_score", "cv_std", "cv_min", "cv_max", "improvement_vs_baseline", "n_trials_completed")})
            if config.log_model:
                import mlflow.sklearn

                mlflow.sklearn.log_model(result.best_model, name="model", signature=False, serialization_format="cloudpickle")
            return parent.info.run_id
    finally:
        if config.tracking_uri is not None:
            mlflow.set_tracking_uri(original_uri)
        for name, value in original_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
