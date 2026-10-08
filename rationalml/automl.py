"""Public facade: split, optimize on train, select by CV, evaluate once."""

from copy import deepcopy
from dataclasses import replace
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .config import AutoMLConfig
from .data import BinaryLabelEncoder, validate_dataframe
from .evaluation import MetricRegistry, evaluate_binary
from .exceptions import ConfigurationError, DataValidationError, UnsupportedTaskError
from .models import ModelRegistry
from .models.base import feature_importance
from .optimization.optimizer import optimize_model
from .preprocessing.builder import build_model_pipeline, transformed_feature_info
from .result import AutoMLResult
from .tasks import resolve_task

logger = logging.getLogger(__name__)


class AutoML:
    """Accept AutoMLConfig or its keyword options, then fit a DataFrame."""

    def __init__(
        self, config: AutoMLConfig | None = None, *,
        model_registry: type[ModelRegistry] = ModelRegistry, **options: Any,
    ) -> None:
        if config is not None:
            if not isinstance(config, AutoMLConfig) or options:
                raise ConfigurationError("Pass either an AutoMLConfig or configuration keyword arguments.")
            self.config = deepcopy(config)
        else:
            self.config = AutoMLConfig(**options)
        self.model_registry = model_registry

    def fit(self, df: pd.DataFrame) -> AutoMLResult:
        config = replace(deepcopy(self.config))  # Revalidate mutable configuration.
        validate_dataframe(df, config.target, preprocessing=config.preprocessing)
        X = df.drop(columns=[config.target]).copy(deep=True)
        y = df[config.target].copy(deep=True)
        task = resolve_task(config.task, y)
        metric = MetricRegistry.get(config.metric)
        if task not in metric.tasks:
            raise UnsupportedTaskError(f"Metric {metric.name!r} does not support {task.value}.")
        names = (
            self.model_registry.available(task) if config.models == "auto"
            else [config.models] if isinstance(config.models, str) else config.models
        )
        if not names:
            raise ConfigurationError(f"No installed models support {task.value}.")
        specs = [self.model_registry.get(name) for name in names]
        for spec in specs:
            if task not in spec.tasks:
                raise UnsupportedTaskError(f"Model {spec.name!r} does not support {task.value}.")
            if not spec.supports_proba:
                raise ConfigurationError(f"Binary evaluation requires predict_proba: {spec.name!r}.")
        try:
            train_rows, test_rows = train_test_split(
                np.arange(len(df)), test_size=config.test_size,
                random_state=config.random_state, stratify=y,
            )
        except (TypeError, ValueError) as error:
            raise DataValidationError(f"Cannot create a stratified train/test split: {error}") from error
        X_train, X_test = X.iloc[train_rows], X.iloc[test_rows]
        original_train, original_test = y.iloc[train_rows], y.iloc[test_rows]
        if original_train.value_counts().min() < config.cv or original_test.nunique() != 2:
            raise DataValidationError(
                "Each train class must have at least cv rows and both classes must be present in test. "
                "Provide more examples or adjust cv/test_size."
            )
        encoder = BinaryLabelEncoder.from_target(original_train, config.positive_class)
        y_train = pd.Series(encoder.transform(original_train), index=original_train.index)
        y_test = pd.Series(encoder.transform(original_test), index=original_test.index)

        optimized = []
        for spec in specs:
            if config.verbose:
                logger.info("Optimizing %s using %s on train only", spec.name, metric.name)
            optimized.append(optimize_model(spec, X_train, y_train, metric, config))
        # Winner selection only sees CV scores. Stable ties retain model order.
        ordered = sorted(optimized, key=lambda item: item.score, reverse=metric.direction == "maximize")
        best = ordered[0]
        leaderboard = pd.DataFrame([{
            "model": item.spec.name, "rank": rank,
            "cv_score": item.score, "cv_std": float(np.std(item.fold_scores)),
        } for rank, item in enumerate(ordered, start=1)]).set_index("model")
        best_model = build_model_pipeline(best.spec, best.params, X_train, config.preprocessing)
        with threadpool_limits(limits=config.n_jobs if config.n_jobs > 0 else None):
            best_model.fit(X_train, y_train)
            test_metrics = evaluate_binary(best_model, X_test, y_test)
        transformed_names, source_features = transformed_feature_info(best_model, X.columns)
        return AutoMLResult(
            task=task, target=config.target, leaderboard=leaderboard,
            best_model=best_model, best_model_name=best.spec.name,
            best_params=best.params.copy(), metrics=test_metrics.copy(),
            cv_results=pd.concat([item.cv_results for item in optimized], ignore_index=True),
            test_metrics=test_metrics, feature_importance=feature_importance(
                best_model.named_steps["estimator"], transformed_names, source_features,
            ),
            config=config, feature_names=tuple(X.columns), label_encoder=encoder,
            train_indices=tuple(int(row) for row in train_rows),
            test_indices=tuple(int(row) for row in test_rows),
            feature_schema=best_model.feature_schema_, transformed_feature_names=transformed_names,
        )
