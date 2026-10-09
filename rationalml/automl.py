"""Public facade: split, optimize on train, select by CV, evaluate once."""

from copy import deepcopy
from dataclasses import replace
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .config import AutoMLConfig
from .data import BinaryLabelEncoder, MulticlassLabelEncoder, validate_dataframe
from .evaluation import MetricRegistry
from .evaluation.baseline import evaluate_baseline
from .evaluation.predictions import evaluate_holdout
from .exceptions import ConfigurationError, DataValidationError, UnsupportedTaskError
from .models import ModelRegistry
from .models.base import feature_importance
from .optimization.cv import make_cv_splits
from .optimization.optimizer import optimize_model
from .preprocessing.builder import build_model_pipeline, transformed_feature_info
from .result import AutoMLResult
from .runtime import TrialProgress, print_run_info
from .tasks import TaskType, resolve_task

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
        started = perf_counter()
        config = replace(deepcopy(self.config))  # Revalidate mutable configuration.
        validate_dataframe(df, config.target, preprocessing=config.preprocessing)
        X = df.drop(columns=[config.target]).copy(deep=True)
        y = df[config.target].copy(deep=True)
        task = resolve_task(config.task, y)
        if task is not TaskType.BINARY and config.positive_class is not None:
            raise ConfigurationError("positive_class is only available for binary classification.")
        metric = MetricRegistry.resolve(config.metric, task)
        config.task, config.metric = task, metric.name
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
            if task is not TaskType.REGRESSION and not spec.supports_proba:
                raise ConfigurationError(f"Classification evaluation requires predict_proba: {spec.name!r}.")
        try:
            train_rows, test_rows = train_test_split(
                np.arange(len(df)), test_size=config.test_size,
                random_state=config.random_state, stratify=None if task is TaskType.REGRESSION else y,
            )
        except (TypeError, ValueError) as error:
            raise DataValidationError(f"Cannot create the train/test split: {error}") from error
        X_train, X_test = X.iloc[train_rows], X.iloc[test_rows]
        original_train, original_test = y.iloc[train_rows], y.iloc[test_rows]
        encoder = None
        if task is TaskType.REGRESSION:
            if len(original_train) < 2 * config.cv or len(original_test) < 2:
                raise DataValidationError(
                    "Regression requires at least two rows per validation fold and in test for finite R2. "
                    "Provide more examples or adjust cv/test_size."
                )
            y_train, y_test = original_train.copy(), original_test.copy()
        else:
            counts = original_train.value_counts()
            if (counts[counts > 0].min() < config.cv
                    or original_train.nunique() != y.nunique() or original_test.nunique() != y.nunique()):
                raise DataValidationError(
                    "Each train class must have at least cv rows and all classes must be present in test. "
                    "Provide more examples or adjust cv/test_size."
                )
            encoder = (
                BinaryLabelEncoder.from_target(original_train, config.positive_class)
                if task is TaskType.BINARY else MulticlassLabelEncoder.from_target(original_train)
            )
            y_train = pd.Series(encoder.transform(original_train), index=original_train.index)
            y_test = pd.Series(encoder.transform(original_test), index=original_test.index)

        folds = make_cv_splits(X_train, y_train, task, config)
        baseline_name, baseline_fold_scores = evaluate_baseline(y_train, metric, task, folds)
        baseline_score = float(np.mean(baseline_fold_scores))
        optimized = []
        model_fit_times = {}
        if config.verbose:
            print_run_info(task.value, metric.name, len(specs), config.cv, config.n_trials, config.n_jobs)
        progress = TrialProgress(len(specs) * config.n_trials) if config.verbose == 1 else None
        interrupted = True
        try:
            for spec in specs:
                model_started = perf_counter()
                callbacks = {"on_trial": progress.update} if progress is not None else {}
                optimized.append(optimize_model(spec, X_train, y_train, metric, config, folds=folds, **callbacks))
                model_fit_times[spec.name] = perf_counter() - model_started
                if progress is not None:
                    progress.model_finished()
            interrupted = False
        finally:
            if progress is not None:
                progress.close(interrupted=interrupted)
        # Winner selection only sees CV scores. Stable ties retain model order.
        ordered = sorted(optimized, key=lambda item: item.score, reverse=metric.direction == "maximize")
        best = ordered[0]
        leaderboard = pd.DataFrame([{
            "model": item.spec.name, "rank": rank,
            "cv_score": item.score, "cv_std": float(np.std(item.fold_scores)),
            "cv_min": float(np.min(item.fold_scores)), "cv_max": float(np.max(item.fold_scores)),
            "improvement_vs_baseline": (
                item.score - baseline_score if metric.direction == "maximize" else baseline_score - item.score
            ),
            "n_trials_completed": len(item.cv_results),
        } for rank, item in enumerate(ordered, start=1)]).set_index("model")
        with threadpool_limits(limits=config.n_jobs if config.n_jobs > 0 else None):
            final_started = perf_counter()
            best_model = build_model_pipeline(best.spec, best.params, X_train, config.preprocessing)
            best_model.fit(X_train, y_train)
            model_fit_times[best.spec.name] += perf_counter() - final_started
            test_metrics, test_predictions = evaluate_holdout(best_model, X_test, y_test, original_test, task, encoder)
        transformed_names, source_features = transformed_feature_info(best_model, X.columns)
        result = AutoMLResult(
            task=task, target=config.target, leaderboard=leaderboard,
            best_model=best_model, best_model_name=best.spec.name,
            best_params=best.params.copy(), metrics=test_metrics.copy(),
            cv_results=pd.concat([item.cv_results for item in optimized], ignore_index=True),
            test_metrics=test_metrics, feature_importance=feature_importance(
                best_model.named_steps["estimator"], transformed_names, source_features,
                classes=encoder.classes_ if encoder is not None else None,
            ),
            config=config, feature_names=tuple(X.columns), label_encoder=encoder,
            train_indices=tuple(int(row) for row in train_rows),
            test_indices=tuple(int(row) for row in test_rows),
            baseline_name=baseline_name, baseline_score=baseline_score,
            baseline_cv_std=float(np.std(baseline_fold_scores)), baseline_fold_scores=baseline_fold_scores,
            _test_predictions=test_predictions,
            feature_schema=best_model.feature_schema_, transformed_feature_names=transformed_names,
            model_best_params={item.spec.name: item.params.copy() for item in optimized},
            model_fit_times=model_fit_times,
        )
        result.fit_time = perf_counter() - started
        if config.verbose:
            print("\n" + result.summary())
        return result
