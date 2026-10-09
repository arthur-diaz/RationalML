"""Optimization accepts training data only; no holdout evaluation lives here."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from threadpoolctl import threadpool_limits

from ..config import AutoMLConfig
from ..data import validate_features
from ..evaluation import MetricSpec
from ..exceptions import OptimizationError
from ..models.base import ModelSpec
from ..preprocessing.builder import build_model_pipeline
from ..runtime import optuna_verbosity
from ..tasks import resolve_task
from .cv import CVSplits, make_cv_splits


@dataclass
class OptimizedModel:
    spec: ModelSpec
    params: dict[str, Any]
    score: float
    fold_scores: tuple[float, ...]
    cv_results: pd.DataFrame


def optimize_model(
    spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series,
    metric: MetricSpec, config: AutoMLConfig,
    *, folds: CVSplits | None = None,
    on_trial: Callable[[optuna.Study, optuna.trial.FrozenTrial], None] | None = None,
) -> OptimizedModel:
    """Select hyperparameters by task-appropriate CV, with seeded sequential trials."""
    validate_features(X_train, preprocessing=config.preprocessing)
    task = resolve_task(config.task, y_train)
    if folds is None:
        folds = make_cv_splits(X_train, y_train, task, config)

    def objective(trial: optuna.Trial) -> float:
        params = spec.parameters(spec.search_space(trial), config.random_state, config.n_jobs, task=task)
        scores = []
        for training_rows, validation_rows in folds:
            fold_train = X_train.iloc[training_rows]
            pipeline = build_model_pipeline(spec, params, fold_train, config.preprocessing)
            pipeline.fit(fold_train, y_train.iloc[training_rows])
            scores.append(metric.evaluate(pipeline, X_train.iloc[validation_rows], y_train.iloc[validation_rows], task=task))
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("estimator_params", params)
        return float(np.mean(scores))

    with optuna_verbosity(config.verbose):
        study = optuna.create_study(
            direction=metric.direction,
            sampler=optuna.samplers.TPESampler(seed=config.random_state),
            pruner=optuna.pruners.NopPruner(),
        )
        try:
            with threadpool_limits(limits=config.n_jobs if config.n_jobs > 0 else None):
                study.optimize(
                    objective, n_trials=config.n_trials, timeout=config.timeout, n_jobs=1,
                    **({"callbacks": [on_trial]} if on_trial is not None else {}),
                )
        finally:
            # Optuna does not call callbacks when an uncaught objective error
            # aborts optimize(). Reconcile terminal states without swallowing it.
            if on_trial is not None:
                for trial in study.trials:
                    on_trial(study, trial)
    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed:
        raise OptimizationError(f"No trial completed for model {spec.name!r}.")
    best = study.best_trial
    # One row per COMPLETE trial; this also supplies the successful trial count.
    rows = [{
        "model": spec.name, "trial": trial.number, "value": trial.value,
        **{f"fold_{i}": score for i, score in enumerate(trial.user_attrs["fold_scores"])},
    } for trial in completed]
    return OptimizedModel(
        spec=spec, params=best.user_attrs["estimator_params"].copy(), score=float(best.value),
        fold_scores=tuple(best.user_attrs["fold_scores"]), cv_results=pd.DataFrame(rows),
    )
