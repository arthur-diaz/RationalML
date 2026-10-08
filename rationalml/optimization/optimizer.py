"""Optimization accepts training data only; no holdout evaluation lives here."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from ..config import AutoMLConfig
from ..data import validate_features
from ..evaluation import MetricSpec
from ..exceptions import OptimizationError
from ..models.base import ModelSpec
from ..preprocessing.builder import build_model_pipeline


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
) -> OptimizedModel:
    """Select hyperparameters by stratified CV, with seeded sequential trials."""
    validate_features(X_train, preprocessing=config.preprocessing)
    folds = list(StratifiedKFold(
        n_splits=config.cv, shuffle=True, random_state=config.random_state,
    ).split(X_train, y_train))

    def objective(trial: optuna.Trial) -> float:
        params = spec.parameters(spec.search_space(trial), config.random_state, config.n_jobs)
        scores = []
        for training_rows, validation_rows in folds:
            fold_train = X_train.iloc[training_rows]
            pipeline = build_model_pipeline(spec, params, fold_train, config.preprocessing)
            pipeline.fit(fold_train, y_train.iloc[training_rows])
            scores.append(metric.evaluate(pipeline, X_train.iloc[validation_rows], y_train.iloc[validation_rows]))
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("estimator_params", params)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction=metric.direction,
        sampler=optuna.samplers.TPESampler(seed=config.random_state),
        pruner=optuna.pruners.NopPruner(),
    )
    with threadpool_limits(limits=config.n_jobs if config.n_jobs > 0 else None):
        study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout, n_jobs=1)
    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed:
        raise OptimizationError(f"No trial completed for model {spec.name!r}.")
    best = study.best_trial
    rows = [{
        "model": spec.name, "trial": trial.number, "value": trial.value,
        **{f"fold_{i}": score for i, score in enumerate(trial.user_attrs["fold_scores"])},
    } for trial in completed]
    return OptimizedModel(
        spec=spec, params=best.user_attrs["estimator_params"].copy(), score=float(best.value),
        fold_scores=tuple(best.user_attrs["fold_scores"]), cv_results=pd.DataFrame(rows),
    )
