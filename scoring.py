"""Legacy scoring names retained with explicit train-only contracts.

Historical spaces now live in rationalml.optimization.spaces. The former
objective mixed threshold fitting, pruning and evaluation and is retired.
"""

import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from rationalml import AutoMLConfig, MetricRegistry
from rationalml.exceptions import LegacyAPIError
from rationalml.legacy import legacy_model_registry
from rationalml.optimization.optimizer import optimize_model
from metrics import Metrics


def optuna_metrics(y, pred_y, threshold: float, init_model) -> tuple:
    """Evaluate a supplied fixed threshold; never choose it from y."""
    predictions = (np.asarray(pred_y) >= threshold).astype(int)
    return (roc_auc_score(y, pred_y), *Metrics.simple_metrics(y, predictions))


def objective(trial, cst_params, X, y, init_model, score, model_type, threshold=0.03):
    raise LegacyAPIError("The old objective is retired; use AutoML.fit or optimization.optimize_model on train.")


def opti(X, y, init_model, score, name: str, *, training_only: bool = False) -> tuple:
    """Optimize only an explicitly declared training partition.

    Returns (estimator parameters, {'cv_std': ...}); former roc/pr variance
    fields were coupled to the retired reporting implementation.
    """
    if not training_only:
        raise LegacyAPIError("opti requires training_only=True and an already isolated train set; prefer AutoML.fit.")
    warnings.warn("opti is legacy; migrate to AutoML.fit.", DeprecationWarning, stacklevel=2)
    if init_model.class_weight == "auto":
        raise LegacyAPIError("Legacy automatic class weights require migration; use a fixed class_weight.")
    names = ("roc_auc", "accuracy", "precision", "recall", "f1")
    metric_name = names[score] if isinstance(score, int) else score
    metric = MetricRegistry.get(metric_name)
    if y.nunique() != 2 or set(y.unique()) != {0, 1} or y.value_counts().min() < init_model.nb_cv:
        raise ValueError("opti requires encoded binary train labels 0/1 and at least nb_cv rows per class.")
    config = AutoMLConfig(
        target=init_model.target_name, models=[name], metric=metric_name,
        cv=init_model.nb_cv, n_trials=init_model.nb_iter, random_state=init_model.seed,
        n_jobs=init_model.n_jobs, timeout=init_model.timeout, verbose=init_model.verbose,
    )
    spec = legacy_model_registry(init_model).get(name)
    optimized = optimize_model(spec, X, y, metric, config)
    return optimized.params, {"cv_std": float(np.std(optimized.fold_scores))}
