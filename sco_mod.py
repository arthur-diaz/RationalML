"""Legacy orchestration routed through the train-only V0.1 engine."""

import warnings

import pandas as pd

from rationalml import AutoML
from rationalml.legacy import legacy_model_registry
from rationalml.exceptions import LegacyAPIError
from strategy import local_optimizer


def sco_mod(init_model: local_optimizer, col_1=None, col_2=None) -> pd.DataFrame:
    """Return a raw CV leaderboard; fitted results live in init_model.results_."""
    warnings.warn("sco_mod is legacy; migrate to AutoML.fit.", DeprecationWarning, stacklevel=2)
    if init_model.excel_report or init_model.model_save is not None or init_model.model_select is not None:
        raise LegacyAPIError("Excel and persistence are outside V0.1; use AutoML.fit and its raw result.")
    if init_model.class_weight == "auto":
        raise LegacyAPIError("Legacy automatic class weights used full-data statistics; use a fixed class_weight.")
    warnings.warn(
        "Legacy thresholds, pruning and early stopping are retired in V0.1; predictions use the estimator defaults.",
        UserWarning, stacklevel=2,
    )
    registry = legacy_model_registry(init_model)
    results, tables = [], []
    for score in init_model.score:
        metric = "f1" if score == "f1_score" else score
        result = AutoML(
            target=init_model.target_name, metric=metric, models=init_model.algo,
            cv=init_model.nb_cv, n_trials=init_model.nb_iter,
            test_size=init_model.test_size, random_state=init_model.seed,
            n_jobs=init_model.n_jobs, timeout=init_model.timeout,
            verbose=init_model.verbose, model_registry=registry,
        ).fit(init_model.df)
        results.append(result)
        table = result.leaderboard.reset_index()
        table["metric"] = metric
        if col_1 is not None:
            table["subset_1"] = col_1
        if col_2 is not None:
            table["subset_2"] = col_2
        tables.append(table)
    if not results:
        raise ValueError("At least one scoring metric is required.")
    init_model.results_ = results
    init_model.result_ = results[-1]
    return pd.concat(tables, ignore_index=True)
