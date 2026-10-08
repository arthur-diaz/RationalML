"""Search spaces adapted from strategy.py/scoring.py, with legacy bounds preserved."""

from copy import deepcopy
from typing import Any

from optuna import Trial

_XGB_BOUNDS = {
    "learning_rate": (0.005, 0.5), "max_depth": (2, 20),
    "colsample_bytree": (0.1, 0.8), "subsample": (0.4, 1),
    "alpha": (1e-8, 1e3), "lambda": (1e-8, 1e3),
    "gamma": (1e-8, 1e2), "max_bin": (2, 400),
}
_LGB_BOUNDS = {
    "max_depth": (2, 20), "reg_alpha": (1e-8, 1e3),
    "reg_lambda": (1e-8, 1e3), "colsample_bytree": (0.1, 0.7),
    "subsample": (0.3, 1), "subsample_freq": (0, 15),
    "min_child_samples": (5, 100), "learning_rate": (0.005, 0.5),
    "min_split_gain": (1e-8, 1e2), "num_leaves": (16, 256),
}
_LR_BOUNDS = {
    "tol": (1e-4, 0.4), "C": (1e-8, 1e9), "max_iter": (100, 300),
    "l1_ratio": (0, 1), "penalty": ["l2", None],
}


def legacy_search_domains() -> dict[str, dict[str, Any]]:
    """Return independent copies of all seven historical search domains."""
    return deepcopy({
        "xgb_gb": _XGB_BOUNDS,
        "lgb_gb": _LGB_BOUNDS,
        "xgb_rf": {**_XGB_BOUNDS, "subsample": (0.3, 0.8), "alpha": (0.01, 1e3)},
        "lgb_rf": {
            **_LGB_BOUNDS, "subsample": (0.3, 0.8), "colsample_bytree": (0.4, 1.0),
            "subsample_freq": (1, 15),
        },
        "xgb_mx": {**_XGB_BOUNDS, "subsample": (0.4, 0.9), "num_parallel_tree": (2, 20)},
        "skl_rl": _LR_BOUNDS,
        "skl_rs": {**_LR_BOUNDS, "penalty": ["l2", "l1", "elasticnet", None]},
    })


def suggest_xgboost(trial: Trial, bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    p = bounds if bounds is not None else _XGB_BOUNDS
    params = {
        "learning_rate": trial.suggest_float("learning_rate", *p["learning_rate"]),
        "max_depth": trial.suggest_int("max_depth", *p["max_depth"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *p["colsample_bytree"], log=True),
        "subsample": trial.suggest_float("subsample", *p["subsample"], log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", *p["alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *p["lambda"], log=True),
        "gamma": trial.suggest_float("gamma", *p["gamma"], log=True),
        "max_bin": trial.suggest_int("max_bin", *p["max_bin"]),
    }
    if "num_parallel_tree" in p:
        params["num_parallel_tree"] = trial.suggest_int("num_parallel_tree", *p["num_parallel_tree"])
    return params


def suggest_lightgbm(trial: Trial, bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    p = bounds if bounds is not None else _LGB_BOUNDS
    return {
        name: (
            trial.suggest_int(name, *p[name]) if name in {
                "max_depth", "subsample_freq", "min_child_samples", "num_leaves"
            } else trial.suggest_float(name, *p[name], log=name in {"reg_alpha", "reg_lambda"})
        )
        for name in p
    }


def suggest_logistic_regression(trial: Trial, bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    # Historical C covered 17 orders of magnitude. The new default remains
    # tunable but avoids near-unregularized, poorly converging configurations.
    p = bounds if bounds is not None else {
        **_LR_BOUNDS, "C": (1e-4, 1e4), "tol": (1e-4, 1e-2), "max_iter": (500, 1000)
    }
    return {
        "C": trial.suggest_float("C", *p["C"], log=True),
        "tol": trial.suggest_float("tol", *p["tol"], log=True),
        "max_iter": trial.suggest_int("max_iter", *p["max_iter"]),
    }


def suggest_ridge(trial: Trial) -> dict[str, Any]:
    """A single regularization parameter on a logarithmic scale."""
    return {"alpha": trial.suggest_float("alpha", 1e-4, 1e4, log=True)}
