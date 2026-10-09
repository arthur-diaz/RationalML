import optuna
import pytest

from rationalml import ModelRegistry, TaskType
from rationalml.optimization.spaces import legacy_search_domains, suggest_lightgbm


BOUNDS = {
    "max_depth": (2, 12), "num_leaves": (4, 128), "learning_rate": (0.01, 0.2),
    "reg_alpha": (1e-8, 10.0), "reg_lambda": (1e-8, 10.0),
    "min_split_gain": (0.0, 0.1), "min_child_samples": (5, 50),
    "colsample_bytree": (0.5, 1.0), "subsample": (0.6, 1.0), "subsample_freq": (1, 5),
}


@pytest.mark.parametrize("depth", [2, 3, 7, 12])
def test_lightgbm_distributions_and_structural_leaf_capacity(depth):
    study = optuna.create_study()
    study.enqueue_trial({"max_depth": depth})
    trial = study.ask()
    params = suggest_lightgbm(trial)
    assert set(params) == set(BOUNDS)
    for name, (low, high) in BOUNDS.items():
        distribution = trial.distributions[name]
        assert distribution.low == low
        assert distribution.high == (min(high, 2**depth) if name == "num_leaves" else high)
        assert low <= params[name] <= high
        assert distribution.log == (name in {"learning_rate", "num_leaves", "reg_alpha", "reg_lambda"})
    assert isinstance(trial.distributions["min_child_samples"], optuna.distributions.IntDistribution)
    assert trial.distributions["min_child_samples"].step == 1
    assert params["num_leaves"] <= 2**params["max_depth"]
    assert "class_weight" not in params and "scale_pos_weight" not in params


@pytest.mark.parametrize("task, model", [(TaskType.BINARY, "lightgbm"), (TaskType.MULTICLASS, "lightgbm"),
                                        (TaskType.REGRESSION, "lightgbm_regressor")])
def test_sampled_lightgbm_parameters_construct_and_fit_for_all_tasks(request, task, model):
    pytest.importorskip("lightgbm")
    spec = ModelRegistry.get(model)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    df = request.getfixturevalue(task.value + "_df")
    X, y = df.drop(columns="target"), df.target
    if task == TaskType.MULTICLASS:
        y = y.astype("category").cat.codes
    draws = []
    for _ in range(12):
        trial = study.ask()
        suggested = spec.search_space(trial)
        estimator = spec.estimator_class(**spec.parameters(suggested, 42, 1, task=task))
        estimator.fit(X, y)
        assert len(estimator.predict(X.iloc[:3])) == 3
        draws.append(suggested)
        study.tell(trial, 0.)  # No dataset performance criterion influences these draws.
    assert len({tuple(sorted(p.items())) for p in draws}) == len(draws)
    assert len({p["max_depth"] for p in draws}) >= 3
    assert len({p["num_leaves"] for p in draws}) >= 3


def test_explicit_legacy_lightgbm_bounds_and_sampling_remain_unchanged():
    bounds = legacy_search_domains()["lgb_gb"]
    assert bounds["reg_alpha"] == (1e-8, 1e3)
    assert bounds["min_split_gain"] == (1e-8, 1e2)
    assert bounds["min_child_samples"] == (5, 100)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    trial = study.ask()
    suggest_lightgbm(trial, bounds)
    for name, (low, high) in bounds.items():
        assert (trial.distributions[name].low, trial.distributions[name].high) == (low, high)
        assert trial.distributions[name].log == (name in {"reg_alpha", "reg_lambda"})
