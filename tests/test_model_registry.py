from dataclasses import replace

import pytest

from rationalml import ModelRegistry, TaskType
from rationalml.exceptions import MissingDependencyError
from rationalml.optimization.spaces import legacy_search_domains


def test_logistic_model_metadata_and_parameters():
    spec = ModelRegistry.get("logistic_regression")
    assert TaskType.BINARY in spec.tasks
    assert spec.supports_proba
    assert callable(spec.search_space)
    estimator = spec.estimator_class(**spec.parameters({"C": 2.0}, 19, 2))
    assert estimator.random_state == 19
    assert estimator.C == 2.0
    assert ModelRegistry.available(TaskType.REGRESSION) == []


@pytest.mark.parametrize("name, module", [("lightgbm", "lightgbm"), ("xgboost", "xgboost")])
def test_boosting_model_metadata(name, module):
    pytest.importorskip(module)
    spec = ModelRegistry.get(name)
    assert name in ModelRegistry.available(TaskType.BINARY)
    assert spec.supports_proba and spec.supports_early_stopping
    params = spec.parameters({}, 13, 2)
    assert params["random_state"] == 13
    assert params["n_jobs"] == 2


def test_missing_optional_dependency_is_explicit(monkeypatch):
    import rationalml.models.registry as registry_module

    def missing_module(name):
        raise ModuleNotFoundError(f"No module named {name}", name=name)

    monkeypatch.setattr(registry_module, "import_module", missing_module)
    with pytest.raises(MissingDependencyError, match="boosting"):
        ModelRegistry.get("lightgbm")


def test_registry_extension_and_unknown_names(monkeypatch):
    monkeypatch.setattr(ModelRegistry, "_specs", ModelRegistry._specs.copy())
    spec = replace(ModelRegistry.get("logistic_regression"), name="custom")
    ModelRegistry.register(spec)
    assert ModelRegistry.get("custom") is spec
    with pytest.raises(ValueError, match="already registered"):
        ModelRegistry.register(spec)
    with pytest.raises(KeyError, match="Unknown model"):
        ModelRegistry.get("unknown")


def test_legacy_domains_are_preserved_without_shared_mutation():
    first = legacy_search_domains()
    assert len(first) == 7
    first["xgb_gb"]["max_depth"] = (1, 1)
    first["skl_rs"]["penalty"].append("invalid")
    second = legacy_search_domains()
    assert second["xgb_gb"]["max_depth"] == (2, 20)
    assert "invalid" not in second["skl_rs"]["penalty"]
