import builtins
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest

from rationalml import AutoML, AutoMLResult, MLflowConfig
from rationalml.exceptions import ConfigurationError, MissingDependencyError


def test_mlflow_config_defaults_frozen_and_copied_tags():
    defaults = MLflowConfig()
    assert defaults.experiment_name == "RationalML"
    assert defaults.run_name is None and defaults.tracking_uri is None and defaults.tags is None
    assert defaults.log_model is True and defaults.log_predictions is False and defaults.nested is False
    tags = {"team": "risk", "number": np.int64(7), "flag": np.bool_(True), "missing": None}
    config = MLflowConfig(tags=tags)
    tags["team"] = "changed"
    assert config.tags == {"team": "risk", "number": "7", "flag": "True", "missing": "None"}
    with pytest.raises(FrozenInstanceError):
        config.nested = True


@pytest.mark.parametrize("field", ["experiment_name", "run_name"])
@pytest.mark.parametrize("value", ["", " ", 3, False, [], {}])
def test_mlflow_names_must_be_nonempty(field, value):
    with pytest.raises(ConfigurationError, match=field):
        MLflowConfig(**{field: value})


@pytest.mark.parametrize("field", ["log_model", "log_predictions", "nested"])
@pytest.mark.parametrize("value", [0, 1, None, "true", np.bool_(True)])
def test_mlflow_flags_are_bools(field, value):
    with pytest.raises(ConfigurationError, match=field):
        MLflowConfig(**{field: value})


@pytest.mark.parametrize("uri", ["", " ", 5, False, [], {}])
def test_mlflow_uri_validation(uri):
    with pytest.raises(ConfigurationError, match="tracking_uri"):
        MLflowConfig(tracking_uri=uri)


@pytest.mark.parametrize("uri", [None, "sqlite:///tracking.db", "http://127.0.0.1:5000", Path("tracking")])
def test_mlflow_uris_accept_none_strings_and_paths(uri):
    assert MLflowConfig(tracking_uri=uri).tracking_uri == uri


@pytest.mark.parametrize("tags", [[], "team", {1: "team"}, {"": "team"}, {" ": "team"},
                                   {"team": []}, {"team": {}}, {"team": object()}])
def test_mlflow_tags_are_simple_named_values(tags):
    with pytest.raises(ConfigurationError, match="[Tt]ag"):
        MLflowConfig(tags=tags)


@pytest.mark.parametrize("tag", ["rationalml.version", "rationalml.task", "rationalml.best_model",
                                  "rationalml.selected", "rationalml.model_name", "mlflow.parentRunId"])
def test_mlflow_technical_tags_cannot_be_overwritten(tag):
    with pytest.raises(ConfigurationError, match="reserved"):
        MLflowConfig(tags={tag: "override"})


def test_mlflow_dependency_error_is_lazy_and_actionable(binary_df, monkeypatch):
    result = AutoML(target="target", models="logistic_regression", n_trials=1, cv=2, verbose=0).fit(binary_df)
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ModuleNotFoundError("No module named 'mlflow'", name="mlflow")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(MissingDependencyError, match=r'MLflow tracking requires MLflow.*pip install "rationalml\[mlflow\]"'):
        result.log_mlflow()


def test_package_import_never_loads_mlflow():
    script = """
import builtins
import sys
original_import = builtins.__import__
def missing(name, *args, **kwargs):
    if name == 'mlflow' or name.startswith('mlflow.'):
        raise AssertionError('MLflow was imported eagerly')
    return original_import(name, *args, **kwargs)
builtins.__import__ = missing
from rationalml import AutoML, AutoMLResult, MLflowConfig
assert 'mlflow' not in sys.modules
assert MLflowConfig().log_predictions is False
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def test_model_best_params_default_preserves_manual_construction(binary_df):
    result = AutoML(target="target", models="logistic_regression", n_trials=1, cv=2, verbose=0).fit(binary_df)
    assert result.model_best_params == {result.best_model_name: result.best_params}
    assert result.model_best_params[result.best_model_name] is not result.best_params
    kwargs = {field.name: getattr(result, field.name) for field in fields(result) if field.name != "model_best_params"}
    first, second = AutoMLResult(**kwargs), AutoMLResult(**kwargs)
    first.model_best_params["manual"] = {}
    assert second.model_best_params == {}
