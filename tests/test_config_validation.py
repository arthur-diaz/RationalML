from dataclasses import asdict

import pytest

from rationalml import AutoML, AutoMLConfig, TaskType
from rationalml.exceptions import ConfigurationError, UnsupportedTaskError


@pytest.mark.parametrize("options", [
    {"target": ""}, {"target": " "}, {"target": None},
    {"test_size": 0}, {"test_size": 1}, {"test_size": float("nan")},
    {"test_size": "0.2"}, {"test_size": True},
    {"cv": 1}, {"cv": 2.5}, {"cv": True},
    {"n_trials": 0}, {"n_trials": False},
    {"n_jobs": 0}, {"n_jobs": -2}, {"n_jobs": 1.5}, {"n_jobs": True},
    {"timeout": 0}, {"timeout": -1}, {"timeout": True}, {"timeout": 1.5},
    {"random_state": -1}, {"random_state": 2**32}, {"random_state": True},
    {"verbose": -1}, {"verbose": False},
    {"metric": ""}, {"models": []}, {"models": ""},
    {"models": ["logistic_regression", "LOGISTIC_REGRESSION"]},
    {"models": ["auto"]}, {"models": [1]}, {"models": None},
])
def test_invalid_configuration(options):
    with pytest.raises(ConfigurationError):
        AutoMLConfig(**({"target": "target"} | options))


def test_config_contains_only_options_and_normalizes_names():
    names = [" Logistic_Regression "]
    config = AutoMLConfig(target="target", task=" BINARY ", models=names, metric=" ROC_AUC ", n_jobs=-1)
    assert config.task is TaskType.BINARY
    assert config.models == ["logistic_regression"]
    assert config.metric == "roc_auc"
    assert names == [" Logistic_Regression "]
    assert "df" not in asdict(config)
    assert config.test_size == 0.2


def test_unknown_task_is_explicit():
    with pytest.raises(UnsupportedTaskError):
        AutoMLConfig(target="target", task="unknown")


def test_config_can_be_passed_directly_and_is_copied():
    config = AutoMLConfig(target="target", models=["logistic_regression"])
    automl = AutoML(config)
    config.models.append("xgboost")
    assert automl.config.models == ["logistic_regression"]
    with pytest.raises(ConfigurationError):
        AutoML(config, cv=2)
