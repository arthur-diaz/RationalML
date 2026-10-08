import builtins
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from rationalml import AutoML, ExcelReportConfig
from rationalml.exceptions import ConfigurationError, MissingDependencyError


def test_excel_dependency_missing_has_actionable_error(binary_df, monkeypatch, tmp_path):
    result = AutoML(target="target", models="logistic_regression", n_trials=1, cv=2, verbose=0).fit(binary_df)
    original_import = builtins.__import__

    def without_openpyxl(name, *args, **kwargs):
        if name == "openpyxl" or name.startswith("openpyxl."):
            raise ModuleNotFoundError("No module named 'openpyxl'", name="openpyxl")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_openpyxl)
    with pytest.raises(MissingDependencyError, match=r'Excel export requires openpyxl.*pip install "rationalml\[excel\]"'):
        result.to_excel(tmp_path / "report.xlsx")
    assert list(tmp_path.iterdir()) == []


def test_package_import_does_not_import_openpyxl():
    script = """
import builtins
import sys
original_import = builtins.__import__
def without_openpyxl(name, *args, **kwargs):
    if name == 'openpyxl' or name.startswith('openpyxl.'):
        raise AssertionError('Optional Excel dependency imported eagerly')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_openpyxl
from rationalml import AutoML, AutoMLResult, ExcelReportConfig
assert 'openpyxl' not in sys.modules
assert ExcelReportConfig().include_predictions is False
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def test_excel_config_defaults_and_immutability():
    config = ExcelReportConfig()
    assert (config.decimal_places, config.percentage_places, config.include_predictions, config.top_fraction) == (3, 2, False, .1)
    with pytest.raises(FrozenInstanceError):
        config.include_predictions = True


@pytest.mark.parametrize("field", ["decimal_places", "percentage_places"])
@pytest.mark.parametrize("value", [True, False, -1, 16, 3.0, "3", None])
def test_excel_config_validates_places(field, value):
    with pytest.raises(ConfigurationError, match=field):
        ExcelReportConfig(**{field: value})


@pytest.mark.parametrize("fraction", [True, False, 0, -1, 1.01, float("nan"), float("inf"), ".1", None])
def test_excel_config_validates_fraction(fraction):
    with pytest.raises(ConfigurationError, match="top_fraction"):
        ExcelReportConfig(top_fraction=fraction)


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_excel_config_validates_predictions_flag(value):
    with pytest.raises(ConfigurationError, match="include_predictions"):
        ExcelReportConfig(include_predictions=value)
