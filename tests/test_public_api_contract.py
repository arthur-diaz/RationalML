import inspect
from pathlib import Path
import re
import subprocess
import sys

import pytest

import rationalml
from rationalml import AutoML, AutoMLResult, ModelRegistry


@pytest.mark.parametrize("api, expected", [
    (AutoML, [("config", "POSITIONAL_OR_KEYWORD", None), ("model_registry", "KEYWORD_ONLY", ModelRegistry),
              ("options", "VAR_KEYWORD", inspect.Parameter.empty)]),
    (AutoML.fit, [("self", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
                  ("df", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty)]),
    *[(getattr(AutoMLResult, name), [("self", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
                                    ("X", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty)])
      for name in ("predict", "predict_proba", "predict_positive_proba")],
    *[(getattr(AutoMLResult, name), [("self", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
                                    ("n_bins", "POSITIONAL_OR_KEYWORD", 10), ("class_label", "KEYWORD_ONLY", None)])
      for name in ("ranking_table", "calibration_table", "calibration_summary")],
    (AutoMLResult.top_segment, [("self", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
                                ("fraction", "POSITIONAL_OR_KEYWORD", .1), ("class_label", "KEYWORD_ONLY", None)]),
    (AutoMLResult.to_excel, [("self", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
                             ("path", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
                             ("config", "KEYWORD_ONLY", None), ("class_label", "KEYWORD_ONLY", None)]),
    (AutoMLResult.log_mlflow, [("self", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
                               ("config", "KEYWORD_ONLY", None), ("class_label", "KEYWORD_ONLY", None)]),
    (AutoMLResult.explain, [("self", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
                            ("X", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
                            ("background", "KEYWORD_ONLY", inspect.Parameter.empty), ("class_label", "KEYWORD_ONLY", None)]),
])
def test_public_call_signatures(api, expected):
    parameters = inspect.signature(api).parameters.values()
    assert [(p.name, p.kind.name, p.default) for p in parameters] == expected


def test_root_exports_are_the_intended_public_api():
    expected = {"AutoML", "AutoMLConfig", "AutoMLResult", "TaskType", "ModelRegistry",
                "ModelSpec", "MetricRegistry", "MetricSpec", "PreprocessingConfig",
                "ExcelReportConfig", "MLflowConfig"}
    assert set(rationalml.__all__) == expected
    assert len(rationalml.__all__) == len(expected)
    assert all(hasattr(rationalml, name) for name in expected)


@pytest.mark.parametrize("removed, module", [
    ("normalize_task", "rationalml.tasks"), ("FeatureSchema", "rationalml.preprocessing"),
    ("infer_schema", "rationalml.preprocessing"), ("build_preprocessor", "rationalml.preprocessing"),
])
def test_advanced_helpers_live_only_in_their_submodules(removed, module):
    from importlib import import_module

    assert removed not in rationalml.__all__
    assert not hasattr(rationalml, removed)
    assert callable(getattr(import_module(module), removed))


def test_source_and_package_versions_agree():
    source = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    # stdlib tomllib is unavailable on Python 3.10; the project version is a literal.
    assert re.search(r'^version = "([^"]+)"$', source, re.MULTILINE).group(1) == rationalml.__version__


def test_import_root_does_not_load_any_optional_runtime():
    script = """
import sys
import rationalml
optional = ('openpyxl', 'mlflow', 'shap', 'lightgbm', 'xgboost')
assert not any(name.split('.')[0] in optional for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
