import builtins
import os
import subprocess
import sys

import numpy as np
import pytest

from rationalml import AutoML
from rationalml.exceptions import DataValidationError, MissingDependencyError


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_missing_shap_is_actionable_without_affecting_fit(request, monkeypatch, task):
    df = request.getfixturevalue(f"{task}_df")
    result = AutoML(target="target", task=task, models="ridge" if task == "regression" else "logistic_regression",
                    n_trials=1, cv=2, verbose=0).fit(df)
    original_import = builtins.__import__

    def without_shap(name, *args, **kwargs):
        if name == "shap" or name.startswith("shap."):
            raise ModuleNotFoundError("No module named 'shap'", name="shap")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_shap)
    X = df.drop(columns="target").iloc[:3]
    label = "gold" if task == "multiclass" else None
    with pytest.raises(MissingDependencyError) as caught:
        result.explain(X, background=X, class_label=label)
    assert str(caught.value) == 'SHAP explainability requires SHAP. Install it with: pip install "rationalml[shap]".'


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
def test_import_and_fit_never_import_shap_or_retain_feature_datasets(task, mode):
    script = '''
import builtins
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.impute import SimpleImputer
original_import = builtins.__import__
def without_shap(name, *args, **kwargs):
    if name == "shap" or name.startswith("shap."):
        raise AssertionError("SHAP was imported outside result.explain")
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_shap
from rationalml import AutoML, AutoMLResult
assert "shap" not in sys.modules
task, mode = sys.argv[1:]
if task == "regression":
    X, y = make_regression(n_samples=60, n_features=3, random_state=17)
else:
    X, y = make_classification(n_samples=60, n_features=4, n_informative=3,
        n_redundant=0, n_classes=3 if task == "multiclass" else 2, n_clusters_per_class=1, random_state=17)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["target"] = y
preprocessing = SimpleImputer() if mode == "custom" else (None if mode == "None" else mode)
result = AutoML(target="target", task=task, preprocessing=preprocessing, n_trials=1, cv=2,
    models="ridge" if task == "regression" else "logistic_regression", verbose=0).fit(df)
assert "shap" not in sys.modules
assert "rationalml.explainability.shap" not in sys.modules
for name in ["X_train", "X_test", "df", "X", "background", "background_", "explainer_", "explanation_", "shap_values_"]:
    assert not hasattr(result, name), name
tables = [value for value in vars(result).values() if isinstance(value, pd.DataFrame)]
assert all(not set(df.columns.drop("target")).intersection(table.columns) for table in tables)
'''
    subprocess.run([sys.executable, "-c", script, task, str(mode)], check=True, capture_output=True, text=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def test_background_is_a_required_keyword_and_never_inferred(binary_df):
    pytest.importorskip("shap")
    result = AutoML(target="target", models="logistic_regression", n_trials=1, cv=2, verbose=0).fit(binary_df)
    X = binary_df.drop(columns="target").iloc[:3]
    with pytest.raises(TypeError, match="background"):
        result.explain(X)
    with pytest.raises(TypeError):
        result.explain(X, X)
    for background in [None, np.empty((0, len(result.feature_names)))]:
        with pytest.raises(DataValidationError):
            result.explain(X, background=background)
