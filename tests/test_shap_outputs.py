from types import SimpleNamespace

import numpy as np
import pytest

from rationalml.exceptions import ConfigurationError
from rationalml.explainability.shap import _single_output
from rationalml.tasks import TaskType


@pytest.mark.parametrize("task,position,n_outputs", [(TaskType.BINARY, 1, 2), (TaskType.BINARY, 1, 1),
    (TaskType.MULTICLASS, 0, 3), (TaskType.MULTICLASS, 2, 3), (TaskType.REGRESSION, 0, 1)])
@pytest.mark.parametrize("constant_base", [False, True])
def test_modern_output_axes_select_one_class_and_matching_base(task, position, n_outputs, constant_base):
    values = np.arange(4 * 2 * n_outputs, dtype=float).reshape(4, 2, n_outputs)
    bases = np.arange(n_outputs, dtype=float) if constant_base else np.arange(4 * n_outputs, dtype=float).reshape(4, n_outputs)
    original = SimpleNamespace(values=values, base_values=bases)
    selected_values, selected_bases = _single_output(original, (4, 2), task, position, 3)
    selected = 0 if n_outputs == 1 else position
    np.testing.assert_array_equal(selected_values, values[:, :, selected])
    np.testing.assert_array_equal(selected_bases, np.repeat(bases[selected], 4) if constant_base else bases[:, selected])
    assert selected_values.shape == (4, 2) and selected_bases.shape == (4,)
    selected_values[0, 0] = -999
    selected_bases[0] = -999
    assert values.min() >= 0 and bases.min() >= 0


@pytest.mark.parametrize("base", [2., np.array([2.]), np.array([2., 3., 4., 5.]), np.array([[2.], [3.], [4.], [5.]])])
@pytest.mark.parametrize("task", [TaskType.BINARY, TaskType.REGRESSION])
def test_single_raw_output_base_scalar_or_per_row(task, base):
    e = SimpleNamespace(values=np.ones((4, 2)), base_values=base)
    values, bases = _single_output(e, (4, 2), task, 1 if task is TaskType.BINARY else 0, 0)
    assert values.shape == (4, 2) and bases.shape == (4,)
    np.testing.assert_array_equal(bases, [2] * 4 if np.asarray(base).size == 1 else [2, 3, 4, 5])


@pytest.mark.parametrize("values,bases,task", [
    (np.zeros((4, 2)), np.zeros(4), TaskType.MULTICLASS),
    (np.zeros((4, 2, 2)), np.zeros((4, 2)), TaskType.MULTICLASS),
    (np.zeros((4, 2, 3)), np.zeros((4, 3)), TaskType.BINARY),
    (np.zeros((4, 2, 2)), np.zeros((4, 2)), TaskType.REGRESSION),
    (np.zeros((4, 2, 3)), np.zeros(4), TaskType.MULTICLASS),
    (np.zeros((4, 2)), np.zeros(3), TaskType.REGRESSION),
    (np.zeros((4, 3)), np.zeros(4), TaskType.BINARY),
    (np.zeros((4, 2, 1, 1)), np.zeros(4), TaskType.BINARY),
])
def test_ambiguous_or_inconsistent_output_shapes_are_explicit(values, bases, task):
    with pytest.raises(ConfigurationError, match="SHAP"):
        _single_output(SimpleNamespace(values=values, base_values=bases), (4, 2), task, 1, 3)
