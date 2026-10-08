"""Retired legacy reports; raw results now come from AutoMLResult.

These signatures carry no reliable provenance for optimized models or test
partitions. Reject them explicitly instead of reproducing data leakage.
"""

from rationalml.exceptions import LegacyAPIError


def prediction(classifier, X, Y, model, lib, init_model, start, start_datetime, auc_var, item) -> dict:
    raise LegacyAPIError("prediction is retired: use AutoML.fit(...).test_metrics and .feature_importance.")


def prediction_next(classifier, X, Y, model, lib, init_model, start, start_datetime, auc_var, item, i, j) -> dict:
    raise LegacyAPIError("prediction_next is retired: use second_step or AutoML.fit on an explicit segment.")


def report_global(init_model, i, j) -> dict:
    raise LegacyAPIError("Saved-model test provenance is unknown; report_global is outside V0.1.")


def domain_stat(init_model) -> tuple[int, float]:
    """Historical positive-label statistics without mutating the source data."""
    _, y = init_model._pandas_sets()
    positive = int((y == 1).sum())
    return positive, positive / len(y)
