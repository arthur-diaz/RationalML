"""Holdout evaluation and an owned prediction artifact from the same calls."""

from typing import Any

import numpy as np
import pandas as pd

from ..data import BinaryLabelEncoder, MulticlassLabelEncoder
from ..tasks import TaskType
from .metrics import evaluate_metrics


def evaluate_holdout(
    estimator: Any, X_test: pd.DataFrame, y_test: pd.Series, original_target: pd.Series,
    task: TaskType, encoder: BinaryLabelEncoder | MulticlassLabelEncoder | None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Predict once per type after winner selection; preserve original row labels."""
    predictions = {"predict": np.asarray(estimator.predict(X_test))}
    if task is not TaskType.REGRESSION:
        predictions["proba"] = np.asarray(estimator.predict_proba(X_test))
    metrics = evaluate_metrics(estimator, X_test, y_test, task, predictions=predictions)
    table = pd.DataFrame({
        "y_true": original_target.to_numpy(copy=True),
        "y_pred": (predictions["predict"] if task is TaskType.REGRESSION
                   else encoder.inverse_transform(predictions["predict"])),
    }, index=X_test.index.copy())
    if task is TaskType.REGRESSION:
        table["residual"] = table["y_true"] - table["y_pred"]
    elif task is TaskType.BINARY:
        table["score"] = predictions["proba"][:, 1]
    else:
        for position in range(len(encoder.classes_)):
            table[f"proba_{position}"] = predictions["proba"][:, position]
    return metrics, table
