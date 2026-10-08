"""Target-only reference evaluated on training CV, outside model selection."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from ..optimization.cv import CVSplits
from ..tasks import TaskType
from .metrics import MetricSpec


def evaluate_baseline(y_train: pd.Series, metric: MetricSpec, task: TaskType, folds: CVSplits) -> tuple[str, tuple[float, ...]]:
    """Fit a fresh dummy per fold; neither features nor holdout enter this function."""
    regression = task is TaskType.REGRESSION
    estimator_class = DummyRegressor if regression else DummyClassifier
    name = "dummy_regressor" if regression else "dummy_classifier"
    strategy = "mean" if regression else "prior"
    # A constant placeholder only supplies row counts/index to sklearn's API.
    X = pd.DataFrame({"_baseline": np.zeros(len(y_train))}, index=y_train.index)
    scores = []
    for training_rows, validation_rows in folds:
        estimator = estimator_class(strategy=strategy)
        estimator.fit(X.iloc[training_rows], y_train.iloc[training_rows])
        scores.append(metric.evaluate(estimator, X.iloc[validation_rows], y_train.iloc[validation_rows], task=task))
    return name, tuple(scores)
