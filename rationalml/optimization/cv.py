"""One deterministic set of train-only CV splits, shared by all candidates."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold, StratifiedKFold

from ..config import AutoMLConfig
from ..tasks import TaskType

CVSplits = tuple[tuple[NDArray[np.intp], NDArray[np.intp]], ...]


def make_cv_splits(X: pd.DataFrame, y: pd.Series, task: TaskType, config: AutoMLConfig) -> CVSplits:
    """Return positional fold indices using the existing V0.3 task rules."""
    splitter = KFold if task is TaskType.REGRESSION else StratifiedKFold
    return tuple(splitter(
        n_splits=config.cv, shuffle=True, random_state=config.random_state,
    ).split(X, y))
