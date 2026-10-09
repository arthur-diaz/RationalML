"""SHAP on fitted estimators and explicitly supplied data; no fitting or storage."""

import json
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge

from ..exceptions import AutoMLError, ConfigurationError, DataValidationError, MissingDependencyError
from ..tasks import TaskType

if TYPE_CHECKING:
    from shap import Explanation
    from ..result import AutoMLResult


def _class_output(result: "AutoMLResult", class_label: object | None) -> tuple[int, str | None]:
    if result.task is TaskType.MULTICLASS:
        if class_label is None:
            raise ConfigurationError("Multiclass SHAP explanations require class_label.")
        labels = result.classes_.tolist()
        try:
            position = labels.index(class_label)
        except (ValueError, TypeError) as error:
            raise ConfigurationError(f"Unknown class_label={class_label!r}; available classes: {labels!r}.") from error
        return position, str(labels[position])
    if class_label is not None:
        raise ConfigurationError("class_label is only used for multiclass SHAP explanations.")
    return (1, str(result.positive_class)) if result.task is TaskType.BINARY else (0, None)


def _matrix(values: object) -> np.ndarray | sparse.csr_matrix:
    """Copy numeric input without densifying sparse matrices."""
    if isinstance(values, pd.DataFrame):
        values = values.to_numpy()
    if getattr(getattr(values, "dtype", None), "kind", None) == "c":
        raise DataValidationError("SHAP transformed features must be real numeric values.")
    try:
        matrix = (sparse.csr_matrix(values, dtype=float, copy=True) if sparse.issparse(values)
                  else np.array(values, dtype=float, copy=True))
    except (TypeError, ValueError) as error:
        raise DataValidationError("SHAP transformed features must be a numeric 2D matrix.") from error
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise DataValidationError("SHAP transformed features must be a nonempty 2D matrix.")
    return matrix


def _single_output(
    explanation: "Explanation", shape: tuple[int, int], task: TaskType, position: int, n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize modern SHAP single/multiple outputs to values 2D and bases 1D."""
    values, bases = np.asarray(explanation.values), np.asarray(explanation.base_values)
    n_rows, n_features = shape
    if values.ndim == 3:
        n_outputs = values.shape[2]
        expected = n_classes if task is TaskType.MULTICLASS else (2 if task is TaskType.BINARY else 1)
        if n_outputs != expected and not (task is TaskType.BINARY and n_outputs == 1):
            raise ConfigurationError("SHAP outputs do not match the RationalML task/classes.")
        selected = 0 if n_outputs == 1 else position
        values = values[:, :, selected]
        if bases.shape == (n_rows, n_outputs):
            bases = bases[:, selected]
        elif bases.shape in {(n_outputs,), (1, n_outputs)}:
            bases = np.full(n_rows, bases.reshape(-1)[selected])
        else:
            raise ConfigurationError("SHAP base values do not match the output dimensions.")
    elif values.ndim != 2 or task is TaskType.MULTICLASS:
        raise ConfigurationError("SHAP did not return the expected feature/class dimensions.")
    if values.shape != (n_rows, n_features):
        raise ConfigurationError("SHAP values do not match the transformed feature dimensions.")
    if bases.ndim == 0 or bases.shape == (1,):
        bases = np.full(n_rows, bases.item())
    elif bases.shape in {(n_rows,), (n_rows, 1)}:
        bases = bases.reshape(n_rows)
    else:
        raise ConfigurationError("SHAP base values do not match the explained rows.")
    return values.copy(), bases.copy()


def explain(
    result: "AutoMLResult", X: object, *, background: object, class_label: object | None = None,
) -> "Explanation":
    """Explain one raw model output with the full explicit background, post-fit."""
    try:
        import shap
    except ImportError as error:
        raise MissingDependencyError(
            'SHAP explainability requires SHAP. Install it with: pip install "rationalml[shap]".'
        ) from error
    position, output_name = _class_output(result, class_label)
    features, reference = result._features(X), result._features(background)
    if features.empty or reference.empty:
        raise DataValidationError("SHAP X and background must both contain at least one row.")
    estimator = result.best_model.named_steps["estimator"]
    linear = isinstance(estimator, (LogisticRegression, Ridge))
    origins = {cls.__module__.split(".")[0] for cls in type(estimator).__mro__}
    if not linear and not origins & {"lightgbm", "xgboost"}:
        raise ConfigurationError(f"SHAP has no supported explainer for model {type(estimator).__name__!r}; no automatic fallback is used.")
    preprocessor = result.best_model.named_steps.get("preprocessing")
    transformed = _matrix(features if preprocessor is None else preprocessor.transform(features))
    transformed_background = _matrix(reference if preprocessor is None else preprocessor.transform(reference))
    if transformed.shape[1] != transformed_background.shape[1]:
        raise DataValidationError("SHAP X and background have different transformed feature counts.")
    if not linear and (sparse.issparse(transformed) or sparse.issparse(transformed_background)):
        raise ConfigurationError("SHAP TreeExplainer requires dense transformed X and background; sparse matrices are not densified automatically.")
    if "xgboost" in origins:
        # SHAP 0.49.x cannot parse XGBoost >=3.1's vector-valued intercepts.
        base_score = json.loads(estimator.get_booster().save_config())["learner"]["learner_model_param"]["base_score"]
        if isinstance(base_score, list) or str(base_score).lstrip().startswith("["):
            raise ConfigurationError("SHAP 0.49.x does not support XGBoost vector base_score (XGBoost >=3.1). Use XGBoost>=2,<3.1 for explain(); training and prediction remain supported.")
    names = result.transformed_feature_names if preprocessor is not None else result.feature_names
    if names is None or len(names) != transformed.shape[1]:
        names = tuple(f"feature_{i}" for i in range(transformed.shape[1]))
        warnings.warn("Transformed feature names are unavailable; SHAP uses positional feature names.", UserWarning, stacklevel=3)
    # Independent defaults to 100 rows; setting this explicitly preserves ALL
    # reference observations instead of silently changing the chosen background.
    masker = shap.maskers.Independent(transformed_background, max_samples=transformed_background.shape[0])
    try:
        explainer = (shap.LinearExplainer(estimator, masker, feature_names=list(names)) if linear else
                     shap.TreeExplainer(estimator, data=masker, model_output="raw",
                                        feature_perturbation="interventional", feature_names=list(names)))
        explanation = explainer(transformed)
    except AutoMLError:
        raise
    except Exception as error:
        raise ConfigurationError(f"SHAP cannot explain this estimator/transformed-data combination: {error}") from error
    n_classes = len(result.classes_) if result.task is TaskType.MULTICLASS else 0
    values, bases = _single_output(explanation, transformed.shape, result.task, position, n_classes)
    return shap.Explanation(values=values, base_values=bases, data=transformed.copy(),
                            feature_names=list(names), output_names=output_name)
