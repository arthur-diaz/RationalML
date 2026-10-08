"""Construct fresh sklearn pipelines; nothing here fits on validation or test."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import quote
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ..exceptions import ConfigurationError
from .config import PreprocessingConfig
from .schema import FeatureSchema, infer_schema

if TYPE_CHECKING:
    from ..models.base import ModelSpec


def _safe_name(name: object) -> str:
    # XGBoost forbids [, ] and < in feature names. Escaping is deterministic.
    return quote(str(name), safe="")


def _normalized_names(transformer: FunctionTransformer, names: Any) -> np.ndarray:
    return np.asarray([_safe_name(name) for name in names], dtype=object)


def _numeric_values(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        X.to_numpy(dtype=float, na_value=np.nan), index=X.index,
        columns=[_safe_name(name) for name in X.columns],
    )


def _categorical_values(X: pd.DataFrame) -> pd.DataFrame:
    values = X.astype(object).copy()
    values = values.where(values.notna(), np.nan)
    values.columns = [_safe_name(name) for name in X.columns]
    return values


def _category_name(feature: str, category: object) -> str:
    # Escaping underscores in categories avoids collisions across raw features.
    return f"{feature}_{_safe_name(category).replace('_', '%5F')}"


def build_preprocessor(
    schema: FeatureSchema, config: PreprocessingConfig | None = None, *, requires_scaling: bool = False,
) -> ColumnTransformer:
    """Return an UNFITTED transformer, with dense pandas output for all models."""
    config = config or PreprocessingConfig()
    scale = requires_scaling if config.scale_numeric == "auto" else config.scale_numeric
    transformers = []
    if schema.numeric:
        steps = [
            ("normalize", FunctionTransformer(_numeric_values, feature_names_out=_normalized_names)),
            ("imputer", SimpleImputer(strategy=config.numeric_imputation, keep_empty_features=True)),
        ]
        if scale:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), list(schema.numeric)))
    if schema.boolean:
        transformers.append(("boolean", Pipeline([
            ("normalize", FunctionTransformer(_numeric_values, feature_names_out=_normalized_names)),
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ]), list(schema.boolean)))
    if schema.categorical:
        transformers.append(("categorical", Pipeline([
            ("normalize", FunctionTransformer(_categorical_values, feature_names_out=_normalized_names)),
            ("imputer", SimpleImputer(strategy=config.categorical_imputation, keep_empty_features=True)),
            ("encoder", OneHotEncoder(
                handle_unknown=config.handle_unknown, sparse_output=False,
                feature_name_combiner=_category_name,
            )),
        ]), list(schema.categorical)))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0).set_output(transform="pandas")


def build_model_pipeline(
    spec: "ModelSpec", params: dict[str, Any], X_train: pd.DataFrame,
    preprocessing: str | PreprocessingConfig | BaseEstimator | None,
) -> Pipeline:
    """Create a fresh estimator and, only when requested, a fresh transformer."""
    schema = None
    steps = []
    if isinstance(preprocessing, PreprocessingConfig) or (
        isinstance(preprocessing, str) and preprocessing == "basic"
    ):
        schema = infer_schema(X_train)
        transformer = build_preprocessor(
            schema, preprocessing if isinstance(preprocessing, PreprocessingConfig) else None,
            requires_scaling=spec.requires_scaling,
        )
        steps.append(("preprocessing", transformer))
    elif preprocessing is not None:
        transformer = clone(preprocessing)
        if transformer is preprocessing:
            raise ConfigurationError("Custom preprocessing must clone to a distinct unfitted instance for each fold.")
        steps.append(("preprocessing", transformer))
    steps.append(("estimator", spec.estimator_class(**params)))
    pipeline = Pipeline(steps)
    pipeline.feature_schema_ = schema
    return pipeline


def transformed_feature_info(
    pipeline: Pipeline, feature_names: Sequence[str],
) -> tuple[tuple[str, ...] | None, tuple[str | None, ...] | None]:
    """Use reliable output names; custom feature lineage is left unspecified."""
    schema = pipeline.feature_schema_
    preprocessor = pipeline.named_steps.get("preprocessing")
    if preprocessor is None:
        return tuple(feature_names), tuple(feature_names)
    if schema is None:
        try:
            getter = getattr(preprocessor, "get_feature_names_out", None)
            if not callable(getter):
                raise ValueError("get_feature_names_out is unavailable")
            try:
                output = getter(list(feature_names))
            except TypeError:
                output = getter()
            output = np.asarray(output, dtype=object)
            if output.ndim != 1 or not len(output) or any(not isinstance(name, str) or not name for name in output):
                raise ValueError("transformed names must be a nonempty vector of strings")
            names = tuple(output)
            count = getattr(pipeline.named_steps["estimator"], "n_features_in_", len(names))
            if len(names) != count or len(set(names)) != len(names):
                raise ValueError("transformed names do not match the fitted features or are duplicated")
        except Exception as error:
            warnings.warn(
                "Custom preprocessing transformed feature names are unavailable; "
                f"feature_importance is unavailable ({error}).",
                UserWarning, stacklevel=2,
            )
            return None, None
        return names, (None,) * len(names)
    names = tuple(str(name) for name in preprocessor.get_feature_names_out())
    sources = [""] * len(names)
    for branch, columns in (("numeric", schema.numeric), ("boolean", schema.boolean)):
        if columns:
            sources[preprocessor.output_indices_[branch]] = columns
    if schema.categorical:
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        sources[preprocessor.output_indices_["categorical"]] = [
            name for name, categories in zip(schema.categorical, encoder.categories_)
            for _ in categories
        ]
    return names, tuple(sources)
