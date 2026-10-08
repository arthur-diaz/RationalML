"""Leakage-safe preprocessing of raw tabular features."""

from .builder import build_preprocessor
from .config import PreprocessingConfig
from .schema import FeatureSchema, infer_schema

__all__ = ["PreprocessingConfig", "FeatureSchema", "infer_schema", "build_preprocessor"]
