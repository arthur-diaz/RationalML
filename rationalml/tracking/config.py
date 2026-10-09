"""Explicit, small configuration for post-fit MLflow tracking."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import ConfigurationError


@dataclass(frozen=True)
class MLflowConfig:
    experiment_name: str = "RationalML"
    run_name: str | None = None
    tracking_uri: str | Path | None = None
    log_model: bool = True
    log_predictions: bool = False
    nested: bool = False
    tags: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("experiment_name", "run_name"):
            value = getattr(self, name)
            if name == "run_name" and value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name} must be a nonempty string" + (" or None." if name == "run_name" else "."))
        if self.tracking_uri is not None and (
            not isinstance(self.tracking_uri, (str, Path)) or not str(self.tracking_uri).strip()
        ):
            raise ConfigurationError("tracking_uri must be None, a nonempty string or a pathlib.Path.")
        for name in ("log_model", "log_predictions", "nested"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"{name} must be a bool.")
        if self.tags is not None:
            if not isinstance(self.tags, Mapping):
                raise ConfigurationError("tags must be a mapping with nonempty string keys and simple scalar values.")
            copied = {}
            for key, value in self.tags.items():
                if not isinstance(key, str) or not key.strip():
                    raise ConfigurationError("tags keys must be nonempty strings.")
                if key.startswith("rationalml.") or key == "mlflow.parentRunId":
                    raise ConfigurationError(f"Tag {key!r} is reserved for RationalML tracking.")
                if isinstance(value, np.generic):
                    value = value.item()
                if value is not None and not isinstance(value, (str, int, float, bool)):
                    raise ConfigurationError(f"Tag {key!r} must have a simple scalar value convertible to str.")
                copied[key] = str(value)
            object.__setattr__(self, "tags", copied)
