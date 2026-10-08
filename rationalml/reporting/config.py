"""Small presentation-only configuration for Excel reports."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from ..exceptions import ConfigurationError


@dataclass(frozen=True)
class ExcelReportConfig:
    decimal_places: int = 3
    percentage_places: int = 2
    include_predictions: bool = False
    top_fraction: float = .10

    def __post_init__(self) -> None:
        for name in ("decimal_places", "percentage_places"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 15:
                raise ConfigurationError(f"{name} must be an integer between 0 and 15, excluding bool.")
        if not isinstance(self.include_predictions, bool):
            raise ConfigurationError("include_predictions must be a bool.")
        if (isinstance(self.top_fraction, bool) or not isinstance(self.top_fraction, Real)
                or not isfinite(self.top_fraction) or not 0 < self.top_fraction <= 1):
            raise ConfigurationError("top_fraction must satisfy 0 < top_fraction <= 1.")
