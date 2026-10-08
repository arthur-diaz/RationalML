"""Small, advanced configuration for the optional basic preprocessing mode."""

from dataclasses import dataclass

from ..exceptions import ConfigurationError


@dataclass(frozen=True)
class PreprocessingConfig:
    numeric_imputation: str = "median"
    categorical_imputation: str = "most_frequent"
    categorical_encoding: str = "onehot"
    scale_numeric: str | bool = "auto"
    handle_unknown: str = "ignore"

    def __post_init__(self) -> None:
        if self.numeric_imputation not in ("median", "mean", "most_frequent"):
            raise ConfigurationError("numeric_imputation must be 'median', 'mean' or 'most_frequent'.")
        if self.categorical_imputation != "most_frequent":
            raise ConfigurationError("categorical_imputation must be 'most_frequent' in V0.2.")
        if self.categorical_encoding != "onehot":
            raise ConfigurationError("categorical_encoding must be 'onehot' in V0.2.")
        if not (isinstance(self.scale_numeric, bool) or self.scale_numeric == "auto"):
            raise ConfigurationError("scale_numeric must be 'auto', True or False.")
        if self.handle_unknown not in ("ignore", "error"):
            raise ConfigurationError("handle_unknown must be 'ignore' or 'error'.")
