"""Explicit configuration, data and migration errors."""


class AutoMLError(Exception):
    """Base error raised by RationalML."""


class ConfigurationError(AutoMLError, ValueError):
    """A configuration value is invalid."""


class DataValidationError(AutoMLError, ValueError):
    """Input data cannot be used without explicit preparation."""


class UnsupportedTaskError(AutoMLError, ValueError):
    """The requested task is ambiguous or not implemented."""


class MissingDependencyError(AutoMLError, ImportError):
    """An explicitly requested optional dependency is missing."""


class OptimizationError(AutoMLError, RuntimeError):
    """Optimization did not produce a usable trial."""


class LegacyAPIError(AutoMLError, ValueError):
    """An old entry point cannot guarantee isolation of the test set."""
