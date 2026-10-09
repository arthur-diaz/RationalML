from importlib import metadata

from rationalml import __version__


def test_installed_distribution_version_matches_public_version():
    assert metadata.version("RationalML") == __version__ == "0.10.0"


def test_distribution_keeps_optional_integrations_in_extras():
    distribution = metadata.metadata("RationalML")
    assert set(distribution.get_all("Provides-Extra")) == {"test", "legacy", "boosting", "excel", "mlflow", "shap"}
    requirements = metadata.requires("RationalML")
    for dependency in ("openpyxl", "mlflow", "shap", "lightgbm", "xgboost"):
        matching = [requirement for requirement in requirements if requirement.startswith(dependency)]
        assert len(matching) == 1 and 'extra ==' in matching[0]
