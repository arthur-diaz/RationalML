"""Offline distribution audit and tag/license guards; never publishes anything."""

import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tarfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_API = {
    "AutoML", "AutoMLConfig", "AutoMLResult", "TaskType",
    "MetricRegistry", "MetricSpec", "ModelRegistry", "ModelSpec",
    "PreprocessingConfig",
    "ExcelReportConfig", "MLflowConfig",
}
LEGACY_MODULES = {"strategy.py", "sco_mod.py", "scoring.py", "report.py", "metrics.py", "second_step.py"}
REQUIREMENTS = {
    None: {"numpy>=1.24,<3", "pandas>=2,<4", "scikit-learn>=1.5,<2", "optuna>=4,<6", "threadpoolctl>=3.5,<4"},
    "boosting": {"lightgbm>=4,<5", "xgboost>=2,<4"}, "legacy": {"tqdm>=4"},
    "excel": {"openpyxl>=3.0.10,<4"}, "mlflow": {"mlflow>=3.1,<4"},
    "shap": {"shap>=0.49.1,<0.50"}, "test": {"pytest>=8,<10"},
}


def project_version():
    source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version = "([^"]+)"$', source, re.MULTILINE).group(1)


def check_tag(tag, version):
    if tag != "v" + version:
        raise ValueError(f"Release tag {tag!r} must equal {'v' + version!r}.")


def require_publication_license(metadata):
    if (metadata.get_all("License-Expression") != ["Apache-2.0"]
            or metadata.get_all("License-File") != ["LICENSE"]):
        raise ValueError("Release license metadata must be Apache-2.0 with LICENSE.")


def _requirement(value):
    name, specs = re.fullmatch(r"([A-Za-z0-9_-]+)(.*)", value.strip()).groups()
    return name.lower().replace("_", "-"), frozenset(specs.replace(" ", "").split(","))


def audit_wheel(wheel):
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Wheel contains duplicate archive entries.")
        candidates = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(candidates) != 1:
            raise ValueError("Wheel must contain exactly one distribution metadata file.")
        metadata = BytesParser().parsebytes(archive.read(candidates[0]))
        if (metadata["Name"] != "RationalML" or metadata["Version"] != project_version()
                or metadata["Requires-Python"] != ">=3.10" or metadata["Description-Content-Type"] != "text/markdown"):
            raise ValueError("Wheel identity/Python/README metadata do not match the project.")
        actual = {extra: set() for extra in REQUIREMENTS}
        for requirement in metadata.get_all("Requires-Dist", []):
            base, _, marker = requirement.partition(";")
            match = re.fullmatch(r"\s*extra == [\"']([^\"']+)[\"']\s*", marker) if marker else None
            if marker and match is None:
                raise ValueError(f"Unexpected dependency marker: {marker}")
            extra = match.group(1) if match else None
            if extra not in actual:
                raise ValueError(f"Unexpected extra: {extra}")
            actual[extra].add(_requirement(base))
        expected = {extra: {_requirement(value) for value in values} for extra, values in REQUIREMENTS.items()}
        if actual != expected or set(metadata.get_all("Provides-Extra", [])) != set(REQUIREMENTS) - {None}:
            raise ValueError("Core dependencies or optional extras differ from the V1 contract.")
        urls = set(metadata.get_all("Project-URL", []))
        if urls != {"Homepage, https://github.com/arthur-diaz/RationalML", "Source, https://github.com/arthur-diaz/RationalML"}:
            raise ValueError("Project URLs differ from the verified owner-provided repository.")
        distribution = candidates[0].rsplit("/", 1)[0]
        allowed = {path.relative_to(ROOT).as_posix() for path in (ROOT / "rationalml").rglob("*.py")}
        allowed |= LEGACY_MODULES | {f"{distribution}/{name}" for name in ("METADATA", "WHEEL", "top_level.txt", "RECORD")}
        for license_file in metadata.get_all("License-File", []):
            if (PurePosixPath(license_file).is_absolute() or PureWindowsPath(license_file).is_absolute()
                    or ".." in PurePosixPath(license_file).parts):
                raise ValueError("Unsafe license file path.")
        require_publication_license(metadata)
        license_path = f"{distribution}/licenses/LICENSE"
        allowed.add(license_path)
        if set(names) != allowed:
            raise ValueError(f"Wheel file mismatch: extra={sorted(set(names) - allowed)}, missing={sorted(allowed - set(names))}")
        if archive.read(license_path) != (ROOT / "LICENSE").read_bytes():
            raise ValueError("Wheel LICENSE differs from the project license text.")
        return metadata, tuple(sorted(names))


def audit_sdist(sdist, version):
    prefix = f"rationalml-{version}/"
    with tarfile.open(sdist) as archive:
        members = archive.getmembers()
        files = {member.name for member in members if member.isfile()}
        forbidden = {".github", ".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist"}
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or forbidden.intersection(path.parts)
                    or member.issym() or member.islnk() or path.suffix in {".pyc", ".ipynb", ".csv", ".parquet", ".xlsx", ".pkl"}):
                raise ValueError(f"Unexpected or unsafe sdist entry: {member.name}")
        required = {"pyproject.toml", "README.md", "CHANGELOG.md", "RELEASE.md", "MANIFEST.in", "LICENSE", "tests/conftest.py",
                    "tests/_robustness.py", "examples/quickstart.py", "tools/check_release.py", "tools/check_wheel.py", "tools/check_sdist.py"}
        required |= LEGACY_MODULES | {path.relative_to(ROOT).as_posix() for path in (ROOT / "rationalml").rglob("*.py")}
        required |= {path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").rglob("*.py")}
        if not {prefix + name for name in required} <= files:
            raise ValueError("sdist omits required source, documentation or test helpers.")
        metadata = BytesParser().parsebytes(archive.extractfile(prefix + "PKG-INFO").read())
        if metadata["Name"] != "RationalML" or metadata["Version"] != version:
            raise ValueError("sdist metadata differs from the wheel.")
        require_publication_license(metadata)
        if archive.extractfile(prefix + "LICENSE").read() != (ROOT / "LICENSE").read_bytes():
            raise ValueError("sdist LICENSE differs from the project license text.")
        return tuple(sorted(files))


def artifacts(directory):
    directory = Path(directory)
    wheels, sdists = list(directory.glob("*.whl")), list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1 or set(directory.iterdir()) != set(wheels + sdists):
        raise ValueError("Distribution directory must contain exactly one wheel and one sdist, with no stale artifacts.")
    return wheels[0], sdists[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--require-license", action="store_true")
    args = parser.parse_args()
    wheel, sdist = artifacts(args.directory)
    metadata, wheel_files = audit_wheel(wheel)
    audit_sdist(sdist, metadata["Version"])
    if args.tag is not None:
        check_tag(args.tag, metadata["Version"])
    if args.require_license:
        require_publication_license(metadata)
    print(f"Validated {metadata['Name']} {metadata['Version']}: {len(wheel_files)} wheel files; sdist complete.")


if __name__ == "__main__":
    main()
