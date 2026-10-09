"""Offline negative tests for release archives, tag and publication guards."""

from email.parser import BytesParser
import importlib.util
from io import BytesIO
from pathlib import Path
import tarfile
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_audit", ROOT / "tools/check_release.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def metadata_bytes(**changes):
    fields = {
        "Metadata-Version": "2.4", "Name": "RationalML", "Version": audit.project_version(),
        "Requires-Python": ">=3.10", "Description-Content-Type": "text/markdown",
        "License-Expression": "Apache-2.0", "License-File": ["LICENSE"],
        "Project-URL": ["Homepage, https://github.com/arthur-diaz/RationalML", "Source, https://github.com/arthur-diaz/RationalML"],
        "Provides-Extra": sorted(set(audit.REQUIREMENTS) - {None}),
        "Requires-Dist": [value + (f'; extra == "{extra}"' if extra else "")
                          for extra, values in audit.REQUIREMENTS.items() for value in sorted(values)],
    } | changes
    lines = [f"{key}: {value}" for key, values in fields.items() if values is not None
             for value in (values if isinstance(values, list) else [values])]
    return ("\n".join(lines) + "\n\n# Local test metadata\n").encode()


def wheel_file(tmp_path, *, extra=(), omit=(), license_text=None, **changes):
    version = audit.project_version()
    path = tmp_path / f"rationalml-{version}-py3-none-any.whl"
    prefix = f"rationalml-{version}.dist-info"
    files = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in (ROOT / "rationalml").rglob("*.py")}
    files |= {name: b"# legacy\n" for name in audit.LEGACY_MODULES}
    files |= {f"{prefix}/{name}": b"" for name in ("METADATA", "WHEEL", "RECORD", "top_level.txt")}
    files[f"{prefix}/METADATA"] = metadata_bytes(**changes)
    files[f"{prefix}/licenses/LICENSE"] = (ROOT / "LICENSE").read_bytes() if license_text is None else license_text
    with ZipFile(path, "w") as archive:
        for name, value in files.items():
            if name not in omit:
                archive.writestr(name, value)
        for name in extra:
            archive.writestr(name, b"unexpected")
    return path


def sdist_file(tmp_path, *, extra=(), omit=(), link=None, license_text=None):
    version = audit.project_version()
    prefix = f"rationalml-{version}/"
    path = tmp_path / f"rationalml-{version}.tar.gz"
    names = {"pyproject.toml", "README.md", "CHANGELOG.md", "RELEASE.md", "MANIFEST.in", "LICENSE",
             "examples/quickstart.py", "tools/check_release.py", "tools/check_wheel.py", "tools/check_sdist.py"}
    names |= audit.LEGACY_MODULES
    names |= {path.relative_to(ROOT).as_posix() for folder in ("rationalml", "tests")
              for path in (ROOT / folder).rglob("*.py")}
    with tarfile.open(path, "w:gz") as archive:
        for name in (names - set(omit)) | {"PKG-INFO"} | set(extra):
            data = metadata_bytes() if name == "PKG-INFO" else b"local test source\n"
            if name == "LICENSE":
                data = (ROOT / "LICENSE").read_bytes() if license_text is None else license_text
            info = tarfile.TarInfo(prefix + name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
        if link is not None:
            info = tarfile.TarInfo(prefix + "linked")
            info.type, info.linkname = tarfile.SYMTYPE, link
            archive.addfile(info)
    return path


def test_wheel_exact_allowlist_and_metadata(tmp_path):
    metadata, files = audit.audit_wheel(wheel_file(tmp_path))
    assert metadata["Version"] == audit.project_version()
    assert metadata.get_all("License-Expression") == ["Apache-2.0"]
    assert metadata.get_all("License-File") == ["LICENSE"]
    assert f"rationalml-{audit.project_version()}.dist-info/licenses/LICENSE" in files
    assert audit.LEGACY_MODULES <= set(files)


@pytest.mark.parametrize("name", ["tests/test_example.py", ".github/workflows/ci.yml", "__pycache__/module.pyc",
                                  "data.csv", "notebook.ipynb", "build/secret.py", "../outside.py", ".git/config",
                                  f"rationalml-{audit.project_version()}.dist-info/licenses/COPYING"])
def test_wheel_rejects_every_unexpected_file(tmp_path, name):
    with pytest.raises(ValueError, match="file mismatch"):
        audit.audit_wheel(wheel_file(tmp_path, extra=[name]))


@pytest.mark.parametrize("name", ["rationalml/__init__.py", "report.py",
                                  f"rationalml-{audit.project_version()}.dist-info/licenses/LICENSE"])
def test_wheel_requires_modern_and_intentional_legacy_sources(tmp_path, name):
    with pytest.raises(ValueError, match="missing"):
        audit.audit_wheel(wheel_file(tmp_path, omit=[name]))


@pytest.mark.parametrize("changes, fragment", [
    ({"Name": "different"}, "identity"), ({"Version": "invalid-version"}, "identity"),
    ({"Requires-Python": ">=3.12"}, "identity"), ({"Description-Content-Type": "text/plain"}, "identity"),
    ({"Requires-Dist": ["numpy>=1"]}, "dependencies"), ({"Provides-Extra": ["shap"]}, "extras"),
    ({"Requires-Dist": ['numpy>=1; python_version > "3.10"']}, "marker"),
    ({"Project-URL": ["Source, https://example.invalid"]}, "URLs"),
    ({"License-Expression": None}, "license metadata"),
    ({"License-Expression": "MIT"}, "license metadata"),
    ({"License-File": None}, "license metadata"),
    ({"License-File": ["COPYING"]}, "license metadata"),
    ({"License-File": ["LICENSE", "COPYING"]}, "license metadata"),
    ({"License-File": ["/absolute/LICENSE"]}, "Unsafe license file path"),
    ({"License-File": ["../LICENSE"]}, "Unsafe license file path"),
    ({"License-File": ["C:/absolute/LICENSE"]}, "Unsafe license file path"),
])
def test_wheel_rejects_changed_metadata(tmp_path, changes, fragment):
    with pytest.raises(ValueError, match=fragment):
        audit.audit_wheel(wheel_file(tmp_path, **changes))


def test_wheel_rejects_duplicate_entries(tmp_path):
    with pytest.warns(UserWarning, match="Duplicate"):
        wheel = wheel_file(tmp_path, extra=["report.py"])
    with pytest.raises(ValueError, match="duplicate"):
        audit.audit_wheel(wheel)


def test_sdist_includes_all_tests_and_helpers(tmp_path):
    files = audit.audit_sdist(sdist_file(tmp_path), audit.project_version())
    assert any(name.endswith("tests/conftest.py") for name in files)
    assert any(name.endswith("tests/_robustness.py") for name in files)
    assert any(name.endswith("tests/test_v1_contract.py") for name in files)
    assert f"rationalml-{audit.project_version()}/LICENSE" in files


@pytest.mark.parametrize("name", [".github/workflows/ci.yml", ".git/config", ".venv/secret", "__pycache__/module.pyc",
                                  "build/old.py", "dist/old.whl", "data.csv", "../outside.py"])
def test_sdist_rejects_unnecessary_or_unsafe_files(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe sdist entry"):
        audit.audit_sdist(sdist_file(tmp_path, extra=[name]), audit.project_version())


@pytest.mark.parametrize("name", ["tests/conftest.py", "tests/_robustness.py", "tests/test_v1_contract.py",
                                  "tools/check_wheel.py", "README.md", "LICENSE"])
def test_sdist_requires_rebuild_and_test_sources(tmp_path, name):
    with pytest.raises(ValueError, match="omits required"):
        audit.audit_sdist(sdist_file(tmp_path, omit=[name]), audit.project_version())


def test_sdist_rejects_links(tmp_path):
    with pytest.raises(ValueError, match="unsafe sdist entry"):
        audit.audit_sdist(sdist_file(tmp_path, link="../../secret"), audit.project_version())


@pytest.mark.parametrize("tag", ["vwrong", "release", "v", "wrong"])
def test_release_tag_must_match_built_metadata(tag):
    with pytest.raises(ValueError, match="must equal"):
        audit.check_tag(tag, audit.project_version())


def test_exact_release_tag_passes():
    audit.check_tag("v" + audit.project_version(), audit.project_version())


@pytest.mark.parametrize("changes", [
    {"License-Expression": None, "License-File": None},
    {"License-Expression": "MIT"},
    {"License-Expression": ["Apache-2.0", "MIT"]},
    {"License-File": None}, {"License-File": ["COPYING"]},
    {"License-File": ["LICENSE", "COPYING"]}, {"License-File": ["LICENSE", "LICENSE"]},
])
def test_publication_is_blocked_until_a_license_is_selected(changes):
    with pytest.raises(ValueError, match="Release license metadata must be Apache-2.0 with LICENSE"):
        audit.require_publication_license(BytesParser().parsebytes(metadata_bytes(**changes)))


def test_apache_license_is_accepted():
    audit.require_publication_license(BytesParser().parsebytes(metadata_bytes()))


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_archives_preserve_the_exact_project_license_text(tmp_path, kind):
    if kind == "wheel":
        check = lambda: audit.audit_wheel(wheel_file(tmp_path, license_text=b"altered text"))
    else:
        check = lambda: audit.audit_sdist(sdist_file(tmp_path, license_text=b"altered text"), audit.project_version())
    with pytest.raises(ValueError, match="LICENSE differs"):
        check()


def test_release_rejects_stale_artifacts(tmp_path):
    wheel, sdist = wheel_file(tmp_path), sdist_file(tmp_path)
    assert audit.artifacts(tmp_path) == (wheel, sdist)
    (tmp_path / "old.whl").write_bytes(b"stale")
    with pytest.raises(ValueError, match="stale artifacts"):
        audit.artifacts(tmp_path)
