from pathlib import Path
from io import BytesIO

import pytest

from rationalml import AutoML

openpyxl = pytest.importorskip("openpyxl")


@pytest.fixture
def result(binary_df):
    return AutoML(target="target", models="logistic_regression", cv=2, n_trials=1, verbose=0).fit(binary_df)


@pytest.mark.parametrize("existing", [False, True])
def test_failed_partial_save_preserves_destination_and_removes_temporary(result, tmp_path, monkeypatch, existing):
    path = tmp_path / "report.xlsx"
    old_bytes = b"previous complete report"
    if existing:
        path.write_bytes(old_bytes)
    before = set(tmp_path.iterdir())
    failure = OSError("simulated interrupted save")

    def partial_save(self, temporary):
        temporary = Path(temporary)
        assert temporary.parent == path.parent
        assert temporary != path
        temporary.write_bytes(b"incomplete workbook")  # Also proves the temporary handle is closed on Windows.
        raise failure

    monkeypatch.setattr(openpyxl.Workbook, "save", partial_save)
    with pytest.raises(OSError) as raised:
        result.to_excel(path)
    assert raised.value is failure
    assert path.exists() == existing
    if existing:
        assert path.read_bytes() == old_bytes
    assert set(tmp_path.iterdir()) == before


def test_replace_failure_preserves_destination_and_cleans_temporary(result, tmp_path, monkeypatch):
    import rationalml.reporting.excel as excel

    path = tmp_path / "report.xlsx"
    path.write_bytes(b"original")

    def fail_replace(source, destination):
        assert Path(source).parent == path.parent
        assert Path(source).read_bytes().startswith(b"PK")
        assert destination == path
        raise PermissionError("destination locked")

    monkeypatch.setattr(excel.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="locked"):
        result.to_excel(path)
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_success_replaces_destination_only_after_complete_save(result, tmp_path, monkeypatch):
    import rationalml.reporting.excel as excel

    path = tmp_path / "report.xlsx"
    path.write_bytes(b"original")
    real_replace = excel.os.replace
    replacements = []

    def replace(source, destination):
        assert path.read_bytes() == b"original"
        workbook = openpyxl.load_workbook(BytesIO(Path(source).read_bytes()))
        try:
            assert "Summary" in workbook.sheetnames
        finally:
            workbook.close()
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(excel.os, "replace", replace)
    assert result.to_excel(path) == path
    assert len(replacements) == 1
    assert path.read_bytes().startswith(b"PK")
    assert list(tmp_path.iterdir()) == [path]
