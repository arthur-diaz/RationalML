import pandas as pd
import pytest

from rationalml import AutoML, ExcelReportConfig
from test_excel_reporting import assert_sheet_matches, column_cells

openpyxl = pytest.importorskip("openpyxl")

CASES = [("binary", None), ("multiclass", None), ("multiclass", "bronze"),
         ("multiclass", "gold"), ("multiclass", 0), ("regression", None)]
SUMMARY_LABELS = {"brier_score": "Brier score", "expected_calibration_error": "Expected calibration error",
                  "max_calibration_error": "Max calibration error"}


@pytest.mark.parametrize("task,label", CASES)
@pytest.mark.parametrize("decimal_places", [3, 5])
def test_excel_calibration_uses_public_diagnostics_numeric_cells_and_canonical_bins(
    request, monkeypatch, tmp_path, task, label, decimal_places,
):
    df = request.getfixturevalue(f"{task}_df").copy()
    if task == "binary":
        df.target = df.target.map({0: "retained", 1: "churn"})
    elif label == 0:
        df.target = df.target.map({"bronze": 0, "gold": 1, "silver": 2})
    df.index = pd.Index([f"client_{j // 2}" for j in range(len(df))], name="client")
    result = AutoML(target="target", task=task, positive_class="churn" if task == "binary" else None,
                    models="ridge" if task == "regression" else "logistic_regression",
                    n_trials=1, cv=2, verbose=0).fit(df)
    original = result.test_predictions
    applicable = task == "binary" or (task == "multiclass" and label is not None)
    calls = []
    if applicable:
        table_method, summary_method = result.calibration_table, result.calibration_summary
        expected_table = table_method(n_bins=10, class_label=label)
        expected_summary = summary_method(n_bins=10, class_label=label)

        def table(n_bins=10, *, class_label=None):
            calls.append(("table", n_bins, class_label))
            return table_method(n_bins=n_bins, class_label=class_label)

        def summary(n_bins=10, *, class_label=None):
            calls.append(("summary", n_bins, class_label))
            return summary_method(n_bins=n_bins, class_label=class_label)
    else:
        def table(*args, **kwargs):
            pytest.fail("Calibration must be omitted for regression and multiclass without class_label")

        summary = table
    monkeypatch.setattr(result, "calibration_table", table)
    monkeypatch.setattr(result, "calibration_summary", summary)

    def forbidden(*args, **kwargs):
        pytest.fail("Excel diagnostics must not access the model")

    for name in ["fit", "predict", "predict_proba"]:
        if callable(getattr(result.best_model, name, None)):
            monkeypatch.setattr(result.best_model, name, forbidden)
    workbook = openpyxl.load_workbook(result.to_excel(tmp_path / "calibration.xlsx", class_label=label,
        config=ExcelReportConfig(decimal_places=decimal_places)))
    try:
        rows = {row[0].value: row[1] for row in workbook["Summary"].iter_rows(min_row=2)}
        if applicable:
            assert sorted(calls) == sorted([("table", 10, label), ("summary", 10, label)])
            assert_sheet_matches(workbook["Calibration"], expected_table)
            assert all(cell.number_format == "0" for cell in column_cells(workbook["Calibration"], "bin"))
            for key, display in SUMMARY_LABELS.items():
                cell = rows[display]
                assert cell.data_type == "n" and isinstance(cell.value, (int, float))
                assert cell.value == pytest.approx(expected_summary[key])
                assert cell.number_format == "0." + "0" * decimal_places
            assert not {"client", "y_true", "y_pred", *result.feature_names} & {cell.value for cell in workbook["Calibration"][1]}
        else:
            assert "Calibration" not in workbook.sheetnames
            assert not set(SUMMARY_LABELS.values()) & rows.keys()
        assert "Predictions" not in workbook.sheetnames
        pd.testing.assert_frame_equal(result.test_predictions, original)
    finally:
        workbook.close()
