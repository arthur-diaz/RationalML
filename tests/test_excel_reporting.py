import json
from io import BytesIO
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from rationalml import AutoML, ExcelReportConfig, __version__
from rationalml.exceptions import ConfigurationError, DataValidationError

openpyxl = pytest.importorskip("openpyxl")
COMMON_SHEETS = ["Summary", "Leaderboard", "Metrics", "Feature Importance", "Hyperparameters"]


@pytest.fixture(params=["binary", "multiclass", "regression"])
def excel_result(request):
    task = request.param
    df = request.getfixturevalue(f"{task}_df").copy(deep=True)
    df.index = pd.Index([f"customer_{i // 2}" for i in range(len(df))], name="customer")
    if task == "binary":
        df.target = df.target.map({0: "retained", 1: "churn"})
    elif task == "regression":
        df["feature_0"] *= -1  # Explicit negative coefficient in the controlled fixture.
    return AutoML(target="target", task=task, models="ridge" if task == "regression" else "logistic_regression",
                  positive_class="churn" if task == "binary" else None, cv=2, n_trials=1,
                  random_state=17, verbose=0).fit(df)


def assert_sheet_matches(sheet, frame, *, include_index=False):
    headers = ([frame.index.name or "index"] if include_index else []) + list(frame.columns)
    rows = list(sheet.iter_rows(values_only=True))
    assert list(rows[0]) == headers
    assert len(rows) == len(frame) + 1
    for actual_row, expected_row in zip(rows[1:], frame.itertuples(index=include_index, name=None)):
        for actual, expected in zip(actual_row, expected_row):
            if isinstance(expected, (float, np.floating)):
                assert isinstance(actual, (int, float))
                assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
            else:
                assert actual == expected


def column_cells(sheet, name):
    position = [cell.value for cell in sheet[1]].index(name) + 1
    return [sheet.cell(row=row, column=position) for row in range(2, sheet.max_row + 1)]


def test_report_common_tables_summary_and_default_sheets(excel_result, tmp_path):
    result = excel_result
    path = tmp_path / "report.xlsx"
    assert result.to_excel(str(path)) == path
    workbook = openpyxl.load_workbook(path)
    additional = [] if result.task.value == "multiclass" else ["Ranking", "Top Segment"]
    if result.task.value == "binary":
        additional.append("Calibration")
    assert workbook.sheetnames == COMMON_SHEETS + additional
    assert_sheet_matches(workbook["Leaderboard"], result.leaderboard.reset_index())
    expected_metrics = pd.DataFrame([(k, v, k == result.primary_metric) for k, v in result.test_metrics.items()],
                                    columns=["metric", "value", "primary"])
    assert_sheet_matches(workbook["Metrics"], expected_metrics)
    assert all(cell.data_type == "b" for cell in column_cells(workbook["Metrics"], "primary"))
    assert_sheet_matches(workbook["Feature Importance"], result.feature_importance)
    assert_sheet_matches(workbook["Hyperparameters"], pd.DataFrame(result.best_params.items(), columns=["parameter", "value"]))
    summary = dict(workbook["Summary"].iter_rows(min_row=2, values_only=True))
    best = result.leaderboard.loc[result.best_model_name]
    assert summary["RationalML version"] == __version__
    assert summary["Task"] == result.task.value
    assert summary["Target"] == result.target
    assert summary["Primary metric"] == result.primary_metric
    assert summary["Best model"] == result.best_model_name
    assert summary["Baseline name"] == result.baseline_name
    for key, value in {"Baseline score": result.baseline_score, "Best CV score": best.cv_score,
                       "CV std": best.cv_std, "Improvement vs baseline": best.improvement_vs_baseline,
                       "Test primary metric": result.test_metrics[result.primary_metric]}.items():
        assert isinstance(summary[key], (int, float))
        assert summary[key] == pytest.approx(value)
    for key, value in {"Train rows": len(result.train_indices), "Test rows": len(result.test_indices),
                       "Number of features": len(result.feature_names), "CV folds": result.config.cv,
                       "Optuna trials requested": result.config.n_trials, "Random state": result.config.random_state}.items():
        assert type(summary[key]) is int and summary[key] == value
    if result.task.value == "binary":
        assert summary["Positive class"] == "churn" and summary["Negative class"] == "retained"
        assert "Classes" not in summary
    elif result.task.value == "multiclass":
        assert summary["Number of classes"] == len(result.classes_)
        assert json.loads(summary["Classes"]) == result.classes_.tolist()
        assert "Positive class" not in summary and "Ranking class" not in summary
    else:
        assert not {"Positive class", "Negative class", "Classes", "Number of classes"} & summary.keys()
    if additional:
        assert_sheet_matches(workbook["Ranking"], result.ranking_table())
        assert_sheet_matches(workbook["Top Segment"], result.top_segment(), include_index=True)
    for sheet in workbook:
        assert sheet.freeze_panes == "A2"
        assert sheet.auto_filter.ref == sheet.dimensions
        assert all(cell.font.bold for cell in sheet[1])
        assert all(12 <= dimension.width <= 50 for dimension in sheet.column_dimensions.values())
    workbook.close()


def test_predictions_opt_in_preserves_original_index_and_raw_columns(excel_result, tmp_path):
    result = excel_result
    config = ExcelReportConfig(include_predictions=True, top_fraction=.25)
    label = "gold" if result.task.value == "multiclass" else None
    workbook = openpyxl.load_workbook(result.to_excel(tmp_path / "full.xlsx", config=config, class_label=label))
    calibration = [] if result.task.value == "regression" else ["Calibration"]
    assert workbook.sheetnames == COMMON_SHEETS + ["Ranking", "Top Segment"] + calibration + ["Predictions"]
    assert_sheet_matches(workbook["Predictions"], result.test_predictions, include_index=True)
    assert_sheet_matches(workbook["Top Segment"], result.top_segment(.25, class_label=label), include_index=True)
    assert_sheet_matches(workbook["Ranking"], result.ranking_table(class_label=label))
    assert not set(result.feature_names) & {cell.value for cell in workbook["Predictions"][1]}
    if label:
        assert dict(workbook["Summary"].iter_rows(min_row=2, values_only=True))["Ranking class"] == label
    workbook.close()


@pytest.mark.parametrize("places,percentages", [(3, 2), (0, 0), (5, 4)])
def test_number_types_formats_and_signed_importance(excel_result, tmp_path, places, percentages):
    result = excel_result
    config = ExcelReportConfig(decimal_places=places, percentage_places=percentages)
    label = "gold" if result.task.value == "multiclass" else None
    workbook = openpyxl.load_workbook(result.to_excel(tmp_path / "formatted.xlsx", config=config, class_label=label))
    decimal = "0" + ("." + "0" * places if places else "")
    percentage = "0" + ("." + "0" * percentages if percentages else "") + "%"
    for sheet_name, column in [("Leaderboard", "cv_score"), ("Metrics", "value"), ("Feature Importance", "importance")]:
        for cell in column_cells(workbook[sheet_name], column):
            assert isinstance(cell.value, (int, float)) and cell.data_type == "n"
            assert cell.number_format == decimal
    for cell in column_cells(workbook["Ranking"], "count"):
        assert type(cell.value) is int and cell.number_format == "0"
    if result.task.value == "regression":
        numeric_columns = ["rmse", "mae", "bias", "pred_mean"]
    else:
        numeric_columns = ["score_mean", "score_min", "score_max"]
        for column in ["positive_rate", "population_share", "positive_capture", "cumulative_positive_capture"]:
            for cell in column_cells(workbook["Ranking"], column):
                assert isinstance(cell.value, (int, float)) and cell.number_format == percentage
        for cell in column_cells(workbook["Ranking"], "lift"):
            assert isinstance(cell.value, (int, float)) and cell.number_format == "0.00"
    for column in numeric_columns:
        for cell in column_cells(workbook["Ranking"], column):
            assert isinstance(cell.value, (int, float)) and cell.number_format == decimal
    actual_importance = [cell.value for cell in column_cells(workbook["Feature Importance"], "importance")]
    np.testing.assert_allclose(actual_importance, result.feature_importance.importance)
    assert any(value < 0 for value in actual_importance)
    workbook.close()


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
def test_excel_never_recalculates_ml_and_does_not_mutate_result(request, monkeypatch, tmp_path, task, mode):
    import rationalml.automl as facade
    import rationalml.evaluation.metrics as metrics
    import optuna

    df = request.getfixturevalue(f"{task}_df")
    preprocessing = SimpleImputer().set_output(transform="pandas") if mode == "custom" else mode
    if mode is not None:
        df.loc[::11, "feature_0"] = np.nan
    result = AutoML(target="target", task=task, preprocessing=preprocessing, n_trials=1, cv=2, verbose=0,
                    models="ridge" if task == "regression" else "logistic_regression").fit(df)
    frames = {name: getattr(result, name).copy(deep=True) for name in ["leaderboard", "cv_results", "feature_importance", "test_predictions"]}
    dictionaries = {name: deepcopy(getattr(result, name)) for name in ["best_params", "test_metrics", "metrics"]}
    label = "gold" if task == "multiclass" else None
    expected_ranking = result.ranking_table(class_label=label)
    expected_top = result.top_segment(class_label=label)
    model = result.best_model

    def unexpected(*args, **kwargs):
        pytest.fail("Excel export must only read stored results")

    for owner in [result, model, *model.named_steps.values()]:
        for name in ["fit", "predict", "predict_proba", "transform", "fit_transform"]:
            if callable(getattr(owner, name, None)):
                monkeypatch.setattr(owner, name, unexpected)
    for name in ["optimize_model", "evaluate_baseline", "evaluate_holdout", "make_cv_splits", "build_model_pipeline"]:
        monkeypatch.setattr(facade, name, unexpected)
    monkeypatch.setattr(metrics, "evaluate_metrics", unexpected)
    monkeypatch.setattr(optuna, "create_study", unexpected)
    # The caller's original dataset is no longer available in a usable form.
    df.drop(df.index, inplace=True)
    workbook = openpyxl.load_workbook(result.to_excel(tmp_path / "stored.xlsx", class_label=label,
                                                    config=ExcelReportConfig(include_predictions=True)))
    assert_sheet_matches(workbook["Ranking"], expected_ranking)
    assert_sheet_matches(workbook["Top Segment"], expected_top, include_index=True)
    assert_sheet_matches(workbook["Predictions"], frames["test_predictions"], include_index=True)
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(getattr(result, name), frame)
    for name, dictionary in dictionaries.items():
        assert getattr(result, name) == dictionary
    assert result.best_model is model and not hasattr(result, "excel_path")
    workbook.close()


@pytest.mark.parametrize("label", ["bronze", "gold", "absent"])
def test_multiclass_selection_is_explicit(multiclass_df, tmp_path, label):
    result = AutoML(target="target", task="multiclass", models="logistic_regression", n_trials=1, cv=2, verbose=0).fit(multiclass_df)
    path = tmp_path / "selected.xlsx"
    if label == "absent":
        path.write_bytes(b"existing report")
        with pytest.raises(ConfigurationError, match="available classes"):
            result.to_excel(path, class_label=label)
        assert path.read_bytes() == b"existing report"
    else:
        workbook = openpyxl.load_workbook(result.to_excel(path, class_label=label))
        assert_sheet_matches(workbook["Ranking"], result.ranking_table(class_label=label))
        assert_sheet_matches(workbook["Top Segment"], result.top_segment(class_label=label), include_index=True)
        workbook.close()


def test_multiclass_zero_label_is_not_treated_as_unspecified(multiclass_df, tmp_path):
    multiclass_df.target = multiclass_df.target.map({"bronze": 0, "gold": 1, "silver": 2})
    result = AutoML(target="target", task="multiclass", models="logistic_regression", n_trials=1, cv=2, verbose=0).fit(multiclass_df)
    workbook = openpyxl.load_workbook(result.to_excel(tmp_path / "zero.xlsx", class_label=0))
    assert_sheet_matches(workbook["Ranking"], result.ranking_table(class_label=0))
    workbook.close()


@pytest.mark.parametrize("empty_headers", [False, True])
def test_unavailable_importance_does_not_break_export(excel_result, tmp_path, empty_headers):
    empty = pd.DataFrame() if empty_headers else excel_result.feature_importance.iloc[:0].copy()
    result = replace(excel_result, feature_importance=empty)
    workbook = openpyxl.load_workbook(result.to_excel(tmp_path / "no-importance.xlsx"))
    sheet = workbook["Feature Importance"]
    if empty_headers:
        assert sheet["A2"].value == "Feature importance unavailable."
    else:
        assert list(sheet.values) == [tuple(empty.columns)]
    workbook.close()


def test_structured_parameters_literal_labels_and_index(excel_result, tmp_path):
    parameters = {"float": np.float64(.2314), "integer": np.int64(3), "bool": np.bool_(True),
                  "none": None, "list": [2, 1], "dict": {"z": 2, "a": 1}, "array": np.array([3, 4]),
                  "object": object(), "literal": "=SUM(A1:A2)"}
    predictions = excel_result.test_predictions
    predictions.index = pd.Index([f"=customer_{i}" for i in range(len(predictions))], name="y_true")
    result = replace(excel_result, best_params=parameters, _test_predictions=predictions)
    workbook = openpyxl.load_workbook(result.to_excel(tmp_path / "literals.xlsx", config=ExcelReportConfig(include_predictions=True)))
    values = dict(workbook["Hyperparameters"].iter_rows(min_row=2, values_only=True))
    assert values["float"] == .2314 and type(values["integer"]) is int
    assert values["bool"] is True and values["none"] is None
    assert json.loads(values["list"]) == [2, 1]
    assert values["dict"] == '{"a": 1, "z": 2}'
    assert json.loads(values["array"]) == [3, 4]
    assert "0x" not in values["object"]
    assert values["literal"] == "=SUM(A1:A2)"
    assert workbook["Hyperparameters"].cell(row=10, column=2).data_type == "s"
    assert workbook["Predictions"]["A2"].value == "=customer_0"
    assert workbook["Predictions"]["A2"].data_type == "s"
    assert_sheet_matches(workbook["Predictions"], predictions, include_index=True)
    workbook.close()


@pytest.mark.parametrize("suffix", ["", ".xls", ".csv", ".xlsx.tmp"])
def test_invalid_extensions_are_explicit(excel_result, tmp_path, suffix):
    with pytest.raises(ConfigurationError, match=".xlsx"):
        excel_result.to_excel(tmp_path / f"report{suffix}")
    assert list(tmp_path.iterdir()) == []


def test_existing_parent_required_and_existing_file_overwritten(excel_result, tmp_path):
    with pytest.raises(FileNotFoundError, match="parent directory"):
        excel_result.to_excel(tmp_path / "missing" / "report.xlsx")
    assert not (tmp_path / "missing").exists()
    path = tmp_path / "report.XLSX"
    path.write_bytes(b"old report")
    assert excel_result.to_excel(path) == path
    workbook = openpyxl.load_workbook(path)
    assert "Summary" in workbook.sheetnames
    workbook.close()
    folder = tmp_path / "folder.xlsx"
    folder.mkdir()
    with pytest.raises(IsADirectoryError):
        excel_result.to_excel(folder)


@pytest.mark.parametrize("option", ["config", "path", "class_label"])
def test_invalid_export_options(excel_result, tmp_path, option):
    kwargs = {"config": {}, "path": 3, "class_label": "absent"}
    with pytest.raises(ConfigurationError):
        excel_result.to_excel(**({"path": tmp_path / "invalid.xlsx"} | {option: kwargs[option]}))
    assert list(tmp_path.iterdir()) == []


def test_excel_row_limits_include_headers_without_truncation(excel_result, monkeypatch, tmp_path):
    import rationalml.reporting.excel as excel

    # Keep Predictions larger than Summary, including its new calibration rows.
    result = replace(excel_result, _test_predictions=pd.concat([excel_result.test_predictions] * 2))
    # Exactly len(predictions)+1 fits; one fewer available row must fail.
    count = len(result.test_predictions)
    config = ExcelReportConfig(include_predictions=True)
    path = tmp_path / "limit.xlsx"
    monkeypatch.setattr(excel, "EXCEL_MAX_ROWS", count + 1)
    result.to_excel(path, config=config)
    # Inspect bytes without retaining openpyxl 3.0's file handles before the
    # subsequent atomic replacement (Windows forbids replacing locked files).
    workbook = openpyxl.load_workbook(BytesIO(path.read_bytes()))
    assert workbook["Predictions"].max_row == count + 1
    workbook.close()
    previous = path.read_bytes()
    monkeypatch.setattr(excel, "EXCEL_MAX_ROWS", count)
    with pytest.raises(DataValidationError, match="Predictions exceed Excel row limit.*disable include_predictions"):
        result.to_excel(path, config=config)
    assert path.read_bytes() == previous
    # Excluding predictions makes the report possible again.
    result.to_excel(path)


def test_other_sheet_row_limit_is_explicit(excel_result, monkeypatch, tmp_path):
    import rationalml.reporting.excel as excel

    monkeypatch.setattr(excel, "EXCEL_MAX_ROWS", 2)
    with pytest.raises(DataValidationError, match="Summary.*row limit.*including header"):
        excel_result.to_excel(tmp_path / "too-small.xlsx")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("index_kind", ["unnamed", "multi", "timezone"])
def test_business_index_representations(excel_result, tmp_path, index_kind):
    predictions = excel_result.test_predictions
    if index_kind == "unnamed":
        predictions.index = pd.Index([i // 2 for i in range(len(predictions))])
        expected = predictions.index.tolist()
    elif index_kind == "multi":
        predictions.index = pd.MultiIndex.from_tuples([("customer", i // 2) for i in range(len(predictions))])
        expected = [json.dumps(list(label)) for label in predictions.index]
    else:
        predictions.index = pd.date_range("2026-01-01", periods=len(predictions), tz="Europe/Paris", name="event")
        expected = [label.isoformat() for label in predictions.index]
    result = replace(excel_result, _test_predictions=predictions)
    workbook = openpyxl.load_workbook(result.to_excel(tmp_path / "index.xlsx", config=ExcelReportConfig(include_predictions=True)))
    assert [row[0] for row in workbook["Predictions"].iter_rows(min_row=2, values_only=True)] == expected
    assert workbook["Predictions"]["A1"].value == ("event" if index_kind == "timezone" else "index")
    workbook.close()


def test_nonrepresentable_values_fail_without_overwriting(excel_result, tmp_path):
    path = tmp_path / "existing.xlsx"
    path.write_bytes(b"existing report")
    for value, message in [(float("inf"), "infinite"), ("x" * 32_768, "character cell limit")]:
        result = replace(excel_result, best_params={"unsupported": value})
        with pytest.raises(DataValidationError, match=message):
            result.to_excel(path)
        assert path.read_bytes() == b"existing report"


def test_excel_column_limit_is_explicit(excel_result, monkeypatch, tmp_path):
    import rationalml.reporting.excel as excel

    monkeypatch.setattr(excel, "EXCEL_MAX_COLUMNS", 1)
    with pytest.raises(DataValidationError, match="Summary.*column limit"):
        excel_result.to_excel(tmp_path / "columns.xlsx")
    assert list(tmp_path.iterdir()) == []
