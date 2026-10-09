import pandas as pd
import pytest

from rationalml import MLflowConfig
from test_mlflow_tracking import assert_table_equal, fit_result, local_tracking, read_table

CASES = [("binary", None), ("multiclass", None), ("multiclass", "bronze"),
         ("multiclass", "gold"), ("multiclass", 0), ("regression", None)]
METRIC_NAMES = {"brier_score": "calibration_brier_score", "expected_calibration_error": "calibration_expected_error",
                "max_calibration_error": "calibration_max_error"}


@pytest.mark.parametrize("task,label", CASES)
def test_mlflow_calibration_is_aggregated_private_parent_only_and_uses_canonical_bins(
    request, local_tracking, monkeypatch, task, label,
):
    uri, client, experiment_id, root = local_tracking
    df = request.getfixturevalue(f"{task}_df").copy()
    if task == "binary":
        df.target = df.target.map({0: "retained", 1: "churn"})
    elif label == 0:
        df.target = df.target.map({"bronze": 0, "gold": 1, "silver": 2})
    df.index = pd.Index([f"secret_client_{j // 2}" for j in range(len(df))], name="client_id")
    result = fit_result(df, task, positive_class="churn" if task == "binary" else None)
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
        pytest.fail("MLflow diagnostics must not access the model")

    for name in ["fit", "predict", "predict_proba"]:
        if callable(getattr(result.best_model, name, None)):
            monkeypatch.setattr(result.best_model, name, forbidden)
    run_id = result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=False), class_label=label)
    parent = client.get_run(run_id)
    artifacts = {item.path for item in client.list_artifacts(run_id)}
    if applicable:
        assert sorted(calls) == sorted([("table", 10, label), ("summary", 10, label)])
        assert "calibration.json" in artifacts
        actual = read_table(client, run_id, "calibration.json", root)
        assert_table_equal(actual, expected_table)
        assert {name: parent.data.metrics[name] for name in METRIC_NAMES.values()} == pytest.approx(
            {name: expected_summary[key] for key, name in METRIC_NAMES.items()})
        assert not {"index", "client_id", "y_true", "y_pred", *result.feature_names} & set(actual.columns)
        assert not any("secret_client" in str(value) for value in actual.to_numpy().flat)
    else:
        assert "calibration.json" not in artifacts
        assert not set(METRIC_NAMES.values()) & parent.data.metrics.keys()
    assert "predictions.json" not in artifacts and "top_segment.json" not in artifacts
    children = client.search_runs([experiment_id], f"tags.`mlflow.parentRunId` = '{run_id}'")
    assert len(children) == len(result.leaderboard)
    for child in children:
        assert not set(METRIC_NAMES.values()) & child.data.metrics.keys()
        assert "calibration.json" not in {item.path for item in client.list_artifacts(child.info.run_id)}
    pd.testing.assert_frame_equal(result.test_predictions, original)
