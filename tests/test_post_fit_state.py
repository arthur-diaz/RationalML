import pickle

import pandas as pd
import pytest

from rationalml import AutoML, MLflowConfig

# This integration test intentionally requires all three reporting extras.
pytest.importorskip("openpyxl")
pytest.importorskip("mlflow")
pytest.importorskip("shap")
from test_mlflow_tracking import local_tracking


def test_sequential_reporting_tracking_and_explanation_add_no_hidden_result_state(binary_df, local_tracking):
    uri, client, _, root = local_tracking
    result = AutoML(target="target", models="logistic_regression", cv=2, n_trials=1, verbose=0).fit(binary_df)
    attributes = set(vars(result))
    saved = {name: getattr(result, name).copy(deep=True) for name in
             ("leaderboard", "cv_results", "feature_importance", "test_predictions")}
    pipeline = pickle.dumps(result.best_model)
    result.ranking_table()
    result.calibration_table()
    result.to_excel(root / "report.xlsx")
    run_id = result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=False))
    assert client.get_run(run_id).info.status == "FINISHED"
    X = binary_df.drop(columns="target")
    result.explain(X.iloc[:3], background=X.iloc[list(result.train_indices)[:10]])
    assert set(vars(result)) == attributes
    assert pickle.dumps(result.best_model) == pipeline
    for name, frame in saved.items():
        pd.testing.assert_frame_equal(getattr(result, name), frame)
