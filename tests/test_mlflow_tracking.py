import json
import os
import socket
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from rationalml import AutoML, MLflowConfig, ModelRegistry, __version__
from rationalml.exceptions import ConfigurationError

mlflow = pytest.importorskip("mlflow")
import mlflow.sklearn


@pytest.fixture
def local_tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("Tracking tests must not connect to a network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    original = mlflow.get_tracking_uri()
    assert mlflow.active_run() is None
    mlflow.set_tracking_uri("sqlite:///" + (tmp_path / "idle.db").as_posix())
    monkeypatch.chdir(tmp_path)  # SQLite's default artifact root is resolved when its store is created.
    uri = "sqlite:///" + (tmp_path / "tracking.db").as_posix()
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment("RationalML", artifact_location=(tmp_path / "artifacts").as_uri())
    try:
        yield uri, client, experiment_id, tmp_path
    finally:
        leaked = mlflow.active_run()
        if leaked is not None:
            mlflow.end_run(status="KILLED")
        mlflow.set_tracking_uri(original)
        assert leaked is None, "A RationalML run leaked out of the logging context"


def fit_result(df, task="binary", **options):
    return AutoML(**({"target": "target", "task": task, "models": "ridge" if task == "regression" else "logistic_regression",
                     "n_trials": 1, "cv": 2, "random_state": 17, "verbose": 0} | options)).fit(df)


def read_table(client, run_id, name, root):
    destination = root / "downloads"
    destination.mkdir(exist_ok=True)
    path = client.download_artifacts(run_id, name, dst_path=str(destination))
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame(data["data"], columns=data["columns"])


def assert_table_equal(actual, expected):
    pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True), check_dtype=False,
                                  check_exact=False, rtol=1e-8, atol=1e-9)


def snapshot(result):
    frames = {name: getattr(result, name).copy(deep=True) for name in
              ["leaderboard", "cv_results", "feature_importance", "test_predictions"]}
    values = {name: deepcopy(getattr(result, name)) for name in
              ["best_params", "model_best_params", "test_metrics", "metrics"]}
    return frames, values


def assert_unchanged(result, saved):
    frames, values = saved
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(getattr(result, name), frame)
    for name, value in values.items():
        assert getattr(result, name) == value
    assert not hasattr(result, "mlflow_run_id")


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_parent_content_tables_and_default_privacy(request, local_tracking, task):
    uri, client, experiment_id, root = local_tracking
    df = request.getfixturevalue(f"{task}_df")
    if task == "binary":
        df.target = df.target.map({0: "retained", 1: "churn"})
    result = fit_result(df, task, positive_class="churn" if task == "binary" else None)
    saved = snapshot(result)
    original_uri = mlflow.get_tracking_uri()
    run_id = result.log_mlflow(config=MLflowConfig(tracking_uri=uri, run_name="business", log_model=False, tags={"team": "risk"}))
    assert isinstance(run_id, str)
    run = client.get_run(run_id)
    assert run.info.experiment_id == experiment_id and run.info.status == "FINISHED"
    assert run.data.tags["mlflow.runName"] == "business"
    assert run.data.tags["team"] == "risk"
    assert {key: run.data.tags[key] for key in ["rationalml.version", "rationalml.task", "rationalml.best_model"]} == {
        "rationalml.version": __version__, "rationalml.task": task, "rationalml.best_model": result.best_model_name}
    expected = {"task": task, "target": "target", "primary_metric": result.primary_metric,
                "best_model": result.best_model_name, "baseline_name": result.baseline_name,
                "cv_folds": "2", "n_trials_requested": "1", "random_state": "17",
                "train_rows": str(len(result.train_indices)), "test_rows": str(len(result.test_indices)),
                "n_features": str(len(result.feature_names))}
    if task == "binary":
        expected.update(positive_class="churn", negative_class="retained")
    elif task == "multiclass":
        expected["n_classes"] = "3"
    assert run.data.params == expected
    best = result.leaderboard.loc[result.best_model_name]
    assert run.data.metrics == pytest.approx({"baseline_score": result.baseline_score, "best_cv_score": best.cv_score,
                                              "best_cv_std": best.cv_std, "improvement_vs_baseline": best.improvement_vs_baseline,
                                              **{f"test_{k}": v for k, v in result.test_metrics.items()}})
    names = {artifact.path for artifact in client.list_artifacts(run_id)}
    required = {"leaderboard.json", "cv_results.json", "feature_importance.json"}
    if task != "multiclass":
        required.add("ranking.json")
    assert names == required
    assert_table_equal(read_table(client, run_id, "leaderboard.json", root), result.leaderboard.reset_index())
    assert_table_equal(read_table(client, run_id, "cv_results.json", root), result.cv_results)
    assert_table_equal(read_table(client, run_id, "feature_importance.json", root), result.feature_importance)
    if task != "multiclass":
        assert_table_equal(read_table(client, run_id, "ranking.json", root), result.ranking_table())
    assert mlflow.active_run() is None and mlflow.get_tracking_uri() == original_uri
    assert_unchanged(result, saved)


def test_three_models_five_trials_create_exactly_four_runs(binary_df, local_tracking, monkeypatch):
    import rationalml.automl as facade

    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    recorded = {}
    optimize = facade.optimize_model

    def record(spec, *args, **kwargs):
        optimized = optimize(spec, *args, **kwargs)
        recorded[spec.name] = optimized.params.copy()
        return optimized

    monkeypatch.setattr(facade, "optimize_model", record)
    result = fit_result(binary_df, models="auto", n_trials=5)
    assert result.model_best_params == recorded
    assert len(result.cv_results) == 15 and len(result.model_best_params) == 3
    uri, client, experiment_id, _ = local_tracking
    run_id = result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=False))
    runs = client.search_runs([experiment_id])
    assert len(runs) == 4
    children = [run for run in runs if run.info.run_id != run_id]
    assert {run.data.tags["mlflow.runName"] for run in children} == set(result.leaderboard.index)
    assert sum(run.data.tags["rationalml.selected"] == "true" for run in children) == 1
    for run in children:
        assert run.info.status == "FINISHED" and run.data.tags["mlflow.parentRunId"] == run_id
        name = run.data.tags["rationalml.model_name"]
        assert run.data.tags["rationalml.selected"] == str(name == result.best_model_name).lower()
        assert run.data.metrics == pytest.approx({key: float(result.leaderboard.loc[name, key]) for key in
                                                  ["cv_score", "cv_std", "cv_min", "cv_max", "improvement_vs_baseline", "n_trials_completed"]})
        for key, value in result.model_best_params[name].items():
            expected = value if isinstance(value, str) else json.dumps(value)
            assert run.data.params[key] == expected


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_predictions_opt_in_and_safe_original_index(request, local_tracking, task):
    uri, client, _, root = local_tracking
    df = request.getfixturevalue(f"{task}_df")
    df.index = pd.Index([f"customer_{i // 2}" for i in range(len(df))], name="y_true")
    result = fit_result(df, task)
    label = "gold" if task == "multiclass" else None
    run_id = result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=False, log_predictions=True), class_label=label)
    expected = result.test_predictions
    expected.insert(0, "y_true_1", expected.index.to_numpy())
    assert_table_equal(read_table(client, run_id, "predictions.json", root), expected)
    assert "top_segment.json" not in {artifact.path for artifact in client.list_artifacts(run_id)}
    if label:
        assert_table_equal(read_table(client, run_id, "ranking.json", root), result.ranking_table(class_label=label))


@pytest.mark.parametrize("label", ["bronze", "gold", "absent", 0])
def test_multiclass_class_selection_precedes_run_creation(multiclass_df, local_tracking, label):
    uri, client, experiment_id, root = local_tracking
    if label == 0:
        multiclass_df.target = multiclass_df.target.map({"bronze": 0, "gold": 1, "silver": 2})
    result = fit_result(multiclass_df, "multiclass")
    config = MLflowConfig(tracking_uri=uri, log_model=False)
    if label == "absent":
        with pytest.raises(ConfigurationError, match="available classes"):
            result.log_mlflow(config=config, class_label=label)
        assert client.search_runs([experiment_id]) == []
    else:
        run_id = result.log_mlflow(config=config, class_label=label)
        assert client.get_run(run_id).data.params["ranking_class"] == str(label)
        assert_table_equal(read_table(client, run_id, "ranking.json", root), result.ranking_table(class_label=label))


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("mode", [None, "basic", "custom"])
@pytest.mark.parametrize("log_model", [False, True])
def test_tracking_never_fits_or_predicts_and_keeps_full_pipeline(request, local_tracking, monkeypatch, task, mode, log_model):
    import optuna
    import rationalml.automl as facade
    import rationalml.evaluation.metrics as metrics

    uri, client, _, root = local_tracking
    df = request.getfixturevalue(f"{task}_df")
    preprocessing = SimpleImputer().set_output(transform="pandas") if mode == "custom" else mode
    if mode is not None:
        df.loc[::11, "feature_0"] = np.nan
    result = fit_result(df, task, preprocessing=preprocessing)
    saved = snapshot(result)
    pipeline = result.best_model
    label = "gold" if task == "multiclass" else None
    ranking = result.ranking_table(class_label=label)

    def forbidden(*args, **kwargs):
        raise AssertionError("Tracking must not trigger ML computation")

    for owner in [result, pipeline, *pipeline.named_steps.values()]:
        for name in ["fit", "predict", "predict_proba", "transform", "fit_transform"]:
            if callable(getattr(owner, name, None)):
                monkeypatch.setattr(owner, name, forbidden)
    monkeypatch.setattr(result, "top_segment", forbidden)
    for name in ["optimize_model", "evaluate_metrics", "evaluate_holdout", "build_model_pipeline", "evaluate_baseline"]:
        if hasattr(facade, name):
            monkeypatch.setattr(facade, name, forbidden)
    monkeypatch.setattr(metrics, "evaluate_metrics", forbidden)
    monkeypatch.setattr(optuna, "create_study", forbidden)
    monkeypatch.setattr(mlflow, "autolog", forbidden)
    monkeypatch.setattr(mlflow.sklearn, "autolog", forbidden)
    monkeypatch.setattr(mlflow, "register_model", forbidden)
    original_log_model = mlflow.sklearn.log_model
    models = []

    def record_model(model, **kwargs):
        assert model is pipeline and kwargs["name"] == "model"
        assert kwargs["signature"] is False
        assert not {"input_example", "artifact_path", "registered_model_name"} & kwargs.keys()
        models.append(original_log_model(model, **kwargs))
        return models[-1]

    monkeypatch.setattr(mlflow.sklearn, "log_model", record_model if log_model else forbidden)
    df.drop(df.index, inplace=True)
    run_id = result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=log_model), class_label=label)
    assert_table_equal(read_table(client, run_id, "ranking.json", root), ranking)
    assert len(models) == int(log_model)
    if log_model:
        artifact_uri = client.get_logged_model(models[0].model_id).artifact_location
        model_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, tracking_uri=uri,
                                                       dst_path=str(root / "model"))
        loaded = mlflow.sklearn.load_model(model_path)
        assert isinstance(loaded, Pipeline)
        assert list(loaded.named_steps) == list(pipeline.named_steps)
        np.testing.assert_allclose(loaded.named_steps["estimator"].coef_, pipeline.named_steps["estimator"].coef_)
        metadata = mlflow.models.Model.load(model_path)
        assert metadata.signature is None and metadata.saved_input_example_info is None
    assert_unchanged(result, saved)


@pytest.mark.parametrize("nested", [False, True])
def test_existing_user_run_is_preserved(binary_df, local_tracking, nested):
    uri, client, experiment_id, _ = local_tracking
    result = fit_result(binary_df)
    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(experiment_id=experiment_id, run_name="user") as user:
        if not nested:
            with pytest.raises(ConfigurationError, match="already active.*nested=True"):
                result.log_mlflow(config=MLflowConfig(log_model=False))
            assert len(client.search_runs([experiment_id])) == 1
        else:
            run_id = result.log_mlflow(config=MLflowConfig(nested=True, log_model=False))
            parent = client.get_run(run_id)
            assert parent.data.tags["mlflow.parentRunId"] == user.info.run_id
            runs = client.search_runs([experiment_id])
            assert len(runs) == 3
            candidate = next(run for run in runs if run.data.tags.get("rationalml.model_name"))
            assert candidate.data.tags["mlflow.parentRunId"] == run_id
        assert mlflow.active_run().info.run_id == user.info.run_id
        assert client.get_run(user.info.run_id).data.params == {}
        assert client.get_run(user.info.run_id).data.metrics == {}


@pytest.mark.parametrize("failure", ["parent", "child", "model"])
def test_backend_failure_closes_own_runs_and_restores_uri(binary_df, local_tracking, monkeypatch, failure):
    uri, client, experiment_id, _ = local_tracking
    result = fit_result(binary_df)
    saved = snapshot(result)
    original_uri = mlflow.get_tracking_uri()
    original = mlflow.log_params
    calls = []

    def fail_params(params):
        calls.append(params)
        if (failure == "parent" and len(calls) == 1) or (failure == "child" and len(calls) == 2):
            raise RuntimeError("backend unavailable")
        return original(params)

    def fail_model(*args, **kwargs):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(mlflow, "log_params", fail_params)
    monkeypatch.setattr(mlflow.sklearn, "log_model", fail_model)
    with pytest.raises(RuntimeError, match="backend unavailable"):
        result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=failure == "model"))
    assert mlflow.active_run() is None and mlflow.get_tracking_uri() == original_uri
    runs = client.search_runs([experiment_id])
    assert all(run.info.status != "RUNNING" for run in runs)
    assert client.get_run(next(run.info.run_id for run in runs if "rationalml.best_model" in run.data.tags)).info.status == "FAILED"
    assert_unchanged(result, saved)


def test_environment_run_id_is_not_resumed(binary_df, local_tracking, monkeypatch):
    uri, client, experiment_id, _ = local_tracking
    previous = client.create_run(experiment_id)
    client.set_terminated(previous.info.run_id)
    monkeypatch.setenv("MLFLOW_RUN_ID", previous.info.run_id)
    result = fit_result(binary_df)
    run_id = result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=False))
    assert run_id != previous.info.run_id
    assert client.get_run(previous.info.run_id).data.params == {}
    assert os.environ["MLFLOW_RUN_ID"] == previous.info.run_id


def test_structured_candidate_parameters_are_deterministic(binary_df, local_tracking):
    uri, client, experiment_id, _ = local_tracking
    result = fit_result(binary_df)
    params = {"text": "value", "int": np.int64(5), "float": np.float32(.25), "bool": np.bool_(True), "none": None,
              "list": [3, 1], "tuple": (2, 4), "dict": {"z": [2], "a": {"nested": True}}, "unknown": object()}
    result = replace(result, model_best_params={result.best_model_name: params})
    with pytest.warns(UserWarning, match="not JSON serializable.*<builtins.object>"):
        parent = result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=False))
    child = next(run for run in client.search_runs([experiment_id]) if run.info.run_id != parent)
    assert child.data.params == {"text": "value", "int": "5", "float": "0.25", "bool": "true", "none": "null",
                                 "list": "[3, 1]", "tuple": "[2, 4]", "dict": '{"a": {"nested": true}, "z": [2]}',
                                 "unknown": "<builtins.object>"}
    assert not any("0x" in value for value in child.data.params.values())


def test_empty_importance_is_optional(binary_df, local_tracking):
    uri, client, _, _ = local_tracking
    result = fit_result(binary_df)
    result = replace(result, feature_importance=result.feature_importance.iloc[:0].copy())
    run_id = result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=False))
    assert "feature_importance.json" not in {item.path for item in client.list_artifacts(run_id)}


def test_invalid_config_and_mutated_reserved_tags_create_no_runs(binary_df, local_tracking):
    uri, client, experiment_id, _ = local_tracking
    result = fit_result(binary_df)
    with pytest.raises(ConfigurationError, match="MLflowConfig"):
        result.log_mlflow(config={})
    config = MLflowConfig(tracking_uri=uri, tags={"team": "risk"}, log_model=False)
    config.tags["rationalml.version"] = "override"
    with pytest.raises(ConfigurationError, match="reserved"):
        result.log_mlflow(config=config)
    assert client.search_runs([experiment_id]) == []


class CustomLogistic(LogisticRegression):
    """Custom registered estimator to exercise complete pipeline serialization."""


@pytest.mark.parametrize("mode", [None, "basic", "custom"])
def test_custom_model_pipeline_roundtrip(binary_df, local_tracking, monkeypatch, mode):
    uri, client, _, root = local_tracking
    spec = replace(ModelRegistry.get("logistic_regression"), name="custom", estimator_class=CustomLogistic)

    class Registry(ModelRegistry):
        _optional = {}
        _specs = {"custom": spec}

    preprocessing = SimpleImputer().set_output(transform="pandas") if mode == "custom" else mode
    if mode is not None:
        binary_df.loc[::11, "feature_0"] = np.nan
    result = fit_result(binary_df, models="custom", model_registry=Registry, preprocessing=preprocessing)
    X = binary_df.drop(columns="target").iloc[:7]
    expected = result.best_model.predict_proba(X)
    infos = []
    original = mlflow.sklearn.log_model

    def record(*args, **kwargs):
        infos.append(original(*args, **kwargs))
        return infos[-1]

    monkeypatch.setattr(mlflow.sklearn, "log_model", record)
    result.log_mlflow(config=MLflowConfig(tracking_uri=uri))
    artifact_uri = client.get_logged_model(infos[0].model_id).artifact_location
    model_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, tracking_uri=uri,
                                                   dst_path=str(root / "roundtrip"))
    loaded = mlflow.sklearn.load_model(model_path)
    assert isinstance(loaded, Pipeline) and isinstance(loaded.named_steps["estimator"], CustomLogistic)
    np.testing.assert_allclose(loaded.predict_proba(X), expected)


def test_existing_uri_and_experiment_environment_are_respected(binary_df, local_tracking, monkeypatch):
    uri, client, experiment_id, root = local_tracking
    user_experiment = client.create_experiment("user", artifact_location=(root / "user-artifacts").as_uri())
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", user_experiment)
    result = fit_result(binary_df)
    run_id = result.log_mlflow(config=MLflowConfig(log_model=False))
    assert client.get_run(run_id).info.experiment_id == experiment_id
    assert mlflow.get_tracking_uri() == uri and os.environ["MLFLOW_EXPERIMENT_ID"] == user_experiment
    with mlflow.start_run() as user:
        assert user.info.experiment_id == user_experiment


def test_requested_experiment_is_created_locally(binary_df, local_tracking, monkeypatch):
    uri, client, _, root = local_tracking
    monkeypatch.chdir(root)
    result = fit_result(binary_df)
    run_id = result.log_mlflow(config=MLflowConfig(tracking_uri=uri, experiment_name="new-experiment", log_model=False))
    experiment = client.get_experiment_by_name("new-experiment")
    assert experiment is not None and client.get_run(run_id).info.experiment_id == experiment.experiment_id
    assert len(client.search_runs([experiment.experiment_id])) == 2


def test_nested_failure_preserves_user_run(binary_df, local_tracking, monkeypatch):
    uri, client, experiment_id, _ = local_tracking
    result = fit_result(binary_df)
    mlflow.set_tracking_uri(uri)

    def fail(*args, **kwargs):
        raise RuntimeError("tracking failure")

    monkeypatch.setattr(mlflow, "log_table", fail)
    with mlflow.start_run(experiment_id=experiment_id) as user:
        with pytest.raises(RuntimeError, match="tracking failure"):
            result.log_mlflow(config=MLflowConfig(nested=True, log_model=False))
        assert mlflow.active_run().info.run_id == user.info.run_id
        logged = next(run for run in client.search_runs([experiment_id]) if run.info.run_id != user.info.run_id)
        assert logged.info.status == "FAILED" and logged.data.tags["mlflow.parentRunId"] == user.info.run_id


def test_nesting_cannot_switch_to_another_backend(binary_df, local_tracking):
    uri, client, experiment_id, root = local_tracking
    result = fit_result(binary_df)
    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(experiment_id=experiment_id) as user:
        with pytest.raises(ConfigurationError, match="active run's tracking URI"):
            result.log_mlflow(config=MLflowConfig(nested=True, tracking_uri=root / "other", log_model=False))
        assert mlflow.active_run().info.run_id == user.info.run_id and mlflow.get_tracking_uri() == uri
        assert len(client.search_runs([experiment_id])) == 1


def test_manual_result_without_candidate_params_uses_known_winner_params(binary_df, local_tracking):
    uri, client, experiment_id, _ = local_tracking
    result = replace(fit_result(binary_df), model_best_params={})
    run_id = result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=False))
    child = next(run for run in client.search_runs([experiment_id]) if run.info.run_id != run_id)
    assert set(child.data.params) == set(result.best_params)


def test_parameter_serialization_failure_creates_no_run(binary_df, local_tracking):
    uri, client, experiment_id, _ = local_tracking
    result = fit_result(binary_df)
    result = replace(result, model_best_params={result.best_model_name: {"collision": {1: "first", "1": "second"}}})
    with pytest.raises(ConfigurationError, match="keys collide"):
        result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=False))
    assert client.search_runs([experiment_id]) == []


@pytest.mark.parametrize("existing_env", [None, "sqlite:///caller.db"])
def test_tracking_uri_environment_is_restored_exactly(binary_df, local_tracking, monkeypatch, existing_env):
    uri, _, _, _ = local_tracking
    if existing_env is None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", existing_env)
    original_uri = mlflow.get_tracking_uri()
    result = fit_result(binary_df)
    result.log_mlflow(config=MLflowConfig(tracking_uri=uri, log_model=False))
    assert mlflow.get_tracking_uri() == original_uri
    assert os.environ.get("MLFLOW_TRACKING_URI") == existing_env
