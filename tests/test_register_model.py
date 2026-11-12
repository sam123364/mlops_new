"""
tests/test_register_model.py
────────────────────────────
Unit tests for the MLflow model registry and promotion pipeline (Workstream A & D).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src.models.register_model import (
    configure_mlflow,
    fetch_best_baseline_run,
    log_runs_from_metrics_file,
    promote_to_production,
    register_model_version,
    run_pipeline,
)


@pytest.fixture
def mlflow_test_env(tmp_path: Path):
    """Create isolated SQLite MLflow tracking registry."""
    db_path = tmp_path / "test_mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    configure_mlflow(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    return client, tracking_uri, tmp_path


def _write_metrics_file(path: Path) -> Path:
    """Small metrics file in the same format as reports/baseline_metrics.json."""
    payload = {
        "target": "monthly_rain_mm",
        "eval_start_year": 1970,
        "eval_end_year": 2025,
        "eval_samples_count": 672,
        "git_commit": "test",
        "models": {
            "B1_climatology_mean": {"mae": 34.0},
            "B2_empirical_quantiles": {
                "pinball_loss_q10": 5.0, "pinball_loss_q50": 16.0, "pinball_loss_q90": 11.0,
                "mae_median": 33.0, "coverage_80pct": 81.0,
            },
            "B4_quantile_regression": {
                "pinball_loss_q10": 6.0, "pinball_loss_q50": 20.0, "pinball_loss_q90": 12.0,
                "mae_median": 40.0, "coverage_80pct": 80.0,
            },
        },
    }
    path.write_text(json.dumps(payload))
    return path


def test_pipeline_logs_runs_from_metrics_file_and_picks_lowest_pinball(mlflow_test_env):
    client, tracking_uri, tmp_path = mlflow_test_env
    exp_name = "test_rainfall_baseline_evaluation"
    metrics_file = _write_metrics_file(tmp_path / "metrics.json")

    result = run_pipeline(
        tracking_uri=tracking_uri,
        experiment_name=exp_name,
        model_name="test_bengaluru_model",
        promote=False,
        repo_root=Path.cwd(),
        metrics_file=metrics_file,
    )

    assert result["stage"] == "Staging"
    assert result["run_name"] == "B2_empirical_quantiles"  # lowest mean pinball loss
    assert "git_commit" in result["tags"]
    assert "dvc_data_version" in result["tags"]

    run = client.get_run(result["run_id"])
    assert run.data.metrics["mae_median"] == 33.0
    assert run.data.tags["metrics_source_md5"]


def test_pipeline_refuses_empty_experiment_without_metrics_file(mlflow_test_env):
    _, tracking_uri, _ = mlflow_test_env
    with pytest.raises(RuntimeError, match="no runs"):
        run_pipeline(tracking_uri=tracking_uri, experiment_name="empty_exp", model_name="m")


def test_metrics_file_not_logged_twice(mlflow_test_env):
    client, _, tmp_path = mlflow_test_env
    metrics_file = _write_metrics_file(tmp_path / "metrics.json")
    assert log_runs_from_metrics_file(client, "dup_exp", metrics_file) == 3
    assert log_runs_from_metrics_file(client, "dup_exp", metrics_file) == 0


def test_selection_ignores_name_order_and_point_only_runs(mlflow_test_env):
    client, _, _ = mlflow_test_env
    exp_id = client.create_experiment("select_exp")
    for name, pinball in [("B4_quantile_regression", 9.0), ("B2_empirical_quantiles", 7.0)]:
        with mlflow.start_run(experiment_id=exp_id, run_name=name):
            for q in ("q10", "q50", "q90"):
                mlflow.log_metric(f"pinball_loss_{q}", pinball)
            mlflow.log_metric("coverage_80pct", 80.0)
    with mlflow.start_run(experiment_id=exp_id, run_name="B1_climatology_mean"):
        mlflow.log_metric("mae", 1.0)  # point model, not eligible

    best = fetch_best_baseline_run(client, "select_exp")
    assert best.info.run_name == "B2_empirical_quantiles"


def test_promote_to_production_gate_fail(mlflow_test_env):
    client, _, tmp_path = mlflow_test_env
    exp_name = "test_promotion_fail"

    client.create_experiment(exp_name)
    exp = client.get_experiment_by_name(exp_name)

    # Log model that fails MAE criteria: MAE = 50.0 > 35.0
    with mlflow.start_run(experiment_id=exp.experiment_id, run_name="B3_persistence_lag1"):
        mlflow.log_param("model_name", "B3_persistence_lag1")
        mlflow.log_metric("mae", 50.0)
        mlflow.log_metric("coverage_80pct", 60.0)
        for q in ("q10", "q50", "q90"):
            mlflow.log_metric(f"pinball_loss_{q}", 20.0)

        dummy_file = tmp_path / "model_spec.json"
        dummy_file.write_text(json.dumps({"name": "B3", "mae": 50.0}))
        mlflow.log_artifact(str(dummy_file), artifact_path="model")

    best_run = fetch_best_baseline_run(client, experiment_name=exp_name)
    model_name = "test_bengaluru_model_fail"

    mv = register_model_version(client, best_run, model_name, repo_root=Path.cwd())
    assert mv.current_stage == "Staging"

    promoted, msg = promote_to_production(client, model_name, mv, run=best_run)
    assert promoted is False
    assert "NOT PROMOTED" in msg

    updated_mv = client.get_model_version(model_name, mv.version)
    assert updated_mv.current_stage == "Staging"


def test_promote_to_production_gate_success(mlflow_test_env):
    client, _, tmp_path = mlflow_test_env
    exp_name = "test_promotion_pass"

    client.create_experiment(exp_name)
    exp = client.get_experiment_by_name(exp_name)

    # Log model that meets benchmark: MAE < 35 mm and Coverage >= 75%
    with mlflow.start_run(experiment_id=exp.experiment_id, run_name="B4_quantile_regression"):
        mlflow.log_param("model_name", "B4_quantile_regression")
        mlflow.log_metric("mae", 28.5)
        mlflow.log_metric("coverage_80pct", 82.0)
        for q in ("q10", "q50", "q90"):
            mlflow.log_metric(f"pinball_loss_{q}", 10.0)

        dummy_file = tmp_path / "model_spec.json"
        dummy_file.write_text(json.dumps({"name": "B4", "mae": 28.5}))
        mlflow.log_artifact(str(dummy_file), artifact_path="model")

    best_run = fetch_best_baseline_run(client, experiment_name=exp_name)
    model_name = "test_promoted_model"

    mv = register_model_version(client, best_run, model_name, repo_root=Path.cwd())
    assert mv.current_stage == "Staging"

    promoted, msg = promote_to_production(client, model_name, mv, run=best_run)
    assert promoted is True
    assert "PROMOTED" in msg

    updated_mv = client.get_model_version(model_name, mv.version)
    assert updated_mv.current_stage == "Production"
