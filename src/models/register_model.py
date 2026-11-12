"""
src/models/register_model.py
────────────────────────────
Model Registry and Promotion Pipeline for Bengaluru Urban Rainwater Harvesting.

Connects to MLflow tracking registry (supporting local SQLite 'sqlite:///mlflow.db'
and DAGsHub remote 'https://dagshub.com/sam123364/mlops_new.mlflow').

Capabilities:
  1. Select the best probabilistic run from 'rainfall_baseline_evaluation' by mean
     pinball loss (Q10, Q50, Q90). An empty experiment is populated only from a real
     backtest metrics file (--metrics-file); metrics are never hard-coded.
  2. Formally register model under 'bengaluru_rainfall_probabilistic_model'.
  3. Log model governance metadata tags: git_commit, dvc_data_version, author,
     target_quantiles ([0.10, 0.50, 0.90]), and methodology.
  4. Automated promotion checks: Transition stage from 'None' to 'Staging',
     and promote_to_production() based on benchmark thresholds:
       - MAE < 35 mm
       - 80% coverage >= 75%
  5. CLI interface for manual and CI/CD automated invocation:
       python -m src.models.register_model --promote
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException, RestException
from mlflow.tracking import MlflowClient

logger = logging.getLogger("register_model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Defaults
DEFAULT_LOCAL_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_DAGSHUB_TRACKING_URI = "https://dagshub.com/sam123364/mlops_new.mlflow"
DEFAULT_EXPERIMENT_NAME = "rainfall_baseline_evaluation"
DEFAULT_REGISTERED_MODEL_NAME = "bengaluru_rainfall_probabilistic_model"

# Benchmark Promotion Criteria
BENCHMARK_MAX_MAE_MM = 35.0
BENCHMARK_MIN_COVERAGE_80PCT = 75.0
TARGET_QUANTILES = [0.10, 0.50, 0.90]


def get_git_commit(repo_root: Optional[Path] = None) -> str:
    """Retrieve current Git commit SHA."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        cwd = repo_root or Path.cwd()
        output = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception as e:
        logger.warning("Could not determine git commit SHA: %s", e)
        return os.environ.get("GIT_COMMIT", "unknown")


def get_dvc_data_version(repo_root: Optional[Path] = None) -> str:
    """Extract primary DVC data hash from .dvc files or return fallback."""
    root = repo_root or Path.cwd()
    data_proc = root / "data" / "processed"
    if data_proc.exists():
        for dvc_file in data_proc.glob("*.dvc"):
            try:
                import yaml

                with open(dvc_file, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
                for out in content.get("outs", []):
                    if "md5" in out:
                        return str(out["md5"])
            except Exception as e:
                logger.debug("Could not read %s: %s", dvc_file, e)

    return os.environ.get("DVC_DATA_VERSION", "v1.0.0-phase1-features")


def configure_mlflow(tracking_uri: Optional[str] = None) -> str:
    """
    Resolve and configure tracking URI and credentials.
    Priority: tracking_uri arg -> MLFLOW_TRACKING_URI env var -> DEFAULT_LOCAL_TRACKING_URI.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_LOCAL_TRACKING_URI

    # DAGsHub token support
    if "dagshub.com" in uri:
        token = os.environ.get("DAGSHUB_USER_TOKEN")
        if token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("MLFLOW_TRACKING_USERNAME", token)
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("MLFLOW_TRACKING_PASSWORD", token)

    mlflow.set_tracking_uri(uri)
    logger.info("Configured MLflow tracking URI: %s", uri)
    return uri


def _file_md5(path: Path) -> str:
    import hashlib

    return hashlib.md5(path.read_bytes()).hexdigest()


def log_runs_from_metrics_file(
    client: MlflowClient,
    experiment_name: str,
    metrics_path: Path,
) -> int:
    """Log one MLflow run per model from a backtest metrics file.

    Used only when the experiment has no runs yet (fresh database, CI). The file is
    written by ``src.evaluation.run_baseline_backtest`` from a real backtest, so every
    logged number is traceable to a computed result. Nothing here is hard-coded.

    Returns the number of runs logged (0 if the experiment already had runs).
    """
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(experiment_name)
        exp = client.get_experiment_by_name(experiment_name)

    if client.search_runs(experiment_ids=[exp.experiment_id], max_results=1):
        return 0

    metrics_path = Path(metrics_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    file_md5 = _file_md5(metrics_path)
    models = payload["models"]

    for model_name, model_metrics in models.items():
        with mlflow.start_run(experiment_id=exp.experiment_id, run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("target", payload.get("target", "monthly_rain_mm"))
            mlflow.log_param("eval_start_year", payload.get("eval_start_year"))
            mlflow.log_param("eval_end_year", payload.get("eval_end_year"))
            mlflow.log_param("eval_samples_count", payload.get("eval_samples_count"))
            for k, v in model_metrics.items():
                mlflow.log_metric(k, float(v))
            mlflow.set_tag("metrics_source", str(metrics_path))
            mlflow.set_tag("metrics_source_md5", file_md5)
            mlflow.set_tag("metrics_git_commit", payload.get("git_commit", "unknown"))
            mlflow.set_tag("workstream", "D_baselines")

            # Model card only: baselines are refit from data, no serialized estimator yet.
            card = {
                "model_name": model_name,
                "metrics": model_metrics,
                "metrics_source": str(metrics_path),
                "metrics_source_md5": file_md5,
                "note": "No serialized estimator. Baselines are refit from the DVC-tracked data.",
            }
            mlflow.log_dict(card, "model/model_card.json")

    logger.info("Logged %d runs from metrics file %s (md5 %s)", len(models), metrics_path, file_md5)
    return len(models)


def _mean_pinball(metrics: Dict[str, float]) -> Optional[float]:
    keys = [f"pinball_loss_q{int(q * 100):02d}" for q in TARGET_QUANTILES]
    if not all(k in metrics for k in keys):
        return None
    return sum(metrics[k] for k in keys) / len(keys)


def fetch_best_baseline_run(
    client: MlflowClient,
    experiment_name: str,
    candidate_names: Optional[List[str]] = None,
) -> Run:
    """Pick the best probabilistic model run by mean pinball loss (Q10, Q50, Q90).

    Only runs that report all three pinball losses and ``coverage_80pct`` are eligible,
    because the registry holds a probabilistic (quantile) model. Ties are broken by MAE.
    ``candidate_names`` optionally restricts the search to those run/model names.
    """
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=500)
    if not runs:
        raise RuntimeError(f"No runs found in experiment '{experiment_name}'.")

    eligible = []
    for r in runs:
        name = r.info.run_name or r.data.params.get("model_name", "")
        if candidate_names and name not in candidate_names and r.data.params.get("model_name") not in candidate_names:
            continue
        m = r.data.metrics
        score = _mean_pinball(m)
        if score is None or "coverage_80pct" not in m:
            continue
        mae = m.get("mae", m.get("mae_median", float("inf")))
        eligible.append((score, mae, r))

    if not eligible:
        raise RuntimeError(
            f"No run in '{experiment_name}' reports pinball losses for quantiles "
            f"{TARGET_QUANTILES} and coverage_80pct; cannot select a probabilistic model."
        )

    eligible.sort(key=lambda t: (t[0], t[1]))
    score, mae, best = eligible[0]
    logger.info(
        "Selected run '%s' (Run ID: %s): mean pinball loss %.3f, MAE %.2f mm",
        best.info.run_name, best.info.run_id, score, mae,
    )
    return best


def register_model_version(
    client: MlflowClient,
    run: Run,
    registered_model_name: str,
    repo_root: Optional[Path] = None,
) -> ModelVersion:
    """
    Register the run's model into the MLflow Model Registry and attach governance metadata tags.
    Transitions initial stage from 'None' to 'Staging'.
    """
    model_exists = False
    try:
        client.get_registered_model(registered_model_name)
        model_exists = True
    except (RestException, MlflowException):
        model_exists = False

    if not model_exists:
        logger.info("Registered model '%s' does not exist yet. Creating it...", registered_model_name)
        try:
            client.create_registered_model(
                name=registered_model_name,
                description="Probabilistic rainfall baseline & forecast models for Bengaluru Urban Rainwater Harvesting.",
            )
        except (RestException, MlflowException) as e:
            logger.debug("create_registered_model handled: %s", e)

    model_uri = f"runs:/{run.info.run_id}/model"
    logger.info("Registering model from URI: %s", model_uri)

    try:
        mv = client.create_model_version(
            name=registered_model_name,
            source=model_uri,
            run_id=run.info.run_id,
            description="Probabilistic model evaluated on historical IMD/KSNDMC rainfall records.",
        )
    except Exception as e:
        logger.warning("Direct create_model_version encountered '%s', using mlflow.register_model", e)
        mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

    version = mv.version

    git_commit = get_git_commit(repo_root)
    dvc_data_version = get_dvc_data_version(repo_root)
    author = "Samiksha / Workstream A & D"
    quantiles_str = json.dumps(TARGET_QUANTILES)
    methodology = "Probabilistic empirical quantiles and quantile regression for urban rainwater harvesting"

    tags = {
        "git_commit": git_commit,
        "dvc_data_version": dvc_data_version,
        "author": author,
        "target_quantiles": quantiles_str,
        "methodology": methodology,
        "run_id": run.info.run_id,
    }

    for k, v in tags.items():
        client.set_model_version_tag(name=registered_model_name, version=version, key=k, value=str(v))

    logger.info("Model version %s registered with metadata tags: %s", version, tags)

    # Transition from None to Staging
    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.info("Model version %s transitioned from 'None' to 'Staging'.", version)

    return client.get_model_version(name=registered_model_name, version=version)


def promote_to_production(
    client: MlflowClient,
    registered_model_name: str,
    model_version: ModelVersion,
    run: Optional[Run] = None,
) -> Tuple[bool, str]:
    """
    Evaluate production promotion criteria:
      - MAE < 35 mm
      - 80% coverage >= 75%
    """
    version = model_version.version
    run_id = model_version.run_id or (run.info.run_id if run else None)

    if not run and run_id:
        run = client.get_run(run_id)

    metrics = run.data.metrics if run else {}
    mae = metrics.get("mae", metrics.get("mae_median", metrics.get("mae_mm", float("inf"))))
    coverage_80 = metrics.get("coverage_80pct", 0.0)

    logger.info(
        "Evaluating promotion benchmark criteria for model version %s: MAE=%.2f mm (max %.2f), Coverage=%.2f%% (min %.2f%%)",
        version,
        mae,
        BENCHMARK_MAX_MAE_MM,
        coverage_80,
        BENCHMARK_MIN_COVERAGE_80PCT,
    )

    passes_mae = mae < BENCHMARK_MAX_MAE_MM
    passes_coverage = coverage_80 >= BENCHMARK_MIN_COVERAGE_80PCT

    if passes_mae and passes_coverage:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        try:
            client.set_registered_model_alias(name=registered_model_name, alias="production", version=str(version))
        except Exception as e:
            logger.debug("Alias setting not supported or failed: %s", e)

        msg = f"PROMOTED: Model version {version} met all criteria (MAE={mae:.2f} < {BENCHMARK_MAX_MAE_MM}, Coverage={coverage_80:.2f}% >= {BENCHMARK_MIN_COVERAGE_80PCT}%)."
        logger.info(msg)
        return True, msg
    else:
        reasons = []
        if not passes_mae:
            reasons.append(f"MAE ({mae:.2f} mm) exceeds threshold ({BENCHMARK_MAX_MAE_MM} mm)")
        if not passes_coverage:
            reasons.append(f"80% coverage ({coverage_80:.2f}%) is below threshold ({BENCHMARK_MIN_COVERAGE_80PCT}%)")

        reason_str = "; ".join(reasons)
        msg = f"NOT PROMOTED: Model version {version} kept in Staging. Benchmark not satisfied: {reason_str}."
        logger.warning(msg)
        return False, msg


def run_pipeline(
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    promote: bool = False,
    repo_root: Optional[Path] = None,
    metrics_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute end-to-end registration and promotion pipeline.

    If the experiment is empty, runs are logged from ``metrics_file`` (a real backtest
    output). With no runs and no metrics file, this fails rather than invent metrics.
    """
    uri = configure_mlflow(tracking_uri)
    client = MlflowClient(tracking_uri=uri)

    if metrics_file is not None:
        log_runs_from_metrics_file(client, experiment_name=experiment_name, metrics_path=Path(metrics_file))

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None or not client.search_runs(experiment_ids=[exp.experiment_id], max_results=1):
        raise RuntimeError(
            f"Experiment '{experiment_name}' has no runs. Run the backtest first "
            "(python -m src.evaluation.run_baseline_backtest) or pass --metrics-file."
        )

    best_run = fetch_best_baseline_run(client, experiment_name=experiment_name)

    mv = register_model_version(
        client=client,
        run=best_run,
        registered_model_name=model_name,
        repo_root=repo_root,
    )

    promoted = False
    promo_msg = "Promotion not requested (--promote flag omitted)."
    if promote:
        promoted, promo_msg = promote_to_production(
            client=client,
            registered_model_name=model_name,
            model_version=mv,
            run=best_run,
        )

    current_mv = client.get_model_version(name=model_name, version=mv.version)

    return {
        "model_name": model_name,
        "model_version": current_mv.version,
        "stage": current_mv.current_stage,
        "run_id": best_run.info.run_id,
        "run_name": best_run.info.run_name,
        "promoted": promoted,
        "promotion_message": promo_msg,
        "tags": current_mv.tags,
    }


def main():
    parser = argparse.ArgumentParser(description="Register and promote rainfall baseline model in MLflow.")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (e.g. sqlite:///mlflow.db or DAGsHub remote URI).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name containing baseline runs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_REGISTERED_MODEL_NAME,
        help="Registered model name in MLflow Model Registry.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Run automated benchmark check to promote from Staging to Production if criteria are met.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Backtest metrics JSON (e.g. reports/baseline_metrics.json) used to populate an empty experiment.",
    )

    args = parser.parse_args()

    result = run_pipeline(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        promote=args.promote,
        metrics_file=args.metrics_file,
    )

    print("\n================ MODEL REGISTRATION REPORT ================")
    print(f"Registered Model:  {result['model_name']}")
    print(f"Model Version:     {result['model_version']}")
    print(f"Current Stage:     {result['stage']}")
    print(f"Source Run ID:     {result['run_id']} ({result['run_name']})")
    print(f"Promotion Status:  {result['promotion_message']}")
    print(f"Tags Recorded:     {json.dumps(result['tags'], indent=2)}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
