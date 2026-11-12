"""Run the baseline expanding-window backtest and write its metrics to a JSON file.

The JSON is the single source of the numbers the model registry uses when it has to
populate an empty MLflow experiment (for example in CI). Settings match
``notebooks/02_baseline_models_and_evaluation.ipynb``.

Usage:
    python -m src.evaluation.run_baseline_backtest
    python -m src.evaluation.run_baseline_backtest --log-mlflow --tracking-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from src.evaluation.backtest import ExpandingWindowEvaluator, log_evaluation_to_mlflow

FEATURE_COLS = [
    "month_sin",
    "month_cos",
    "rain_lag1m",
    "rain_lag2m",
    "wet_days_lag1m",
    "rolling_3m_sum",
    "rolling_12m_sum",
    "departure_pct_lag1m",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(
    features_path: Path = Path("data/processed/features_bengaluru_monthly.csv"),
    output_path: Path = Path("reports/baseline_metrics.json"),
    eval_start_year: int = 1970,
    eval_end_year: int = 2025,
    min_train_years: int = 30,
) -> dict:
    """Run the backtest and write ``output_path``. Returns the written payload."""
    df = pd.read_csv(features_path)
    evaluator = ExpandingWindowEvaluator(
        min_train_years=min_train_years,
        eval_start_year=eval_start_year,
        eval_end_year=eval_end_year,
        quantiles=[0.10, 0.50, 0.90],
    )
    pred_df, metrics = evaluator.run_monthly_evaluation(df, feature_cols=FEATURE_COLS)

    payload = {
        "target": "monthly_rain_mm",
        "eval_start_year": int(pred_df["year"].min()),
        "eval_end_year": int(pred_df["year"].max()),
        "eval_samples_count": len(pred_df),
        "min_train_years": min_train_years,
        "feature_cols": FEATURE_COLS,
        "git_commit": _git_commit(),
        "models": {k: {m: round(float(v), 4) for m, v in d.items()} for k, d in metrics.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"payload": payload, "pred_df": pred_df, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=Path("data/processed/features_bengaluru_monthly.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/baseline_metrics.json"))
    parser.add_argument("--eval-start-year", type=int, default=1970)
    parser.add_argument("--eval-end-year", type=int, default=2025)
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    out = run(args.features, args.output, args.eval_start_year, args.eval_end_year)
    print(f"Wrote {args.output}")
    for name, m in out["payload"]["models"].items():
        print(f"  {name}: {m}")
    if args.log_mlflow:
        log_evaluation_to_mlflow(out["metrics"], out["pred_df"], tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
