"""Expanding-window backtest of the main LightGBM quantile model against the baselines.

Protocol (same as the baselines in notebooks/02):
- For each test year T in [eval_start, eval_end], fit on all months of years < T and
  predict the 12 months of year T. No random splits.
- Hyperparameters are tuned only on validation years before eval_start (default
  1940-1969) with the same expanding window, then frozen for the test period.

Usage:
    python -m src.evaluation.main_model_backtest            # tune, backtest, write reports/
    python -m src.evaluation.main_model_backtest --log-mlflow
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.evaluation.backtest import (
    ExpandingWindowEvaluator,
    compute_interval_coverage,
    compute_mae,
    compute_pinball_loss,
)
from src.features.model_features import (
    CLIMATE_FEATURES,
    RAINFALL_FEATURES,
    build_model_features,
)
from src.ingest.climate_indices import load_climate_indices
from src.models.main_model import QUANTILES, LGBMQuantileModel, LGBMQuantileParams

BASELINE_FEATURES = [
    "month_sin", "month_cos", "rain_lag1m", "rain_lag2m", "wet_days_lag1m",
    "rolling_3m_sum", "rolling_12m_sum", "departure_pct_lag1m",
]
SEASONS = {1: "Winter", 2: "Winter", 3: "Pre-Monsoon", 4: "Pre-Monsoon", 5: "Pre-Monsoon",
           6: "SW-Monsoon", 7: "SW-Monsoon", 8: "SW-Monsoon", 9: "SW-Monsoon",
           10: "NE-Post-Monsoon", 11: "NE-Post-Monsoon", 12: "NE-Post-Monsoon"}


def load_params(path: Path = Path("params.yaml")) -> dict:
    return yaml.safe_load(path.read_text())["main_model"]


def load_feature_table(
    monthly_path: Path = Path("data/processed/features_bengaluru_monthly.csv"),
    daily_path: Path = Path("data/processed/clean_bengaluru_imd_daily_1901_2025.csv"),
) -> pd.DataFrame:
    monthly = pd.read_csv(monthly_path)
    daily = pd.read_csv(daily_path, usecols=["date", "rain_urban_mean_mm"])
    return build_model_features(monthly, daily, load_climate_indices())


def backtest_model(
    df: pd.DataFrame, features: list[str], params: LGBMQuantileParams,
    years: list[int], calib_years: int = 0, init_from_climatology: bool = False,
) -> pd.DataFrame:
    """Expanding-window predictions for ``years``; returns year, month, actual, q10/q50/q90."""

    def one_year(t: int) -> pd.DataFrame:
        train, test = df[df["year"] < t], df[df["year"] == t]
        model = LGBMQuantileModel(features=features, params=params, calib_years=calib_years,
                                  init_from_climatology=init_from_climatology).fit(train)
        p = model.predict_all_quantiles(test)
        return pd.DataFrame({"year": t, "month": test["month"].to_numpy(),
                             "actual": test["monthly_rain_mm"].to_numpy(),
                             "q10": p[0.10], "q50": p[0.50], "q90": p[0.90]})

    parts = Parallel(n_jobs=-1)(delayed(one_year)(t) for t in years)
    return pd.concat(parts, ignore_index=True)


def score(pred: pd.DataFrame, prefix: str = "") -> dict[str, float]:
    y = pred["actual"].to_numpy()
    m = {f"pinball_loss_q{int(q * 100):02d}": compute_pinball_loss(y, pred[f"{prefix}q{int(q * 100):02d}"].to_numpy(), q)
         for q in QUANTILES}
    m["mean_pinball_loss"] = float(np.mean(list(m.values())))
    m["mae_median"] = compute_mae(y, pred[f"{prefix}q50"].to_numpy())
    m["coverage_80pct"] = compute_interval_coverage(y, pred[f"{prefix}q10"].to_numpy(), pred[f"{prefix}q90"].to_numpy())
    return m


def tune(df: pd.DataFrame, features: list[str], grid: dict, val_years: list[int],
         init_from_climatology: bool = False) -> tuple[dict, list[dict]]:
    """Grid search on validation years only. Returns best param dict and all results."""
    keys = list(grid)
    results = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = dict(zip(keys, combo))
        pred = backtest_model(df, features, LGBMQuantileParams(**p), val_years,
                              init_from_climatology=init_from_climatology)
        results.append({**p, "val_mean_pinball": score(pred)["mean_pinball_loss"]})
    best = min(results, key=lambda r: r["val_mean_pinball"])
    return {k: best[k] for k in keys}, results


def year_block_bootstrap(loss_a: pd.Series, loss_b: pd.Series, years: pd.Series, n: int = 2000, seed: int = 42) -> dict:
    """95% CI of mean(loss_a - loss_b), resampling whole years (keeps within-year dependence)."""
    diff = (loss_a - loss_b).groupby(years).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    sums, counts = diff["sum"].to_numpy()[idx].sum(1), diff["count"].to_numpy()[idx].sum(1)
    boot = sums / counts
    point = diff["sum"].sum() / diff["count"].sum()
    return {"mean_diff": float(point), "ci95_low": float(np.percentile(boot, 2.5)),
            "ci95_high": float(np.percentile(boot, 97.5)), "p_model_worse_or_equal": float(np.mean(boot >= 0))}


def row_pinball(pred: pd.DataFrame, prefix: str) -> pd.Series:
    y = pred["actual"]
    total = 0
    for q in QUANTILES:
        e = y - pred[f"{prefix}q{int(q * 100):02d}"]
        total = total + np.maximum(q * e, (q - 1) * e)
    return total / len(QUANTILES)


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(params_path: Path = Path("params.yaml"), out_dir: Path = Path("reports")) -> dict:
    cfg = load_params(params_path)
    df = load_feature_table()
    test_years = list(range(cfg["eval_start_year"], cfg["eval_end_year"] + 1))
    val_years = list(range(cfg["tune_start_year"], cfg["eval_start_year"]))

    # Baselines: the existing harness, unchanged (same numbers as notebook 02).
    ev = ExpandingWindowEvaluator(min_train_years=30, eval_start_year=test_years[0], eval_end_year=test_years[-1])
    base_pred, base_metrics = ev.run_monthly_evaluation(pd.read_csv("data/processed/features_bengaluru_monthly.csv"),
                                                        feature_cols=BASELINE_FEATURES)

    variants = {
        "M1_lgbm_rainfall_features": RAINFALL_FEATURES,
        "M2_lgbm_with_climate_indices": RAINFALL_FEATURES + CLIMATE_FEATURES,
    }
    tuned, preds, tuning_log = {}, {}, {}
    for name, feats in variants.items():
        best, log = tune(df, feats, cfg["tuning_grid"], val_years)
        tuned[name], tuning_log[name] = best, log
        preds[name] = backtest_model(df, feats, LGBMQuantileParams(**best), test_years)
    calib_name = "M3_lgbm_climate_conformal"
    tuned[calib_name] = tuned["M2_lgbm_with_climate_indices"]
    preds[calib_name] = backtest_model(df, variants["M2_lgbm_with_climate_indices"],
                                       LGBMQuantileParams(**tuned[calib_name]), test_years,
                                       calib_years=cfg["conformal_calib_years"])

    # Iteration 2: boost from the past-only climatological quantiles (B2) instead of from scratch.
    boosted = {
        "M4_lgbm_boosted_from_climatology": RAINFALL_FEATURES + CLIMATE_FEATURES,
        "M5_lgbm_boosted_rainfall_only": RAINFALL_FEATURES,
    }
    for name, feats in boosted.items():
        best, log = tune(df, feats, cfg["tuning_grid_boosted"], val_years, init_from_climatology=True)
        tuned[name], tuning_log[name] = best, log
        preds[name] = backtest_model(df, feats, LGBMQuantileParams(**best), test_years, init_from_climatology=True)
    variants.update(boosted)

    # Align every model on the same 672 rows.
    merged = base_pred[["year", "month", "actual", "b2_q10", "b2_q50", "b2_q90", "b4_q10", "b4_q50", "b4_q90"]].copy()
    for name, p in preds.items():
        tag = name.split("_")[0].lower()
        merged = merged.merge(p.drop(columns="actual").rename(columns={c: f"{tag}_{c}" for c in ("q10", "q50", "q90")}),
                              on=["year", "month"], how="left")
    assert merged.notna().all().all(), "prediction rows did not align"

    models = {"B2_empirical_quantiles": "b2_", "B4_quantile_regression": "b4_"}
    models.update({n: n.split("_")[0].lower() + "_" for n in preds})
    metrics = {n: score(merged, p) for n, p in models.items()}
    b2 = metrics["B2_empirical_quantiles"]
    for m in metrics.values():
        for q in ("q10", "q50", "q90"):
            m[f"skill_vs_b2_{q}"] = 1 - m[f"pinball_loss_{q}"] / b2[f"pinball_loss_{q}"]
        m["skill_vs_b2_mean"] = 1 - m["mean_pinball_loss"] / b2["mean_pinball_loss"]

    merged["season"] = merged["month"].map(SEASONS)
    by_season = {n: {s: score(g, p)["mean_pinball_loss"] for s, g in merged.groupby("season")} for n, p in models.items()}
    b2_loss = row_pinball(merged, "b2_")
    significance = {n: year_block_bootstrap(row_pinball(merged, p), b2_loss, merged["year"])
                    for n, p in models.items() if n != "B2_empirical_quantiles"}

    # Final main model (M4: boosted from climatology, all features) refit on all years
    # through eval_end_year; its boosters are the registered artifact.
    main_name = "M4_lgbm_boosted_from_climatology"
    final = LGBMQuantileModel(features=variants[main_name], params=LGBMQuantileParams(**tuned[main_name]),
                              init_from_climatology=True)
    final.fit(df[df["year"] <= test_years[-1]])
    gain = final.feature_importance(0.5)
    importance = (gain / gain.sum()).round(4).to_dict() if gain.sum() > 0 else {}

    payload = {
        "target": "monthly_rain_mm",
        "eval_start_year": test_years[0], "eval_end_year": test_years[-1],
        "eval_samples_count": len(merged),
        "tuning_years": [val_years[0], val_years[-1]],
        "git_commit": _git_commit(),
        "data_md5": {p: _md5(p) for p in ("data/processed/features_bengaluru_monthly.csv",
                                          "data/processed/clean_bengaluru_imd_daily_1901_2025.csv",
                                          "data/raw/climate_indices/nina34.anom.data",
                                          "data/raw/climate_indices/dmi.had.long.data")},
        "tuned_params": tuned,
        "models": {n: {k: round(v, 4) for k, v in m.items()} for n, m in metrics.items()},
        "mean_pinball_by_season": {n: {s: round(v, 3) for s, v in d.items()} for n, d in by_season.items()},
        "vs_b2_year_block_bootstrap": {n: {k: round(v, 4) for k, v in d.items()} for n, d in significance.items()},
        "main_model": main_name,
        "feature_importance_q50_gain_share": importance,
        "baseline_reference_metrics": {k: {a: round(b, 4) for a, b in v.items()} for k, v in base_metrics.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "main_model_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    merged.round(3).to_csv(out_dir / "main_model_predictions.csv", index=False)
    (out_dir / "main_model_tuning.json").write_text(json.dumps(tuning_log, indent=2) + "\n")
    return {"payload": payload, "merged": merged, "final_model": final}


def log_to_mlflow(result: dict, tracking_uri: str, experiment_name: str = "rainfall_baseline_evaluation") -> None:
    """Log each main-model variant as an MLflow run; the final M4 boosters are saved as artifacts."""
    import tempfile

    import mlflow

    payload = result["payload"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    for name, m in payload["models"].items():
        if not name.startswith("M"):
            continue
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_params({f"lgbm_{k}": v for k, v in payload["tuned_params"][name].items()})
            mlflow.log_param("eval_start_year", payload["eval_start_year"])
            mlflow.log_param("eval_end_year", payload["eval_end_year"])
            mlflow.log_param("eval_samples_count", payload["eval_samples_count"])
            mlflow.log_metrics({k: float(v) for k, v in m.items()})
            mlflow.set_tags({"git_commit": payload["git_commit"], "workstream": "D_main_model",
                             **{f"data_md5:{Path(p).name}": h for p, h in payload["data_md5"].items()}})
            mlflow.log_artifact("reports/main_model_metrics.json", artifact_path="evaluation")
            if name == payload["main_model"]:
                mlflow.set_tag("role", "main_model_candidate")
                with tempfile.TemporaryDirectory() as tmp:
                    for q, booster in result["final_model"].models_.items():
                        path = Path(tmp) / f"lgbm_q{int(q * 100):02d}.txt"
                        booster.booster_.save_model(str(path))
                        mlflow.log_artifact(str(path), artifact_path="model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-mlflow", action="store_true")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()
    result = run()
    p = result["payload"]
    cols = ["pinball_loss_q10", "pinball_loss_q50", "pinball_loss_q90", "mean_pinball_loss",
            "mae_median", "coverage_80pct", "skill_vs_b2_mean"]
    print(pd.DataFrame(p["models"]).T[cols].round(3).to_string())
    print("\nvs B2 (year-block bootstrap, mean pinball difference; negative = better than B2):")
    print(pd.DataFrame(p["vs_b2_year_block_bootstrap"]).T.to_string())
    print("\nTuned params:", json.dumps(p["tuned_params"]))
    if args.log_mlflow:
        log_to_mlflow(result, args.tracking_uri)


if __name__ == "__main__":
    main()
