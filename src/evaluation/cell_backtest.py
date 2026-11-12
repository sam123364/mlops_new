"""Per-grid-cell backtest: baselines (B2, B4) vs the main model (M4) for each IMD cell.

Same protocol as the city-mean backtest (``main_model_backtest.py``):
- For each test year T in 1970-2025, fit on years < T and predict the 12 months of T.
- M4 hyperparameters are tuned per cell on 1940-1969 only, then frozen.
- B2 and B4 come from the existing harness (``ExpandingWindowEvaluator``), unchanged.

Champion rule (decided before looking at the results, applied to all cells at once):
M4 serves only if its pooled mean pinball loss is lower than B2's with the year-block
bootstrap 95% CI of the difference entirely below zero. Otherwise B2 serves. Choosing
per cell would pick winners by chance across 18 comparisons, so it is not done.

Usage:
    python -m src.evaluation.cell_backtest
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.evaluation.backtest import ExpandingWindowEvaluator
from src.evaluation.main_model_backtest import (
    BASELINE_FEATURES,
    _git_commit,
    _md5,
    backtest_model,
    load_params,
    row_pinball,
    score,
    tune,
    year_block_bootstrap,
)
from src.features.cell_features import build_cell_feature_table
from src.features.model_features import ALL_FEATURES
from src.ingest.climate_indices import load_climate_indices
from src.models.main_model import LGBMQuantileParams

CELLS_DAILY = Path("data/processed/imd_cells_daily_1901_2025.csv")
CELLS_META = Path("data/processed/imd_cells.csv")
MAIN_MODEL_FEATURES = ALL_FEATURES  # M4: all features, boosted from climatology
OUT_METRICS = Path("reports/cell_model_metrics.json")
OUT_PREDS = Path("reports/cell_model_predictions.csv")


def load_cell_features(with_forecast_row: bool = False) -> pd.DataFrame:
    daily = pd.read_csv(CELLS_DAILY)
    return build_cell_feature_table(daily, load_climate_indices(), with_forecast_row=with_forecast_row)


def _baselines_one_cell(cell_df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    ev = ExpandingWindowEvaluator(min_train_years=30, eval_start_year=start, eval_end_year=end)
    pred, _ = ev.run_monthly_evaluation(cell_df, feature_cols=BASELINE_FEATURES)
    return pred[["year", "month", "actual", "b2_q10", "b2_q50", "b2_q90", "b4_q10", "b4_q50", "b4_q90"]]


def run(params_path: Path = Path("params.yaml")) -> dict:
    cfg = load_params(params_path)
    feats = load_cell_features()
    test_years = list(range(cfg["eval_start_year"], cfg["eval_end_year"] + 1))
    val_years = list(range(cfg["tune_start_year"], cfg["eval_start_year"]))
    cells = sorted(feats["cell_id"].unique())
    by_cell = {c: feats[feats["cell_id"] == c].reset_index(drop=True) for c in cells}

    base = Parallel(n_jobs=-1)(delayed(_baselines_one_cell)(by_cell[c], test_years[0], test_years[-1]) for c in cells)

    tuned, parts = {}, []
    for c, b in zip(cells, base):
        best, _ = tune(by_cell[c], MAIN_MODEL_FEATURES, cfg["tuning_grid_boosted"], val_years,
                       init_from_climatology=True)
        tuned[c] = best
        m4 = backtest_model(by_cell[c], MAIN_MODEL_FEATURES, LGBMQuantileParams(**best), test_years,
                            init_from_climatology=True)
        m4 = m4.drop(columns="actual").rename(columns={q: f"m4_{q}" for q in ("q10", "q50", "q90")})
        merged = b.merge(m4, on=["year", "month"], how="left")
        merged.insert(0, "cell_id", c)
        parts.append(merged)
        print(f"  {c}: tuned {best}")

    preds = pd.concat(parts, ignore_index=True)
    assert preds.notna().all().all(), "prediction rows did not align"

    prefixes = {"B2_empirical_quantiles": "b2_", "B4_quantile_regression": "b4_",
                "M4_lgbm_boosted_from_climatology": "m4_"}
    pooled = {n: score(preds, p) for n, p in prefixes.items()}
    per_cell = {c: {n: round(score(g, p)["mean_pinball_loss"], 4) for n, p in prefixes.items()}
                for c, g in preds.groupby("cell_id")}

    # Year-block bootstrap on the pooled loss: resample whole years, all cells together,
    # because cells share the same weather in a given year.
    b2_loss = row_pinball(preds, "b2_")
    pooled_vs_b2 = {n: year_block_bootstrap(row_pinball(preds, p), b2_loss, preds["year"])
                    for n, p in prefixes.items() if not n.startswith("B2")}
    per_cell_vs_b2 = {}
    for c, g in preds.groupby("cell_id"):
        boot = year_block_bootstrap(row_pinball(g, "m4_"), row_pinball(g, "b2_"), g["year"])
        per_cell_vs_b2[c] = {k: round(v, 4) for k, v in boot.items()}

    m4_boot = pooled_vs_b2["M4_lgbm_boosted_from_climatology"]
    champion = "M4" if m4_boot["ci95_high"] < 0 else "B2"

    payload = {
        "target": "monthly_rain_mm per IMD 0.25 degree cell",
        "n_cells": len(cells),
        "eval_start_year": test_years[0], "eval_end_year": test_years[-1],
        "eval_samples_count": len(preds),
        "tuning_years": [val_years[0], val_years[-1]],
        "git_commit": _git_commit(),
        "data_md5": {str(p): _md5(p) for p in (CELLS_DAILY, CELLS_META,
                                               Path("data/raw/climate_indices/nina34.anom.data"),
                                               Path("data/raw/climate_indices/dmi.had.long.data"))},
        "champion_rule": "M4 only if pooled year-block bootstrap 95% CI of (M4 - B2) mean pinball is entirely < 0",
        "champion": champion,
        "pooled_metrics": {n: {k: round(v, 4) for k, v in m.items()} for n, m in pooled.items()},
        "pooled_vs_b2_year_block_bootstrap": {n: {k: round(v, 4) for k, v in d.items()} for n, d in pooled_vs_b2.items()},
        "per_cell_mean_pinball": per_cell,
        "per_cell_m4_vs_b2_bootstrap": per_cell_vs_b2,
        "m4_tuned_params_per_cell": tuned,
    }
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.write_text(json.dumps(payload, indent=2) + "\n")
    preds.round(3).to_csv(OUT_PREDS, index=False)
    return {"payload": payload, "preds": preds}


def main() -> None:
    p = run()["payload"]
    print(pd.DataFrame(p["pooled_metrics"]).T[["mean_pinball_loss", "mae_median", "coverage_80pct"]].round(3).to_string())
    print("\nPooled vs B2 (negative = better than B2):")
    print(pd.DataFrame(p["pooled_vs_b2_year_block_bootstrap"]).T.to_string())
    better = sum(d["ci95_high"] < 0 for d in p["per_cell_m4_vs_b2_bootstrap"].values())
    worse = sum(d["ci95_low"] > 0 for d in p["per_cell_m4_vs_b2_bootstrap"].values())
    print(f"\nCells where M4 is significantly better / worse than B2: {better} / {worse} of {p['n_cells']}")
    print(f"Champion: {p['champion']}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
