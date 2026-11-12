"""Package the per-cell B2 and M4 models as MLflow pyfunc models and register them.

Reads the backtest outputs of ``src.evaluation.cell_backtest`` (run that first), builds
one bundle per model kind (see ``cell_model_bundle.py``), logs each as an MLflow run with
its metrics, git commit and data hashes, registers both as versions of
``bengaluru_rainfall_by_cell``, and points the ``production`` alias at the champion
chosen by the backtest's pre-set rule (the other gets ``challenger``).

Usage:
    python -m src.models.package_cell_model
    python -m src.models.package_cell_model --tracking-uri sqlite:///mlflow.db --no-register
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.evaluation.cell_backtest import (
    CELLS_META,
    MAIN_MODEL_FEATURES,
    OUT_METRICS,
    OUT_PREDS,
    load_cell_features,
)
from src.models.baseline_models import EmpiricalQuantileBaseline
from src.models.cell_model_bundle import CellRainfallBundle
from src.models.main_model import QUANTILES, LGBMQuantileModel, LGBMQuantileParams

REGISTERED_MODEL_NAME = "bengaluru_rainfall_by_cell"
EXPERIMENT_NAME = "rainfall_cell_models"
BOUNDARY = Path("data/raw/bengaluru_full_urban_region_imdweb.geojson")
KIND_NAMES = {"B2": "B2_empirical_quantiles", "M4": "M4_lgbm_boosted_from_climatology"}
LIMITATIONS = [
    ("Rainfall is the IMD 0.25 degree gridded product (about 27 km cells), interpolated from "
     "sparse stations: every location in a cell gets the same answer."),
    "No KSNDMC gauge adjustment yet, so neighbourhood-scale differences are not captured.",
    "Monthly totals only; a monthly time step hides intense short storms.",
    "Forecasts are 1 month ahead of the last observed month; later months are climatology.",
]


def _qname(q: float) -> str:
    return f"q{round(q * 100):02d}"


def build_bundle(kind: str, out_dir: Path, feats: pd.DataFrame, report: dict, preds: pd.DataFrame) -> Path:
    """Write a bundle directory for ``kind`` ('B2' or 'M4') and return it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    observed = feats[~feats["is_forecast_row"]]
    fc_rows = feats[feats["is_forecast_row"]]
    last = observed.sort_values(["year", "month"]).iloc[-1]
    first_fc = fc_rows.iloc[0]

    cells = pd.read_csv(CELLS_META)
    cells.to_csv(out_dir / "cells.csv", index=False)
    shutil.copy(BOUNDARY, out_dir / "boundary.geojson")

    clim_parts = []
    for cid, g in observed.groupby("cell_id"):
        b2 = EmpiricalQuantileBaseline(quantiles=list(QUANTILES)).fit(g, "month", "monthly_rain_mm")
        for m, qs in b2.quantile_table_.items():
            clim_parts.append({"cell_id": cid, "month": int(m), **{_qname(q): v for q, v in qs.items()}})
    pd.DataFrame(clim_parts).to_csv(out_dir / "climatology.csv", index=False)

    p = kind.lower() + "_"
    back = preds[["cell_id", "year", "month", "actual", f"{p}q10", f"{p}q50", f"{p}q90",
                  "b2_q10", "b2_q50", "b2_q90"]].copy()
    back.columns = ["cell_id", "year", "month", "observed_mm", "q10", "q50", "q90", "b2_q10", "b2_q50", "b2_q90"]
    back.to_csv(out_dir / "backtest.csv", index=False)

    meta = {
        "kind": kind,
        "model_name": KIND_NAMES[kind],
        "data_end": {"year": int(last["year"]), "month": int(last["month"])},
        "forecast_month": {"year": int(first_fc["year"]), "month": int(first_fc["month"])},
        "backtest_start_year": report["eval_start_year"],
        "backtest_end_year": report["eval_end_year"],
        "resolution_deg": 0.25,
        "features": MAIN_MODEL_FEATURES if kind == "M4" else [],
        "pooled_backtest_metrics": report["pooled_metrics"][KIND_NAMES[kind]],
        "b2_pooled_backtest_metrics": report["pooled_metrics"][KIND_NAMES["B2"]],
        "m4_vs_b2_bootstrap": report["pooled_vs_b2_year_block_bootstrap"][KIND_NAMES["M4"]],
        "champion": report["champion"],
        "champion_rule": report["champion_rule"],
        "git_commit": report["git_commit"],
        "data_md5": report["data_md5"],
        "limitations": LIMITATIONS,
    }

    if kind == "M4":
        (out_dir / "boosters").mkdir(exist_ok=True)
        fc_rows[["cell_id", "year", "month", *MAIN_MODEL_FEATURES]].to_csv(out_dir / "forecast_features.csv", index=False)
        for cid, g in observed.groupby("cell_id"):
            params = LGBMQuantileParams(**report["m4_tuned_params_per_cell"][cid])
            model = LGBMQuantileModel(features=MAIN_MODEL_FEATURES, params=params, init_from_climatology=True).fit(g)
            for q, est in model.models_.items():
                est.booster_.save_model(str(out_dir / "boosters" / f"{cid}_{_qname(q)}.txt"))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def _pip_requirements() -> list[str]:
    pkgs = ["mlflow", "lightgbm", "pandas", "pyarrow", "numpy", "geopandas", "shapely"]  # inference only
    return [f"{p}=={md.version(p)}" for p in pkgs]


def register(bundles: dict[str, Path], report: dict, tracking_uri: str, do_register: bool = True) -> dict:
    import mlflow
    from mlflow import MlflowClient

    from src.models.cell_model_bundle import CellRainfallPyfunc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    versions = {}
    for kind, path in bundles.items():
        bundle = CellRainfallBundle.load(path)
        example = pd.DataFrame({"cell_id": [bundle.cells.loc[bundle.cells["in_service"], "cell_id"].iloc[0]],
                                "year": [bundle.meta["forecast_month"]["year"]],
                                "month": [bundle.meta["forecast_month"]["month"]]})
        with mlflow.start_run(run_name=f"{KIND_NAMES[kind]}_by_cell") as run:
            mlflow.log_params({"kind": kind, "n_cells": report["n_cells"],
                               "data_end": f"{bundle.meta['data_end']['year']}-{bundle.meta['data_end']['month']:02d}",
                               "backtest_years": f"{report['eval_start_year']}-{report['eval_end_year']}"})
            mlflow.log_metrics({k: float(v) for k, v in bundle.meta["pooled_backtest_metrics"].items()})
            mlflow.set_tags({"git_commit": report["git_commit"], "champion": str(report["champion"] == kind),
                             **{f"data_md5:{Path(p).name}": h for p, h in report["data_md5"].items()}})
            mlflow.log_artifact(str(OUT_METRICS), artifact_path="evaluation")
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=CellRainfallPyfunc(),
                artifacts={"bundle": str(path)},
                code_paths=["src"],
                input_example=example,
                pip_requirements=_pip_requirements(),
                registered_model_name=REGISTERED_MODEL_NAME if do_register else None,
            )
        versions[kind] = {"run_id": run.info.run_id, "model_uri": info.model_uri,
                          "version": getattr(info, "registered_model_version", None)}
    if do_register:
        champ = report["champion"]
        for kind, v in versions.items():
            alias = "production" if kind == champ else "challenger"
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, str(v["version"]))
            client.set_model_version_tag(REGISTERED_MODEL_NAME, str(v["version"]), "kind", kind)
            client.set_model_version_tag(REGISTERED_MODEL_NAME, str(v["version"]), "champion_rule", report["champion_rule"])
            v["alias"] = alias
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--no-register", action="store_true", help="log runs but do not register versions")
    args = parser.parse_args()

    report = json.loads(OUT_METRICS.read_text())
    preds = pd.read_csv(OUT_PREDS)
    feats = load_cell_features(with_forecast_row=True)
    with tempfile.TemporaryDirectory() as tmp:
        bundles = {k: build_bundle(k, Path(tmp) / k, feats, report, preds) for k in ("B2", "M4")}
        versions = register(bundles, report, args.tracking_uri, do_register=not args.no_register)
    print(f"Champion by rule: {report['champion']}")
    for kind, v in versions.items():
        print(f"  {kind}: version {v.get('version')} alias {v.get('alias', '-')} run {v['run_id']}")


if __name__ == "__main__":
    main()
