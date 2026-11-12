"""Servable, location-aware monthly rainfall model (one bundle per registered version).

A bundle is a directory holding everything needed to answer a query, frozen at
packaging time so that a registered version always gives the same answers:

- ``meta.json``: model kind (``B2`` or ``M4``), last observed month, forecast month,
  backtest metrics, champion rule, git commit and data hashes.
- ``cells.csv``: IMD 0.25 degree cells (centre, bounds, ``in_service``).
- ``boundary.geojson``: Bengaluru urban-region boundary used to accept locations.
- ``climatology.csv``: per cell and calendar month, empirical Q10/Q50/Q90 over all
  observed years (the B2 model fitted on the full record).
- ``backtest.csv``: out-of-sample predictions for 1970 to the last observed year
  (each year predicted from earlier years only) with the observed rainfall.
- M4 only: ``forecast_features.csv`` (one feature row per cell for the forecast month)
  and ``boosters/<cell_id>_qXX.txt`` (LightGBM models trained on all observed months).

``predict`` takes rows of ``cell_id, year, month`` and returns quantiles plus a ``basis``:

- ``forecast``: the month right after the last observed month (1-month lead), from the
  bundle's model.
- ``climatology``: later months. Past-month predictors do not exist yet, so only the
  climatological quantiles are honest.
- ``backtest``: observed months from the first backtest year on; the stored
  out-of-sample prediction and the observed value.
- ``unavailable``: months before the backtest period.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.main_model import QUANTILES, sqrt_to_mm_quantiles

QCOLS = ["q10", "q50", "q90"]


def _qname(q: float) -> str:
    return f"q{round(q * 100):02d}"


def _ym(year: int, month: int) -> int:
    return int(year) * 12 + int(month) - 1


@dataclass
class CellRainfallBundle:
    meta: dict
    cells: pd.DataFrame
    boundary: gpd.GeoDataFrame
    climatology: pd.DataFrame
    backtest: pd.DataFrame
    forecast: pd.DataFrame  # cell_id, q10, q50, q90 for meta["forecast_month"]

    # ---------- loading ----------
    @classmethod
    def load(cls, root: Path | str) -> CellRainfallBundle:
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text())
        cells = pd.read_csv(root / "cells.csv")
        clim = pd.read_csv(root / "climatology.csv")
        backtest = pd.read_csv(root / "backtest.csv")
        boundary = gpd.read_file(root / "boundary.geojson")
        if meta["kind"] == "M4":
            forecast = cls._run_boosters(root, meta)
        elif meta["kind"] == "B2":
            m = meta["forecast_month"]["month"]
            forecast = clim[clim["month"] == m][["cell_id", *QCOLS]].reset_index(drop=True)
        else:
            raise ValueError(f"unknown model kind {meta['kind']!r}")
        return cls(meta, cells, boundary, clim, backtest, forecast)

    @staticmethod
    def _run_boosters(root: Path, meta: dict) -> pd.DataFrame:
        """Run the stored LightGBM boosters on the stored forecast-month features."""
        feats = pd.read_csv(root / "forecast_features.csv")
        features = meta["features"]
        rows = []
        for r in feats.itertuples(index=False):
            x = pd.DataFrame([r._asdict()])
            raw = {}
            for q in QUANTILES:
                booster = lgb.Booster(model_file=str(root / "boosters" / f"{r.cell_id}_{_qname(q)}.txt"))
                init = np.sqrt(np.clip(x[f"clim_{_qname(q)}_exp"].to_numpy(dtype=float), 0, None))
                raw[q] = booster.predict(x[features]) + init
            mm = sqrt_to_mm_quantiles(raw)
            rows.append({"cell_id": r.cell_id, **{_qname(q): float(mm[q][0]) for q in QUANTILES}})
        return pd.DataFrame(rows)

    # ---------- location ----------
    def locate(self, lat: float, lon: float) -> dict | None:
        """The in-service cell containing (lat, lon), or None if outside the service area."""
        from shapely.geometry import Point

        if not self.boundary.union_all().covers(Point(lon, lat)):
            return None
        c = self.cells[self.cells["in_service"]]
        inside = c[(c["lat_min"] <= lat) & (lat <= c["lat_max"]) & (c["lon_min"] <= lon) & (lon <= c["lon_max"])]
        if inside.empty:
            return None
        # On a shared edge, take the nearest centre (ties: first in table order).
        d = (inside["lat"] - lat).abs() + (inside["lon"] - lon).abs()
        return inside.loc[d.idxmin()].to_dict()

    # ---------- prediction ----------
    def predict(self, query: pd.DataFrame) -> pd.DataFrame:
        """Quantiles for each ``cell_id, year, month`` row (see module docstring)."""
        fm, start = self.meta["forecast_month"], self.meta["backtest_start_year"]
        f_ym = _ym(fm["year"], fm["month"])
        clim = self.climatology.set_index(["cell_id", "month"])
        back = self.backtest.set_index(["cell_id", "year", "month"])
        fc = self.forecast.set_index("cell_id")
        out = []
        for r in query.itertuples(index=False):
            cid, y, m = str(r.cell_id), int(r.year), int(r.month)
            ref = clim.loc[(cid, m), QCOLS].astype(float).tolist()
            row = {"cell_id": cid, "year": y, "month": m, "observed_mm": np.nan,
                   "ref_q10": ref[0], "ref_q50": ref[1], "ref_q90": ref[2]}
            ym = _ym(y, m)
            if ym == f_ym:
                row.update(basis="forecast", **fc.loc[cid, QCOLS].astype(float).to_dict())
            elif ym > f_ym:
                row.update(basis="climatology", q10=ref[0], q50=ref[1], q90=ref[2])
            elif y >= start and (cid, y, m) in back.index:
                b = back.loc[(cid, y, m)]
                row.update(basis="backtest", q10=float(b["q10"]), q50=float(b["q50"]), q90=float(b["q90"]),
                           observed_mm=float(b["observed_mm"]),
                           ref_q10=float(b["b2_q10"]), ref_q50=float(b["b2_q50"]), ref_q90=float(b["b2_q90"]))
            else:
                row.update(basis="unavailable", q10=np.nan, q50=np.nan, q90=np.nan)
            out.append(row)
        return pd.DataFrame(out, columns=["cell_id", "year", "month", "basis", *QCOLS, "observed_mm",
                                          "ref_q10", "ref_q50", "ref_q90"])


try:  # MLflow wrapper; the bundle itself does not need MLflow.
    import mlflow.pyfunc

    class CellRainfallPyfunc(mlflow.pyfunc.PythonModel):
        """MLflow pyfunc wrapper around a ``CellRainfallBundle`` (artifact key ``bundle``)."""

        def load_context(self, context) -> None:
            self.bundle = CellRainfallBundle.load(context.artifacts["bundle"])

        def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
            return self.bundle.predict(model_input)

except ImportError:  # pragma: no cover
    CellRainfallPyfunc = None  # type: ignore[assignment,misc]
