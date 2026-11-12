"""Load the registered rainfall model and answer location-based queries."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.cell_model_bundle import CellRainfallBundle
from src.serving.schemas import (
    CellInfo,
    ModelCard,
    ModelInfo,
    Prediction,
    PredictRequest,
    Quantiles,
)

DEFAULT_MODEL_URI = "models:/bengaluru_rainfall_by_cell@production"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

BASIS_NOTES = {
    "forecast": "1-month-ahead forecast from the month after the last observed month.",
    "climatology": "Beyond the 1-month lead: climatological quantiles for this cell and calendar month.",
    "backtest": "Observed month: out-of-sample prediction made from earlier years only, with the observed value.",
}


class OutsideServiceArea(ValueError):
    pass


class PeriodUnavailable(ValueError):
    pass


@dataclass
class Predictor:
    bundle: CellRainfallBundle
    registered_name: str | None = None
    version: str | None = None
    alias: str | None = None

    @classmethod
    def from_registry(cls, model_uri: str | None = None, tracking_uri: str | None = None) -> Predictor:
        """Load ``model_uri`` (default: the ``production`` alias) through MLflow."""
        import mlflow
        from mlflow import MlflowClient

        model_uri = model_uri or os.getenv("RAINFALL_MODEL_URI", DEFAULT_MODEL_URI)
        if model_uri.startswith("models:/"):  # an exported local model needs no tracking server
            mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
        pyfunc = mlflow.pyfunc.load_model(model_uri)
        bundle = pyfunc.unwrap_python_model().bundle
        name = version = alias = None
        info_file = Path(model_uri) / "registry_info.json"
        if not model_uri.startswith("models:/") and info_file.exists():
            # Exported model (Docker image): registry details were saved at export time.
            info = json.loads(info_file.read_text())
            name, version, alias = info.get("registered_name"), info.get("version"), info.get("alias")
        elif model_uri.startswith("models:/"):
            ref = model_uri.removeprefix("models:/")
            client = MlflowClient()
            if "@" in ref:
                name, alias = ref.split("@", 1)
                version = client.get_model_version_by_alias(name, alias).version
            else:
                name, version = ref.split("/", 1)
        return cls(bundle=bundle, registered_name=name, version=str(version) if version else None, alias=alias)

    # ---------- info ----------
    def model_info(self) -> ModelInfo:
        m = self.bundle.meta
        return ModelInfo(
            registered_name=self.registered_name, version=self.version, alias=self.alias,
            kind=m["kind"], model_name=m["model_name"],
            data_end=f"{m['data_end']['year']}-{m['data_end']['month']:02d}",
            forecast_month=f"{m['forecast_month']['year']}-{m['forecast_month']['month']:02d}",
        )

    def model_card(self) -> ModelCard:
        m = self.bundle.meta
        return ModelCard(
            model=self.model_info(), champion_rule=m["champion_rule"], champion=m["champion"],
            backtest_years=f"{m['backtest_start_year']}-{m['backtest_end_year']}",
            pooled_backtest_metrics=m["pooled_backtest_metrics"],
            b2_pooled_backtest_metrics=m["b2_pooled_backtest_metrics"],
            m4_vs_b2_bootstrap=m["m4_vs_b2_bootstrap"],
            git_commit=m["git_commit"], data_md5=m["data_md5"], limitations=m["limitations"],
        )

    def cells(self) -> list[CellInfo]:
        c = self.bundle.cells[self.bundle.cells["in_service"]]
        return [self._cell_info(r) for r in c.to_dict("records")]

    def _cell_info(self, c: dict) -> CellInfo:
        return CellInfo(cell_id=c["cell_id"], centre_lat=c["lat"], centre_lon=c["lon"],
                        lat_min=c["lat_min"], lat_max=c["lat_max"], lon_min=c["lon_min"], lon_max=c["lon_max"],
                        resolution_deg=self.bundle.meta["resolution_deg"])

    # ---------- prediction ----------
    def predict(self, requests: list[PredictRequest]) -> list[Prediction]:
        located = []
        for r in requests:
            cell = self.bundle.locate(r.lat, r.lon)
            if cell is None:
                raise OutsideServiceArea(f"({r.lat}, {r.lon}) is outside the Bengaluru service area")
            located.append(cell)
        query = pd.DataFrame({"cell_id": [c["cell_id"] for c in located],
                              "year": [r.year for r in requests], "month": [r.month for r in requests]})
        out = self.bundle.predict(query)
        info = self.model_info()
        preds = []
        for r, cell, row in zip(requests, located, out.to_dict("records")):
            if row["basis"] == "unavailable":
                start = self.bundle.meta["backtest_start_year"]
                raise PeriodUnavailable(f"{r.year}-{r.month:02d} is before the backtest period ({start} onwards)")
            obs = row["observed_mm"]
            preds.append(Prediction(
                lat=r.lat, lon=r.lon, year=r.year, month=r.month, cell=self._cell_info(cell),
                basis=row["basis"], basis_note=BASIS_NOTES[row["basis"]],
                rainfall_mm=Quantiles(q10=round(row["q10"], 1), q50=round(row["q50"], 1), q90=round(row["q90"], 1)),
                interval_80pct_mm=(round(row["q10"], 1), round(row["q90"], 1)),
                climatology_mm=Quantiles(q10=round(row["ref_q10"], 1), q50=round(row["ref_q50"], 1),
                                         q90=round(row["ref_q90"], 1)),
                observed_mm=None if obs is None or math.isnan(obs) else round(obs, 1),
                model=info,
            ))
        return preds
