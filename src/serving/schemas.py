"""Request and response models for the rainfall inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Basis = Literal["forecast", "climatology", "backtest"]


class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees (WGS84)", examples=[12.93])
    lon: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees (WGS84)", examples=[77.58])
    year: int = Field(..., ge=1900, le=2100, examples=[2026])
    month: int = Field(..., ge=1, le=12, examples=[1])


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=240)


class Quantiles(BaseModel):
    q10: float = Field(..., description="Dry outcome: 10% chance of less rain (mm)")
    q50: float = Field(..., description="Median (mm)")
    q90: float = Field(..., description="Wet outcome: 10% chance of more rain (mm)")


class CellInfo(BaseModel):
    cell_id: str
    centre_lat: float
    centre_lon: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution_deg: float


class ModelInfo(BaseModel):
    registered_name: str | None
    version: str | None
    alias: str | None
    kind: str
    model_name: str
    data_end: str
    forecast_month: str


class Prediction(BaseModel):
    lat: float
    lon: float
    year: int
    month: int
    cell: CellInfo
    basis: Basis
    basis_note: str
    rainfall_mm: Quantiles
    interval_80pct_mm: tuple[float, float]
    climatology_mm: Quantiles = Field(..., description="B2 reference for the same cell and month")
    observed_mm: float | None = Field(None, description="Observed IMD rainfall (backtest months only)")
    model: ModelInfo


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]


class ModelCard(BaseModel):
    model: ModelInfo
    champion_rule: str
    champion: str
    backtest_years: str
    pooled_backtest_metrics: dict[str, float]
    b2_pooled_backtest_metrics: dict[str, float]
    m4_vs_b2_bootstrap: dict[str, float]
    git_commit: str
    data_md5: dict[str, str]
    limitations: list[str]


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str | None
