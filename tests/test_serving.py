"""Tests for the per-cell model bundle, its features, and the FastAPI service.

All tests use small synthetic bundles, so they run without the DVC data or MLflow registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import box

from src.features.cell_features import build_one_cell
from src.features.model_features import ALL_FEATURES, RAINFALL_FEATURES
from src.models.cell_model_bundle import CellRainfallBundle
from src.models.main_model import LGBMQuantileModel, LGBMQuantileParams
from src.serving.app import create_app
from src.serving.predictor import Predictor
from src.serving.schemas import PredictRequest

CENTRAL = "c13.00_77.50"
EAST = "c13.00_77.75"
JAYANAGAR = (12.93, 77.58)
WHITEFIELD = (12.97, 77.75)
MYSURU = (12.30, 76.64)


def _cell(cid: str, lat: float, lon: float) -> dict:
    return {"cell_id": cid, "lat": lat, "lon": lon, "lat_min": lat - 0.125, "lat_max": lat + 0.125,
            "lon_min": lon - 0.125, "lon_max": lon + 0.125, "boundary_overlap": 1.0,
            "has_data": True, "in_service": True}


def _meta(kind: str = "B2", features: list[str] | None = None) -> dict:
    metrics = {"mean_pinball_loss": 11.0, "mae_median": 33.0, "coverage_80pct": 80.0}
    return {
        "kind": kind, "model_name": f"{kind}_test",
        "data_end": {"year": 2025, "month": 12}, "forecast_month": {"year": 2026, "month": 1},
        "backtest_start_year": 1970, "backtest_end_year": 2025, "resolution_deg": 0.25,
        "features": features or [], "pooled_backtest_metrics": metrics, "b2_pooled_backtest_metrics": metrics,
        "m4_vs_b2_bootstrap": {"mean_diff": 0.0, "ci95_low": -0.1, "ci95_high": 0.1},
        "champion": kind, "champion_rule": "test rule", "git_commit": "abc", "data_md5": {"x": "y"},
        "limitations": ["synthetic test bundle"],
    }


def write_bundle(root: Path, meta: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_cell(CENTRAL, 13.0, 77.5), _cell(EAST, 13.0, 77.75)]).to_csv(root / "cells.csv", index=False)
    gpd.GeoDataFrame({"name": ["test"]}, geometry=[box(77.40, 12.88, 77.85, 13.10)], crs="EPSG:4326").to_file(
        root / "boundary.geojson", driver="GeoJSON")
    clim = [{"cell_id": c, "month": m, "q10": 10.0 * m + off, "q50": 20.0 * m + off, "q90": 40.0 * m + off}
            for c, off in ((CENTRAL, 0.0), (EAST, 1.0)) for m in range(1, 13)]
    pd.DataFrame(clim).to_csv(root / "climatology.csv", index=False)
    back = [{"cell_id": c, "year": y, "month": m, "observed_mm": 55.0, "q10": 5.0, "q50": 50.0, "q90": 150.0,
             "b2_q10": 4.0, "b2_q50": 45.0, "b2_q90": 140.0}
            for c in (CENTRAL, EAST) for y in (1970, 2025) for m in range(1, 13)]
    pd.DataFrame(back).to_csv(root / "backtest.csv", index=False)
    (root / "meta.json").write_text(json.dumps(meta))
    return root


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    bundle = CellRainfallBundle.load(write_bundle(tmp_path / "b2", _meta("B2")))
    app = create_app(Predictor(bundle=bundle, registered_name="test_model", version="7", alias="production"))
    with TestClient(app) as c:
        yield c


def _post(client: TestClient, lat: float, lon: float, year: int, month: int):
    return client.post("/predict", json={"lat": lat, "lon": lon, "year": year, "month": month})


# ---------- API ----------
def test_health_reports_loaded_model(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "7"}


def test_forecast_month_uses_model(client: TestClient) -> None:
    r = _post(client, *JAYANAGAR, 2026, 1)
    assert r.status_code == 200
    body = r.json()
    assert body["basis"] == "forecast"
    assert body["cell"]["cell_id"] == CENTRAL
    # B2 bundle: the forecast is the January climatology of that cell.
    assert body["rainfall_mm"] == {"q10": 10.0, "q50": 20.0, "q90": 40.0}
    assert body["model"]["version"] == "7" and body["model"]["alias"] == "production"


def test_locations_in_different_cells_get_different_answers(client: TestClient) -> None:
    a = _post(client, *JAYANAGAR, 2026, 1).json()
    b = _post(client, *WHITEFIELD, 2026, 1).json()
    assert a["cell"]["cell_id"] == CENTRAL and b["cell"]["cell_id"] == EAST
    assert a["rainfall_mm"] != b["rainfall_mm"]


def test_months_after_forecast_month_fall_back_to_climatology(client: TestClient) -> None:
    body = _post(client, *JAYANAGAR, 2026, 5).json()
    assert body["basis"] == "climatology"
    assert body["rainfall_mm"] == {"q10": 50.0, "q50": 100.0, "q90": 200.0}
    assert body["observed_mm"] is None


def test_observed_month_returns_backtest_and_observation(client: TestClient) -> None:
    body = _post(client, *WHITEFIELD, 2025, 7).json()
    assert body["basis"] == "backtest"
    assert body["observed_mm"] == 55.0
    assert body["rainfall_mm"] == {"q10": 5.0, "q50": 50.0, "q90": 150.0}
    assert body["climatology_mm"] == {"q10": 4.0, "q50": 45.0, "q90": 140.0}


def test_quantiles_are_ordered_for_every_basis(client: TestClient) -> None:
    for year, month in ((2026, 1), (2026, 8), (2025, 3)):
        q = _post(client, *JAYANAGAR, year, month).json()["rainfall_mm"]
        assert q["q10"] <= q["q50"] <= q["q90"]


def test_month_before_backtest_period_is_rejected(client: TestClient) -> None:
    r = _post(client, *JAYANAGAR, 1950, 6)
    assert r.status_code == 422
    assert "before the backtest period" in r.json()["detail"]


def test_location_outside_service_area_is_rejected(client: TestClient) -> None:
    r = _post(client, *MYSURU, 2026, 1)
    assert r.status_code == 422
    assert "outside" in r.json()["detail"]


@pytest.mark.parametrize("payload", [
    {"lat": 12.93, "lon": 77.58, "year": 2026, "month": 13},
    {"lat": 95.0, "lon": 77.58, "year": 2026, "month": 1},
    {"lat": 12.93, "lon": 77.58, "year": 2026},
])
def test_invalid_requests_are_rejected(client: TestClient, payload: dict) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_matches_single_predictions(client: TestClient) -> None:
    items = [{"lat": JAYANAGAR[0], "lon": JAYANAGAR[1], "year": 2026, "month": 1},
             {"lat": WHITEFIELD[0], "lon": WHITEFIELD[1], "year": 2025, "month": 7}]
    batch = client.post("/predict/batch", json={"items": items}).json()["predictions"]
    singles = [client.post("/predict", json=i).json() for i in items]
    assert batch == singles


def test_model_card_and_cells(client: TestClient) -> None:
    card = client.get("/model").json()
    assert card["model"]["kind"] == "B2" and card["limitations"]
    assert {c["cell_id"] for c in client.get("/cells").json()} == {CENTRAL, EAST}


# ---------- M4 bundle: serving gives the same numbers as training-time prediction ----------
def _synthetic_feature_table(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1901-01-01", "1960-12-31", freq="D")
    season = 1 + 4 * np.exp(-((dates.month - 9) ** 2) / 6)
    rain = rng.gamma(0.3, 8.0 * season) * (rng.random(len(dates)) < 0.35)
    daily = pd.DataFrame({"date": dates, "rain_mm": rain})
    months = pd.period_range("1900-01", "1961-12", freq="M")
    indices = pd.DataFrame({"year": months.year, "month": months.month,
                            "nino34": rng.normal(0, 1, len(months)), "dmi": rng.normal(0, 0.4, len(months))})
    return build_one_cell(daily, indices, with_forecast_row=True)


def test_m4_bundle_reproduces_training_predictions(tmp_path: Path) -> None:
    feats = _synthetic_feature_table()
    observed = feats[~feats["is_forecast_row"]]
    fc = feats[feats["is_forecast_row"]].copy()
    model = LGBMQuantileModel(features=ALL_FEATURES, params=LGBMQuantileParams(n_estimators=20, num_leaves=4),
                              init_from_climatology=True).fit(observed)
    expected = model.predict_all_quantiles(fc)

    meta = _meta("M4", features=ALL_FEATURES)
    meta["data_end"], meta["forecast_month"] = {"year": 1960, "month": 12}, {"year": 1961, "month": 1}
    root = write_bundle(tmp_path / "m4", meta)
    (root / "boosters").mkdir()
    for cid in (CENTRAL, EAST):
        fc_c = fc.assign(cell_id=cid)
        fc_c[["cell_id", "year", "month", *ALL_FEATURES]].to_csv(
            root / "forecast_features.csv", mode="a", header=cid == CENTRAL, index=False)
        for q, est in model.models_.items():
            est.booster_.save_model(str(root / "boosters" / f"{cid}_q{round(q * 100):02d}.txt"))

    bundle = CellRainfallBundle.load(root)
    got = bundle.predict(pd.DataFrame({"cell_id": [CENTRAL], "year": [1961], "month": [1]})).iloc[0]
    assert got["basis"] == "forecast"
    for q in (0.10, 0.50, 0.90):
        assert got[f"q{round(q * 100):02d}"] == pytest.approx(float(expected[q][0]), rel=1e-9)


# ---------- leakage: the forecast row only uses months before it ----------
def test_forecast_row_features_use_only_earlier_months() -> None:
    """Features of month T built as a forecast row (data cut before T) must equal those of
    month T in the full table. Any use of month T or later would make them differ."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("1901-01-01", "1950-12-31", freq="D")
    daily = pd.DataFrame({"date": dates, "rain_mm": rng.gamma(0.3, 10.0, len(dates)) * (rng.random(len(dates)) < 0.4)})
    months = pd.period_range("1900-01", "1951-12", freq="M")
    indices = pd.DataFrame({"year": months.year, "month": months.month,
                            "nino34": rng.normal(0, 1, len(months)), "dmi": rng.normal(0, 0.4, len(months))})
    full = build_one_cell(daily, indices).set_index(["year", "month"])
    for y, m in ((1940, 1), (1945, 7), (1950, 12)):
        cut = daily[daily["date"] < pd.Timestamp(year=y, month=m, day=1)]
        row = build_one_cell(cut, indices, with_forecast_row=True).iloc[-1]
        assert (int(row["year"]), int(row["month"])) == (y, m)
        pd.testing.assert_series_equal(
            row[ALL_FEATURES].astype(float), full.loc[(y, m), ALL_FEATURES].astype(float),
            check_names=False, rtol=1e-9)


def test_forecast_row_has_no_target_and_complete_rainfall_features() -> None:
    feats = _synthetic_feature_table(seed=3)
    fc = feats[feats["is_forecast_row"]]
    assert len(fc) == 1
    assert fc["monthly_rain_mm"].isna().all()
    assert fc[RAINFALL_FEATURES].notna().all().all()


# ---------- exported model (what the Docker image loads) ----------
def test_exported_model_loads_with_registry_info(tmp_path: Path) -> None:
    from tests.make_synthetic_model import main as make_model

    out = tmp_path / "model"
    make_model(str(out))
    p = Predictor.from_registry(str(out))
    assert (p.registered_name, p.version, p.alias) == ("synthetic_ci_model", "0", "ci")
    pred = p.predict([PredictRequest(lat=JAYANAGAR[0], lon=JAYANAGAR[1], year=2026, month=1)])[0]
    assert pred.basis == "forecast" and pred.model.version == "0"
