"""Extract per-cell daily rainfall from the IMD 0.25 degree gridded NetCDF.

The raw file covers a 4 x 5 grid around Bengaluru. A cell is kept when it has data and
its 0.25 degree box overlaps the Bengaluru urban-region boundary. Each kept cell gets an
id like ``c13.00_77.50`` (centre latitude and longitude).

Output (long format): ``data/processed/imd_cells_daily_1901_2025.csv`` with columns
``date, cell_id, rain_mm`` and ``data/processed/imd_cells.csv`` with one row per cell.

Usage:
    python -m src.ingest.imd_cells
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

RAW_NC = Path("data/raw/imdweb_pTxNQ39i.nc")
BOUNDARY = Path("data/raw/bengaluru_full_urban_region_imdweb.geojson")
OUT_DAILY = Path("data/processed/imd_cells_daily_1901_2025.csv")
OUT_CELLS = Path("data/processed/imd_cells.csv")
HALF = 0.125  # half of the 0.25 degree cell size


def cell_id(lat: float, lon: float) -> str:
    return f"c{lat:.2f}_{lon:.2f}"


def cell_table(ds: xr.Dataset, boundary: gpd.GeoDataFrame) -> pd.DataFrame:
    """One row per grid cell: centre, bounds, overlap with the boundary, data availability."""
    poly = boundary.to_crs("EPSG:4326").union_all()
    has_data = ds["rain"].notnull().any("time")
    rows = []
    for lat in ds["lat"].values:
        for lon in ds["lon"].values:
            cell = box(lon - HALF, lat - HALF, lon + HALF, lat + HALF)
            rows.append({
                "cell_id": cell_id(float(lat), float(lon)),
                "lat": float(lat), "lon": float(lon),
                "lat_min": float(lat - HALF), "lat_max": float(lat + HALF),
                "lon_min": float(lon - HALF), "lon_max": float(lon + HALF),
                "boundary_overlap": round(cell.intersection(poly).area / cell.area, 4),
                "has_data": bool(has_data.sel(lat=lat, lon=lon)),
            })
    cells = pd.DataFrame(rows)
    cells["in_service"] = cells["has_data"] & (cells["boundary_overlap"] > 0)
    return cells


def extract_cells(nc_path: Path = RAW_NC, boundary_path: Path = BOUNDARY) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = xr.open_dataset(nc_path, group="rain")
    cells = cell_table(ds, gpd.read_file(boundary_path))
    parts = []
    for c in cells[cells["in_service"]].itertuples():
        s = ds["rain"].sel(lat=c.lat, lon=c.lon).to_series()
        # Same rule as notebook 01: isolated missing days are treated as rainless.
        parts.append(pd.DataFrame({"date": s.index.normalize(), "cell_id": c.cell_id,
                                   "rain_mm": np.round(s.fillna(0.0).clip(lower=0).to_numpy(), 4)}))
    daily = pd.concat(parts, ignore_index=True)
    return cells, daily


def main() -> None:
    cells, daily = extract_cells()
    OUT_CELLS.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT_CELLS, index=False)
    daily.to_csv(OUT_DAILY, index=False, date_format="%Y-%m-%d")
    kept = cells[cells["in_service"]]
    print(f"Kept {len(kept)} of {len(cells)} cells; wrote {OUT_DAILY} ({len(daily)} rows) and {OUT_CELLS}")


if __name__ == "__main__":
    main()
