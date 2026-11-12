"""Per-grid-cell monthly feature tables for the location-aware rainfall model.

Uses the same feature code as the city-mean model (``build_monthly_features`` and
``build_model_features``) on each IMD cell's own daily series, so training, backtest and
serving share one code path.

``with_forecast_row=True`` appends one row per cell for the month after the last
observed month. Its target columns are NaN; its predictors are built only from earlier
months, exactly as they would be for an observed month. The serving model uses this row
for the 1-month-lead forecast.
"""

from __future__ import annotations

import pandas as pd

from src.features.build_features import build_monthly_features
from src.features.model_features import build_model_features

TARGET_COLS = ["monthly_rain_mm", "wet_days_count", "max_daily_storm_mm"]


def next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def build_one_cell(
    daily: pd.DataFrame,
    indices: pd.DataFrame | None,
    with_forecast_row: bool = False,
) -> pd.DataFrame:
    """Feature table for one cell. ``daily`` has columns ``date`` and ``rain_mm``."""
    d = daily[["date", "rain_mm"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)
    if with_forecast_row:
        last = d["date"].iloc[-1]
        y, m = next_month(last.year, last.month)
        # A placeholder day so the aggregation creates the month; its values are blanked below.
        d = pd.concat([d, pd.DataFrame({"date": [pd.Timestamp(year=y, month=m, day=1)], "rain_mm": [0.0]})],
                      ignore_index=True)

    monthly = build_monthly_features(d, target_col="rain_mm")
    if with_forecast_row:
        monthly.loc[monthly.index[-1], TARGET_COLS] = float("nan")
    feats = build_model_features(monthly, d, indices, rain_col="rain_mm")
    feats["is_forecast_row"] = False
    if with_forecast_row:
        feats.loc[feats.index[-1], "is_forecast_row"] = True
    return feats


def build_cell_feature_table(
    daily_long: pd.DataFrame,
    indices: pd.DataFrame | None,
    with_forecast_row: bool = False,
) -> pd.DataFrame:
    """Stack per-cell feature tables. ``daily_long`` has ``date, cell_id, rain_mm``."""
    parts = []
    for cid, g in daily_long.groupby("cell_id", sort=True):
        f = build_one_cell(g, indices, with_forecast_row=with_forecast_row)
        f.insert(0, "cell_id", cid)
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


__all__ = ["build_cell_feature_table", "build_one_cell", "next_month"]
