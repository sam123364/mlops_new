"""Leak-free predictors for the main probabilistic monthly rainfall model.

Every feature for month (y, m) uses only information available before month (y, m)
starts:
- "Expanding" climatology uses only years before y for the same calendar month. This
  replaces ``historical_normal_rain_mm``/``departure_pct_lag1m`` from build_features.py,
  which average over the whole 1901-2025 record (test years included).
- Anomaly lags and rolling sums are shifted by at least one month.
- Climate indices are lagged by at least one month.
"""

from __future__ import annotations

import pandas as pd

MIN_CLIM_YEARS = 10
WET_MONTH_MM = 10.0


def _expanding_by_month(monthly: pd.DataFrame, col: str, func: str, q: float | None = None) -> pd.Series:
    """Statistic of ``col`` over strictly earlier years, per calendar month."""
    grouped = monthly.groupby("month")[col]
    shifted = grouped.shift(1)  # rows are sorted by date, so shift(1) within a month = previous year
    by_month = shifted.groupby(monthly["month"])
    if func == "mean":
        return by_month.transform(lambda s: s.expanding(MIN_CLIM_YEARS).mean())
    if func == "quantile":
        return by_month.transform(lambda s: s.expanding(MIN_CLIM_YEARS).quantile(q))
    if func == "wetfrac":
        return by_month.transform(lambda s: (s >= WET_MONTH_MM).astype(float).where(s.notna()).expanding(MIN_CLIM_YEARS).mean())
    raise ValueError(func)


def add_expanding_climatology(monthly: pd.DataFrame) -> pd.DataFrame:
    df = monthly.sort_values(["year", "month"]).reset_index(drop=True).copy()
    df["clim_mean_exp"] = _expanding_by_month(df, "monthly_rain_mm", "mean")
    for q in (0.10, 0.50, 0.90):
        df[f"clim_q{int(q * 100):02d}_exp"] = _expanding_by_month(df, "monthly_rain_mm", "quantile", q)
    df["clim_wetfrac_exp"] = _expanding_by_month(df, "monthly_rain_mm", "wetfrac")
    return df


def add_anomaly_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Past rainfall anomalies relative to the expanding (past-only) normal."""
    df = df.copy()
    anom = df["monthly_rain_mm"] - df["clim_mean_exp"]
    past = anom.shift(1)
    df["anom_lag1m"] = past
    df["anom_roll3m"] = past.rolling(3, min_periods=2).sum()
    df["anom_roll6m"] = past.rolling(6, min_periods=4).sum()
    df["anom_roll12m"] = past.rolling(12, min_periods=8).sum()
    df["rain_lag12m"] = df["monthly_rain_mm"].shift(12)
    df["max_storm_lag1m"] = df["max_daily_storm_mm"].shift(1)
    return df


def monthly_max_dry_spell(daily: pd.DataFrame, rain_col: str = "rain_urban_mean_mm", threshold: float = 1.0) -> pd.DataFrame:
    """Longest run of days with rain < threshold inside each calendar month."""
    d = daily[["date", rain_col]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["year"], d["month"] = d["date"].dt.year, d["date"].dt.month
    d["dry"] = (d[rain_col] < threshold).astype(int)
    breaks = (d["dry"] == 0) | (d["month"] != d["month"].shift()) | (d["year"] != d["year"].shift())
    run_id = breaks.cumsum()
    d["run_len"] = d.groupby(run_id)["dry"].cumsum() * d["dry"]
    return d.groupby(["year", "month"])["run_len"].max().rename("max_dry_spell").reset_index()


def add_dry_spell_lag(df: pd.DataFrame, daily: pd.DataFrame, rain_col: str = "rain_urban_mean_mm") -> pd.DataFrame:
    spells = monthly_max_dry_spell(daily, rain_col=rain_col)
    df = df.merge(spells, on=["year", "month"], how="left")
    df["max_dry_spell_lag1m"] = df["max_dry_spell"].shift(1)
    return df.drop(columns=["max_dry_spell"])


def add_climate_index_lags(df: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    """Nino 3.4 and DMI known before month (y, m): lag 1, 3-month mean of lags 1-3, tendency."""
    idx = indices.sort_values(["year", "month"]).reset_index(drop=True).copy()
    out = idx[["year", "month"]].copy()
    for name in ("nino34", "dmi"):
        if name not in idx:
            continue
        s = idx[name]
        out[f"{name}_lag1"] = s.shift(1)
        out[f"{name}_mean3_lag1"] = s.shift(1).rolling(3, min_periods=3).mean()
        out[f"{name}_tend_lag1"] = s.shift(1) - s.shift(4)
    return df.merge(out, on=["year", "month"], how="left")


def build_model_features(
    monthly: pd.DataFrame,
    daily: pd.DataFrame,
    indices: pd.DataFrame | None = None,
    rain_col: str = "rain_urban_mean_mm",
) -> pd.DataFrame:
    """Full feature table for the main model (one row per year-month)."""
    df = add_expanding_climatology(monthly)
    df = add_anomaly_memory(df)
    df = add_dry_spell_lag(df, daily, rain_col=rain_col)
    if indices is not None:
        df = add_climate_index_lags(df, indices)
    df["month_cat"] = df["month"].astype(int)
    return df.sort_values(["year", "month"]).reset_index(drop=True)


RAINFALL_FEATURES = [
    "month_cat",
    "month_sin",
    "month_cos",
    "clim_mean_exp",
    "clim_q10_exp",
    "clim_q50_exp",
    "clim_q90_exp",
    "clim_wetfrac_exp",
    "rain_lag1m",
    "rain_lag2m",
    "wet_days_lag1m",
    "rolling_3m_sum",
    "rolling_12m_sum",
    "anom_lag1m",
    "anom_roll3m",
    "anom_roll6m",
    "anom_roll12m",
    "rain_lag12m",
    "max_storm_lag1m",
    "max_dry_spell_lag1m",
]
CLIMATE_FEATURES = [
    "nino34_lag1",
    "nino34_mean3_lag1",
    "nino34_tend_lag1",
    "dmi_lag1",
    "dmi_mean3_lag1",
    "dmi_tend_lag1",
]
ALL_FEATURES = RAINFALL_FEATURES + CLIMATE_FEATURES

__all__ = ["ALL_FEATURES", "CLIMATE_FEATURES", "RAINFALL_FEATURES", "build_model_features"]
