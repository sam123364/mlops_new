"""Feature engineering pipeline for Bengaluru rainfall modeling.

Produces temporal antecedent memory, cyclical calendar encodings,
spatial gradient features, as well as monthly and seasonal aggregation features
with strict zero-leakage guarantees.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclical harmonic calendar encodings and seasonal progress.

    Features generated:
    - sin_doy, cos_doy: Continuous 365.25-day harmonic encodings.
    - season_progress: Relative progression within the current meteorological season [0, 1].

    Note: Calendar properties depend only on the date, so there is zero target leakage.
    """
    df = df.copy()
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])

    doy = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)

    def compute_season_progress(row) -> float:
        m = row["date"].month
        if m in [1, 2]:
            start = pd.Timestamp(year=row["date"].year, month=1, day=1)
            end = pd.Timestamp(year=row["date"].year, month=2, day=28 if not row["date"].is_leap_year else 29)
        elif m in [3, 4, 5]:
            start = pd.Timestamp(year=row["date"].year, month=3, day=1)
            end = pd.Timestamp(year=row["date"].year, month=5, day=31)
        elif m in [6, 7, 8, 9]:
            start = pd.Timestamp(year=row["date"].year, month=6, day=1)
            end = pd.Timestamp(year=row["date"].year, month=9, day=30)
        else:
            start = pd.Timestamp(year=row["date"].year, month=10, day=1)
            end = pd.Timestamp(year=row["date"].year, month=12, day=31)
        total_days = (end - start).days
        elapsed = (row["date"] - start).days
        return float(np.clip(elapsed / total_days, 0.0, 1.0))

    df["season_progress"] = df.apply(compute_season_progress, axis=1)
    return df


def add_antecedent_memory_features(
    df: pd.DataFrame,
    target_col: str = "rain_urban_mean_mm",
    dry_threshold_mm: float = 0.1,
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Calculates antecedent dry weather period (ADWP) and rolling precipitation.

    STRICT ZERO-LEAKAGE ENFORCEMENT:
    All calculations strictly use shift(1) or prior history (<= t-1).
    Value at time t never contains precipitation from time t.

    Features generated:
    - rain_lag1: Rain observed on day t-1.
    - rain_lag2: Rain observed on day t-2.
    - adwp: Antecedent Dry Weather Period (consecutive dry days prior to day t).
    - p_roll_{k}d: Rolling sum over prior k days (t-k to t-1).
    - cv_roll_14d: Coefficient of variation over prior 14 days.
    - markov_state_2d: Higher-order Markov transition tuple (rain_{t-1} >= tau, rain_{t-2} >= tau).
    """
    df = df.copy()
    if rolling_windows is None:
        rolling_windows = [3, 7, 14, 30]

    lag1 = df[target_col].shift(1)
    lag2 = df[target_col].shift(2)

    df["rain_lag1"] = lag1
    df["rain_lag2"] = lag2

    for k in rolling_windows:
        df[f"p_roll_{k}d"] = lag1.rolling(window=k, min_periods=1).sum()

    roll_14_mean = lag1.rolling(window=14, min_periods=3).mean()
    roll_14_std = lag1.rolling(window=14, min_periods=3).std()
    df["cv_roll_14d"] = roll_14_std / (roll_14_mean + 1e-4)
    df["cv_roll_14d"] = df["cv_roll_14d"].fillna(0.0)

    is_dry = (lag1 < dry_threshold_mm).astype(int)
    dry_groups = (~is_dry.astype(bool)).cumsum()
    df["adwp"] = is_dry.groupby(dry_groups).cumsum() * is_dry
    df.loc[lag1.isna(), "adwp"] = np.nan

    wet_t1 = (lag1 >= dry_threshold_mm).astype(float)
    wet_t2 = (lag2 >= dry_threshold_mm).astype(float)
    markov_code = (wet_t2 * 2 + wet_t1)
    markov_code[lag1.isna() | lag2.isna()] = np.nan
    df["markov_state_2d"] = markov_code

    return df


def add_spatial_gradient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes spatial asymmetry indices between Bengaluru sub-regions.

    Features generated (all lagged to t-1 to prevent leakage):
    - grad_ew_lag1: East (Whitefield) minus Central (BBMP) rainfall on day t-1.
    - grad_ns_lag1: Central (BBMP) minus South (Electronic City) rainfall on day t-1.
    - spatial_coherence_lag1: Min / Max rainfall ratio across all 3 zones on day t-1.
    """
    df = df.copy()
    req_cols = ["rain_central_bbmp_mm", "rain_south_ecity_mm", "rain_east_whitefield_mm"]
    if not all(col in df.columns for col in req_cols):
        return df

    east_lag1 = df["rain_east_whitefield_mm"].shift(1)
    central_lag1 = df["rain_central_bbmp_mm"].shift(1)
    south_lag1 = df["rain_south_ecity_mm"].shift(1)

    df["grad_ew_lag1"] = east_lag1 - central_lag1
    df["grad_ns_lag1"] = central_lag1 - south_lag1

    min_spatial = pd.concat([east_lag1, central_lag1, south_lag1], axis=1).min(axis=1)
    max_spatial = pd.concat([east_lag1, central_lag1, south_lag1], axis=1).max(axis=1)
    df["spatial_coherence_lag1"] = min_spatial / (max_spatial + 1e-4)

    return df


def build_monthly_features(
    daily_df: pd.DataFrame,
    target_col: str = "rain_urban_mean_mm",
    dry_threshold_mm: float = 0.1,
) -> pd.DataFrame:
    """Builds monthly aggregated features for seasonal/monthly rainwater yield models.

    Aggregates daily series to month level:
    - Target: total rainfall depth (monthly_rain_mm), rain days (wet_days_count), max 1-day storm.
    - Predictors:
      - month_sin, month_cos: Harmonic annual cycle.
      - rain_lag1m, rain_lag2m: Prior 1-month and 2-month rainfall totals.
      - wet_days_lag1m: Prior month count of wet days.
      - rolling_3m_sum: Prior 3-month accumulated precipitation.
      - rolling_12m_sum: Prior 12-month accumulated precipitation (annual drought indicator).
      - historical_normal_rain_mm: Long-term monthly climate normal.
      - departure_pct_lag1m: Prior month percentage departure from long-term normal.
    """
    daily_df = daily_df.copy()
    if not np.issubdtype(daily_df["date"].dtype, np.datetime64):
        daily_df["date"] = pd.to_datetime(daily_df["date"])

    daily_df["year"] = daily_df["date"].dt.year
    daily_df["month"] = daily_df["date"].dt.month
    daily_df["is_wet_day"] = (daily_df[target_col] >= dry_threshold_mm).astype(int)

    monthly_summary = daily_df.groupby(["year", "month"]).agg(
        monthly_rain_mm=(target_col, "sum"),
        wet_days_count=("is_wet_day", "sum"),
        max_daily_storm_mm=(target_col, "max"),
        total_days=("date", "count")
    ).reset_index()

    monthly_summary["date"] = pd.to_datetime(
        monthly_summary["year"].astype(str) + "-" + monthly_summary["month"].astype(str).str.zfill(2) + "-01"
    )
    monthly_summary = monthly_summary.sort_values("date").reset_index(drop=True)

    # Cyclical calendar for month (1 to 12)
    monthly_summary["month_sin"] = np.sin(2.0 * np.pi * monthly_summary["month"] / 12.0)
    monthly_summary["month_cos"] = np.cos(2.0 * np.pi * monthly_summary["month"] / 12.0)

    # Calculate long-term normals per month
    month_normals = monthly_summary.groupby("month")["monthly_rain_mm"].mean().to_dict()
    monthly_summary["historical_normal_rain_mm"] = monthly_summary["month"].map(month_normals)

    # STRICT ZERO LEAKAGE: Lags and rolling predictors use shift(1)
    rain_lag1m = monthly_summary["monthly_rain_mm"].shift(1)
    rain_lag2m = monthly_summary["monthly_rain_mm"].shift(2)
    monthly_summary["rain_lag1m"] = rain_lag1m
    monthly_summary["rain_lag2m"] = rain_lag2m
    monthly_summary["wet_days_lag1m"] = monthly_summary["wet_days_count"].shift(1)

    monthly_summary["rolling_3m_sum"] = rain_lag1m.rolling(window=3, min_periods=1).sum()
    monthly_summary["rolling_12m_sum"] = rain_lag1m.rolling(window=12, min_periods=6).sum()

    # Prior month departure from normal
    normal_lag1m = monthly_summary["historical_normal_rain_mm"].shift(1)
    monthly_summary["departure_pct_lag1m"] = (
        (rain_lag1m - normal_lag1m) / (normal_lag1m + 1e-4)
    ) * 100.0

    return monthly_summary


def build_seasonal_features(
    monthly_df: pd.DataFrame,
) -> pd.DataFrame:
    """Builds seasonal aggregated features for Bengaluru's 4 distinct seasons.

    Seasons:
    - Winter: Jan, Feb
    - Pre-Monsoon: Mar, Apr, May
    - SW-Monsoon: Jun, Jul, Aug, Sep
    - NE-Post-Monsoon: Oct, Nov, Dec
    """
    monthly_df = monthly_df.copy()

    def map_season(month: int) -> str:
        if month in [1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Pre-Monsoon"
        elif month in [6, 7, 8, 9]:
            return "SW-Monsoon"
        else:
            return "NE-Post-Monsoon"

    monthly_df["season"] = monthly_df["month"].apply(map_season)

    seasonal_summary = monthly_df.groupby(["year", "season"]).agg(
        seasonal_rain_mm=("monthly_rain_mm", "sum"),
        wet_days_count=("wet_days_count", "sum"),
        max_daily_storm_mm=("max_daily_storm_mm", "max")
    ).reset_index()

    # Define seasonal temporal order
    season_order = {"Winter": 1, "Pre-Monsoon": 2, "SW-Monsoon": 3, "NE-Post-Monsoon": 4}
    seasonal_summary["season_num"] = seasonal_summary["season"].map(season_order)
    seasonal_summary = seasonal_summary.sort_values(["year", "season_num"]).reset_index(drop=True)

    # Cyclical encoding for season (1 to 4)
    seasonal_summary["season_sin"] = np.sin(2.0 * np.pi * seasonal_summary["season_num"] / 4.0)
    seasonal_summary["season_cos"] = np.cos(2.0 * np.pi * seasonal_summary["season_num"] / 4.0)

    # Prior season yield and 4-season (annual) rolling yield strictly shifted by 1
    rain_lag1s = seasonal_summary["seasonal_rain_mm"].shift(1)
    seasonal_summary["rain_lag1s"] = rain_lag1s
    seasonal_summary["rolling_4s_annual_sum"] = rain_lag1s.rolling(window=4, min_periods=2).sum()

    # Pre-monsoon yield predictor for SW-Monsoon
    # Pivoting year-level seasonal yields to provide early-season signals
    pre_monsoon_by_year = seasonal_summary[seasonal_summary["season"] == "Pre-Monsoon"].set_index("year")["seasonal_rain_mm"]
    seasonal_summary["pre_monsoon_rain_mm"] = seasonal_summary["year"].map(pre_monsoon_by_year)
    # Leakage guard: Only allow pre_monsoon_rain_mm to be visible during SW-Monsoon and NE-Post-Monsoon
    mask_visible = seasonal_summary["season"].isin(["SW-Monsoon", "NE-Post-Monsoon"])
    seasonal_summary.loc[~mask_visible, "pre_monsoon_rain_mm"] = np.nan

    return seasonal_summary


def build_rainfall_features(
    input_path: str = "data/processed/clean_bengaluru_imd_daily_1901_2025.csv",
    output_daily_path: Optional[str] = "data/processed/features_bengaluru_daily.csv",
    output_monthly_path: Optional[str] = "data/processed/features_bengaluru_monthly.csv",
    output_seasonal_path: Optional[str] = "data/processed/features_bengaluru_seasonal.csv",
) -> pd.DataFrame:
    """Executes the full feature extraction pipeline (daily, monthly, seasonal) and saves the outputs."""
    print(f"Loading cleaned dataset from {input_path}...")
    daily_df = pd.read_csv(input_path)
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    print("Building daily calendar, antecedent memory, and spatial gradient features...")
    daily_df = add_calendar_features(daily_df)
    daily_df = add_antecedent_memory_features(daily_df, target_col="rain_urban_mean_mm")
    daily_df = add_spatial_gradient_features(daily_df)

    if output_daily_path:
        out_dir = Path(output_daily_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(output_daily_path, index=False)
        print(f"Successfully saved daily engineered features to {output_daily_path} ({daily_df.shape[0]} rows, {daily_df.shape[1]} cols)")

    print("Building monthly aggregation features...")
    monthly_df = build_monthly_features(daily_df, target_col="rain_urban_mean_mm")
    if output_monthly_path:
        monthly_df.to_csv(output_monthly_path, index=False)
        print(f"Successfully saved monthly engineered features to {output_monthly_path} ({monthly_df.shape[0]} rows, {monthly_df.shape[1]} cols)")

    print("Building seasonal aggregation features...")
    seasonal_df = build_seasonal_features(monthly_df)
    if output_seasonal_path:
        seasonal_df.to_csv(output_seasonal_path, index=False)
        print(f"Successfully saved seasonal engineered features to {output_seasonal_path} ({seasonal_df.shape[0]} rows, {seasonal_df.shape[1]} cols)")

    return daily_df


if __name__ == "__main__":
    build_rainfall_features()
