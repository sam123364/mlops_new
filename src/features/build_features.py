"""Feature engineering pipeline for Bengaluru rainfall modeling.

Produces temporal antecedent memory, cyclical calendar encodings,
and spatial gradient features with strict zero-leakage guarantees.
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

    # Season progression calculation
    # Bengaluru seasons:
    # Winter: Jan 1 - Feb 28/29 (DOY 1 to ~59)
    # Pre-Monsoon: Mar 1 - May 31 (~DOY 60 to ~151)
    # SW-Monsoon: Jun 1 - Sep 30 (~DOY 152 to ~273)
    # NE-Post-Monsoon: Oct 1 - Dec 31 (~DOY 274 to 365/366)
    def compute_season_progress(row) -> float:
        m = row["date"].month
        d = row["date"].day
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

    # Shifted series (t-1)
    lag1 = df[target_col].shift(1)
    lag2 = df[target_col].shift(2)

    df["rain_lag1"] = lag1
    df["rain_lag2"] = lag2

    # Multi-horizon rolling cumulative precipitation (t-k to t-1)
    for k in rolling_windows:
        df[f"p_roll_{k}d"] = lag1.rolling(window=k, min_periods=1).sum()

    # Rolling volatility and coefficient of variation (14 days)
    roll_14_mean = lag1.rolling(window=14, min_periods=3).mean()
    roll_14_std = lag1.rolling(window=14, min_periods=3).std()
    df["cv_roll_14d"] = roll_14_std / (roll_14_mean + 1e-4)
    df["cv_roll_14d"] = df["cv_roll_14d"].fillna(0.0)

    # Antecedent Dry Weather Period (ADWP): count of consecutive prior days < dry_threshold_mm
    is_dry = (lag1 < dry_threshold_mm).astype(int)
    # Vectorized run-length counter
    # Groups reset whenever is_dry is 0 (i.e. wet day)
    dry_groups = (~is_dry.astype(bool)).cumsum()
    df["adwp"] = is_dry.groupby(dry_groups).cumsum() * is_dry
    # On the very first row, lag1 is NaN; initialize adwp as NaN or 0
    df.loc[lag1.isna(), "adwp"] = np.nan

    # 2-day Markov state
    wet_t1 = (lag1 >= dry_threshold_mm).astype(float)
    wet_t2 = (lag2 >= dry_threshold_mm).astype(float)
    # 0: dry-dry, 1: dry-wet, 2: wet-dry, 3: wet-wet
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


def build_rainfall_features(
    input_path: str = "data/processed/clean_bengaluru_imd_daily_1901_2025.csv",
    output_path: Optional[str] = "data/processed/features_bengaluru_daily.csv",
) -> pd.DataFrame:
    """Executes the full feature extraction pipeline and saves the output."""
    print(f"Loading cleaned dataset from {input_path}...")
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"])

    print("Building calendar features...")
    df = add_calendar_features(df)

    print("Building antecedent memory and persistence features...")
    df = add_antecedent_memory_features(df, target_col="rain_urban_mean_mm")

    print("Building spatial gradient features...")
    df = add_spatial_gradient_features(df)

    if output_path:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Successfully saved engineered features to {output_path} ({df.shape[0]} rows, {df.shape[1]} cols)")

    return df


if __name__ == "__main__":
    build_rainfall_features()
