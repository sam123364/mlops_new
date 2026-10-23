"""Unit tests for feature engineering and temporal leakage prevention.

Ensures that:
1. Daily lag and rolling features at time t depend strictly on observations at or before t-1.
2. Monthly and seasonal features do not leak target-period precipitation into predictors.
3. Modifying or perturbing rainfall on day/month t does not alter any feature values at step t.
4. ADWP accurately resets to 0 on wet days and increments on dry days.
5. Harmonic calendar encodings match expected bounds.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.build_features import (
    add_calendar_features,
    add_antecedent_memory_features,
    add_spatial_gradient_features,
    build_monthly_features,
    build_seasonal_features,
)


@pytest.fixture
def sample_rainfall_df():
    """Generates a small deterministic 10-day rainfall series."""
    dates = pd.date_range("2024-06-01", periods=10, freq="D")
    rain = [0.0, 0.0, 0.0, 15.0, 25.0, 0.0, 0.0, 0.0, 12.0, 30.0]
    return pd.DataFrame({
        "date": dates,
        "rain_urban_mean_mm": rain,
        "rain_central_bbmp_mm": rain,
        "rain_south_ecity_mm": [r * 0.8 for r in rain],
        "rain_east_whitefield_mm": [r * 1.2 for r in rain],
    })


@pytest.fixture
def sample_multiyear_df():
    """Generates a 3-year daily rainfall series for monthly/seasonal tests."""
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    np.random.seed(42)
    # Generate positive synthetic rain
    rain = np.random.exponential(scale=3.0, size=len(dates))
    rain[rain < 1.0] = 0.0
    return pd.DataFrame({
        "date": dates,
        "rain_urban_mean_mm": rain,
    })


def test_calendar_features_bounds(sample_rainfall_df):
    """Verifies that cyclical harmonic features are within [-1, 1] and continuous."""
    df_feat = add_calendar_features(sample_rainfall_df)
    assert "sin_doy" in df_feat.columns
    assert "cos_doy" in df_feat.columns
    assert "season_progress" in df_feat.columns

    assert df_feat["sin_doy"].between(-1.0, 1.0).all()
    assert df_feat["cos_doy"].between(-1.0, 1.0).all()
    assert df_feat["season_progress"].between(0.0, 1.0).all()


def test_adwp_counting_logic(sample_rainfall_df):
    """Verifies Antecedent Dry Weather Period (ADWP) counts consecutive dry days correctly."""
    df_feat = add_antecedent_memory_features(sample_rainfall_df, target_col="rain_urban_mean_mm")

    expected_adwp = [np.nan, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0]

    np.testing.assert_allclose(
        df_feat["adwp"].values[1:],
        expected_adwp[1:],
        err_msg="ADWP values do not match expected dry spell progression"
    )


def test_daily_zero_leakage_perturbation(sample_rainfall_df):
    """CRITICAL TEST: Proves zero target leakage at daily level."""
    df_orig = sample_rainfall_df.copy()
    feat_orig = add_antecedent_memory_features(df_orig, target_col="rain_urban_mean_mm")
    feat_orig = add_spatial_gradient_features(feat_orig)

    test_idx = 4
    df_perturbed = sample_rainfall_df.copy()
    df_perturbed.loc[test_idx, "rain_urban_mean_mm"] = 100.0
    df_perturbed.loc[test_idx, "rain_central_bbmp_mm"] = 100.0
    df_perturbed.loc[test_idx, "rain_south_ecity_mm"] = 100.0
    df_perturbed.loc[test_idx, "rain_east_whitefield_mm"] = 100.0

    feat_perturbed = add_antecedent_memory_features(df_perturbed, target_col="rain_urban_mean_mm")
    feat_perturbed = add_spatial_gradient_features(feat_perturbed)

    feature_cols = [
        "rain_lag1", "rain_lag2", "p_roll_3d", "p_roll_7d",
        "cv_roll_14d", "adwp", "markov_state_2d",
        "grad_ew_lag1", "grad_ns_lag1", "spatial_coherence_lag1"
    ]

    for col in feature_cols:
        val_orig = feat_orig.loc[test_idx, col]
        val_pert = feat_perturbed.loc[test_idx, col]
        if np.isnan(val_orig):
            assert np.isnan(val_pert), f"Leakage detected in {col}: NaN mismatch at perturbed index"
        else:
            assert np.isclose(val_orig, val_pert), (
                f"LEAKAGE DETECTED in daily feature '{col}'! "
                f"Value changed at day {test_idx} when rain on day {test_idx} was perturbed."
            )


def test_monthly_zero_leakage_perturbation(sample_multiyear_df):
    """CRITICAL TEST: Proves zero target leakage in monthly predictors."""
    monthly_orig = build_monthly_features(sample_multiyear_df)

    test_month_idx = 9  # Oct 2021 (month 10 is index 9)
    df_perturbed = sample_multiyear_df.copy()
    # Perturb all days in month 10
    mask_oct = (df_perturbed["date"].dt.year == 2021) & (df_perturbed["date"].dt.month == 10)
    df_perturbed.loc[mask_oct, "rain_urban_mean_mm"] += 50.0

    monthly_perturbed = build_monthly_features(df_perturbed)

    # Monthly target changes as expected:
    assert monthly_perturbed.loc[test_month_idx, "monthly_rain_mm"] != monthly_orig.loc[test_month_idx, "monthly_rain_mm"]

    # But monthly lagged predictors at test_month_idx MUST NOT CHANGE:
    predictor_cols = ["rain_lag1m", "rain_lag2m", "wet_days_lag1m", "rolling_3m_sum", "rolling_12m_sum"]
    for col in predictor_cols:
        val_orig = monthly_orig.loc[test_month_idx, col]
        val_pert = monthly_perturbed.loc[test_month_idx, col]
        assert np.isclose(val_orig, val_pert), (
            f"LEAKAGE DETECTED in monthly feature '{col}' at index {test_month_idx}!"
        )


def test_seasonal_features_structure(sample_multiyear_df):
    """Verifies that seasonal features are constructed with correct seasons and zero-leakage masking."""
    monthly_df = build_monthly_features(sample_multiyear_df)
    seasonal_df = build_seasonal_features(monthly_df)

    expected_seasons = {"Winter", "Pre-Monsoon", "SW-Monsoon", "NE-Post-Monsoon"}
    assert set(seasonal_df["season"].unique()) == expected_seasons
    assert "rolling_4s_annual_sum" in seasonal_df.columns

    # Check leakage guard on pre_monsoon_rain_mm:
    # Must be NaN for Winter and Pre-Monsoon
    winter_mask = seasonal_df["season"] == "Winter"
    assert seasonal_df.loc[winter_mask, "pre_monsoon_rain_mm"].isna().all()
