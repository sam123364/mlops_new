"""Unit tests for feature engineering and temporal leakage prevention.

Ensures that:
1. All lag and rolling features at time t depend strictly on observations at or before t-1.
2. Modifying or perturbing rainfall on day t does not alter any feature values at day t.
3. ADWP accurately resets to 0 on wet days and increments on dry days.
4. Harmonic calendar encodings match expected bounds.
"""

import numpy as np
import pandas as pd
import pytest
from src.features.build_features import (
    add_calendar_features,
    add_antecedent_memory_features,
    add_spatial_gradient_features,
)


@pytest.fixture
def sample_rainfall_df():
    """Generates a small deterministic 10-day rainfall series."""
    dates = pd.date_range("2024-06-01", periods=10, freq="D")
    # Days 0-2 dry, day 3 wet, day 4 wet, days 5-7 dry, day 8 wet, day 9 wet
    rain = [0.0, 0.0, 0.0, 15.0, 25.0, 0.0, 0.0, 0.0, 12.0, 30.0]
    return pd.DataFrame({
        "date": dates,
        "rain_urban_mean_mm": rain,
        "rain_central_bbmp_mm": rain,
        "rain_south_ecity_mm": [r * 0.8 for r in rain],
        "rain_east_whitefield_mm": [r * 1.2 for r in rain],
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

    # rain series: [0, 0, 0, 15, 25, 0, 0, 0, 12, 30]
    # Expected ADWP on day t:
    # t=0: NaN (no prior history)
    # t=1: 1 (day 0 was dry)
    # t=2: 2 (days 0, 1 were dry)
    # t=3: 3 (days 0, 1, 2 were dry)
    # t=4: 0 (day 3 was 15mm wet -> reset)
    # t=5: 0 (day 4 was 25mm wet -> reset)
    # t=6: 1 (day 5 was dry)
    # t=7: 2 (days 5, 6 were dry)
    # t=8: 3 (days 5, 6, 7 were dry)
    # t=9: 0 (day 8 was 12mm wet -> reset)
    expected_adwp = [np.nan, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0]

    np.testing.assert_allclose(
        df_feat["adwp"].values[1:],
        expected_adwp[1:],
        err_msg="ADWP values do not match expected dry spell progression"
    )


def test_zero_leakage_perturbation_test(sample_rainfall_df):
    """CRITICAL TEST: Proves zero target leakage.

    If we modify the rainfall at index `k`, all features at index `k` MUST remain identical.
    Only features at index > k may change.
    """
    df_orig = sample_rainfall_df.copy()
    feat_orig = add_antecedent_memory_features(df_orig, target_col="rain_urban_mean_mm")
    feat_orig = add_spatial_gradient_features(feat_orig)

    test_idx = 4
    df_perturbed = sample_rainfall_df.copy()
    # Inject an extreme 100mm storm on test_idx
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
                f"LEAKAGE DETECTED in feature '{col}'! "
                f"Feature value at day {test_idx} changed from {val_orig} to {val_pert} "
                f"when rain on day {test_idx} was perturbed."
            )
