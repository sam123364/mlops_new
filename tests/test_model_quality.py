"""
Pytest Test Suite for Model Quality Assurance, Invariants, and Risk Checks.
"""

import numpy as np
import pandas as pd
import pytest
from src.evaluation.quality_checks import ModelQualityChecker
from src.models.baseline_models import (
    EmpiricalQuantileBaseline,
    QuantileRegressionBaseline,
)


@pytest.fixture
def mock_predictions():
    n = 100
    q10 = np.linspace(5.0, 50.0, n)
    q50 = q10 + 20.0
    q90 = q50 + 40.0
    return {0.10: q10, 0.50: q50, 0.90: q90}


@pytest.fixture
def real_monthly_features():
    df = pd.read_csv("data/processed/features_bengaluru_monthly.csv")
    return df


def test_non_negativity_check(mock_predictions):
    checker = ModelQualityChecker()
    res = checker.check_non_negativity(mock_predictions)
    assert res["passed"] is True
    assert res["total_negative_predictions"] == 0

    # Inject negative prediction failure
    corrupt_preds = mock_predictions.copy()
    corrupt_preds[0.10] = corrupt_preds[0.10].copy()
    corrupt_preds[0.10][5] = -4.5
    res_fail = checker.check_non_negativity(corrupt_preds)
    assert res_fail["passed"] is False
    assert res_fail["total_negative_predictions"] == 1


def test_monotonic_quantile_ordering(mock_predictions):
    checker = ModelQualityChecker()
    res = checker.check_quantile_monotonicity(mock_predictions)
    assert res["passed"] is True
    assert res["total_quantile_crossings"] == 0

    # Inject quantile crossing (Q10 > Q50)
    corrupt_preds = mock_predictions.copy()
    corrupt_preds[0.10] = corrupt_preds[0.10].copy()
    corrupt_preds[0.10][10] = 500.0  # Crosses Q50 and Q90
    res_fail = checker.check_quantile_monotonicity(corrupt_preds)
    assert res_fail["passed"] is False
    assert res_fail["q10_gt_q50_violations"] >= 1


def test_empirical_quantiles_calibration_on_real_data(real_monthly_features):
    checker = ModelQualityChecker(coverage_tolerance_pct=5.0)
    
    # Train B2 up to 1980, evaluate on 1981-2025
    train_df = real_monthly_features[real_monthly_features["year"] <= 1980]
    test_df = real_monthly_features[real_monthly_features["year"] > 1980]

    b2 = EmpiricalQuantileBaseline(quantiles=[0.10, 0.50, 0.90])
    b2.fit(train_df, "month", "monthly_rain_mm")
    preds = b2.predict_all_quantiles(test_df, "month")

    # 1. Non-negativity check
    non_neg = checker.check_non_negativity(preds)
    assert non_neg["passed"] is True

    # 2. Monotonicity check
    monotonic = checker.check_quantile_monotonicity(preds)
    assert monotonic["passed"] is True

    # 3. Interval coverage check (Nominal 80% with +/- 5% tolerance)
    cov = checker.check_interval_coverage(
        test_df["monthly_rain_mm"].values, preds[0.10], preds[0.90]
    )
    assert cov["passed"] is True
    assert 75.0 <= cov["empirical_coverage_pct"] <= 85.0


def test_input_corruption_resilience(real_monthly_features):
    checker = ModelQualityChecker()
    features = [
        "month_sin", "month_cos", "rain_lag1m", "rain_lag2m", 
        "wet_days_lag1m", "rolling_3m_sum", "rolling_12m_sum", "departure_pct_lag1m"
    ]
    train_df = real_monthly_features[real_monthly_features["year"] <= 1980]
    test_df = real_monthly_features[real_monthly_features["year"] > 1980].copy()

    X_train = train_df[features].fillna(0.0)
    y_train = train_df["monthly_rain_mm"]
    X_test = test_df[features].fillna(0.0)

    b4 = QuantileRegressionBaseline(quantiles=[0.10, 0.50, 0.90], alpha=0.01)
    b4.fit(X_train, y_train)

    res = checker.run_input_corruption_resilience(b4, X_test.head(50))
    assert res["passed"] is True
    assert res["extreme_spike_handled"] is True
    assert res["zero_drought_handled"] is True
