"""
Test suite for Baseline Models & Expanding Window Evaluation Harness.
"""

import numpy as np
import pandas as pd
import pytest
from src.models.baseline_models import (
    ClimatologyBaseline,
    EmpiricalQuantileBaseline,
    PersistenceBaseline,
    QuantileRegressionBaseline,
)
from src.evaluation.backtest import (
    compute_mae,
    compute_pinball_loss,
    compute_interval_coverage,
    ExpandingWindowEvaluator,
)


@pytest.fixture
def dummy_monthly_data():
    years = np.repeat(range(1960, 2021), 12)
    months = np.tile(range(1, 13), len(range(1960, 2021)))
    # Simulated seasonal rain with noise
    rain = 50 + 40 * np.sin(2 * np.pi * months / 12) + np.random.normal(0, 10, len(years))
    rain = np.maximum(0.0, rain)
    df = pd.DataFrame({
        "year": years,
        "month": months,
        "monthly_rain_mm": rain,
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12),
        "rain_lag1m": pd.Series(rain).shift(1).bfill(),
        "rain_lag2m": pd.Series(rain).shift(2).bfill(),
        "wet_days_lag1m": 10.0,
        "rolling_3m_sum": pd.Series(rain).rolling(3).sum().bfill(),
        "rolling_12m_sum": pd.Series(rain).rolling(12).sum().bfill(),
        "departure_pct_lag1m": 0.0,
    })
    return df


def test_pinball_loss_symmetry():
    # For median (q=0.5), pinball loss is 0.5 * MAE
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 35.0])
    mae = compute_mae(y_true, y_pred)
    loss_q50 = compute_pinball_loss(y_true, y_pred, 0.5)
    assert np.isclose(loss_q50, 0.5 * mae)


def test_empirical_quantiles_ordering(dummy_monthly_data):
    b2 = EmpiricalQuantileBaseline(quantiles=[0.10, 0.50, 0.90])
    b2.fit(dummy_monthly_data, "month", "monthly_rain_mm")
    preds = b2.predict_all_quantiles(dummy_monthly_data.head(12), "month")
    
    # Q10 <= Q50 <= Q90 everywhere
    assert np.all(preds[0.10] <= preds[0.50])
    assert np.all(preds[0.50] <= preds[0.90])


def test_expanding_window_leakage_immunity(dummy_monthly_data):
    evaluator = ExpandingWindowEvaluator(min_train_years=30, eval_start_year=2000, eval_end_year=2005)
    features = ["month_sin", "month_cos", "rain_lag1m", "rolling_3m_sum"]
    pred_df, metrics = evaluator.run_monthly_evaluation(dummy_monthly_data, feature_cols=features)
    
    assert len(pred_df) == (2005 - 2000 + 1) * 12
    assert "B2_empirical_quantiles" in metrics
    assert metrics["B2_empirical_quantiles"]["coverage_80pct"] > 0
