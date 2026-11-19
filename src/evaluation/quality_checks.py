"""
Model Quality Assurance, Stress Testing, and Risk Checks for Probabilistic Rainfall Models.

Implements rigorous physical, statistical, and operational quality checks per AGENTS.md:
1. Physical Invariants (Non-negativity: P >= 0)
2. Monotonic Quantile Ordering (Q10 <= Q50 <= Q90, zero quantile crossing)
3. Prediction Interval Calibration (Nominal 80% coverage check within tolerance)
4. Extreme Climate Stress Testing (e.g., 2022 historic flood anomaly & drought years)
5. Out-of-Distribution (OOD) Input & Data Corruption Immunity
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class ModelQualityChecker:
    """Evaluates behavioral, physical, and distributional risks of rainfall models."""

    def __init__(
        self,
        quantiles: List[float] = [0.10, 0.50, 0.90],
        coverage_tolerance_pct: float = 8.0,  # e.g., 80% +/- 8%
    ):
        self.quantiles = sorted(quantiles)
        self.coverage_tolerance_pct = coverage_tolerance_pct

    def check_non_negativity(self, predictions: Dict[float, np.ndarray]) -> Dict[str, any]:
        """Physical Invariant Check: Precipitation depth can never be negative."""
        failures = {}
        for q, preds in predictions.items():
            neg_count = int(np.sum(preds < 0.0))
            failures[f"q{int(q*100)}_neg_count"] = neg_count

        total_neg = sum(failures.values())
        return {
            "passed": bool(total_neg == 0),
            "total_negative_predictions": total_neg,
            "details": failures,
        }

    def check_quantile_monotonicity(self, predictions: Dict[float, np.ndarray]) -> Dict[str, any]:
        """Monotonicity Check: Quantile crossing violation check.
        Ensures Q10 <= Q50 <= Q90 for all test samples.
        """
        q10 = predictions[0.10]
        q50 = predictions[0.50]
        q90 = predictions[0.90]

        q10_gt_q50 = int(np.sum(q10 > q50))
        q50_gt_q90 = int(np.sum(q50 > q90))
        q10_gt_q90 = int(np.sum(q10 > q90))

        total_violations = q10_gt_q50 + q50_gt_q90 + q10_gt_q90

        return {
            "passed": bool(total_violations == 0),
            "total_quantile_crossings": total_violations,
            "q10_gt_q50_violations": q10_gt_q50,
            "q50_gt_q90_violations": q50_gt_q90,
        }

    def check_interval_coverage(
        self, y_true: np.ndarray, q10_preds: np.ndarray, q90_preds: np.ndarray
    ) -> Dict[str, any]:
        """Statistical Calibration Check: Empirical coverage should match nominal target (80%)."""
        in_band = (y_true >= q10_preds) & (y_true <= q90_preds)
        coverage_pct = float(np.mean(in_band) * 100.0)
        target = 80.0
        deviation = abs(coverage_pct - target)

        return {
            "passed": bool(deviation <= self.coverage_tolerance_pct),
            "empirical_coverage_pct": round(coverage_pct, 2),
            "target_coverage_pct": target,
            "deviation_pct": round(deviation, 2),
            "tolerance_pct": self.coverage_tolerance_pct,
        }

    def run_extreme_event_stress_test(
        self,
        df_eval: pd.DataFrame,
        extreme_years: List[int] = [2022, 2017, 2023],
    ) -> Dict[str, any]:
        """Stress Testing on Historically Extreme Climatic Events.
        
        2022: Historic urban flood anomaly in Bengaluru.
        2017: Extreme pre-monsoon convective storm year.
        2023: Weak monsoon El Nino deficit condition.
        """
        stress_results = {}
        for yr in extreme_years:
            yr_df = df_eval[df_eval["year"] == yr]
            if len(yr_df) == 0:
                continue

            y_actual = yr_df["actual"].values
            q10 = yr_df["b2_q10"].values
            q90 = yr_df["b2_q90"].values

            in_band = (y_actual >= q10) & (y_actual <= q90)
            underestimate_drought = int(np.sum(y_actual < q10))
            exceed_flood = int(np.sum(y_actual > q90))

            stress_results[f"year_{yr}"] = {
                "months_evaluated": len(yr_df),
                "in_band_coverage_pct": round(float(np.mean(in_band) * 100.0), 2),
                "drought_underestimates": underestimate_drought,
                "flood_exceedances": exceed_flood,
                "max_monthly_actual_mm": round(float(np.max(y_actual)), 2),
            }

        return stress_results

    def run_input_corruption_resilience(
        self,
        model,
        X_sample: pd.DataFrame,
    ) -> Dict[str, any]:
        """Operational Robustness Check: Model behavior under corrupted or missing inputs."""
        # Test 1: High rainfall anomaly injection
        X_corrupt_extreme = X_sample.copy()
        for col in ["rain_lag1m", "rolling_3m_sum"]:
            if col in X_corrupt_extreme.columns:
                X_corrupt_extreme[col] = X_corrupt_extreme[col] * 10.0

        preds_corrupt = model.predict_all_quantiles(X_corrupt_extreme)
        non_neg = self.check_non_negativity(preds_corrupt)
        monotonic = self.check_quantile_monotonicity(preds_corrupt)

        # Test 2: Zero antecedent rainfall (drought scenario)
        X_corrupt_zero = X_sample.copy()
        for col in ["rain_lag1m", "rolling_3m_sum", "rolling_12m_sum"]:
            if col in X_corrupt_zero.columns:
                X_corrupt_zero[col] = 0.0

        preds_zero = model.predict_all_quantiles(X_corrupt_zero)
        non_neg_zero = self.check_non_negativity(preds_zero)
        monotonic_zero = self.check_quantile_monotonicity(preds_zero)

        resilient = bool(
            non_neg["passed"]
            and monotonic["passed"]
            and non_neg_zero["passed"]
            and monotonic_zero["passed"]
        )

        return {
            "passed": resilient,
            "extreme_spike_handled": bool(non_neg["passed"] and monotonic["passed"]),
            "zero_drought_handled": bool(non_neg_zero["passed"] and monotonic_zero["passed"]),
        }
