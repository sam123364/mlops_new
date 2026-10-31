"""
Evaluation Harness for Rainfall Models.

Implements:
1. Pinball Loss / Quantile Loss
2. Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
3. Prediction Interval Coverage Probability (PICP)
4. Expanding-Window Time-Series Cross Validation (1901 to 2025)
5. Local and Remote MLflow Experiment Tracking
"""

from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import mlflow


def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Computes standard pinball loss (quantile loss) for a given quantile."""
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_interval_coverage(
    y_true: np.ndarray, lower_pred: np.ndarray, upper_pred: np.ndarray
) -> float:
    """Computes empirical coverage probability within [lower_pred, upper_pred]."""
    in_interval = (y_true >= lower_pred) & (y_true <= upper_pred)
    return float(np.mean(in_interval) * 100.0)


class ExpandingWindowEvaluator:
    """Expanding-window time-series backtesting engine.
    
    Ensures zero temporal leakage by training strictly on observations up to
    year T, and predicting on year T+1.
    """

    def __init__(
        self,
        min_train_years: int = 30,
        eval_start_year: int = 1950,
        eval_end_year: int = 2025,
        quantiles: List[float] = [0.10, 0.50, 0.90],
    ):
        self.min_train_years = min_train_years
        self.eval_start_year = eval_start_year
        self.eval_end_year = eval_end_year
        self.quantiles = quantiles

    def run_monthly_evaluation(
        self,
        df_monthly: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "monthly_rain_mm",
        group_col: str = "month",
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """Runs expanding window backtest on monthly feature dataset."""
        from src.models.baseline_models import (
            ClimatologyBaseline,
            EmpiricalQuantileBaseline,
            PersistenceBaseline,
            QuantileRegressionBaseline,
        )

        years = sorted(df_monthly["year"].unique())
        test_years = [y for y in years if self.eval_start_year <= y <= self.eval_end_year]

        records = []

        for test_yr in test_years:
            df_train = df_monthly[df_monthly["year"] < test_yr].copy()
            df_test = df_monthly[df_monthly["year"] == test_yr].copy()

            if len(df_train["year"].unique()) < self.min_train_years:
                continue

            # Model 1: Climatological Mean / Median
            b1_mean = ClimatologyBaseline(use_median=False).fit(df_train, group_col, target_col)
            b1_median = ClimatologyBaseline(use_median=True).fit(df_train, group_col, target_col)
            pred_b1_mean = b1_mean.predict(df_test, group_col)
            pred_b1_median = b1_median.predict(df_test, group_col)

            # Model 2: Empirical Quantiles
            b2 = EmpiricalQuantileBaseline(quantiles=self.quantiles).fit(df_train, group_col, target_col)
            preds_b2 = b2.predict_all_quantiles(df_test, group_col)

            # Model 3: Persistence
            b3 = PersistenceBaseline().fit(df_train, group_col, target_col)
            pred_b3 = b3.predict(df_test, group_col)

            # Model 4: Quantile Regression on features
            X_train = df_train[feature_cols].fillna(0.0)
            y_train = df_train[target_col]
            X_test = df_test[feature_cols].fillna(0.0)

            b4 = QuantileRegressionBaseline(quantiles=self.quantiles, alpha=0.01).fit(X_train, y_train)
            preds_b4 = b4.predict_all_quantiles(X_test)

            # Store step evaluation
            for i, (_, row) in enumerate(df_test.iterrows()):
                records.append({
                    "year": test_yr,
                    "month": row["month"],
                    "actual": row[target_col],
                    "b1_mean": pred_b1_mean[i],
                    "b1_median": pred_b1_median[i],
                    "b2_q10": preds_b2[0.10][i],
                    "b2_q50": preds_b2[0.50][i],
                    "b2_q90": preds_b2[0.90][i],
                    "b3_persistence": pred_b3[i],
                    "b4_q10": preds_b4[0.10][i],
                    "b4_q50": preds_b4[0.50][i],
                    "b4_q90": preds_b4[0.90][i],
                })

        pred_df = pd.DataFrame(records)

        # Aggregate overall benchmark metrics
        metrics = self._calculate_metrics(pred_df)
        return pred_df, metrics

    def _calculate_metrics(self, pred_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        y = pred_df["actual"].values

        # B1 Climatology
        metrics = {
            "B1_climatology_mean": {
                "mae": compute_mae(y, pred_df["b1_mean"].values),
            },
            "B1_climatology_median": {
                "mae": compute_mae(y, pred_df["b1_median"].values),
            },
            "B2_empirical_quantiles": {
                "pinball_loss_q10": compute_pinball_loss(y, pred_df["b2_q10"].values, 0.10),
                "pinball_loss_q50": compute_pinball_loss(y, pred_df["b2_q50"].values, 0.50),
                "pinball_loss_q90": compute_pinball_loss(y, pred_df["b2_q90"].values, 0.90),
                "mae_median": compute_mae(y, pred_df["b2_q50"].values),
                "coverage_80pct": compute_interval_coverage(y, pred_df["b2_q10"].values, pred_df["b2_q90"].values),
            },
            "B3_persistence_lag1": {
                "mae": compute_mae(y, pred_df["b3_persistence"].values),
            },
            "B4_quantile_regression": {
                "pinball_loss_q10": compute_pinball_loss(y, pred_df["b4_q10"].values, 0.10),
                "pinball_loss_q50": compute_pinball_loss(y, pred_df["b4_q50"].values, 0.50),
                "pinball_loss_q90": compute_pinball_loss(y, pred_df["b4_q90"].values, 0.90),
                "mae_median": compute_mae(y, pred_df["b4_q50"].values),
                "coverage_80pct": compute_interval_coverage(y, pred_df["b4_q10"].values, pred_df["b4_q90"].values),
            },
        }
        return metrics


def log_evaluation_to_mlflow(
    metrics: Dict[str, Dict[str, float]],
    pred_df: pd.DataFrame,
    experiment_name: str = "rainfall_baseline_evaluation",
    tracking_uri: str = "sqlite:///mlflow.db",
):
    """Logs baseline evaluation metrics and prediction artifacts to MLflow."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    os.makedirs("models", exist_ok=True)
    pred_csv_path = "models/baseline_predictions_backtest.csv"
    pred_df.to_csv(pred_csv_path, index=False)

    for model_name, model_metrics in metrics.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("eval_start_year", int(pred_df["year"].min()))
            mlflow.log_param("eval_end_year", int(pred_df["year"].max()))
            mlflow.log_param("eval_samples_count", len(pred_df))

            for metric_k, metric_v in model_metrics.items():
                mlflow.log_metric(metric_k, metric_v)

            mlflow.log_artifact(pred_csv_path, artifact_path="predictions")
            mlflow.set_tag("workstream", "D_baselines")
            mlflow.set_tag("project", "Urban_Rainwater_Harvesting_Bengaluru")
            print(f"Logged {model_name} to MLflow successfully.")
