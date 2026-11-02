"""
One-click sync script to push baseline evaluation and model versioning to DAGsHub MLflow.
Reads credentials from environment variables ($env:MLFLOW_TRACKING_USERNAME / $env:MLFLOW_TRACKING_PASSWORD)
or .env file.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

from src.evaluation.backtest import ExpandingWindowEvaluator, log_evaluation_to_mlflow

def main():
    dagshub_uri = "https://dagshub.com/sam123364/mlops_new.mlflow"
    print(f"Connecting to DAGsHub MLflow: {dagshub_uri}...")

    # Load processed monthly features
    csv_path = "data/processed/features_bengaluru_monthly.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run dvc pull first.")
        sys.exit(1)

    df_monthly = pd.read_csv(csv_path)
    features = [
        "month_sin", "month_cos", 
        "rain_lag1m", "rain_lag2m", 
        "wet_days_lag1m", 
        "rolling_3m_sum", "rolling_12m_sum", 
        "departure_pct_lag1m"
    ]

    print("Running expanding-window backtesting (1970 - 2025)...")
    evaluator = ExpandingWindowEvaluator(min_train_years=30, eval_start_year=1970, eval_end_year=2025)
    pred_df, metrics = evaluator.run_monthly_evaluation(df_monthly, feature_cols=features)

    print("Logging runs, parameters, metrics, and prediction artifacts to DAGsHub MLflow...")
    log_evaluation_to_mlflow(
        metrics=metrics,
        pred_df=pred_df,
        experiment_name="rainfall_baseline_evaluation",
        tracking_uri=dagshub_uri
    )
    print("\nSUCCESS: All baseline models and metrics pushed to DAGsHub MLflow!")
    print("Go to https://dagshub.com/sam123364/mlops_new and click the 'Experiments' tab to see your models live!")

if __name__ == "__main__":
    main()
