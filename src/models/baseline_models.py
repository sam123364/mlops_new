"""
Baseline Models for Bengaluru Probabilistic Rainfall Estimation.

Implements benchmark models compliant with docs/PROJECT_BRIEF.md section 7.3:
1. Climatology Mean / Median
2. Empirical Historical Quantiles (P10, P50, P90)
3. Persistence (Lag-1)
4. Quantile Regression on Engineered Antecedent & Harmonic Features
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor


class ClimatologyBaseline:
    """Baseline B1: Historical Climatological Central Tendency.
    
    Predicts the expanding historical mean or median for each temporal slice
    (month or season).
    """

    def __init__(self, use_median: bool = False):
        self.use_median = use_median
        self.climatology_: Dict[int, float] = {}
        self.overall_stat_: float = 0.0

    def fit(self, df_train: pd.DataFrame, group_col: str, target_col: str):
        agg_func = "median" if self.use_median else "mean"
        self.climatology_ = (
            df_train.groupby(group_col)[target_col].agg(agg_func).to_dict()
        )
        self.overall_stat_ = (
            float(df_train[target_col].median())
            if self.use_median
            else float(df_train[target_col].mean())
        )
        return self

    def predict(self, df_test: pd.DataFrame, group_col: str) -> np.ndarray:
        return np.array(
            [
                self.climatology_.get(val, self.overall_stat_)
                for val in df_test[group_col]
            ]
        )


class EmpiricalQuantileBaseline:
    """Baseline B2: Historical Empirical Quantile Distribution.
    
    Computes empirical percentiles (e.g., Q10 dry, Q50 normal, Q90 wet)
    per seasonal or monthly cycle from past observations.
    """

    def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90]):
        self.quantiles = quantiles
        self.quantile_table_: Dict[int, Dict[float, float]] = {}

    def fit(self, df_train: pd.DataFrame, group_col: str, target_col: str):
        self.quantile_table_ = {}
        for group_val, grp in df_train.groupby(group_col):
            self.quantile_table_[group_val] = {
                q: float(np.percentile(grp[target_col], q * 100))
                for q in self.quantiles
            }
        return self

    def predict_quantile(
        self, df_test: pd.DataFrame, group_col: str, q: float
    ) -> np.ndarray:
        return np.array(
            [self.quantile_table_[val][q] for val in df_test[group_col]]
        )

    def predict_all_quantiles(
        self, df_test: pd.DataFrame, group_col: str
    ) -> Dict[float, np.ndarray]:
        return {
            q: self.predict_quantile(df_test, group_col, q)
            for q in self.quantiles
        }


class PersistenceBaseline:
    """Baseline B3: Lag-1 Autoregressive Persistence.
    
    Assumes rainfall for season/month t equals rainfall observed in the same
    season/month in year t-1.
    """

    def __init__(self):
        self.last_observed_: Dict[int, float] = {}

    def fit(self, df_train: pd.DataFrame, group_col: str, target_col: str):
        # Store most recent observation for each cycle group
        sorted_df = df_train.sort_values(by="year")
        for group_val, grp in sorted_df.groupby(group_col):
            self.last_observed_[group_val] = float(grp[target_col].iloc[-1])
        return self

    def predict(self, df_test: pd.DataFrame, group_col: str) -> np.ndarray:
        return np.array(
            [self.last_observed_.get(val, 0.0) for val in df_test[group_col]]
        )


class QuantileRegressionBaseline:
    """Baseline B4: Linear Quantile Regression on Engineered Antecedent Features.
    
    Models target conditional quantiles as linear functions of antecedent
    rolling sums, anomalies, and harmonic cyclic phenology.
    """

    def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90], alpha: float = 0.0):
        self.quantiles = quantiles
        self.alpha = alpha
        self.models_: Dict[float, QuantileRegressor] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        self.models_ = {}
        for q in self.quantiles:
            reg = QuantileRegressor(
                quantile=q,
                alpha=self.alpha,
                solver="highs",
            )
            reg.fit(X_train, y_train)
            self.models_[q] = reg
        return self

    def predict_all_quantiles(self, X_test: pd.DataFrame) -> Dict[float, np.ndarray]:
        raw_predictions = {}
        for q, model in self.models_.items():
            preds = model.predict(X_test)
            # Physical hydrology constraint: rainfall cannot be negative
            raw_predictions[q] = np.maximum(0.0, preds)

        # Monotonic sorting to eliminate quantile crossings
        # (Standard post-processing for independent linear quantile models)
        q_keys = sorted(self.quantiles)
        stacked = np.column_stack([raw_predictions[q] for q in q_keys])
        sorted_stacked = np.sort(stacked, axis=1)

        predictions = {}
        for idx, q in enumerate(q_keys):
            predictions[q] = sorted_stacked[:, idx]

        return predictions
