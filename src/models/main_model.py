"""Main probabilistic rainfall model: gradient-boosted quantile regression (LightGBM).

One LightGBM model per quantile (Q10, Q50, Q90) with the pinball (quantile) loss.

Design choices:
- Target is sqrt(rainfall). Quantiles are equivariant under monotone transforms, so
  squaring a predicted quantile of sqrt(y) gives the same quantile of y; the transform
  only tames the heavy right tail during training.
- Predictions are clipped at 0 mm and sorted so that Q10 <= Q50 <= Q90.
- Optional boosting from climatology (``init_from_climatology``): each quantile model
  starts from the past-only empirical quantile of that calendar month (the B2 baseline,
  features ``clim_qXX_exp``) via LightGBM ``init_score`` and only learns corrections.
  With few trees the model stays close to B2; tuning decides how far to move.
- Optional split-conformal calibration: each quantile is shifted by the empirical
  quantile of its residuals on the most recent ``calib_years`` of the training window
  (models fitted on the earlier years), then the models are refit on the full window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

QUANTILES = (0.10, 0.50, 0.90)


@dataclass
class LGBMQuantileParams:
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 7
    min_child_samples: int = 30
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = 42

    def to_lgbm(self, alpha: float) -> dict:
        return {
            "objective": "quantile",
            "alpha": alpha,
            "verbose": -1,
            "n_jobs": 1,
            **self.__dict__,
        }


@dataclass
class LGBMQuantileModel:
    features: list[str]
    params: LGBMQuantileParams = field(default_factory=LGBMQuantileParams)
    quantiles: tuple[float, ...] = QUANTILES
    calib_years: int = 0
    categorical: tuple[str, ...] = ("month_cat",)
    init_from_climatology: bool = False
    models_: dict[float, lgb.LGBMRegressor] = field(default_factory=dict, init=False)
    offsets_: dict[float, float] = field(default_factory=dict, init=False)

    @staticmethod
    def _init_col(q: float) -> str:
        return f"clim_q{round(q * 100):02d}_exp"

    def _init(self, df: pd.DataFrame, q: float) -> np.ndarray:
        if not self.init_from_climatology:
            return np.zeros(len(df))
        return np.sqrt(df[self._init_col(q)].clip(lower=0).to_numpy())

    def _usable(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.init_from_climatology:
            return df
        cols = [self._init_col(q) for q in self.quantiles]
        return df.dropna(subset=cols)  # first years of the record have no past climatology

    def _fit_models(self, df: pd.DataFrame, target_col: str) -> dict[float, lgb.LGBMRegressor]:
        df = self._usable(df)
        x = df[self.features]
        y = np.sqrt(df[target_col].clip(lower=0).to_numpy())
        cats = [c for c in self.categorical if c in self.features]
        models = {}
        for q in self.quantiles:
            m = lgb.LGBMRegressor(**self.params.to_lgbm(q))
            init = self._init(df, q) if self.init_from_climatology else None
            m.fit(x, y, init_score=init, categorical_feature=cats or "auto")
            models[q] = m
        return models

    def _predict_sqrt(self, models: dict[float, lgb.LGBMRegressor], df: pd.DataFrame) -> dict[float, np.ndarray]:
        # LightGBM predictions exclude init_score, so add it back.
        return {q: m.predict(df[self.features]) + self._init(df, q) for q, m in models.items()}

    def fit(self, df_train: pd.DataFrame, target_col: str = "monthly_rain_mm") -> LGBMQuantileModel:
        self.offsets_ = {q: 0.0 for q in self.quantiles}
        if self.calib_years > 0:
            last = df_train["year"].max()
            proper = df_train[df_train["year"] <= last - self.calib_years]
            calib = df_train[df_train["year"] > last - self.calib_years]
            models = self._fit_models(proper, target_col)
            calib = self._usable(calib)
            preds = self._predict_sqrt(models, calib)
            y_sqrt = np.sqrt(calib[target_col].clip(lower=0).to_numpy())
            for q in self.quantiles:
                # Shift so that a fraction q of calibration points lies below the prediction.
                self.offsets_[q] = float(np.quantile(y_sqrt - preds[q], q))
        self.models_ = self._fit_models(df_train, target_col)
        return self

    def predict_all_quantiles(self, df: pd.DataFrame) -> dict[float, np.ndarray]:
        raw = self._predict_sqrt(self.models_, df)
        stacked = np.column_stack([np.maximum(raw[q] + self.offsets_[q], 0.0) for q in self.quantiles])
        stacked = np.sort(stacked, axis=1) ** 2  # non-crossing, back to mm
        return {q: stacked[:, i] for i, q in enumerate(self.quantiles)}

    def feature_importance(self, q: float = 0.5) -> pd.Series:
        m = self.models_[q]
        return pd.Series(m.booster_.feature_importance("gain"), index=self.features).sort_values(ascending=False)
