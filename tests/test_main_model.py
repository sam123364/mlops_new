"""Tests for the main model features (leakage), the PSL parser and the LightGBM quantile model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_monthly_features
from src.features.model_features import (
    ALL_FEATURES,
    RAINFALL_FEATURES,
    build_model_features,
    monthly_max_dry_spell,
)
from src.ingest.climate_indices import parse_psl_file
from src.models.main_model import LGBMQuantileModel, LGBMQuantileParams


def _synthetic(n_years: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1950-01-01", f"{1950 + n_years - 1}-12-31", freq="D")
    seasonal = 2 + 6 * np.clip(np.sin(2 * np.pi * (dates.dayofyear - 120) / 365.25), 0, None)
    rain = rng.gamma(0.4, seasonal) * (rng.random(len(dates)) < 0.4)
    daily = pd.DataFrame({"date": dates, "rain_urban_mean_mm": rain})
    monthly = build_monthly_features(daily)
    months = pd.MultiIndex.from_product([range(1949, 1950 + n_years), range(1, 13)], names=["year", "month"])
    indices = pd.DataFrame(index=months).reset_index()
    indices["nino34"] = rng.normal(size=len(indices))
    indices["dmi"] = rng.normal(size=len(indices))
    return daily, monthly, indices


def test_features_do_not_use_current_or_future_rainfall():
    daily, monthly, indices = _synthetic()
    base = build_model_features(monthly, daily, indices)

    cut_year, cut_month = 1975, 6
    future = (daily["date"] >= pd.Timestamp(cut_year, cut_month, 1))
    daily2 = daily.copy()
    daily2.loc[future, "rain_urban_mean_mm"] *= 7.0
    daily2.loc[future, "rain_urban_mean_mm"] += 3.0
    monthly2 = build_monthly_features(daily2)
    changed = build_model_features(monthly2, daily2, indices)

    key = (base["year"] < cut_year) | ((base["year"] == cut_year) & (base["month"] <= cut_month))
    # historical_normal-based columns are excluded from the model on purpose; check model features only.
    pd.testing.assert_frame_equal(base.loc[key, ALL_FEATURES], changed.loc[key, ALL_FEATURES])
    # Sanity: the perturbation does change later features.
    later = (base["year"] == cut_year) & (base["month"] == cut_month + 1)
    assert not np.allclose(base.loc[later, "rain_lag1m"], changed.loc[later, "rain_lag1m"])


def test_expanding_climatology_excludes_current_year():
    daily, monthly, indices = _synthetic()
    df = build_model_features(monthly, daily, indices)
    row = df[(df["year"] == 1980) & (df["month"] == 9)].iloc[0]
    past = df[(df["year"] < 1980) & (df["month"] == 9)]["monthly_rain_mm"]
    assert row["clim_mean_exp"] == pytest.approx(past.mean())
    assert row["clim_q50_exp"] == pytest.approx(past.quantile(0.5))


def test_climate_index_features_are_lagged():
    daily, monthly, indices = _synthetic()
    df = build_model_features(monthly, daily, indices)
    row = df[(df["year"] == 1970) & (df["month"] == 3)].iloc[0]
    feb = indices[(indices["year"] == 1970) & (indices["month"] == 2)]["nino34"].iloc[0]
    nov_to_jan = indices[((indices["year"] == 1969) & (indices["month"].isin([12])))
                         | ((indices["year"] == 1970) & (indices["month"].isin([1, 2])))]["nino34"]
    assert row["nino34_lag1"] == pytest.approx(feb)
    assert row["nino34_mean3_lag1"] == pytest.approx(nov_to_jan.mean())


def test_monthly_max_dry_spell_hand_case():
    dates = pd.date_range("2000-01-01", "2000-02-29", freq="D")
    rain = np.full(len(dates), 5.0)
    rain[3:10] = 0.0  # 7 dry days in January
    rain[29:35] = 0.0  # Jan 30-31 dry (2 days), Feb 1-4 dry (4 days): split at month boundary
    spells = monthly_max_dry_spell(pd.DataFrame({"date": dates, "rain_urban_mean_mm": rain}))
    assert spells.set_index("month")["max_dry_spell"].to_dict() == {1: 7, 2: 4}


def test_parse_psl_file(tmp_path):
    text = "  2000 2001\n 2000 1 2 3 4 5 6 7 8 9 10 11 12\n 2001 1 2 3 -99.99 5 6 7 8 9 10 11 12\n  -99.99\n notes\n"
    path = tmp_path / "idx.data"
    path.write_text(text)
    df = parse_psl_file(path, "x")
    assert len(df) == 24
    assert np.isnan(df[(df.year == 2001) & (df.month == 4)]["x"].iloc[0])
    assert df["x"].notna().sum() == 23


@pytest.mark.parametrize("init", [False, True])
def test_model_quantiles_ordered_and_non_negative(init):
    daily, monthly, indices = _synthetic()
    df = build_model_features(monthly, daily, indices)
    train, test = df[df["year"] < 1985], df[df["year"] == 1985]
    model = LGBMQuantileModel(features=RAINFALL_FEATURES, params=LGBMQuantileParams(n_estimators=30),
                              init_from_climatology=init).fit(train)
    p = model.predict_all_quantiles(test)
    assert all((p[q] >= 0).all() for q in p)
    assert (p[0.10] <= p[0.50]).all() and (p[0.50] <= p[0.90]).all()


def test_boosting_from_climatology_starts_at_climatology():
    daily, monthly, indices = _synthetic()
    df = build_model_features(monthly, daily, indices)
    train, test = df[df["year"] < 1985], df[df["year"] == 1985]
    params = LGBMQuantileParams(n_estimators=1, learning_rate=1e-9)
    p = LGBMQuantileModel(features=RAINFALL_FEATURES, params=params, init_from_climatology=True).fit(train)
    pred = p.predict_all_quantiles(test)
    expected = np.sort(test[["clim_q10_exp", "clim_q50_exp", "clim_q90_exp"]].to_numpy(), axis=1)
    np.testing.assert_allclose(np.column_stack([pred[0.1], pred[0.5], pred[0.9]]), expected, rtol=1e-4, atol=1e-3)
