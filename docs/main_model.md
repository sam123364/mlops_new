# Main Rainfall Model: Gradient-Boosted Quantile Regression

**Code:** `src/models/main_model.py`, `src/features/model_features.py`, `src/evaluation/main_model_backtest.py`
**Config:** `params.yaml` → `main_model`
**Results:** `reports/main_model_metrics.json` (all numbers below), `reports/main_model_predictions.csv`, `reports/main_model_tuning.json`

## 1. Choice of model

**LightGBM quantile regression**: one gradient-boosted tree model per quantile (Q10 dry, Q50 normal, Q90 wet), trained with the pinball loss on √(monthly rainfall).

Why:
- The project needs a *distribution* (dry/normal/wet), not one forecast. Quantile boosting predicts the quantiles directly, with no Gaussian assumption on a skewed, zero-heavy target.
- Trees capture month-specific, non-linear effects (for example, ENSO mattering only in certain seasons) that linear quantile regression (B4) cannot.
- LightGBM handles missing values natively (Niño 3.4 starts in 1948), is fast enough to refit every year of a 56-year backtest, and is already in `requirements.txt`.
- √ transform: quantiles are equivariant under monotone transforms, so squaring a predicted quantile of √y gives the same quantile of y. It only stabilises training on the heavy tail.
- Post-processing: clip at 0 mm and sort so Q10 ≤ Q50 ≤ Q90 (no crossing).

Alternatives considered: NGBoost / distributional regression (parametric, needs a new package), quantile regression forests (needs a new package), deep models (far too little data: 1,500 months).

## 2. Two designs tested

| ID | Design |
| :--- | :--- |
| M1 | LightGBM from scratch, rainfall-derived features only |
| M2 | LightGBM from scratch, + ENSO and IOD indices |
| M3 | M2 + split-conformal calibration on the last 15 training years |
| **M4** | **LightGBM boosted from climatology** (each quantile starts from the past-only empirical quantile for that month, i.e. B2, via `init_score`), all features |
| M5 | M4 without the climate indices (ablation) |

M1–M3 were the first iteration. They lost to B2, so M4/M5 were added: starting from B2 means the trees only learn corrections, and tuning decides how large those corrections should be. **All variants are reported.**

## 3. Features (all leak-free; see `tests/test_main_model.py`)

Existing engineered features: `month_sin`, `month_cos`, `rain_lag1m`, `rain_lag2m`, `wet_days_lag1m`, `rolling_3m_sum`, `rolling_12m_sum`.

New features:
| Feature | Meaning |
| :--- | :--- |
| `month_cat` | Month as a categorical variable |
| `clim_mean_exp`, `clim_q10/q50/q90_exp`, `clim_wetfrac_exp` | Mean, quantiles and share of months ≥ 10 mm for this calendar month, from **earlier years only** |
| `anom_lag1m`, `anom_roll3m/6m/12m` | Past rainfall anomalies relative to that past-only normal |
| `rain_lag12m` | Same month last year |
| `max_storm_lag1m` | Largest 1-day rainfall last month |
| `max_dry_spell_lag1m` | Longest run of days < 1 mm last month |
| `nino34_lag1`, `nino34_mean3_lag1`, `nino34_tend_lag1` | ENSO (Niño 3.4 SST anomaly): last month, mean of last 3 months, 3-month change |
| `dmi_lag1`, `dmi_mean3_lag1`, `dmi_tend_lag1` | Indian Ocean Dipole (DMI), same lags |

**Leakage fix:** `departure_pct_lag1m` (used by B4) is computed from monthly normals averaged over 1901–2025, test years included. The main model does not use it; the past-only `anom_*` features replace it. B4 is reported unchanged so its numbers match notebook 02.

## 4. Protocol

- Expanding window: for each test year T in 1970–2025, train on all years < T, predict the 12 months of T (672 months). Same rows as the baselines.
- Hyperparameters tuned **only on 1940–1969** with the same expanding window, then frozen. Grids are in `params.yaml`.
- Metrics: pinball loss per quantile, mean pinball loss, MAE of Q50, 80% interval coverage, skill vs B2 = 1 − loss/loss_B2.
- Significance: 95% confidence interval of the mean pinball-loss difference vs B2, by year-block bootstrap (2,000 resamples of the 56 test years).

## 5. Results (1970–2025, 672 months)

| Model | Pinball Q10 | Pinball Q50 | Pinball Q90 | Mean pinball | MAE (mm) | 80% coverage | Skill vs B2 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **B2 Empirical quantiles** | 5.849 | 16.903 | 11.004 | **11.252** | 33.81 | 80.95% | 0 |
| B4 Linear quantile regression | 6.139 | 20.416 | 12.454 | 13.003 | 40.83 | 79.76% | −15.6% |
| M1 LGBM, rainfall features | 5.930 | 17.383 | 11.267 | 11.527 | 34.77 | 75.45% | −2.4% |
| M2 LGBM, + climate indices | 5.954 | 17.326 | 11.293 | 11.524 | 34.65 | 76.04% | −2.4% |
| M3 M2 + conformal | 5.950 | 17.270 | 10.949 | 11.389 | 34.54 | 77.83% | −1.2% |
| **M4 LGBM boosted from climatology** | 5.856 | 16.913 | 10.991 | **11.253** | 33.83 | 80.51% | −0.0% |
| M5 M4 without climate indices | 5.863 | 16.968 | 11.008 | 11.280 | 33.94 | 80.21% | −0.2% |

Difference in mean pinball loss vs B2 (negative = better than B2), 95% CI:

| Model | Mean diff | 95% CI |
| :--- | ---: | :--- |
| M1 | +0.275 | [+0.077, +0.467] (worse) |
| M2 | +0.272 | [+0.056, +0.480] (worse) |
| M3 | +0.138 | [−0.239, +0.488] |
| **M4** | +0.001 | [−0.037, +0.037] (no difference) |
| M5 | +0.028 | [−0.022, +0.075] |

## 6. Conclusion (honest)

- **No model beats B2.** The best ML model, M4, is statistically tied with B2. Tuning on 1940–1969 chose the smallest correction offered (10 trees), meaning the data say "stay with climatology".
- Training trees from scratch (M1–M3) is clearly worse: they spend capacity re-learning the seasonal cycle from about 800 training months.
- Climate indices add almost nothing at a 1-month lead for this IMD cell: M4 vs M5 differs by 0.03 pinball. Diagnostic correlations for 1970–2025: Oct–Dec total vs September Niño 3.4, −0.03; vs September DMI, −0.03. Jun–Sep total vs May Niño 3.4, −0.10.
- Feature importance (Q50, gain share): in the from-scratch M2, the seasonal cycle dominates (`month_cat` about 64%, climatology features most of the rest). In M4, whose trees only learn corrections to climatology, those corrections use mostly `nino34_tend_lag1` (36%), `dmi_lag1` (13%) and `anom_lag1m` (10%). The corrections are tiny (10 trees at learning rate 0.03) and give no measurable skill.
- **Recommendation:** keep B2 as the production model (the registry gate selects it). Keep M4 as the main ML candidate: it never does meaningfully worse than B2, stays calibrated, and will pick up skill automatically if more informative predictors are added. The most promising place for ML in this project is the spatial adjustment with KSNDMC gauges (brief §7.2), not the 1-month-ahead temporal forecast.

## 7. Limitations

- The climate indices are reanalysed series (ERSST v6, HadISST1.1), not what was available in real time. The Niño 3.4 anomaly uses a fixed 1981–2010 base period. Both could slightly overstate the skill available in practice, yet skill is still zero.
- Monthly time step on a ~25 km IMD cell average; hides intense sub-daily storms and local variation.
- Only the 1-month-ahead monthly target is tested. The brief's seasonal totals and longest dry spell targets are not yet modelled.
- The saved M4 boosters predict corrections. A full prediction is (booster output + √clim_q)², where clim_q is the past-only empirical quantile for that month.

## 8. Reproduce

```bash
python -m src.ingest.climate_indices            # parse NOAA files -> data/processed/climate_indices_monthly.csv
python -m src.evaluation.main_model_backtest     # tune + backtest, writes reports/
python -m src.evaluation.main_model_backtest --log-mlflow --tracking-uri sqlite:///mlflow.db
```
