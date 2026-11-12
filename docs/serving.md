# Rainfall inference service (FastAPI)

Location-aware monthly rainfall quantiles for Bengaluru, served from the MLflow model registry.

## What it answers

Given a latitude, longitude, year and month, it returns Q10 (dry), Q50 (median) and Q90 (wet)
monthly rainfall in mm for the IMD 0.25° grid cell (about 27 km) that contains the point,
plus the climatology (B2) for the same cell and month as a reference.

The `basis` field says what kind of answer it is:

| basis | When | Source |
|---|---|---|
| `forecast` | The month right after the last observed month (currently 2026-01) | The registered model, 1-month lead |
| `climatology` | Later months | Climatological quantiles for that cell and calendar month (past-month predictors do not exist yet) |
| `backtest` | Observed months from 1970 onwards | The stored out-of-sample prediction (made from earlier years only), with the observed rainfall |

Months before 1970 and points outside the Bengaluru urban-region boundary return HTTP 422.

## Model

- Registered model: `bengaluru_rainfall_by_cell` in the local MLflow registry (`sqlite:///mlflow.db`).
- Each version is a self-contained `mlflow.pyfunc` bundle (`src/models/cell_model_bundle.py`):
  grid cells, boundary, climatology, backtest predictions and, for M4, the LightGBM boosters and
  the forecast-month features. A version always gives the same answers.
- Version 1: B2 (empirical quantiles per cell), alias `challenger`.
- Version 2: M4 (LightGBM quantile regression boosted from climatology, all features, tuned per cell
  on 1940–1969), alias `production`.
- Champion rule, fixed in `src/evaluation/cell_backtest.py` before the results were seen: M4 serves
  only if its pooled mean pinball loss beats B2 with the year-block bootstrap 95% CI of the
  difference entirely below zero. It is applied to all cells together, not per cell.

### Backtest, 18 cells × 672 months (1970–2025)

| Model | Mean pinball | MAE of Q50 (mm) | 80% coverage |
|---|---|---|---|
| B2 empirical quantiles | 12.493 | 38.09 | 79.8% |
| B4 quantile regression | 13.866 | 43.48 | 78.5% |
| M4 LightGBM | **12.424** | **38.02** | 79.1% |

M4 − B2 = −0.069 (95% CI −0.109 to −0.034): better than B2, but only by about 0.6%. Per cell,
M4 is significantly better in 4 of 18 cells (all on the western edge: 76.75°E and 77.00°E) and
significantly worse in none. In the cells containing Jayanagar (`c13.00_77.50`) and Whitefield
(`c13.00_77.75`), M4 is slightly better but not significantly. In practice the forecasts stay close
to climatology. Full numbers: `reports/cell_model_metrics.json`.

The city-mean model (`docs/main_model.md`) tied B2. Per cell there are 18 times more test months,
so the same small improvement becomes detectable.

## Run it

From the repository root:

```bash
python -m src.ingest.imd_cells              # per-cell daily series from the IMD NetCDF
python -m src.evaluation.cell_backtest      # per-cell backtest (about 3 minutes), writes reports/
python -m src.models.package_cell_model     # build bundles, log and register, set aliases
uvicorn src.serving.app:app --port 8000     # start the API
```

Interactive docs: http://localhost:8000/docs

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"lat": 12.93, "lon": 77.58, "year": 2026, "month": 1}'
```

Environment variables: `RAINFALL_MODEL_URI` (default `models:/bengaluru_rainfall_by_cell@production`;
use `@challenger` to serve B2) and `MLFLOW_TRACKING_URI` (default `sqlite:///mlflow.db`).

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | Status and loaded model version |
| GET | `/model` | Model card: version, alias, data end, backtest metrics vs B2, champion rule, git commit, data hashes, limitations |
| GET | `/cells` | The 18 grid cells in service, with bounds |
| POST | `/predict` | One prediction |
| POST | `/predict/batch` | Up to 240 predictions |

## Updating with new data

When newer IMD data arrives: replace the raw NetCDF (DVC), rerun the four commands above, and a new
version is registered. The `production` alias moves only if the champion rule is met again.

## Limitations

- About 27 km resolution: every point in a cell gets the same answer. The IMD grid is interpolated
  from sparse stations. KSNDMC gauge adjustment (next step) is needed for neighbourhood scale.
- Monthly totals only. A monthly time step hides intense short storms.
- Only 1 month ahead. Beyond that, climatology.
- The data ends in 2025-12, so the "forecast" month (2026-01) is already in the past. It becomes a
  real forecast once data is updated regularly.
- No Docker image or authentication yet. Run locally only.

## Tests

`tests/test_serving.py` (synthetic bundles, no data needed; runs in CI as `Serving API CI`):

- API behaviour for each basis.
- Different cells give different answers.
- Rejection of out-of-area points, pre-1970 months and invalid input.
- Batch gives the same results as single requests.
- Quantile ordering.
- M4 served from saved boosters gives exactly the same numbers as at training time.
- A forecast row's features equal those of the same month built with the full record, which proves
  they use only earlier months.
