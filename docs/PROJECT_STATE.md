# PROJECT_STATE.md: Current State & Tracking

## 1. Overview
- **Project**: Urban Rainwater Harvesting Estimation (Bengaluru)
- **Course**: MLOps Course Project at RVCE
- **Team**: Samarth, Sambhav, Samiksha, Samith

## 2. Workstream Status (Phase 1)
- **Workstream A: Infrastructure** (Samarth): In Progress (DVC initialized, MLflow tracking registry integrated).
- **Workstream B: Data** (Sambhav): Done (Raw IMD daily & NC gridded datasets collected and cleaned).
- **Workstream C: Exploration and Features** (Samiksha): Done (EDA executed, feature engineering module and zero-leakage tests built).
- **Workstream D: Baselines and Evaluation** (Samith): Done (Baseline models, expanding-window backtesting, MLflow experiment tracking, and executed notebook complete).

## 3. What is Done
- [x] Initialized Git repository and connected to remote (`https://github.com/sam123364/mlops_new.git`).
- [x] Defined repository structure according to AGENTS.md guidelines.
- [x] Moved `PROJECT_BRIEF.md` to `docs/`.
- [x] Placed raw IMD datasets under `data/raw/` and committed to GitHub.
- [x] Configured baseline configs: `regulations.yaml`, `demand.yaml`, `roof_params.yaml`, and `params.yaml`.
- [x] Cleaned daily district CSV and extracted 125-year (1901-2025) IMD NetCDF gridded timeseries to `data/processed/`.
- [x] Executed comprehensive EDA notebook (`notebooks/01_bengaluru_rainfall_eda_cleaning.ipynb`) validating 4 hydrological hypotheses with rich plots and unit simulations.
- [x] Built modular feature extraction script (`src/features/build_features.py`) computing harmonic calendar, antecedent memory, and spatial predictors.
- [x] Generated daily, monthly, and seasonal engineered feature datasets and tracked with DVC (`.dvc` pointer files).
- [x] Implemented zero-leakage test suite (`tests/test_features.py`) - all passed.
- [x] Initialized DVC repository (`dvc init`) for pipeline and data versioning.
- [x] Implemented 4 baseline models (`src/models/baseline_models.py`): Climatology Mean/Median (B1), Empirical Quantiles (B2), Persistence (B3), Quantile Regression (B4).
- [x] Built expanding-window time-series evaluation harness (`src/evaluation/backtest.py`) across 1970–2025 out-of-sample data.
- [x] Integrated MLflow tracking (`sqlite:///mlflow.db`) logging parameters, Pinball losses, MAE, coverage intervals, and artifacts.
- [x] Executed evaluation notebook (`notebooks/02_baseline_models_and_evaluation.ipynb`) rendering comparison tables and time-series envelope charts.
- [x] Authored baseline architecture and DAGsHub remote sync guide in `docs/baseline_models.md`.
- [x] Implemented Model Registry and automated Promotion Pipeline (`src/models/register_model.py`) supporting local SQLite and DAGsHub remote tracking.
- [x] Configured automated model promotion gate: stage transition from `None` -> `Staging`, and `Production` promotion checking MAE < 35 mm and 80% coverage >= 75%.
- [x] Tagged models with governance metadata (`git_commit`, `dvc_data_version`, `author`, `target_quantiles`, `methodology`).
- [x] Added unit tests for model registration and promotion gate (`tests/test_register_model.py`) - all 3 passed.
- [x] Created GitHub Actions CI/CD pipeline (`.github/workflows/model_registry_ci.yml`) verifying dependencies and running test suite on PRs.
- [x] Documented model registry best practices, lineage, and DAGsHub inspection guide in `docs/model_registry_guide.md`.

- [x] Registry fix: removed hard-coded seeded metrics; empty experiments are populated only from `reports/baseline_metrics.json` (written by `src/evaluation/run_baseline_backtest.py`); model chosen by lowest mean pinball loss. Selects B2 (MAE 33.81 mm, coverage 80.95%), which passes the gate.

- [x] Main model (`feat/main-model`): LightGBM quantile regression (Q10/Q50/Q90) with new leak-free features (past-only climatology, anomaly memory, dry-spell lag) and lagged ENSO (Niño 3.4) and IOD (DMI) indices from NOAA PSL. Tuned on 1940–1969, backtested 1970–2025 against the baselines. See `docs/main_model.md`.
- [x] Result: no ML variant beats B2. Best (M4, boosted from climatology) ties B2 (mean pinball 11.253 vs 11.252; 95% CI of difference [−0.037, +0.037]). Climate indices add no detectable skill at a 1-month lead.
- [x] Added pinned `requirements.txt` (Python 3.12; data, geospatial, modelling, DVC, MLflow, pytest, ruff, Jupyter). Local setup: `uv python install 3.12`, `uv venv --python 3.12 .venv`, `uv pip install --python .venv/bin/python -r requirements.txt`. On macOS, LightGBM also needs `brew install libomp`.

## 4. In Progress / Next Steps
- [ ] Configure DAGsHub remote credentials for remote DVC push and web MLflow sync.
- [ ] Implement deterministic hydrology balance engine (`src/hydrology/`) for tank sizing and demand offset simulation.
- [ ] Spatial leave-one-gauge-out cross-validation harness for KSNDMC micro-climate adjustments.

## 5. Open Questions / Blockers
- `departure_pct_lag1m` (used by B4) uses monthly normals from all of 1901–2025, including test years: a small look-ahead leak. The main model uses past-only anomalies instead.
- Licences for the NOAA PSL Niño 3.4 (ERSST v6) and DMI (HadISST1.1) files are not yet verified.
- Climate-index data is tracked with DVC locally but not pushed (DAGsHub credentials needed).
- Regenerated processed CSVs reproduce every baseline metric exactly but differ byte-wise from the DVC hashes (float formatting under pandas 3.1). Re-run `dvc add` after agreeing on the pinned environment.
- Registered baseline versions have no serialized estimator (model card only).
- Next modelling targets from the brief: seasonal totals and longest dry spell, and the KSNDMC spatial adjustment (leave-one-gauge-out).
- The IMD NetCDF stores its variables in a `rain` group: open with `xr.open_dataset(path, group="rain")`.
