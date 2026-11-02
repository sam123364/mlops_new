# PROJECT_STATE.md: Current State & Tracking

## 1. Overview
- **Project**: Urban Rainwater Harvesting Estimation (Bengaluru)
- **Course**: MLOps Course Project at RVCE
- **Team**: Samarth, Sambhav, Samiksha, Samith

## 2. Workstream Status (Phase 1)
- **Workstream A: Infrastructure** (Samarth): In Progress (DVC initialized, DAGsHub synced, MLflow tracking registry integrated).
- **Workstream B: Data** (Sambhav): Done (Raw IMD daily & NC gridded datasets collected, cleaned, and DVC-versioned).
- **Workstream C: Exploration and Features** (Samiksha): Done (EDA executed, feature engineering module and zero-leakage tests built).
- **Workstream D: Baselines, Quality & Evaluation** (Samith): Done (Baseline models, expanding-window backtesting, MLflow experiment tracking, physical quality checks & stress testing suite complete).

## 3. What is Done
- [x] Initialized Git repository and connected to remote (`https://github.com/sam123364/mlops_new.git`).
- [x] Defined repository structure according to AGENTS.md guidelines.
- [x] Moved `PROJECT_BRIEF.md` to `docs/`.
- [x] Placed raw IMD datasets under `data/raw/` and tracked end-to-end via DVC on DAGsHub.
- [x] Configured baseline configs: `regulations.yaml`, `demand.yaml`, `roof_params.yaml`, and `params.yaml`.
- [x] Cleaned daily district CSV and extracted 125-year (1901-2025) IMD NetCDF gridded timeseries to `data/processed/`.
- [x] Executed comprehensive EDA notebook (`notebooks/01_bengaluru_rainfall_eda_cleaning.ipynb`) validating 4 hydrological hypotheses with rich plots and unit simulations.
- [x] Built modular feature extraction script (`src/features/build_features.py`) computing harmonic calendar, antecedent memory, and spatial predictors.
- [x] Generated daily, monthly, and seasonal engineered feature datasets and tracked with DVC (`.dvc` pointer files on DAGsHub).
- [x] Implemented zero-leakage test suite (`tests/test_features.py`) - all passed.
- [x] Initialized DVC repository (`dvc init`) with DAGsHub cloud remote.
- [x] Implemented 4 baseline models (`src/models/baseline_models.py`): Climatology Mean/Median (B1), Empirical Quantiles (B2), Persistence (B3), Quantile Regression (B4).
- [x] Built expanding-window time-series evaluation harness (`src/evaluation/backtest.py`) across 1970–2025 out-of-sample data.
- [x] Integrated MLflow tracking (`sqlite:///mlflow.db` and DAGsHub remote) logging parameters, Pinball losses, MAE, coverage intervals, and artifacts.
- [x] Executed evaluation notebook (`notebooks/02_baseline_models_and_evaluation.ipynb`) rendering comparison tables and time-series envelope charts.
- [x] Authored baseline architecture and DAGsHub remote sync guide in `docs/baseline_models.md`.
- [x] Implemented Model Quality, Physical Invariants, and Stress Testing suite (`src/evaluation/quality_checks.py`).
- [x] Verified zero quantile crossings, physical non-negativity ($P \ge 0$), and prediction interval calibration ($80.95\%$ vs $80\%$ nominal).
- [x] Added historical climate stress tests (2022 historic flood, 2017 pre-monsoon storm, 2023 El Niño deficit).
- [x] Authored Model Risk & Stress Testing Report (`docs/model_risk_assessment.md`).
- [x] Complete pytest test suite passing: 12 tests passed (`tests/test_baselines.py`, `tests/test_features.py`, `tests/test_model_quality.py`).

## 4. In Progress / Next Steps
- [ ] Person 2 (Partner): MLflow Model Registry, model staging, and CI/CD promotion gate (`feat/model-registry`).
- [ ] Implement deterministic hydrology balance engine (`src/hydrology/`) for tank sizing and demand offset simulation.
- [ ] Spatial leave-one-gauge-out cross-validation harness for KSNDMC micro-climate adjustments.

## 5. Open Questions / Blockers
- None currently.
