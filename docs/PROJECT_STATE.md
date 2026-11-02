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

## 4. In Progress / Next Steps
- [ ] Configure DAGsHub remote credentials for remote DVC push and web MLflow sync.
- [ ] Implement deterministic hydrology balance engine (`src/hydrology/`) for tank sizing and demand offset simulation.
- [ ] Spatial leave-one-gauge-out cross-validation harness for KSNDMC micro-climate adjustments.

## 5. Open Questions / Blockers
- None currently.
