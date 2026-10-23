# PROJECT_STATE.md: Current State & Tracking

## 1. Overview
- **Project**: Urban Rainwater Harvesting Estimation (Bengaluru)
- **Course**: MLOps Course Project at RVCE
- **Team**: Samarth, Sambhav, Samiksha, Samith

## 2. Workstream Status (Phase 1)
- **Workstream A: Infrastructure** (Samarth): In Progress (Repo initialization & folder layout established).
- **Workstream B: Data** (Sambhav): In Progress (Raw IMD daily & NC gridded datasets collected).
- **Workstream C: Exploration and Features** (Samiksha): Pending setup.
- **Workstream D: Baselines and Evaluation** (Samith): Pending harness setup.

## 3. What is Done
- [x] Initialized Git repository and connected to remote (`https://github.com/sam123364/mlops_new.git`).
- [x] Defined repository structure according to AGENTS.md guidelines.
- [x] Moved `PROJECT_BRIEF.md` to `docs/`.
- [x] Placed raw IMD datasets under `data/raw/` and committed to GitHub.
- [x] Configured baseline configs: `regulations.yaml`, `demand.yaml`, `roof_params.yaml`, and `params.yaml`.
- [x] Cleaned daily district CSV and extracted 125-year (1901-2025) IMD NetCDF gridded timeseries to `data/processed/`.
- [x] Executed comprehensive EDA notebook (`notebooks/01_bengaluru_rainfall_eda_cleaning.ipynb`) validating 4 hydrological hypotheses with rich plots and unit simulations.

## 4. In Progress / Next Steps
- [ ] Initialize DVC (`dvc init`) for data versioning.
- [ ] Baseline temporal models (Workstream D: long-term mean, empirical quantiles, Markov chain).
- [ ] Spatial leave-one-gauge-out cross-validation harness.
- [ ] Hydrology module implementation in `src/hydrology/` with pytest suite.

## 5. Open Questions / Blockers
- None currently.
