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
- [x] Added pinned `requirements.txt` (Python 3.12; data, geospatial, modelling, DVC, MLflow, pytest, ruff, Jupyter). Local setup: `uv python install 3.12`, `uv venv --python 3.12 .venv`, `uv pip install --python .venv/bin/python -r requirements.txt`. On macOS, LightGBM also needs `brew install libomp`.

## 4. In Progress / Next Steps
- [ ] Initialize DVC (`dvc init`) for data versioning.
- [ ] Baseline temporal models (Workstream D: long-term mean, empirical quantiles, Markov chain).
- [ ] Spatial leave-one-gauge-out cross-validation harness.
- [ ] Hydrology module implementation in `src/hydrology/` with pytest suite.

## 5. Open Questions / Blockers
- `data/raw/` files were committed directly to Git (commit `fac3e85`); AGENTS.md says raw data must be DVC-tracked. Move them to DVC during `dvc init`.
- The IMD NetCDF stores its variables in a `rain` group: open with `xr.open_dataset(path, group="rain")`.
