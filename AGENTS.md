# AGENTS.md: Urban Rainwater Harvesting Estimation (Bengaluru)

Rules and essentials for AI coding agents (Antigravity, Claude Code, Cursor, Codex). Keep this file short. Full detail lives in `docs/PROJECT_BRIEF.md`.

## Start of every session
1. Read `docs/PROJECT_BRIEF.md` (problem, methodology, datasets, risks).
2. Read `docs/PROJECT_STATE.md` (what is done, in progress, blocked, next).
3. Confirm which workstream and task you are doing. If unclear, ask.

## End of every session
Update `docs/PROJECT_STATE.md`: done, in progress, blocked, next steps, open questions. Commit.

## Project in five lines
- MLOps course project at RVCE, team of four (Samarth, Sambhav, Samiksha, Samith). Humans create accounts and credentials; you write code.
- Goal: for a Bengaluru building, estimate harvestable rooftop rainwater (normal year and dry year), recommend tank or recharge size, and report how much demand it offsets, with uncertainty.
- ML parts: (1) probabilistic rainfall scenarios from the long IMD record; (2) spatial adjustment of rainfall using KSNDMC gauges, validated by leave-one-gauge-out.
- Deterministic parts: roof yield, daily water balance, tank sizing, compliance checklist.
- Phase 1 only: dataset acquisition, EDA, features, baselines, Git, DVC, MLflow, initial evaluation. Do not build the UI or serving layer yet.

## Decisions that are fixed (ask before changing)
- Rainfall is the ML component. Report distributions (quantiles), not a single forecast.
- Every ML model must beat simple baselines, and results are reported honestly if it does not.
- Time-based splits only for temporal data. Spatial validation uses leave-one-gauge-out with spatial blocking. Never random splits.
- Show uncertainty (normal and dry year; low and high coefficients). Never output one unqualified number.
- Legal thresholds are NOT confirmed. Keep them in `config/regulations.yaml` with `value`, `source`, `confidence`, `last_verified`. Never hard-code them in logic.
- Runoff coefficient, first-flush depth and per-capita demand must come from cited sources, stored in config files. Never hard-code from memory.

## Core equations (for unit tests)
- Daily roof inflow (m3): `Q_t = max(0, P_t - FF) * A_roof * C / 1000` (P and FF in mm, A in m2).
- Balance: `available_t = S_(t-1) + Q_t`; `use_t = min(demand_t, available_t)`; `S_t = min(S_cap, available_t - use_t)`; `overflow_t = max(0, available_t - use_t - S_cap)`.
- Demand offset = sum(use) / sum(demand).
- 1 mm of rain on 1 m2 = 1 litre.

## Datasets (verify every link; never invent URLs)
IMD gridded rainfall (imdpune.gov.in); KSNDMC rainfall and gauge locations via OpenCity (data.opencity.in); Google Open Buildings and VIDA combined footprints (Earth Engine / gee-community-catalog.org); Sentinel-2 (dataspace.copernicus.eu); SoilGrids (soilgrids.org); SRTM (earthexplorer.usgs.gov). Full table with caveats is in the brief, section 8. Record source, licence, download date and limitations in `docs/data_sources.md`.

## Repository layout (target)
```
AGENTS.md  README.md  params.yaml  dvc.yaml
config/   regulations.yaml  demand.yaml  roof_params.yaml
data/raw/  data/processed/     (DVC-tracked, never in Git)
src/      ingest/ features/ models/ evaluation/ hydrology/
tests/    notebooks/ (EDA only)   docs/
```

## Tooling
Python; pandas, xarray, geopandas, rasterio; scikit-learn, statsmodels, XGBoost or LightGBM; Pandera (validation); DVC (data versioning, pipelines); MLflow (tracking, local `mlruns` first); pytest; ruff; GitHub Actions for CI.

## Rules
- **Secrets:** never read, print or commit `.env` or any credential. Credentials are created by humans and kept in a git-ignored `.env`. Never put secrets in Git or DVC.
- **Never invent** data, URLs, metrics, citations or results. If a source is unreachable, say so and stop that step.
- **Ask first** before installing packages, deleting files, force-pushing, rewriting history, or changing the fixed decisions above.
- **Leakage:** write tests proving no feature uses information from the period being predicted.
- **Tests with code:** run `pytest` and `ruff` before committing. Use hand-calculated cases for the hydrology equations.
- **Reproducibility:** configuration in `params.yaml` and `config/`; pipelines in `dvc.yaml`; every run logged to MLflow with the git commit and DVC data hash.
- **Commits:** small and frequent, clear messages. One branch per workstream: `feat/infra`, `feat/data`, `feat/eda-features`, `feat/baselines`.
- **Style:** type hints where practical, short docstrings, no large notebooks.
- **Honest reporting:** state limitations (coarse IMD grid, few open KSNDMC years, footprint is not the roof, daily time step hides intense storms).

## Workstreams (Phase 1)
- A Infrastructure (Samarth): repo, CI skeleton, DVC, MLflow, integration.
- B Data (Sambhav): download scripts, KSNDMC request, `data_sources.md`, validation.
- C Exploration and features (Samiksha): EDA, feature script, leakage tests.
- D Baselines and evaluation (Samith): metrics, backtest and leave-one-gauge-out harness, baselines, MLflow logging.
