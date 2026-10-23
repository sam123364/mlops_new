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
- [x] Placed raw IMD datasets under `data/raw/`.
- [x] Configured `.gitignore` to prevent committing raw data, secrets, and large artifacts.
- [x] Created baseline configs: `regulations.yaml`, `demand.yaml`, `roof_params.yaml`, and `params.yaml`.

## 4. In Progress / Next Steps
- [ ] Initialize DVC (`dvc init`) for data versioning.
- [ ] Ingest script for parsing IMD gridded and daily data.
- [ ] Set up basic CI pipeline via GitHub Actions.
- [ ] Write hydrology test cases based on equations in `AGENTS.md`.

## 5. Open Questions / Blockers
- None currently.
