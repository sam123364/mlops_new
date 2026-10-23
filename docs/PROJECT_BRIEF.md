# Project Brief: Urban Rainwater Harvesting Estimation for Buildings in Bengaluru

| | |
|---|---|
| **Version** | v0.1 (draft for team review) |
| **Date** | 23 October 2026 |
| **Course** | MLOps, RV College of Engineering (RVCE), Bengaluru |
| **Team** | Samarth, Sambhav, Samiksha, Samith |
| **Scope status** | Urban variant is the document's subject. The alternative (rural farm variant, eastern dry belt) shares the rainfall model, water-balance engine and MLOps pipeline; only the catchment and demand modules differ. Confirm the final choice as a team. |

**How to use this document.** Humans: read sections 1 to 9 so everyone shares the same picture. AI agents: read this file before any task, then `docs/PROJECT_STATE.md` for current status. Rules for agents are in `AGENTS.md`. If this brief and a chat instruction conflict, ask the team before acting.

**Evidence labels used below.** *Verified* = confirmed from a retrieved source. *Reported* = from news or blogs, not primary text. *Unverified* = from general knowledge or an assumption; check before relying on it.

---

## 1. Problem statement

Many buildings in Bengaluru are legally required to harvest rainwater, yet owners often install systems that are poorly sized, or collect the water and then drain it away. Owners and designers lack a quick way to answer: how much rainwater can this roof and plot actually yield in a normal year and in a dry year, what tank or recharge structure is worth building given the space available, and how much of the building's water demand will it offset? Generic rules of thumb ignore the specific building, its location within a city where rainfall varies, and its occupancy.

## 2. Proposed solution

A location-aware decision-support tool that, for a given building, estimates harvestable rainwater, recommends a storage or recharge structure, and reports how much demand it can offset, with uncertainty shown plainly.

**Inputs from the user:** map location; plot area; roof area (pre-filled from building footprints, editable); roof type; paved versus open ground; number of occupants or building type; existing sump or borewell; space available for a tank or recharge pit.

**Outputs to the user:**
1. Annual harvestable volume in a normal year and a dry year (1 in 5).
2. Recommended tank size (and/or recharge option) with a size-versus-benefit curve showing where extra capacity stops paying off.
3. Share of the building's demand (especially non-potable use) offset by harvested water.
4. A compliance checklist against BWSSB rules (see section 5; thresholds must be verified).
5. Estimated bill savings, only if BWSSB tariffs are sourced and cited.

## 3. Scope

**In scope:** rooftop harvesting for individual buildings in Bengaluru; probabilistic local rainfall; roof yield; daily water balance; tank sizing; storage-versus-recharge guidance; compliance checklist.

**Out of scope for this version:** treatment and water-quality design; structural or plumbing design drawings; city-scale flood modelling; areas outside Bengaluru; automated legal advice (the checklist is informational and must say so).

## 3.1 Where the ML sits (and where it does not)

Roof yield and the water balance are deterministic physics and arithmetic. The ML components are:
1. **Rainfall scenario model (temporal):** probabilistic seasonal and dry-year rainfall from the long IMD record.
2. **Spatial rainfall adjustment (spatial):** use the KSNDMC gauge network to adjust the coarse IMD estimate to the building's location, validated by holding out gauges.

Honest limits: a neural network will not reliably beat simple statistics on a few decades of seasonal totals. Every ML model must be compared against baselines (section 7.3) and reported honestly if it does not win.

## 4. Why this problem

- Compliance is mandatory for many buildings, so there is a clear user need.
- The catchment is a roof you can measure, so the runoff step is far less uncertain than for farmland (no curve number guesswork).
- Rainfall varies across the city: KSNDMC runs about 180 automated weather stations in Bengaluru Urban (reported), and in 2022 the east of the city received much more rain than other parts (reported). Average annual rainfall is about 855 mm, with plus or minus 20 percent counted as normal (reported).
- The rainfall model, water-balance engine and MLOps pipeline are reusable for other regions.

## 5. Regulatory context (read this carefully)

Sources are news articles, blogs and a 2009 draft of the regulations. **The Act's primary legal text was not retrieved. Verify every threshold with BWSSB before the checklist feature is trusted.**

| Item | What we found | Confidence |
|---|---|---|
| Core requirement | Rainwater harvesting is mandatory under the BWSSB Act as amended in 2009 (Section 72A). Thresholds commonly cited: new construction on sites of 1,200 sq ft (30x40 ft) and above; existing buildings on sites of 2,400 sq ft (40x60 ft) and above. A 2024 report says the 2021 amendment extended the requirement to older buildings on 40x60 ft and larger sites. | Reported, multiple sources agree |
| Conflicting source | One calculator site states different dates (40x60 ft and above since 2009; 30x40 ft and above from 2021). | Conflict, unresolved |
| What counts | The 2009 draft regulations describe roof-based and land-based harvesting: storage in a tank or recharge through an open well or borewell. | Draft text, final wording unconfirmed |
| Large buildings | News report that the Cabinet approved an amendment requiring buildings of 10,000 sq ft and above to use harvested rainwater via a dual-pipe system. Final status unconfirmed. | Reported, unconfirmed |
| Enforcement | A 2024 report says over 39,000 residential buildings were regularly fined, and newer buildings' plans are not sanctioned without rainwater harvesting. No penalty amounts found. | Reported |
| Design standards | Secondary sources cite IS 15797:2008 and the CGWB Rainwater Harvesting Manual (2023) for calculations. Primary documents not retrieved. | Unverified |
| Not checked | Municipal building bylaws (BBMP or successor); BWSSB tariffs; penalty amounts. | Unknown |

**Implementation rule:** keep all legal thresholds in `config/regulations.yaml`, each with fields `value`, `source`, `confidence`, `last_verified`. The app must display them as "per BWSSB rules, verify before relying".

## 6. Methodology

### 6.1 Pipeline
1. **Input handling.** Coordinates, plot and roof area, roof type, occupancy, space and existing structures.
2. **Local rainfall.** Combine the long IMD record (temporal behaviour) with KSNDMC gauge data (spatial pattern) to get rainfall scenarios for the building's location.
3. **Roof yield.** Daily inflow `Q_t = max(0, P_t - FF) x A_roof x C / 1000` in cubic metres, where `P_t` is daily rainfall (mm), `FF` is the first-flush depth discarded (mm), `A_roof` is roof area (m^2) and `C` is the runoff coefficient for the roof type. (1 mm over 1 m^2 = 1 litre.) Take `C` and `FF` from IS 15797 or the CGWB manual; **do not hard-code values from memory**. Carry a low and high choice through the model.
4. **Demand.** `demand = occupants x per-capita use x days`, split into potable and non-potable (flushing, gardening, cleaning). Per-capita norms must come from a cited source and live in `config/demand.yaml`. Real consumption data is not open; state this as a limitation.
5. **Daily water balance** for each candidate tank size `S_cap`: `available_t = S_(t-1) + Q_t`; `use_t = min(demand_t, available_t)`; `S_t = min(S_cap, available_t - use_t)`; `overflow_t = max(0, available_t - use_t - S_cap)`. Report **demand offset** = total `use` / total demand, and the overflow volume. Run for normal-year and dry-year rainfall scenarios.
6. **Sizing.** Plot demand offset against tank size; recommend the smallest size beyond which the marginal gain is small, capped by available space.
7. **Storage versus recharge.** Rule-based choice using space, soil infiltration and groundwater depth (inputs partly unavailable; see section 8). Not an ML claim.
8. **Outputs and compliance checklist** (section 2).

### 6.2 Known limitation: storm intensity
A daily time step can hide a roof tank overflowing within a single downpour. Sub-daily rainfall data may be needed for accurate small-tank sizing. Treat daily results as an approximation and say so.

### 6.3 Uncertainty
Report normal-year and dry-year results, plus low/high ranges for roof coefficient, first-flush loss and roof area. Never present a single unqualified number.

## 7. Modelling and evaluation plan

### 7.1 Temporal rainfall model
Target: seasonal rainfall total and longest dry spell for Bengaluru's IMD cell(s), as quantiles (distribution), not a single forecast. Season windows live in `params.yaml`.

### 7.2 Spatial adjustment
Estimate a location-specific adjustment relative to the IMD cell using KSNDMC gauge totals (and optionally elevation). Validate with **leave-one-gauge-out** cross-validation, using spatial blocking so that neighbouring gauges do not leak information.

### 7.3 Baselines every ML model must beat
Temporal: (a) long-term mean, (b) empirical distribution of past seasons, (c) persistence, (d) linear or quantile regression on lagged features. Spatial: (a) nearest gauge, (b) inverse-distance weighting, (c) the plain IMD cell value.

### 7.4 Evaluation
Time-based expanding-window backtests only (never random splits). Metrics: MAE for point estimates; pinball loss and interval coverage for quantiles; skill score versus climatology; for spatial: MAE and bias on held-out gauges. Log every run in MLflow with the git commit and the DVC data hash.

### 7.5 End-to-end validation
Check the water-balance code against hand-calculated examples (unit tests). If permission can be obtained, measure one real roof and tank (for example on campus) as a case study. Do not claim accuracy that has not been tested.

## 8. Datasets

Links come from web search results; **verify each works and record its licence in `docs/data_sources.md`. Never invent a dataset URL.**

| Dataset | Purpose | Link | Caveats |
|---|---|---|---|
| KSNDMC Bengaluru Urban rainfall, taluk and hobli, 2017 to 2024 (via OpenCity) | Local seasonal rainfall totals | https://data.opencity.in/dataset/bengaluru-urban-annual-rainfall-taluks-and-hoblis | Seasonal split (pre-monsoon, SW, NE). Hobli level only since 2020. Only about 8 years: too short for ML on its own. |
| KSNDMC automated weather station list (CSV and KML) | Gauge locations for spatial modelling | https://data.opencity.in/dataset/list-of-automated-weater-stations-aws-in-bengaluru-urban-rural-districts | Locations only. |
| KSNDMC telemetric stations and rain gauges group | Other station datasets | https://data.opencity.in/dataset/groups/karnataka-telemetric-weather-stations-and-rain-gauges | Public domain (open data) per listing. |
| IMD gridded daily rainfall, 0.25 deg, 1901 to present | Long temporal record | https://imdpune.gov.in (R helper: https://github.com/Subhradip25/imdR) | Binary files; about 25 km grid (only a cell or two over Bengaluru); server may be temporarily down; download once and version with DVC. |
| Google Open Buildings | Building footprints as roof-area proxy | https://developers.google.com/earth-engine/datasets (ID `GOOGLE/Research/open-buildings/v3/polygons`) | Footprint is not exactly the roof; let users correct it. |
| Combined Google and Microsoft footprints (VIDA) | More complete footprint coverage | https://gee-community-catalog.org/projects/global_buildings/ | Check source label per footprint. |
| Open Buildings Extractor (pulls Google, Microsoft, OSM, Overture footprints) | Scripted footprint download | https://pypi.org/project/obe/ | Python tool. |
| Sentinel-2 imagery | Paved versus open ground | https://dataspace.copernicus.eu/browser/ | Free account needed. |
| SoilGrids 250 m | Rough infiltration for recharge feasibility | https://soilgrids.org | Weak in built-up areas; indicator only. |
| SRTM 1 arc-second DEM | Elevation for spatial adjustment, low-lying sites | https://earthexplorer.usgs.gov (also https://opentopography.org/) | Free account needed. |

**Not found, no verified link (to request or research):** daily KSNDMC gauge data (ask KSNDMC); groundwater levels (CGWB or the state groundwater directorate); BWSSB tariffs and consumption data; plot and building sanction records; stormwater drain network; primary text of the BWSSB Act and regulations; IS 15797:2008 and the CGWB manual; local per-capita water use norms.

## 9. MLOps lifecycle and tools (free, student friendly)

| Stage | Tools |
|---|---|
| Planning | GitHub Issues and Projects, Markdown docs, model card, data card |
| Ingestion | Python: pandas, xarray, geopandas, rasterio; Earth Engine where useful |
| Validation | Pandera |
| Data versioning | DVC (remote: Google Drive or DagsHub) |
| Pipelines | DVC pipelines (`dvc.yaml`); Prefect later |
| Modelling | scikit-learn, statsmodels, XGBoost or LightGBM; deep learning only if it beats baselines |
| Tracking | MLflow (local first) |
| Quality | pytest, ruff, pre-commit |
| CI/CD | GitHub Actions |
| Serving | FastAPI, Docker |
| Frontend | Streamlit or Gradio with folium |
| Monitoring | Evidently AI; Prometheus and Grafana (local) |

Real data arrives once a year, so demonstrate retraining by **replaying history**: train to year N, release year N+1, monitor drift, retrain, repeat.

## 10. Phase 1 scope and work split

**Phase 1 deliverables:** dataset acquisition and approval; data exploration; feature engineering; baseline models; Git repository; DVC dataset versioning; MLflow experiment tracking; initial model evaluation.

**Phase 1 modelling tasks:** (1) temporal rainfall baselines on IMD data for Bengaluru; (2) spatial baselines on KSNDMC gauges with leave-one-gauge-out validation. The roof yield and water-balance modules are Phase 2 unless time allows, but unit-test the equations in section 6.1 early.

| Member | Workstream | Phase 1 tasks | Depends on |
|---|---|---|---|
| Samarth | A. Repo and MLOps infrastructure; integration lead | Git repo, branch rules, folder structure, CI skeleton; DVC init, remote, `dvc.yaml`; MLflow setup; merging; `PROJECT_STATE.md` | Starts first (days 1 to 2) |
| Sambhav | B. Data acquisition | Accounts and approvals; IMD and KSNDMC download scripts; email KSNDMC for daily gauge data; `docs/data_sources.md`; Pandera checks | A |
| Samiksha | C. Exploration and features | EDA report; feature engineering script; leakage tests | B (use a small sample until B delivers) |
| Samith | D. Baselines and evaluation | Metrics code; backtest and leave-one-gauge-out harness; baselines; MLflow logging; evaluation summary | A, C (build the harness on dummy data first) |

This split is a proposal; swap roles by skill. One branch per workstream, pull requests reviewed by the next member in the chain (A, B, C, D, A). Everyone must understand the whole pipeline for the viva.

## 11. Risks and open questions

| # | Risk or question | Why it matters | Mitigation |
|---|---|---|---|
| 1 | Daily gauge data may not be available | Without it the spatial and temporal ML is weak (only about 8 years of open seasonal totals) | Request from KSNDMC now; fall back to IMD for time and KSNDMC totals for spatial scaling |
| 2 | No open building water-demand data | The "how much is offset" half of the output rests on assumptions | Use cited per-capita norms, label as assumption, show sensitivity |
| 3 | Legal thresholds unconfirmed | A wrong checklist misleads users | `config/regulations.yaml` with sources; verify with BWSSB; disclaimer |
| 4 | Footprint is not the roof | Roof area error propagates linearly into yield | User-editable roof area; show range |
| 5 | Daily time step hides intense storms | Small tanks may be mis-sized | State limitation; seek sub-daily data later |
| 6 | No ground truth for harvested volumes | Cannot claim accuracy end to end | Hand-calculated tests; one measured campus roof if permitted |
| 7 | Urban versus rural final choice | Rework risk | Confirm as a team; keep shared modules location-agnostic |

## 12. Glossary

BWSSB: Bangalore Water Supply and Sewerage Board. KSNDMC: Karnataka State Natural Disaster Monitoring Centre. IMD: India Meteorological Department. AWS: automated weather station. DEM: digital elevation model. DVC: Data Version Control. MLflow: experiment tracking and model registry tool. First flush: the initial roof runoff discarded because it carries dirt. Demand offset: share of demand met from harvested water.

## 13. Change log

- v0.1 (23 Oct 2026): first draft.
