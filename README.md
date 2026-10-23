# Urban Rainwater Harvesting Estimation (Bengaluru)

MLOps Course Project at RVCE.

## Objective
For any Bengaluru building, estimate harvestable rooftop rainwater (normal and dry years), recommend tank or recharge well sizing, and report how much domestic demand it offsets, accompanied by uncertainty estimates.

## Repository Layout
```text
├── AGENTS.md                  # Coding agent rules and project specification
├── README.md                  # Project overview
├── params.yaml                # Global pipeline parameters
├── dvc.yaml                   # DVC pipeline stages
├── config/                    # Domain and regulatory configurations
│   ├── demand.yaml            # Per-capita water demand figures
│   ├── regulations.yaml       # BWSSB/local RWH regulatory thresholds
│   └── roof_params.yaml       # Runoff coefficients & first flush values
├── data/
│   ├── raw/                   # Raw data files (DVC-tracked)
│   └── processed/             # Cleaned & feature-engineered data
├── docs/                      # Documentation
│   ├── PROJECT_BRIEF.md       # Comprehensive methodology & brief
│   ├── PROJECT_STATE.md       # Current progress & task tracker
│   └── data_sources.md        # Provenance, licences, and download dates
├── notebooks/                 # Exploratory Data Analysis (EDA)
├── src/                       # Source code modules
│   ├── ingest/                # Data collection and raw parsing
│   ├── features/              # Feature engineering & preprocessing
│   ├── models/                # Probabilistic rainfall & spatial ML models
│   ├── evaluation/            # Backtesting & leave-one-gauge-out harness
│   └── hydrology/             # Deterministic yield & water balance equations
└── tests/                     # Unit tests & data leakage checks
```

## Quick Start
1. Clone repo:
   ```bash
   git clone https://github.com/sam123364/mlops_new.git
   cd mlops_new
   ```
2. Set up virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
