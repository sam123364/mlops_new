# Model Registry & Governance Guide

This guide describes the Model Registry architecture, lifecycle governance, and stage promotion policies implemented for the **Bengaluru Urban Rainwater Harvesting** project.

---

## 1. Why a Model Registry is Needed in MLOps

In real-world machine learning systems, training a model is only the first step. Without centralized model governance, organizations face operational hazards:

- **Strict Model Versioning & Artifact Traceability:**
  Production models cannot rely on volatile local weights or arbitrary file paths. The MLflow Model Registry provides an immutable, chronological versioning system (`v1`, `v2`, `v3`...) linking serialized model weights directly to the exact code commit, training configuration, and metrics that produced them.
- **Data & Feature Lineage:**
  Through our dual DVC–MLflow integration, every registered model is tagged with:
  - `git_commit`: The exact code commit in Git.
  - `dvc_data_version`: The cryptographic hash of the training data from `.dvc` files.
  This guarantees 100% reproducible training lineage from raw IMD/KSNDMC observations to registered models.
- **Stage Progression & Audit Governance (Staging vs Production):**
  Decoupling training from serving enables clear operational stages:
  - `None` $\rightarrow$ Untested candidate models.
  - `Staging` $\rightarrow$ Models undergoing automated validation, leakage checks, and shadow backtesting.
  - `Production` $\rightarrow$ Verified models authorized for active downstream yield estimation and decision-support systems.
  - `Archived` $\rightarrow$ Deprecated models retained for regulatory auditability and instant rollback capability.

---

## 2. Inspecting Models on DAGsHub MLflow UI

Our project configures both local SQLite storage and remote hosted tracking on DAGsHub:
- **DAGsHub Repository:** `https://dagshub.com/sam123364/mlops_new`
- **Remote Tracking URI:** `https://dagshub.com/sam123364/mlops_new.mlflow`

### How to View Registered Models on DAGsHub:
1. Navigate to your repository page on DAGsHub (`https://dagshub.com/sam123364/mlops_new`).
2. Click the **MLflow** tab on the repository navigation bar or access the experiment tracker URL directly.
3. In the MLflow UI top navigation bar, click on **Models**.
4. Select the registered model:
   ```
   bengaluru_rainfall_probabilistic_model
   ```
5. On the Model Overview page, you can inspect:
   - **Version History:** Every logged version, timestamp, and author.
   - **Stage / Alias:** Current stage (`Staging`, `Production`, `Archived`) and modern MLflow aliases (`@production`).
   - **Tags & Metadata:**
     - `git_commit`: Links directly to GitHub commit diff.
     - `dvc_data_version`: Verifies the data snapshot used.
     - `target_quantiles`: Probabilistic bounds `[0.10, 0.50, 0.90]`.
     - `methodology`: Modeling approach (e.g. empirical quantiles / quantile regression).
   - **Source Run:** Click on the source run to inspect all logged hyperparameters, test metrics (`mae`, `coverage_80pct`, `pinball_loss`), and evaluation artifacts.

---

## 3. Production Promotion Policy & Benchmark Criteria

Model promotion is governed by an automated quality gate in [`src/models/register_model.py`](file:///Users/samikshadadarya/Documents/antigravity/fearless-kepler/mlops_new/src/models/register_model.py).

When a model is registered:
1. It is automatically initialized in the **`Staging`** stage.
2. The promotion gate `promote_to_production(model_version)` evaluates two strict objective benchmarks:

| Benchmark Criterion | Threshold Required | Rationale |
| :--- | :--- | :--- |
| **Point Estimate Error (`MAE`)** | **`MAE < 35.0 mm`** | Ensures rainfall errors do not excessively distort tank sizing and rooftop rainwater harvesting yield calculations. |
| **80% Prediction Interval Coverage** | **`Coverage >= 75.0%`** | Validates that probabilistic bounds capture semi-arid monsoon variability without under-predicting extreme dry or wet periods. |

### Gate Execution Workflow:
- **Pass:** If both criteria are satisfied, the model version transitions from `Staging` to `Production`, older production models are archived, and the `@production` alias is updated.
- **Fail:** If either criterion fails, promotion is blocked. The model remains in `Staging` with clear explanatory warning logs recorded for engineering review.

---

## 4. Usage Instructions

### Run Registration Locally:
```bash
python -m src.models.register_model --tracking-uri sqlite:///mlflow.db
```

### Run Registration with Production Promotion Gate:
```bash
python -m src.models.register_model --promote --tracking-uri sqlite:///mlflow.db
```

### Run Remote DAGsHub Sync:
```bash
export DAGSHUB_USER_TOKEN="<your_token>"
python -m src.models.register_model --promote --tracking-uri https://dagshub.com/sam123364/mlops_new.mlflow
```
