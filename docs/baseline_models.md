# Baseline Models & Evaluation Protocol Reference

**Project**: Urban Rainwater Harvesting Estimation (Bengaluru)  
**Workstream D**: Baselines, Evaluation & MLflow Tracking  
**Target Goal**: Sizing storage tanks and recharge pits under Normal ($P_{50}$) and 1-in-5 Drought ($P_{10}$) scenarios.

---

## 1. Why Baseline Models are Mandatory in MLOps
In climate and rainfall modelling, non-stationarity and natural intra-annual variability make it easy for complex machine learning models (e.g. neural networks or boosting trees) to overfit or produce illusionary skill. 

Per `docs/PROJECT_BRIEF.md` (Section 7.3) and `AGENTS.md`:
> *"Every ML model must beat simple baselines, and results are reported honestly if it does not."*

---

## 2. Selected Baseline Models & Rationale

| ID | Model Name | Category | Theoretical & Hydrological Rationale |
| :--- | :--- | :--- | :--- |
| **B1** | **Historical Climatological Mean / Median** | Non-parametric central benchmark | Represents standard civil engineering practice (using long-term monthly normals). Provides the baseline Mean Absolute Error (MAE) for point estimates. |
| **B2** | **Empirical Climatological Quantiles ($P_{10}, P_{50}, P_{90}$)** | Non-parametric distribution | RWH design requires **probabilistic safety bounds** ($P_{10}$ for dry year drought security, $P_{50}$ for median balance). Past empirical quantiles serve as the primary distributional baseline. |
| **B3** | **Lag-1 Persistence** | Autoregressive memory | Standard meteorological benchmark assuming rainfall this season will mimic the same season last year. Proves whether year-to-year persistence contains predictive power. |
| **B4** | **Linear Quantile Regression on Engineered Features** | Parametric linear model | Uses pre-selected engineered predictors (`rain_lag1m`, `rolling_3m_sum`, `rolling_12m_sum`, `month_sin`, `month_cos`) to assess whether antecedent moisture memory improves upon static historical percentiles. |

---

## 3. Strict Time-Based Expanding-Window Protocol
- **No Random Shuffling**: Random train/test splits destroy temporal dependency and cause future-to-past information leakage.
- **Protocol**:
  - Minimum warm-up training period: **30 years** ($1901 \to 1930$).
  - Out-of-sample evaluation period: **1970 to 2025** ($56\text{ years} = 672\text{ monthly forecasts}$).
  - For each test year $T$, the models are trained strictly on data prior to year $T$, predicting year $T$.

---

## 4. Evaluation Metrics
1. **Pinball Loss (Quantile Loss)**:
   $$L_q(y, \hat{y}_q) = \max(q(y - \hat{y}_q), (q - 1)(y - \hat{y}_q))$$
   Evaluated at $q \in \{0.10, 0.50, 0.90\}$.
2. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
3. **Prediction Interval Coverage Probability (PICP)**:
   Empirical proportion of observations falling within the $[P_{10}, P_{90}]$ interval (nominal target: $80\%$).

---

## 5. Benchmark Performance Results (1970 – 2025 Out-of-Sample)

| Model | MAE (mm) | Pinball $Q_{10}$ | Pinball $Q_{50}$ | Pinball $Q_{90}$ | 80% Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B1 Climatological Mean** | $34.57$ | — | — | — | — |
| **B1 Climatological Median** | $33.81$ | — | — | — | — |
| **B2 Empirical Quantiles** | $33.81$ | **$5.85$** | **$16.90$** | **$11.00$** | **$80.95\%$** |
| **B3 Lag-1 Persistence** | $48.13$ | — | — | — | — |
| **B4 Quantile Regression** | $40.83$ | $6.14$ | $20.42$ | $12.45$ | $79.76\%$ |

### Key Hydrological Findings:
1. **Persistence is Inferior ($MAE = 48.13\text{ mm}$)**: Year-to-year rainfall in Bengaluru has low autoregressive memory, failing against simple climatology.
2. **Empirical Quantiles Provide Exact Calibration ($80.95\%$ Coverage)**: Empirical quantiles reliably capture the nominal $80\%$ uncertainty band without underestimating drought extremes.
3. **Linear Regression Challenge**: A basic linear quantile regression on past lags underperforms empirical seasonal distribution ($MAE = 40.83$ vs $33.81$). Any future ML model (e.g. LightGBM or Quantile Trees) must be tested against **B2** to prove real value!

---

## 6. MLOps Versioning: DVC & DAGsHub + MLflow Architecture

```
[Processed Features] ---> [DVC Versioning (.dvc files in Git)] ---> [DAGsHub DVC Remote]
                                 |
[Baseline Models]   ---> [MLflow SQLite Registry]             ---> [DAGsHub Hosted MLflow]
```

### Remote Connection Instructions:
To link local tracking with DAGsHub:
```bash
# 1. Add DAGsHub DVC remote
dvc remote add origin https://dagshub.com/sam123364/mlops_new.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-username>
dvc remote modify origin --local password <your-token>
dvc push

# 2. Point MLflow to DAGsHub
export MLFLOW_TRACKING_URI="https://dagshub.com/sam123364/mlops_new.mlflow"
export MLFLOW_TRACKING_USERNAME="<your-username>"
export MLFLOW_TRACKING_PASSWORD="<your-token>"
```
