# Model Governance & Regulatory Compliance Framework
**Project:** Urban Rainwater Harvesting Estimation (Bengaluru)  
**Phase:** 3 – Productionization & Deployment  
**Role:** Model Governance, Compliance & Risk Management  

---

## 1. Executive Summary & Purpose
In municipal-scale urban water planning, machine learning models directly impact capital allocation (sizing storage tanks, recharge pits, and public subsidies) and homeowner legal compliance under the Bangalore Water Supply and Sewerage Board (BWSSB) regulations.

This document formalizes the **Governance and Risk Framework** governing model training, automated promotion gates, runtime provenance, and regulatory alignment.

---

## 2. Model Governance Lifecycle Architecture

```
[ Data Ingest & DVC Hash ] 
            │
            ▼
[ Time-Blocked Cross Validation (1940–2025) ]
            │
            ▼
[ Automated Quality Gate (src/models/register_model.py) ]
      ├── MAE Floor Check (< 35.0 mm)
      ├── 80% Coverage Tolerance (>= 75.0%)
      └── Block-Bootstrap Pinball Superiority Test
            │
      ┌─────┴──────────────────────┐
      │ Fails                      │ Passes
      ▼                            ▼
[ Flagged & Rejected ]      [ Registered in MLflow ]
                             (@challenger / @production)
                                   │
                                   ▼
                            [ OCI Labeled Container ]
                             (Non-root UID 10001, /model)
```

---

## 3. Four Core Pillars of Project Governance

### Pillar I: Automated Quality Gates & Rejection Criteria
To prevent unvetted or degraded models from entering production, candidate models must pass three automated gates defined in `src/models/register_model.py`:

| Gate | Threshold Metric | Technical Rationale |
|---|---|---|
| **Zero-Skill MAE Floor** | $\text{MAE} < 35.0\text{ mm}$ | Derived from the B1 Climatology Mean benchmark (~34.56 mm). Any model worse than basic climatology represents negative predictive skill. |
| **Quantile Coverage Guardrail** | $\text{Coverage} \ge 75.0\%$ | For nominal $80\%$ interval ($Q_{10}$ to $Q_{90}$), allows max $5\%$ empirical miscoverage across extreme dry/wet years. |
| **Statistical Superiority Gate** | Year-Block Bootstrap 95% CI $< 0$ | A candidate model only replaces the champion if the difference in pinball loss is statistically significant at $\alpha = 0.05$. |

> **Current Result:** Baseline B2 (Empirical Quantiles) achieved $\text{MAE} = 33.81\text{ mm}$ and $\text{Coverage} = 80.95\%$, passing all gates. Complex models (M1–M5) tied or performed worse, hence B2 is properly crowned as the **Production Champion**.

---

### Pillar II: Model Card, Lineage & Runtime Transparency
Every deployed model exposes a governance card via the live HTTP endpoint `GET /model`:
- **Commit Traceability:** Git commit SHA (`org.opencontainers.image.revision`) baked into Docker labels and registry metadata.
- **Data Lineage:** DVC data hash of the source IMD 0.25° grid and KSNDMC telemetry.
- **Quantile Semantics:** Explicit definition of output quantiles ($Q_{10}$ dry, $Q_{50}$ median, $Q_{90}$ wet).
- **Declared Limitations:**
  1. *Spatial resolution:* 27 km grid cells smooth local micro-climate variability.
  2. *Temporal resolution:* Monthly totals do not capture sub-daily cloudbursts.
  3. *Forecast lead:* Maximum 1-month lead forecast; subsequent months fall back to climatology.

---

### Pillar III: Regulatory Decoupling (`config/regulations.yaml`)
Municipal bylaws (such as BWSSB mandatory RWH mandates for sites $\ge 1200\text{ sq ft}$ / $108\text{ m}^2$) are frequently amended by urban local bodies.

**Governance Principle:** Legal regulations must **never be hardcoded** inside application or ML logic.
- Managed in `config/regulations.yaml`.
- Each rule carries metadata:
  - `value` & `unit`
  - `source` (official government gazette/act)
  - `confidence` (e.g., medium/high)
  - `last_verified` (date of audit)

---

### Pillar IV: Security & Container Hardening
- **Non-Root Execution:** Container runs strictly as `app` (UID 10001, without login shell) under the principle of least privilege.
- **Read-Only Filesystem:** Container root filesystem is mounted read-only; transient scratch space is confined to an in-memory `tmpfs` at `/tmp`.
- **Zero Secret Exposure:** Credentials are never committed to Git or baked into Docker layers. Environment injection is handled via orchestration secrets.
