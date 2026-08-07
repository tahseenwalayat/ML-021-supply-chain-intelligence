# MLOps & Automated Retraining Governance Policy

## 1. Executive Summary & Objectives

This document establishes the operational governance, model registry lifecycle policy, automated retraining trigger rules, and strict promotion quality gates for the **Enterprise Supply Chain Demand Forecasting Platform**.

The objective of this policy is to guarantee continuous forecast accuracy, prevent silent data drift degradation, enforce strict holdout validation quality gates, and maintain complete audit traceability for every model stage transition.

---

## 2. Model Registry Governance & Lifecycle Stages

All demand forecasting model artifacts (LightGBM, XGBoost, Prophet) are tracked via the Model Registry (`src/mlops/registry.py`) and assigned explicit operational stages:

| Stage Alias | Purpose | Promotion / Demotion Rules |
|---|---|---|
| **Staging** | Validation & Candidate Evaluation | Newly trained candidate models default to `Staging` while undergoing holdout validation testing. |
| **Production** | Live REST API Serving | Only **one active version** per hierarchy level (e.g., `sku_region`) holds the `Production` alias. |
| **Archived** | Historic Baseline Retention | Superceded or demoted Production models are transitioned to `Archived` for audit rollback capability. |

### Promotion & Demotion Logging Requirement:
Every stage transition (`promote_model`, `demote_model`) **must log a explicit textual justification reason**, recording:
- Timestamp & Operator / Automated Service ID
- Model Name & Version ID (`YYYYMMDD_HHMMSS`)
- Source Stage & Target Stage
- Holdout Validation Metrics (WMAPE, Naive WMAPE, RMSE)
- Textual Justification Reason

---

## 3. Retraining Trigger Policy

Automated model retraining is triggered via two complementary mechanisms:

### A. Scheduled Retraining Cadence
- **Frequency**: Every Sunday at 02:00 UTC (Weekly Cadence).
- **Scope**: Re-runs feature engineering (`src.features.build_feature_table`) across the latest 30-day sliding window and retrains models across all hierarchy levels (`sku_region`, `category_region`, `region_total`).

### B. Event & Drift-Triggered Retraining
Retraining is dynamically triggered whenever operational drift monitors detect statistically significant shifts:

1. **Feature Data Drift Trigger**:
   - **Metric**: Population Stability Index (PSI) computed via `src.mlops.drift_detection.py`.
   - **Threshold**: $\text{PSI} \ge 0.25$ or Kolmogorov-Smirnov Test $p$-value $< 0.05$ across key demand velocity/lead-time features.
2. **Forecast Error Degradation Trigger**:
   - **Metric**: Relative WMAPE degradation computed via `src.mlops.forecast_drift.py`.
   - **Threshold**: Rolling actual-vs-predicted WMAPE degrades by $\ge 20\%$ relative to the baseline training WMAPE (e.g., WMAPE increases from 11.8% to $\ge 14.2\%$).

---

## 4. Strict Promotion Gate Constraint

> [!CRITICAL]
> **Mandatory Quality Gate**: The automated retraining pipeline (`src/mlops/retraining_pipeline.py`) **MUST NEVER** promote a candidate model to `Production` if it performs worse than or equal to the current active `Production` model on the same holdout validation dataset.

### Evaluation Protocol:
1. Candidate model is trained and evaluated on expanding window cross-validation splits to calculate `candidate_wmape`.
2. The current active Production model's holdout performance is fetched from the registry (`production_wmape`).
3. **Decision Logic**:
   $$\text{Promote to Production} \iff \text{Candidate WMAPE} < \text{Production WMAPE}$$
4. **Rejection Action**: If $\text{Candidate WMAPE} \ge \text{Production WMAPE}$, the candidate model is tagged as `REJECTED_STAGING_ONLY`, remains in `Staging`, and a rejection justification log is recorded.

---

## 5. Prediction Monitoring & Operational SLAs

Live inference requests served by the API are monitored in real time via `src/mlops/monitoring.py`:

- **Latency SLA**: 95th Percentile ($P_{95}$) prediction latency must remain $\le 50\text{ ms}$.
- **Schema Validation**: Inputs are scanned for null value spikes, missing required columns, and data type mismatches.
- **Prediction Distribution Stats**: Min, Max, Mean, Std Dev, and Zero-Sales prediction counts are logged to `models/prediction_monitoring_logs.json`.
