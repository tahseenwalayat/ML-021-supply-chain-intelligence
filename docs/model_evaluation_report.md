# Demand Forecasting Engine - Model Evaluation & Explainability Report

## 1. Executive Summary

This report documents the empirical holdout evaluation, performance benchmarking, model explainability analysis, and MLflow Model Registry staging selection for the Enterprise Demand Forecasting Engine.

> **Evaluation status - August 2026:** The historic numerical results in this report were produced before the feature-store target was corrected to `target_next_day_sales`. They must not be used for acceptance, promotion, or comparison. Re-run training and holdout evaluation before publishing replacement metrics.

To evaluate model generalization and prevent temporal lookahead leakage, all candidate models—spanning **Tabular Gradient Boosted Decision Trees (LightGBM, XGBoost)** and **Univariate Trend Baselines (Prophet)**—were evaluated on a **strictly held-out 28-day time period** (`2018-08-07` to `2018-09-03`). This window was never used during model training, expanding-window cross-validation, or hyperparameter optimization.

### Current Results Summary:
- **All hierarchy levels**: Pending retraining and target-aligned holdout evaluation.
- **MLflow Model Registry**: Do not promote existing artifacts on the basis of the historic metrics below. New artifacts must declare `target_col=target_next_day_sales`.

---

## 2. Evaluation Methodology & Holdout Strategy

### 2.1 Strictly Held-Out Temporal Window
Per `docs/model_plan.md`, evaluating time-series forecasting models with random K-Fold cross-validation is strictly invalid due to temporal autocorrelation and lookahead leakage. The evaluation scheme enforces a strict **28-day temporal holdout split**:

$$\text{Holdout Window} = [\text{2018-08-07} \text{ to } \text{2018-09-03}]$$

- **No Future Leakage**: All feature windows (lags, rolling averages, seasonality indices) were generated strictly up to the training cutoff.
- **Sample Sizes**:
  - Level 1 (SKU-Region): **4,786** holdout observation rows
  - Level 2 (Category-Region): **4,662** holdout observation rows
  - Level 3 (Region-Total): **4,630** holdout observation rows

### 2.2 Standard Evaluation Metrics
Four core complementary metrics are computed across all models and hierarchy levels:

1. **WMAPE (Weighted Mean Absolute Percentage Error)** *(Primary Ranking Metric)*:
   $$\text{WMAPE} = \frac{\sum_{t=1}^{N} |y_t - \hat{y}_t|}{\sum_{t=1}^{N} y_t} \times 100\%$$
2. **RMSE (Root Mean Squared Error)**: Penalizes high-magnitude outlier forecast errors.
3. **MAPE (Mean Absolute Percentage Error)**: Mean percentage deviation on non-zero sales days.
4. **Forecast Bias (Percentage Mean Error)**: Identifies systematic over-forecasting ($>0$) or under-forecasting ($<0$).

---

## 3. Historical Holdout Evaluation Results - Superseded

The figures in this section are retained only as an audit record. They are not comparable with the current next-day forecasting pipeline and do not demonstrate the case-study WAPE target.

### 3.1 Overall Model Comparison Table

| Hierarchy Level | Candidate Model | Holdout WMAPE (%) | RMSE | MAPE (%) | Bias (%) | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Level 3 (Region-Total)** | **LightGBM** | **25.49%** | **1.28** | **21.38%** | **-17.51%** | **SELECTED (Staging)** |
| | XGBoost | 40.00% | 0.60 | 41.25% | +26.68% | Benchmark |
| | Seasonal Naive (7d) | 99.81% | 1.85 | 100.00% | - | Baseline |
| **Level 2 (Category-Region)** | **XGBoost** | **45.17%** | **0.62** | **45.63%** | **+32.48%** | **SELECTED (Staging)** |
| | LightGBM | 71.40% | 6.70 | 68.12% | +62.29% | Benchmark |
| | Seasonal Naive (7d) | 99.98% | 1.18 | 100.00% | - | Baseline |
| **Level 1 (SKU-Region)** | **LightGBM** | **93.67%** | **1.22** | **97.94%** | **+79.65%** | **SELECTED (Staging)** |
| | XGBoost | 101.55% | 1.25 | 105.12% | +96.29% | Benchmark |
| | Prophet (Top 10 Series CV)*| 7.19% | 0.45 | 6.85% | +2.14% | Benchmark (Macro) |
| | Seasonal Naive (7d) | 100.00% | 1.50 | 100.00% | - | Baseline |

*\*Note: Prophet was trained and validated on top high-volume series as a macro seasonality baseline.*

---

## 4. Historical Segment-Level Performance Analysis - Superseded

Evaluating overall performance alone hides critical operational edge-case risks. Model performance was dissected across operational segments: **Overall**, **Cold-Start Products** ($\le 30$ days history), **Promotional Periods**, and **Holiday Periods**.

### Segment Performance Breakdown for Winning Staging Models

#### 1. Level 3 (Region-Total) - LightGBM
- **Overall**: WMAPE = **25.49%**, RMSE = **1.28**, Bias = **-17.51%** (4,630 rows)
- **Cold-Start Segment**: WMAPE = **22.14%**, RMSE = **0.49**, Bias = **-22.10%** (1,491 rows)
- *Insight*: Regional aggregate demand exhibits stable seasonality; cold-start items benefit strongly from regional category mean imputation, achieving lower error variance.

#### 2. Level 2 (Category-Region) - XGBoost
- **Overall**: WMAPE = **45.17%**, RMSE = **0.62**, Bias = **+32.48%** (4,662 rows)
- **Cold-Start Segment**: WMAPE = **45.24%**, RMSE = **0.62**, Bias = **+32.40%** (1,508 rows)
- *Insight*: XGBoost's exact depth-wise tree growth provides robust generalization across category groupings, maintaining consistent error performance between mature and cold-start products.

#### 3. Level 1 (SKU-Region) - LightGBM
- **Overall**: WMAPE = **93.67%**, RMSE = **1.22**, Bias = **+79.65%** (4,786 rows)
- **Cold-Start Segment**: WMAPE = **98.29%**, RMSE = **1.25**, Bias = **+93.01%** (1,563 rows)
- *Insight*: Granular SKU-level demand contains high intermittency and zero-demand periods. LightGBM captures non-zero velocity signals significantly better than XGBoost (93.67% vs 101.55%).

---

## 5. Model Explainability & Feature Drivers

Global and local explainability analysis was conducted to establish trust in model predictions and verify feature hygiene.

### 5.1 Global Feature Importance Summaries

Global feature importances were extracted and saved into `docs/shap_summary/`:

- **SKU-Region Level Summary Plot**: [shap_summary_sku_region.png](file:///D:/ML-021/docs/shap_summary/shap_summary_sku_region.png)
- **Category-Region Level Summary Plot**: [shap_summary_category_region.png](file:///D:/ML-021/docs/shap_summary/shap_summary_category_region.png)
- **Region-Total Level Summary Plot**: [shap_summary_region_total.png](file:///D:/ML-021/docs/shap_summary/shap_summary_region_total.png)

#### Key Global Feature Insights:
1. **Short-Term Demand Velocity**: `sales_velocity_7d`, `sales_velocity_14d`, and `sales_velocity_30d` are the single most dominant drivers across all hierarchy levels.
2. **Temporal & Seasonal Signals**: `days_until_next_holiday`, `days_since_last_holiday`, and `seasonality_index_dow` provide critical macro adjustments for holiday spikes and day-of-week demand patterns.
3. **Categorical Entity Identifiers**: High-cardinality features like `product_id` and `category` heavily influence baseline demand levels in LightGBM.

### 5.2 Local Explanation Examples

Local explanations analyze specific prediction scenarios to show how feature values push predictions away from the historical base mean:

#### Scenario A: High-Demand Spike (Level 1 SKU-Region)
- **Actual Demand**: 3,498.0 units | **Predicted Demand**: 3,159.9 units | **Base Mean**: 567.0 units
- **Top Positive Drivers**:
  1. `sales_velocity_30d = 3106.3` $\rightarrow$ Impact: **+542.9 units**
  2. `sales_velocity_14d = 3033.6` $\rightarrow$ Impact: **+475.7 units**
  3. `sales_velocity_7d = 2869.9` $\rightarrow$ Impact: **+422.9 units**
- *Takeaway*: High historical momentum over recent rolling windows correctly drives predictions up toward peak sales volume.

#### Scenario B: Promotion Period (Level 2 Category-Region)
- **Actual Demand**: 5.0 units | **Predicted Demand**: 3.4 units | **Base Mean**: 1,868.8 units
- **Top Drivers**:
  1. `weather_temperature_c = 32.9°C` $\rightarrow$ Impact: **-520.6 units** (High heat suppression)
  2. `sales_acceleration_7d = 0.51` $\rightarrow$ Impact: **+255.5 units** (Promo momentum uplift)
  3. `promo_days_in_last_14d = 14.0` $\rightarrow$ Impact: **-76.0 units** (Promo fatigue attenuation)

#### Scenario C: Low/Zero Demand Day (Level 3 Region-Total)
- **Actual Demand**: 1.0 units | **Predicted Demand**: 0.86 units | **Base Mean**: 3,527.9 units
- **Top Drivers**:
  1. `day_of_week = 6 (Sunday)` $\rightarrow$ Impact: **-196.9 units** (Closed store/low weekend traffic)
  2. `month = 11 (November)` $\rightarrow$ Impact: **-121.8 units** (Pre-holiday trough)
  3. `seasonality_index_dow = 0.74` $\rightarrow$ Impact: **+115.8 units**

---

## 6. Historical Strengths, Weaknesses, and Model Selection Rationale - Superseded

### 6.1 Model Architecture Comparison

| Model Architecture | Strengths | Weaknesses | Best Use Case |
| :--- | :--- | :--- | :--- |
| **LightGBM** | Fast training speed, native categorical feature handling, superior handling of zero-inflated high-cardinality series. | Sensitive to hyperparameter tuning on noisy aggregate trends. | SKU-Region & Region-Total forecasting |
| **XGBoost** | Depth-wise tree growth prevents overfitting; robust regularized splits on aggregated category series. | Slower training on high cardinality; higher memory overhead. | Category-Region forecasting |
| **Prophet** | Interpretable trend decomposition, explicit holiday effect curve fitting. | Scalability bottleneck; cannot handle cross-series tabular features natively. | Macro trend benchmarking |

### 6.2 Final Model Selection Rationale
1. **Level 3 (`region_total`)**: Selected **LightGBM** due to lowest holdout WMAPE (25.49%) and low prediction latency.
2. **Level 2 (`category_region`)**: Selected **XGBoost** due to lowest holdout WMAPE (45.17%) and tight error bounds across category aggregates.
3. **Level 1 (`sku_region`)**: Selected **LightGBM** due to superior handling of sparse SKU data (WMAPE 93.67% vs XGBoost 101.55%).

---

## 7. Historical MLflow Model Registry Status - Superseded

All three selected winning models have been registered in the MLflow Model Registry and tagged with the production-ready `staging` alias:

| Registered Model Name | Version | Model Type | Hierarchy Level | Assigned Alias | Holdout WMAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Demand_Forecasting_REGION_TOTAL` | `v2` | LightGBM | `region_total` | `staging` | **25.49%** |
| `Demand_Forecasting_CATEGORY_REGION` | `v2` | XGBoost | `category_region` | `staging` | **45.17%** |
| `Demand_Forecasting_SKU_REGION` | `v2` | LightGBM | `sku_region` | `staging` | **93.67%** |

---

## 8. Limitations & Recommendations for Future Improvements

### 8.1 Current Limitations
1. **Intermittent Demand Bias**: At Level 1 (SKU-Region), high proportions of zero-demand days lead to positive forecast bias (+79.65%).
2. **Extreme Promotion Spikes**: Fast 1-day promo spikes occasionally exhibit promo fatigue under-forecasting.

### 8.2 Future Improvement Roadmap
1. **Two-Stage Croston / Classification-Regression Pipeline**: Implement a binary classification model ($P(\text{sale} > 0)$) combined with a regression model ($E[y \mid y > 0]$) for intermittent SKUs.
2. **Hierarchical Reconciliation (MinT)**: Apply Minimum Trace linear covariance matrix reconciliation to guarantee that bottom-up SKU predictions perfectly sum to category and regional totals.
3. **Deep Learning Sequence Models (TFT)**: Evaluate Temporal Fusion Transformers for multi-horizon joint sequence modeling.
