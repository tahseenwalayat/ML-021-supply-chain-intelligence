# Changelog

All notable changes to the **Enterprise Supply Chain Demand Forecasting & Risk Intelligence Engine** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-08-07

### 🌟 Release Overview
Version 1.0.0 marks the final enterprise release of the Supply Chain Demand Forecasting & Risk Intelligence Engine. This release delivers an end-to-end platform featuring multi-horizon AI forecasting, mathematical safety stock optimization, a 5D operational risk engine, interactive scenario stress testing, an API-First Streamlit dashboard, automated MLOps retraining, and high-availability Docker/Kubernetes deployment.

---

### Added

#### Phase 1: Data Ingestion & Context Synthesis
- Ingested 4 real-world benchmark datasets: Olist E-Commerce, DataCo Supply Chain, Rossmann Store Sales, and M5 Walmart Sales (`data/raw/`).
- Built ETL ingestion pipeline (`src/ingestion/build_unified_dataset.py`) generating cleaned interim parquet tables in `data/interim/`.
- Developed context synthesizer (`src/ingestion/synthesize_context.py`) integrating synthetic weather, promotional campaigns, and regional event indicators.
- Modeled Star Schema entity-relationship data model (`sales_fact.parquet`, `product_dim.parquet`, `supplier_dim.parquet`, `warehouse_dim.parquet`, `calendar_dim.parquet`).

#### Phase 2: Feature Engineering & Unified Feature Store
- Implemented 9 domain-specific feature engineering modules in `src/features/`:
  1. `sales_velocity.py`: Short/long-term moving averages, velocity trends.
  2. `demand_seasonality.py`: Day-of-week, month, quarter, Fourier cyclical features.
  3. `volatility_metrics.py`: Demand coefficient of variation ($CV$), rolling standard deviation.
  4. `lead_time_features.py`: Lead time mean, standard deviation, variance.
  5. `supplier_reliability.py`: On-time delivery rates, late delivery frequencies.
  6. `promotion_impact.py`: Discount depth, active campaign flags.
  7. `product_lifecycle.py`: Days since launch, zero-sales week counters.
  8. `regional_demand.py`: Regional aggregation and warehouse level demand ratios.
  9. `holiday_effects.py`: Proximity counters to past/future holidays, weather parameters.
- Built unified feature store assembly pipeline (`src/features/build_feature_table.py`) generating a 1,112,929 row $\times$ 54 column feature store saved at `data/processed/feature_store.parquet`.

#### Phase 3: Demand Forecasting & Evaluation Engine
- Designed expanding-window time-series cross-validation splitter (`src/forecasting/dataset_split.py`).
- Implemented GBDT forecasting models (`train_lightgbm.py`, `train_xgboost.py`) and Prophet additive time-series models (`train_prophet.py`) across 3 hierarchy levels (`sku_region`, `category_region`, `region_total`).
- Built model evaluator (`src/forecasting/evaluate.py`) benchmarking WMAPE, MAE, RMSE, MAPE, and Bias against Seasonal Naive (Lag 7) baselines.
- Integrated SHAP feature explainability engine (`src/forecasting/explainability.py`).
- Implemented Optuna TPE hyperparameter optimization (`src/forecasting/hyperparam_search.py`).

#### Phase 4: Mathematical Inventory Optimization Engine
- Developed Safety Stock ($SS = Z \cdot \sigma_d \cdot \sqrt{L}$) and Reorder Point ($ROP = d \cdot L + SS$) calculators (`src/inventory/safety_stock.py`, `reorder_point.py`).
- Built Economic Order Quantity ($EOQ = \sqrt{2DS/H}$) calculator (`src/inventory/eoq.py`).
- Implemented multi-warehouse inventory allocation engine (`src/inventory/allocation.py`).
- Automated batch inventory recommendation pipeline (`src/inventory/run_inventory_engine.py`) outputting `data/processed/inventory_recommendations.parquet`.

#### Phase 5: 5D Supply Chain Risk Engine & Alert Center
- Created composite 5D Risk Engine (`src/risk/risk_engine.py`) integrating 5 risk modules:
  1. `supplier_delay_risk.py`: Lead-time variance and late delivery frequency scorecards.
  2. `stockout_risk.py`: Stockout probability and days of inventory remaining.
  3. `overstock_risk.py`: Excess inventory capital tied up above ROP thresholds.
  4. `inventory_health_risk.py`: Slow-moving and dead stock capital risk evaluation.
  5. `demand_anomaly_risk.py`: Demand surge and statistical Z-score anomaly detector.
- Built operational Alert Center engine (`src/risk/alert_center.py`) generating prioritized alert feeds across 6 categories (`LOW_INVENTORY`, `OVERSTOCK`, `DEMAND_SPIKE`, `SUPPLIER_DELAY`, `WAREHOUSE_CAPACITY`, `FORECAST_DRIFT`).

#### Phase 6: Scenario Stress Simulator
- Built interactive scenario simulation sandbox (`src/simulation/scenario_simulator.py`) supporting 6 supply chain perturbation shocks:
  1. Supplier Failure / Lead Time Delay (`supplier_failure.py`)
  2. Price Increase & Elasticity (`price_increase.py`)
  3. Holiday Sales Surge (`holiday_sales.py`)
  4. New Product Launch Cold-Start (`new_product_launch.py`)
  5. Regional Transport Disruption (`transport_delay.py`)
  6. Macroeconomic Demand Surge (`demand_surge.py`)

#### Phase 7: REST API Service Layer & Security
- Built FastAPI application (`api/main.py`) exposing REST endpoints across 5 domain routers (`risk_router.py`, `inventory_router.py`, `simulation_router.py`, `mlops_router.py`, `alerts_router.py`).
- Enforced `X-API-Key` header authentication middleware (`src/api/auth.py`).
- Included Pydantic data validation schemas across all request/response payloads.

#### Phase 8: Executive Streamlit Dashboard
- Built multi-page Streamlit dashboard (`dashboard/app.py`, `api_client.py`) adhering strictly to API-First access rules.
- Developed 7 focus module pages in `dashboard/pages/`:
  1. `kpi_overview.py`: Executive metrics, total inventory valuation, high-risk counts.
  2. `forecast_accuracy.py`: Hierarchy WMAPE charts, actual vs predicted trends.
  3. `inventory_health.py`: Safety stock levels, ROP indicators, capital tied up.
  4. `warehouse_utilization.py`: Storage unit accumulation and facility capacity gauges.
  5. `fill_rate_stockout.py`: Estimated fill rate curves and stockout risk distribution.
  6. `procurement_recommendations.py`: Optimized order quantities (EOQ) & reorder triggers.
  7. `alert_center.py`: Prioritized alert feed with severity filters (`CRITICAL`, `HIGH`, `MEDIUM`).

#### Phase 9: MLOps, Drift Detection & Automated Retraining
- Developed Population Stability Index (PSI $\ge 0.25$) feature drift detector and forecast WMAPE degradation detector (`src/mlops/drift_detector.py`).
- Built real-time prediction logger and latency monitor (`src/mlops/monitoring.py`).
- Built automated retraining pipeline (`src/mlops/retraining_pipeline.py`).
- Implemented model registry (`src/mlops/registry.py`) with strict holdout promotion quality gates (`Candidate WMAPE < Production WMAPE`).

#### Phase 10: Production HA Deployment & Kubernetes
- Configured NGINX load balancer (`nginx/nginx.conf`) distributing traffic across a 3-replica API pool (`api1`, `api2`, `api3`) on port 80.
- Authored production Docker Compose stack (`docker-compose.prod.yml`) integrating FastAPI, Streamlit, NGINX, MLflow tracking database, PostgreSQL, and Redis.
- Authored Kubernetes production manifests (`k8s/`).

#### Phase 11 & 12: Documentation Consolidation & Verification
- Consolidated master documentation set: [`README.md`](README.md), [`docs/architecture_diagram.md`](docs/architecture_diagram.md), [`docs/index.md`](docs/index.md), [`docs/known_limitations.md`](docs/known_limitations.md), and [`docs/presentation.md`](docs/presentation.md).
- Added Google/Sphinx style docstrings to all 26 public functions across `src/` and `api/`.
- Verified complete test suite passing (67 tests, 100% pass rate).
