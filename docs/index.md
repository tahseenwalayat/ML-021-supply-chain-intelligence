# Enterprise Platform Documentation Master Index

Welcome to the master sitemap for the **Enterprise Supply Chain Demand Forecasting & Risk Intelligence Platform**. This index links to all technical documentation produced across all platform lifecycle phases.

---

## 1. Core Architecture & System Specifications

- 🏗️ [**System Architecture Diagram**](architecture_diagram.md): Mermaid flowchart of end-to-end data ingestion, feature store, forecasting models, 5D risk engines, REST API, HA deployment, and MLOps retraining loop.
- 📋 [**Requirements Specification**](requirements.md): Functional and non-functional system requirements across all platform modules.
- 📐 [**Data Model Specification**](data_model.md): Star Schema entity-relationship design (`sales_fact`, `product_dim`, `supplier_dim`, `warehouse_dim`, `calendar_dim`).
- 🔌 [**REST API Specification**](api_spec.md): OpenAPI / FastAPI endpoint definitions, request/response schemas, and `X-API-Key` authentication details.
- ⚠️ [**Known Limitations & Roadmap**](known_limitations.md): Prototype-scale caveats, synthetic data notes, mathematical model assumptions, and future enhancement roadmap.
- ⛔ [**Out-of-Scope Boundaries**](out_of_scope.md): Boundaries and features explicitly excluded from the current project scope.
- 🎯 [**Requirements Traceability Matrix**](traceability.md): Mapping of functional requirements to implementation source code modules and automated test cases.

---

## 2. Ingestion, EDA & Feature Engineering

- 📊 [**Dataset Profile Report**](dataset_profile.md): Comprehensive data profiling of benchmark datasets (Olist E-Commerce, DataCo Supply Chain, Rossmann Store, M5 Walmart).
- 🔍 [**Exploratory Data Analysis (EDA) Findings**](eda_findings.md): Demand seasonality, lead time variance, supplier reliability, and price elasticity insights.
- 📜 [**Pipeline Execution Run Log**](pipeline_run_log.md): Operational execution log of raw data processing, context synthesis, and feature store assembly.

---

## 3. Modeling, Risk & Simulation Engines

- 🔮 [**Demand Forecasting Model Plan**](model_plan.md): GBDT (LightGBM, XGBoost) and Statistical Time Series (Prophet) modeling strategy and expanding-window cross-validation design.
- 📈 [**Model Evaluation Report**](model_evaluation_report.md): WMAPE benchmarks, seasonal naive baselines, cross-validation metrics, and SHAP feature importance analysis.
- ⚠️ [**5D Supply Chain Risk Engine Specification**](phase_12_supply_chain_risk_engine.md): Supplier delay, stockout exposure, overstock capital, slow-moving/dead stock, and demand anomaly risk formulations.
- 🎮 [**Scenario Stress Simulator Specification**](phase_13_scenario_simulation.md): Interactive "what-if" simulator for supplier delays, price changes, holiday surges, product launches, transport delays, and demand shocks.

---

## 4. Executive Dashboard, MLOps & Production Operations

- 🖥️ [**Dashboard Stack Selection & Rationale**](dashboard_stack.md): Streamlit & Plotly framework selection, multi-page layout, and strict API-First data access rules.
- ⚡ [**MLOps & Retraining Policy**](mlops_policy.md): Model registry promotion/demotion workflow, feature Population Stability Index (PSI) drift detection, rolling error tracking, and strict holdout promotion gates.
- 🚢 [**Production Deployment & Operations Guide**](deployment_guide.md): High-availability API scaling, NGINX load balancer configuration, Docker Compose, Kubernetes manifests, and emergency rollback procedures.
