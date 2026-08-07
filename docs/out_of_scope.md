# Out-of-Scope & Prototype Simplifications

## Overview
This document outlines the operational boundaries, technical simplifications, and mocked features for the **Enterprise Supply Chain Demand Forecasting & Inventory Optimization Platform** prototype. To maintain focus on core machine learning forecasting accuracy, statistical inventory optimization, and supply chain risk evaluation within prototype constraints, full enterprise-scale streaming infrastructure, live vendor integrations, and hardware-specific automation are deliberately scoped out or simplified.

---

## Simplified & Mocked Components

### 1. External Live Data Feeds & APIs
* **Live Weather Data API**: Real-time integrations with weather services (e.g., OpenWeatherMap, NOAA) are replaced by static historical calendar attributes and seasonal flags derived from the **Rossmann** and **M5** datasets. Environmental impact on demand is modeled via static seasonal indices rather than live weather polling.
* **Live Supplier & ERP Feeds**: Real-time SAP/Oracle ERP webhooks and live vendor EDI feeds for purchase order updates are mocked. Supplier lead times, delivery delays, and fulfillment statuses are simulated using static distribution profiles extracted from the **DataCo** and **Olist** datasets.
* **Real-Time Vehicle & GPS Tracking**: Live IoT telematics, fleet GPS tracking, and carrier API webhooks are omitted. Transit durations and delivery risk factors are calculated based on historical fulfillment records in **DataCo** and **Olist**.

### 2. Enterprise Infrastructure & Deployment
* **Distributed Streaming Infrastructure**: Real-time event-streaming architectures (e.g., Apache Kafka, Apache Flink) are simplified to Python-based batch processing and local SQLite/PostgreSQL database ETL pipelines.
* **Multi-Tenant Authentication & Enterprise RBAC**: Production-grade Single Sign-On (SSO), Active Directory/OAuth2 multi-tenant RBAC, and granular audit logging are out of scope. The prototype exposes lightweight REST endpoints protected by basic API keys or local session states.
* **Distributed Multi-Node Training Clusters**: Multi-GPU/TPU distributed model training (e.g., Ray, Spark ML, Horovod) is simplified to local, single-node execution of scikit-learn, LightGBM, XGBoost, and Prophet models.

### 3. Physical Warehouse & Logistics Automation
* **Automated Purchase Order Execution**: Direct, automated issuance of binding purchase orders to external vendors is out of scope. The platform generates actionable procurement quantity recommendations, requiring manual approval or simulated API triggers.
* **Robotics & AGV Dispatch Systems**: Physical warehouse dispatch, Automated Guided Vehicle (AGV) routing, and physical bin placement optimization are excluded. Warehouse optimization is limited to mathematical allocation targets, safety stock, and inventory fill rates.
* **Customs, Tariff & Currency Services**: Live international cross-border customs tracking, tariff rate calculators, and real-time forex exchange rate APIs are mocked using static regional currency mappings and fixed freight structures.

### 4. Advanced AI Extensions (Future / Bonus Scope)
* **Full Multi-Echelon Reinforcement Learning (RL)**: Deep RL agents (e.g., PPO/DQN) for dynamic multi-echelon policy control are kept as optional extensions. The core prototype baseline relies on provable statistical inventory logic (Safety Stock, ROP, EOQ).
* **Graph Neural Networks (GNN) for Network Modeling**: End-to-end GNN topologies for supply chain network propagation are simplified to relational database tables and graph-like adjacency mappings.
* **3D Digital Twin Warehouse Simulation**: Full 3D CAD/WebGL physics rendering of warehouse interiors is replaced by 2D dashboard metric indicators, capacity heatmaps, and parameter sliders.
