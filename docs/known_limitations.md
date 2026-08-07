# System Known Limitations, Assumptions & Future Roadmap

This document outlines prototype-scale caveats, synthetic data generation notes, mathematical model assumptions, infrastructure constraints, and future architectural enhancements for the **Enterprise Supply Chain Demand Forecasting & Risk Intelligence Engine**.

---

## 1. Prototype Scale & Synthetic Data Notes

1. **Synthetic Context Integration**:
   - The platform integrates four real-world benchmark datasets: **Olist E-Commerce**, **DataCo Supply Chain**, **Rossmann Store Sales**, and **M5 Walmart Sales**.
   - To enable multi-dimensional risk evaluation (e.g., weather impact, regional events, supplier reliability), contextual features were synthesized using seed-controlled statistical distributions (`src/ingestion/synthesize_context.py`).
   - *Impact*: While statistical relationships (correlations, seasonal demand spikes, lead-time variances) accurately reflect real-world supply chain dynamics, domain behavior in live enterprise production will depend on direct ERP/telemetry feeds.

2. **Supplier Lead Time & Delivery Variance Logs**:
   - Supplier late delivery rates and lead time standard deviations were derived from historical delivery logs. In a live production deployment, these should stream directly from supplier Electronic Data Interchange (EDI) or API integrations.

3. **Prototype Scope & Benchmark Dataset Scale**:
   - Data cleaning, feature store building, and forecasting models are evaluated on benchmark datasets sized for single-node memory footprint. Scaling to multi-terabyte production telemetry will require distributed data processing frameworks.

---

## 2. Mathematical & Model Assumptions

1. **Safety Stock Normal Distribution Assumption**:
   - Safety stock is calculated using standard normal distribution theory ($SS = Z \cdot \sigma_d \cdot \sqrt{L}$).
   - *Limitation*: High-volatility SKUs with non-Gaussian or highly skewed, intermittent demand (e.g. Poisson / Tweedie distributions) may experience minor safety stock underestimation.

2. **Economic Order Quantity (EOQ) Cost Constants**:
   - EOQ calculations ($EOQ = \sqrt{2DS/H}$) assume static ordering cost ($S$) and annual holding cost rate ($H$).
   - *Limitation*: Volume tier discounts, dynamic freight rates, and tiered warehouse storage pricing are not modeled in the core EOQ formula.

3. **Warehouse Capacity Limits**:
   - Warehouse utilization metrics currently compute capacity limits dynamically as $1.2 \times \text{total stock units}$ per facility for prototype demonstration purposes. In production, this connects to physical warehouse square footage / bin volume data models.

---

## 3. Infrastructure & Compute Limitations

1. **Single-Node Processing**:
   - Feature engineering and model training pipelines run using Pandas, NumPy, and LightGBM in single-node memory.
   - *Scale Limit*: Suitable for datasets up to ~10 million rows. Datasets exceeding 100M rows require migration to PySpark or Ray distributed compute.

2. **MLflow Storage Backend**:
   - Default local MLflow tracking uses a local SQLite database (`sqlite:///mlflow.db`) and local artifact storage (`./mlruns`).
   - *Production Path*: Should be configured with an AWS RDS PostgreSQL tracking database and S3 / GCS artifact buckets.

3. **In-Memory API Caching**:
   - Fast response times for batch risk evaluation and scenario simulation in prototype mode rely on local Parquet reads and in-memory evaluation. Production deployment utilizes Redis for distributed endpoint caching.

---

## 4. Future System Enhancements & Roadmap

| Feature Area | Planned Upgrade | Business Value |
|---|---|---|
| **Deep Learning Models** | Temporal Fusion Transformer (TFT), N-BEATS, DeepAR | Improved multi-horizon intermittent demand forecasting accuracy. |
| **Real-Time Streaming** | Apache Kafka / Apache Flink Ingestion Pipeline | Real-time stockout risk alerts triggered on live POS transactions. |
| **ERP System Connectors** | Native SAP S/4HANA & Oracle SCM REST/OData Connectors | Automated purchase order placement directly into enterprise ERP systems. |
| **Cold Chain IoT** | Telemetry ingestion for temperature & humidity sensors | Automated spoilage and transit condition risk evaluation. |
| **Multi-Echelon Optimization** | Multi-Echelon Inventory Optimization (MEIO) | Joint safety stock optimization across central DCs and regional stores. |
