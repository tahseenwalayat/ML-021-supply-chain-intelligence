# Executive Dashboard Architecture & Stack Rationale

## 1. Selected Stack & Rationale

For the frontend of the **Enterprise Supply Chain Demand Forecasting & Risk Intelligence Engine**, **Streamlit** (supported by **Plotly**) was selected as the core dashboarding framework.

### Key Drivers & Rationale:
1. **Rapid Development Velocity**: Streamlit enables rapid prototyping and production deployment of data-intensive executive interfaces entirely in pure Python.
2. **Native Interactive Visualizations**: Plotly provides dark-themed, high-performance rendering for multi-horizon demand forecast confidence bands, 5D risk radar charts, and warehouse capacity utilization gauges.
3. **Native Multi-Page Routing**: Streamlit's file-based `pages/` directory architecture natively supports modular, maintainable multi-page navigation across all executive focus areas.
4. **Decoupled Architecture**: By completely isolating the frontend dashboard from backend persistence, the dashboard operates strictly as a REST API consumer, demonstrating system modularity and scalability.

---

## 2. Strict API-First Data Access Strategy

To enforce enterprise data governance, security, and single-source-of-truth principles:
- **Zero Direct Storage Access**: The dashboard code **never** reads Parquet files (`.parquet`), SQLite/PostgreSQL databases, or local ML artifact storage directly.
- **HTTP REST Protocol**: All metrics, inventory recommendations, risk scores, scenario simulations, and operational alerts are fetched dynamically via HTTP `GET` and `POST` calls to the FastAPI server (`http://localhost:8000`).
- **API Key Security**: Every API request includes the `X-API-Key` authentication header, retrieved securely from the `API_KEY` environment variable (defaulting to `sc-key-secret-2026`).

---

## 3. Module & Page Directory Mapping

| Page File | Module Covered | Key Backend API Endpoint |
|---|---|---|
| `dashboard/pages/kpi_overview.py` | Supply Chain KPIs & Executive Overview | `/health`, `/api/v1/inventory/health`, `/api/v1/risk/evaluate-batch` |
| `dashboard/pages/forecast_accuracy.py` | Forecast Accuracy & Model Evaluation | `/api/v1/mlops/models`, `/api/v1/mlops/detect-degradation` |
| `dashboard/pages/inventory_health.py` | Inventory Health & Working Capital | `/api/v1/inventory/health`, `/api/v1/inventory/recommendation` |
| `dashboard/pages/warehouse_utilization.py` | Warehouse Capacity & Utilization | `/api/v1/inventory/utilization` |
| `dashboard/pages/fill_rate_stockout.py` | Fill Rate & Stockout Risk | `/api/v1/risk/stockout`, `/api/v1/simulation/simulate-sku` |
| `dashboard/pages/procurement_recommendations.py` | Procurement & Safety Stock (ROP/EOQ) | `/api/v1/inventory/recommendation`, `/api/v1/risk/supplier-delay` |
| `dashboard/pages/alert_center.py` | 6 Operational Alert Types | `/api/v1/alerts/scan` |

---

## 4. Graceful Error Handling & API Resilience

The dashboard implements automated API state checking:
- If the FastAPI backend is unreachable or returns HTTP errors (e.g., 403, 404, 500, or connection timeout), pages catch the exception cleanly.
- Instead of throwing raw Python tracebacks, an executive warning banner (`st.error`) displays diagnostic information and instructions to launch the backend server (`uvicorn api.main:app --port 8000`).
