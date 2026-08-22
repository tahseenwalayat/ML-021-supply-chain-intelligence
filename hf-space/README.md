---
title: ML021 Supply Chain API
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# ML021 Enterprise Supply Chain API

FastAPI REST API service providing Demand Forecasting, Inventory Optimization, 5D Risk Engine, Scenario Stress Simulation, Alert Center, and MLOps metrics.

## Platform Deployment Architecture

This Hugging Face Space hosts the **FastAPI Backend (Docker SDK)**.
The companion **Streamlit Executive Dashboard** is deployed on **Streamlit Community Cloud** and communicates with this Space via secure REST API calls.

## API Endpoints

- **Health Probe**: `GET /health` (Public)
- **Interactive Documentation**: `GET /docs` (Swagger UI) / `GET /redoc`
- **Demand Forecasting**: `POST /api/v1/forecast/predict`
- **Inventory Recommendations**: `GET /api/v1/inventory/recommendation`
- **Inventory Health**: `GET /api/v1/inventory/health`
- **Warehouse Utilization**: `GET /api/v1/inventory/utilization`
- **5D Risk Evaluation**: `POST /api/v1/risk/evaluate-batch`
- **Scenario Stress Testing**: `POST /api/v1/simulation/simulate-sku`
- **Alert Center**: `POST /api/v1/alerts/scan`
- **MLOps Models & Drift**: `GET /api/v1/mlops/models`, `POST /api/v1/mlops/detect-degradation`

## Authentication

All data and decisioning endpoints require the `X-API-Key` HTTP header:
```http
X-API-Key: sc-key-secret-2026
```

## Streamlit Cloud Integration

In your Streamlit Community Cloud app settings, add the following to **Secrets**:
```toml
API_BASE_URL = "https://<your-username>-<your-space-name>.hf.space"
API_KEY = "sc-key-secret-2026"
```
