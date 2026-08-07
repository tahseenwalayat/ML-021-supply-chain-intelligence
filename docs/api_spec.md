# API Contract Specification

## Overview

The Enterprise Supply Chain REST API service provides high-performance, secured, real-time endpoints for demand forecasting, inventory optimization, risk intelligence, scenario simulation, MLOps, and automated alerts.

---

## Base URL & Authentication

- **Base URL**: `http://localhost:8000/api/v1`
- **OpenAPI / Interactive Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Authentication**: `X-API-Key` HTTP Header required for all protected endpoints.
  - Default Test Key: `sc-key-secret-2026`

---

## Endpoint Specifications

### 1. Health Check (`GET /health`)
Public health probe endpoint for container orchestration and load balancers.
- **Request**: `GET /health` (No authentication required)
- **Response** (`200 OK`):
```json
{
  "status": "healthy",
  "service": "supply-chain-api",
  "version": "1.0.0",
  "modules": ["forecast", "inventory", "risk", "simulation", "mlops", "alerts"]
}
```

### 2. Demand Forecast (`GET /api/v1/forecast/{product_id}/{warehouse_id}`)
Retrieves historical demand and model forecast trajectories for a specific SKU-warehouse pair. Uses Redis caching for high throughput and low latency.
- **Parameters**: `product_id` (str), `warehouse_id` (str), `horizon_days` (int, default=30)
- **Header**: `X-API-Key: sc-key-secret-2026`
- **Response** (`200 OK`):
```json
{
  "product_id": "SKU_0001",
  "warehouse_id": "WH_NORTH",
  "forecast_horizon_days": 30,
  "allocated_daily_demand": 12.5,
  "confidence_interval_95": {"lower": 9.2, "upper": 15.8},
  "cached": true
}
```

### 3. Inventory Recommendations (`GET /api/v1/inventory/recommendation`)
Retrieves optimized inventory allocation, safety stock, reorder point, EOQ, and procurement quantities.
- **Query Params**: `product_id` (optional), `warehouse_id` (optional), `region` (optional)
- **Header**: `X-API-Key: sc-key-secret-2026`
- **Response** (`200 OK`):
```json
{
  "total_items": 1,
  "recommendations": [
    {
      "product_id": "SKU_0001",
      "warehouse_id": "WH_NORTH",
      "safety_stock": 25.0,
      "reorder_point": 85.0,
      "eoq": 150.0,
      "current_stock": 40.0,
      "recommended_procurement_qty": 195.0,
      "procurement_status": "REORDER_REQUIRED"
    }
  ]
}
```

### 4. Supply Chain Risk Scores (`POST /api/v1/risk/evaluate-batch`)
Calculates 5-dimensional composite risk scores across inventory nodes.
- **Header**: `X-API-Key: sc-key-secret-2026`
- **Request Body**:
```json
{
  "items": [
    {
      "product_id": "SKU_0001",
      "warehouse_id": "WH_NORTH",
      "current_stock": 10.0,
      "reorder_point": 50.0,
      "safety_stock": 20.0,
      "avg_daily_demand": 5.0
    }
  ]
}
```
- **Response** (`200 OK`): Bounded risk scores ($[0.0, 1.0]$), severity levels (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`), and recommended buffer days.

### 5. Stress Test Scenario Simulation (`POST /api/v1/simulation/simulate-sku`)
Executes interactive what-if operational stress testing.
- **Header**: `X-API-Key: sc-key-secret-2026`
- **Request Body**:
```json
{
  "base_daily_demand": 10.0,
  "current_stock": 100.0,
  "reorder_point": 50.0,
  "safety_stock": 20.0,
  "base_lead_time": 7.0,
  "unit_cost": 15.0,
  "scenario_parameters": {
    "scenario_name": "Supplier Delay Test",
    "supplier_delay_days": 5.0,
    "demand_surge_multiplier": 1.25,
    "simulation_horizon_days": 30
  }
}
```
- **Response** (`200 OK`): Baseline vs scenario fill rates, stockout exposure days, extra safety stock, and capital requirements.

---

## Security, Caching & Rate Limiting

- **API Key Security**: Requests missing `X-API-Key` or with an invalid key return `401 Unauthorized`.
- **Redis Caching**: Hot forecast responses cached with configurable TTL (default: 300 seconds).
- **Rate Limiting**: IP and API-key rate limiting middleware restricts requests to prevent service overload.
