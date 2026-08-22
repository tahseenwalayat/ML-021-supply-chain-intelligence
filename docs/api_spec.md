# API Contract Specification

## Overview

The FastAPI service exposes authenticated inventory, risk, simulation, MLOps, alert, and one-step-ahead demand forecasting operations. `GET /health` is the only public endpoint, for load balancers and orchestration probes.

## Authentication

Every `/api/v1/*` request must include `X-API-Key`. Local development uses the key from `.env`; production refuses the built-in development key. Generate a production key with `openssl rand -hex 32`.

## Endpoints

### Health check

`GET /health` returns `200` without authentication.

### One-step demand forecast

`POST /api/v1/forecast/predict` uses the newest feature-store row for the requested hierarchy and returns the next-day demand forecast. Weekly and monthly values are clearly labelled daily extrapolations, not independently trained horizon models.

```json
{
  "product_id": "P100",
  "region": "North",
  "hierarchy_level": "sku_region",
  "model_type": "lightgbm",
  "feature_overrides": {"sales_velocity_7d": 120.0}
}
```

Forecast serving requires a model artifact whose `target_col` is `target_next_day_sales`; legacy artifacts return `409` until retrained.

### Inventory, risk, simulation, and alerts

- `GET /api/v1/inventory/recommendation`
- `GET /api/v1/inventory/health`
- `GET /api/v1/inventory/utilization`
- `POST /api/v1/risk/*`
- `POST /api/v1/simulation/simulate-sku`
- `POST /api/v1/alerts/scan`

### MLOps

- `GET /api/v1/mlops/models` lists model WMAPE, target label, and serving readiness.
- `POST /api/v1/mlops/detect-degradation` evaluates current WMAPE degradation.
- `POST /api/v1/mlops/retrain` queues a background retraining job and returns a job ID.
- `GET /api/v1/mlops/retrain/{job_id}` returns its in-process status.

The interactive OpenAPI schema at `/docs` is the authoritative request and response definition.
