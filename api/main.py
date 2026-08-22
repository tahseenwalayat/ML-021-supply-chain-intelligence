import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.auth import verify_api_key
from src.api.risk_router import router as risk_router
from src.api.simulation_router import router as simulation_router
from src.api.mlops_router import router as mlops_router
from src.api.alerts_router import router as alerts_router
from src.api.inventory_router import router as inventory_router
from src.api.forecast_router import router as forecast_router

app = FastAPI(
    title="Enterprise Supply Chain Demand Forecasting & Risk Intelligence API",
    description="REST API service for Demand Forecasting, Inventory Optimization, 5D Risk Engine, Scenario Simulation, Alert Center, and MLOps",
    version="1.0.0"
)

_cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_origins == "*" else [o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep /health public for orchestration probes. Every decisioning and data route
# requires the configured X-API-Key.
secured_router_dependencies = [Depends(verify_api_key)]
app.include_router(risk_router, dependencies=secured_router_dependencies)
app.include_router(simulation_router, dependencies=secured_router_dependencies)
app.include_router(mlops_router, dependencies=secured_router_dependencies)
app.include_router(alerts_router, dependencies=secured_router_dependencies)
app.include_router(inventory_router, dependencies=secured_router_dependencies)
app.include_router(forecast_router, dependencies=secured_router_dependencies)


@app.get("/health")
def health_check():
    """Returns service health status, version, and active platform modules."""
    return {
        "status": "healthy",
        "service": "supply-chain-api",
        "version": "1.0.0",
        "modules": [
            "supplier_delay_risk",
            "stockout_risk",
            "overstock_risk",
            "inventory_health_risk",
            "demand_anomaly_risk",
            "risk_engine",
            "scenario_simulation",
            "mlops_drift_detector",
            "alert_center",
            "inventory_optimization"
        ]
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=port,
        reload=os.getenv("ENVIRONMENT", "development").lower() == "development",
    )
