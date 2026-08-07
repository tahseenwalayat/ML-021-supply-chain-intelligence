from fastapi import FastAPI
from src.api.risk_router import router as risk_router
from src.api.simulation_router import router as simulation_router
from src.api.mlops_router import router as mlops_router
from src.api.alerts_router import router as alerts_router
from src.api.inventory_router import router as inventory_router

app = FastAPI(
    title="Enterprise Supply Chain Demand Forecasting & Risk Intelligence API",
    description="REST API service for Demand Forecasting, Inventory Optimization, 5D Risk Engine, Scenario Simulation, Alert Center, and MLOps",
    version="1.0.0"
)

app.include_router(risk_router)
app.include_router(simulation_router)
app.include_router(mlops_router)
app.include_router(alerts_router)
app.include_router(inventory_router)


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

