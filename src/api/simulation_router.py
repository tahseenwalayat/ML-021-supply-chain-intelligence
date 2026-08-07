from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.simulation.scenario_simulator import ScenarioSimulationEngine, ScenarioParameters

router = APIRouter(prefix="/api/v1/simulation", tags=["Scenario Simulation Engine"])
sim_engine = ScenarioSimulationEngine()


class SKUSimulationRequest(BaseModel):
    base_daily_demand: float = Field(10.0, gt=0.0)
    current_stock: float = Field(100.0, ge=0.0)
    reorder_point: float = Field(50.0, ge=0.0)
    safety_stock: float = Field(20.0, ge=0.0)
    base_lead_time: float = Field(7.0, gt=0.0)
    unit_cost: float = Field(15.0, ge=0.0)
    scenario_parameters: ScenarioParameters


@router.post("/simulate-sku", summary="Run Interactive Stress-Test Scenario Simulation for SKU")
def simulate_sku(req: SKUSimulationRequest) -> Dict[str, Any]:
    """Executes an interactive scenario stress test simulation for a specific SKU."""
    try:
        return sim_engine.simulate_sku_scenario(
            base_daily_demand=req.base_daily_demand,
            current_stock=req.current_stock,
            reorder_point=req.reorder_point,
            safety_stock=req.safety_stock,
            base_lead_time=req.base_lead_time,
            unit_cost=req.unit_cost,
            params=req.scenario_parameters
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
