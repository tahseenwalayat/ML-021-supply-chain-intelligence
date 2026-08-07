from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.risk.alert_center import AlertCenterEngine
from src.api.risk_router import BatchRiskEvaluationRequest

router = APIRouter(prefix="/api/v1/alerts", tags=["Alert Center Engine"])
alert_engine = AlertCenterEngine()


class ScanAlertsRequest(BaseModel):
    items_evaluation: BatchRiskEvaluationRequest
    capacity_utilization_pct: float = Field(85.0, ge=0.0, le=100.0)
    forecast_drift_pct: float = Field(0.0, ge=0.0)


@router.post("/scan", summary="Scan Operational Risks and Generate Prioritized Alert Feed")
def scan_alerts(req: ScanAlertsRequest) -> Dict[str, Any]:
    """Scans inventory and 5D risk inputs to generate a prioritized operational alert feed."""
    try:
        from src.risk.risk_engine import SupplyChainRiskEngine
        engine = SupplyChainRiskEngine()

        evaluated_items = []
        for item in req.items_evaluation.items:
            res = engine.evaluate_item_risk(
                product_id=item.product_id,
                warehouse_id=item.warehouse_id,
                region=item.region,
                supplier_id=item.supplier_id,
                current_stock=item.current_stock,
                reorder_point=item.reorder_point,
                safety_stock=item.safety_stock,
                avg_daily_demand=item.avg_daily_demand,
                avg_lead_time=item.avg_lead_time,
                lead_time_std_days=item.lead_time_std_days,
                supplier_reliability_score=item.supplier_reliability_score,
                late_delivery_rate=item.late_delivery_rate,
                unit_cost=item.unit_cost,
                sales_velocity=item.sales_velocity,
                zero_sales_weeks=item.zero_sales_weeks,
                demand_val=item.demand_val,
                mean_demand=item.mean_demand,
                std_demand=item.std_demand
            )
            evaluated_items.append(res)

        alerts = alert_engine.scan_inventory_and_risk_alerts(
            evaluated_items=evaluated_items,
            capacity_utilization_pct=req.capacity_utilization_pct,
            forecast_drift_pct=req.forecast_drift_pct
        )

        return {
            "total_alerts": len(alerts),
            "critical_count": sum(1 for a in alerts if a["severity"] == "CRITICAL"),
            "high_count": sum(1 for a in alerts if a["severity"] == "HIGH"),
            "medium_count": sum(1 for a in alerts if a["severity"] == "MEDIUM"),
            "info_count": sum(1 for a in alerts if a["severity"] == "INFO"),
            "alerts": alerts
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
