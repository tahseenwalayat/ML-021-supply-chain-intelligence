from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.risk.supplier_delay_risk import (
    calculate_supplier_delay_risk,
    evaluate_supplier_delay_details,
    classify_risk_level
)
from src.risk.stockout_risk import (
    calculate_stockout_risk,
    evaluate_stockout_details
)
from src.risk.overstock_risk import (
    calculate_overstock_risk,
    evaluate_overstock_details
)
from src.risk.inventory_health_risk import (
    calculate_inventory_health_risk
)
from src.risk.demand_anomaly_risk import (
    calculate_demand_anomaly_risk
)
from src.risk.risk_engine import SupplyChainRiskEngine

router = APIRouter(prefix="/api/v1/risk", tags=["Supply Chain Risk Engine"])
risk_engine = SupplyChainRiskEngine()


# --- Pydantic Request Models ---

class SupplierDelayRiskRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    reliability_score: float = Field(1.0, ge=0.0, le=1.0, description="Supplier reliability score [0.0, 1.0]")
    lead_time_std: float = Field(0.0, ge=0.0, description="Lead time standard deviation in days")
    lead_time_avg: float = Field(7.0, gt=0.0, description="Mean lead time in days")
    late_delivery_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Late delivery rate [0.0, 1.0]")
    w_late: float = Field(0.6, ge=0.0, description="Weight for late delivery rate")
    w_var: float = Field(0.4, ge=0.0, description="Weight for lead time variance")


class StockoutRiskRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    current_stock: float = Field(..., ge=0.0, description="Current stock on hand")
    reorder_point: float = Field(..., ge=0.0, description="Reorder point (ROP)")
    safety_stock: float = Field(..., ge=0.0, description="Safety stock buffer level")
    avg_daily_demand: float = Field(1.0, ge=0.0, description="Average daily sales rate")
    lead_time_days: float = Field(7.0, gt=0.0, description="Replenishment lead time in days")


class OverstockRiskRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    current_stock: float = Field(..., ge=0.0, description="Current stock on hand")
    reorder_point: float = Field(..., ge=0.0, description="Reorder point (ROP)")
    unit_cost: float = Field(10.0, ge=0.0, description="Unit purchasing cost ($)")
    avg_daily_demand: float = Field(1.0, ge=0.0, description="Average daily sales rate")
    overstock_rop_multiplier: Optional[float] = Field(3.0, gt=1.0, description="Multiplier threshold above ROP")


class InventoryHealthRiskRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sales_velocity: float = Field(1.5, ge=0.0, description="Average daily sales units")
    zero_sales_weeks: int = Field(0, ge=0, description="Consecutive weeks with zero sales")
    current_stock: float = Field(100.0, ge=0.0, description="Current stock on hand")
    unit_cost: float = Field(10.0, ge=0.0, description="Unit cost ($)")


class DemandAnomalyRiskRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    demand_val: float = Field(..., ge=0.0, description="Observed or forecasted demand value")
    mean_demand: float = Field(100.0, ge=0.0, description="Historical mean demand")
    std_demand: float = Field(15.0, gt=0.0, description="Historical standard deviation of demand")
    anomaly_z_score_threshold: Optional[float] = Field(3.0, gt=0.0, description="Z-score anomaly threshold")


class ItemRiskEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    product_id: str = Field("P1", description="SKU / Product identifier")
    warehouse_id: str = Field("W1", description="Warehouse location identifier")
    region: str = Field("North", description="Geographic region")
    supplier_id: str = Field("SUP1", description="Supplier identifier")
    current_stock: float = Field(50.0, ge=0.0)
    reorder_point: float = Field(100.0, ge=0.0)
    safety_stock: float = Field(30.0, ge=0.0)
    avg_daily_demand: float = Field(10.0, ge=0.0)
    avg_lead_time: float = Field(7.0, gt=0.0)
    lead_time_std_days: float = Field(2.0, ge=0.0)
    supplier_reliability_score: float = Field(0.85, ge=0.0, le=1.0)
    late_delivery_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    unit_cost: float = Field(15.0, ge=0.0)
    sales_velocity: float = Field(10.0, ge=0.0)
    zero_sales_weeks: int = Field(0, ge=0)
    demand_val: Optional[float] = Field(None, ge=0.0)
    mean_demand: Optional[float] = Field(None, ge=0.0)
    std_demand: Optional[float] = Field(None, gt=0.0)


class BatchRiskEvaluationRequest(BaseModel):
    items: List[ItemRiskEvaluationRequest] = Field(..., description="List of items to evaluate")


# --- REST Endpoints ---

@router.post("/supplier-delay", summary="Evaluate Supplier Delay Risk")
def evaluate_supplier_delay(req: SupplierDelayRiskRequest) -> Dict[str, Any]:
    """Evaluates supplier lead-time delay risk score and classification based on delivery metrics."""
    try:
        return evaluate_supplier_delay_details(
            reliability_score=req.reliability_score,
            lead_time_std=req.lead_time_std,
            lead_time_avg=req.lead_time_avg,
            late_delivery_rate=req.late_delivery_rate,
            w_late=req.w_late,
            w_var=req.w_var
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stockout", summary="Evaluate Stockout Risk")
def evaluate_stockout(req: StockoutRiskRequest) -> Dict[str, Any]:
    """Evaluates stockout probability and days of inventory remaining given current stock and demand."""
    try:
        return evaluate_stockout_details(
            current_stock=req.current_stock,
            reorder_point=req.reorder_point,
            safety_stock=req.safety_stock,
            avg_daily_demand=req.avg_daily_demand,
            lead_time_days=req.lead_time_days
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/overstock", summary="Evaluate Overstock Risk")
def evaluate_overstock(req: OverstockRiskRequest) -> Dict[str, Any]:
    """Evaluates overstock exposure score and excess inventory capital tied up above ROP thresholds."""
    try:
        return evaluate_overstock_details(
            current_stock=req.current_stock,
            reorder_point=req.reorder_point,
            unit_cost=req.unit_cost,
            avg_daily_demand=req.avg_daily_demand,
            overstock_rop_multiplier=req.overstock_rop_multiplier
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/inventory-health", summary="Evaluate Inventory Health Risk (Slow-Moving & Dead Stock)")
def evaluate_inventory_health(req: InventoryHealthRiskRequest) -> Dict[str, Any]:
    """Evaluates inventory health risk identifying slow-moving and dead stock capital risks."""
    try:
        return calculate_inventory_health_risk(
            sales_velocity=req.sales_velocity,
            zero_sales_weeks=req.zero_sales_weeks,
            current_stock=req.current_stock,
            unit_cost=req.unit_cost
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/demand-anomaly", summary="Evaluate Demand Volatility & Anomaly Risk")
def evaluate_demand_anomaly(req: DemandAnomalyRiskRequest) -> Dict[str, Any]:
    """Evaluates demand surge and statistical anomaly risks using Z-score thresholding."""
    try:
        return calculate_demand_anomaly_risk(
            demand_val=req.demand_val,
            mean_demand=req.mean_demand,
            std_demand=req.std_demand,
            anomaly_z_score_threshold=req.anomaly_z_score_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/evaluate-item", summary="Evaluate Full 5D Risk for Single Item")
def evaluate_item_risk(req: ItemRiskEvaluationRequest) -> Dict[str, Any]:
    """Evaluates full composite 5D supply chain risk scorecard for a single SKU item."""
    try:
        return risk_engine.evaluate_item_risk(
            product_id=req.product_id,
            warehouse_id=req.warehouse_id,
            region=req.region,
            supplier_id=req.supplier_id,
            current_stock=req.current_stock,
            reorder_point=req.reorder_point,
            safety_stock=req.safety_stock,
            avg_daily_demand=req.avg_daily_demand,
            avg_lead_time=req.avg_lead_time,
            lead_time_std_days=req.lead_time_std_days,
            supplier_reliability_score=req.supplier_reliability_score,
            late_delivery_rate=req.late_delivery_rate,
            unit_cost=req.unit_cost,
            sales_velocity=req.sales_velocity,
            zero_sales_weeks=req.zero_sales_weeks,
            demand_val=req.demand_val,
            mean_demand=req.mean_demand,
            std_demand=req.std_demand
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/evaluate-batch", summary="Evaluate Risk for Batch of Items & Generate Summary")
def evaluate_batch_risk(req: BatchRiskEvaluationRequest) -> Dict[str, Any]:
    """Evaluates composite 5D supply chain risk scorecards across a batch of SKU items."""
    try:
        evaluated_items = []
        for item in req.items:
            res = risk_engine.evaluate_item_risk(
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

        total_items = len(evaluated_items)
        critical_cnt = sum(1 for x in evaluated_items if x["overall_risk_level"] == "CRITICAL")
        high_cnt = sum(1 for x in evaluated_items if x["overall_risk_level"] == "HIGH")
        med_cnt = sum(1 for x in evaluated_items if x["overall_risk_level"] == "MEDIUM")
        low_cnt = sum(1 for x in evaluated_items if x["overall_risk_level"] == "LOW")

        avg_score = float(sum(x["composite_risk_score"] for x in evaluated_items) / max(1, total_items))

        summary = {
            "total_items_evaluated": total_items,
            "overall_mean_risk_score": float(round(avg_score, 4)),
            "risk_level_counts": {
                "CRITICAL": critical_cnt,
                "HIGH": high_cnt,
                "MEDIUM": med_cnt,
                "LOW": low_cnt
            }
        }

        return {
            "summary": summary,
            "items": evaluated_items
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
