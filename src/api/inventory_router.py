import os
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from src.api.auth import verify_api_key

router = APIRouter(
    prefix="/api/v1/inventory",
    tags=["Inventory Optimization Engine"],
    dependencies=[Depends(verify_api_key)]
)

DATA_PATH = "data/processed/inventory_recommendations.parquet"

def load_inventory_data() -> pd.DataFrame:
    """Loads precomputed inventory optimization recommendations from parquet storage."""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Inventory recommendations data not found.")
    return pd.read_parquet(DATA_PATH)

@router.get("/recommendation", summary="Get Optimized Inventory Recommendations")
def get_recommendations(
    product_id: Optional[str] = Query(None),
    warehouse_id: Optional[str] = Query(None),
    region: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Retrieves optimized safety stock, reorder point, and EOQ recommendations with optional filtering."""
    try:
        df = load_inventory_data()
        
        if product_id:
            df = df[df["product_id"] == product_id]
        if warehouse_id:
            df = df[df["warehouse_id"] == warehouse_id]
        if region:
            df = df[df["region"] == region]
            
        return {
            "total_items": len(df),
            "recommendations": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", summary="Get Inventory Health Aggregate Metrics")
def get_inventory_health() -> Dict[str, Any]:
    """Computes aggregate inventory health KPIs including stockouts, reorders, and tied-up capital."""
    try:
        df = load_inventory_data()
        
        total_items = len(df)
        stockout_items = len(df[df["current_stock"] == 0])
        below_safety_items = len(df[df["current_stock"] < df["safety_stock"]])
        reorder_items = len(df[df["procurement_status"] == "REORDER_REQUIRED"])
        
        # Calculate total capital tied up (assuming unit_cost exists)
        total_capital = 0.0
        if "unit_cost" in df.columns:
            total_capital = float((df["current_stock"] * df["unit_cost"]).sum())
            
        return {
            "summary": {
                "total_items": total_items,
                "stockout_count": stockout_items,
                "below_safety_count": below_safety_items,
                "reorder_count": reorder_items,
                "total_inventory_value_usd": round(total_capital, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/utilization", summary="Get Warehouse Capacity Utilization Metrics")
def get_warehouse_utilization() -> Dict[str, Any]:
    """Computes warehouse-level storage unit accumulation and capacity utilization percentages."""
    try:
        df = load_inventory_data()
        
        # Aggregating by warehouse
        wh_stats = df.groupby("warehouse_id").agg({
            "current_stock": "sum",
            "product_id": "count"
        }).reset_index()
        wh_stats.columns = ["warehouse_id", "total_stock_units", "sku_count"]
        
        # Simulating capacity limit if not available (prototype logic)
        # In a real system, this would come from a warehouse_dim table
        wh_stats["capacity_limit"] = wh_stats["total_stock_units"] * 1.2
        wh_stats["utilization_pct"] = (wh_stats["total_stock_units"] / wh_stats["capacity_limit"] * 100).round(1)
        
        return {
            "warehouse_utilization": wh_stats.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
