import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from src.risk.stockout_risk import calculate_stockout_risk, compute_stockout_risk_df
from src.risk.overstock_risk import calculate_overstock_risk, compute_overstock_risk_df
from src.risk.inventory_health_risk import calculate_inventory_health_risk, compute_inventory_health_risk_df
from src.risk.demand_anomaly_risk import calculate_demand_anomaly_risk, compute_demand_anomaly_risk_df
from src.risk.risk_engine import SupplyChainRiskEngine, evaluate_full_supply_chain_risk


client = TestClient(app)


def test_stockout_risk_critical():
    """Stockout risk must equal 1.0 when inventory is zero."""
    risk = calculate_stockout_risk(
        current_stock=0.0,
        reorder_point=50.0,
        safety_stock=20.0,
        avg_daily_demand=5.0
    )
    assert risk == 1.0

    # Safety stock breached
    risk_ss = calculate_stockout_risk(
        current_stock=10.0,
        reorder_point=50.0,
        safety_stock=20.0,
        avg_daily_demand=5.0
    )
    assert risk_ss >= 0.75


def test_overstock_risk_excess_capital():
    """Overstock risk must flag excess inventory and compute tied-up capital."""
    risk_score, excess_units, capital, _ = calculate_overstock_risk(
        current_stock=500.0,
        reorder_point=100.0,
        unit_cost=20.0,
        overstock_rop_multiplier=3.0
    )
    # Threshold = 3.0 * 100.0 = 300.0
    # Excess units = 500 - 300 = 200
    # Capital = 200 * 20.0 = 4000.0
    assert excess_units == 200.0
    assert capital == 4000.0
    assert risk_score > 0.25


def test_inventory_health_dead_stock():
    """12+ weeks of zero sales must categorize inventory as DEAD_STOCK."""
    res = calculate_inventory_health_risk(
        sales_velocity=0.0,
        zero_sales_weeks=14,
        current_stock=100.0,
        unit_cost=10.0
    )
    assert res["inventory_health_status"] == "DEAD_STOCK"
    assert res["inventory_health_risk_score"] >= 0.75
    assert res["is_dead_stock"] is True


def test_demand_anomaly_spike():
    """Demand values with Z-score >= 3.0 must trigger DEMAND_SPIKE anomaly."""
    res = calculate_demand_anomaly_risk(
        demand_val=250.0,
        mean_demand=100.0,
        std_demand=20.0,
        anomaly_z_score_threshold=3.0
    )
    # Z-score = (250 - 100) / 20 = 7.5
    assert res["is_anomaly"] is True
    assert res["anomaly_type"] == "DEMAND_SPIKE"
    assert res["demand_anomaly_risk_score"] >= 0.50


def test_supply_chain_risk_engine_item_evaluation():
    """Full Supply Chain Risk Engine must calculate composite score and recommendations."""
    engine = SupplyChainRiskEngine()
    result = engine.evaluate_item_risk(
        product_id="SKU_1001",
        warehouse_id="WH_NORTH",
        current_stock=5.0,  # Below safety stock
        reorder_point=50.0,
        safety_stock=20.0,
        avg_daily_demand=10.0,
        avg_lead_time=7.0,
        supplier_reliability_score=0.60
    )

    assert result["product_id"] == "SKU_1001"
    assert 0.0 <= result["composite_risk_score"] <= 1.0
    assert result["overall_risk_level"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert len(result["recommendations"]) > 0


def test_evaluate_full_supply_chain_risk_df():
    """Batch evaluation wrapper must compute scores across DataFrame and generate summary."""
    df = pd.DataFrame([
        {
            "product_id": "P1", "current_stock": 0.0, "reorder_point": 50.0,
            "safety_stock": 20.0, "avg_daily_demand": 5.0, "avg_lead_time": 7.0,
            "supplier_reliability_score": 0.5, "unit_cost": 10.0
        },
        {
            "product_id": "P2", "current_stock": 200.0, "reorder_point": 50.0,
            "safety_stock": 20.0, "avg_daily_demand": 5.0, "avg_lead_time": 7.0,
            "supplier_reliability_score": 0.95, "unit_cost": 10.0
        }
    ])

    df_eval, summary = evaluate_full_supply_chain_risk(df)

    assert "composite_risk_score" in df_eval.columns
    assert "overall_risk_level" in df_eval.columns
    assert summary["total_items_evaluated"] == 2
    assert "risk_level_counts" in summary


# --- API Endpoint Integration Tests ---

def test_api_supplier_delay_endpoint():
    """POST /api/v1/risk/supplier-delay endpoint integration test."""
    response = client.post(
        "/api/v1/risk/supplier-delay",
        json={
            "reliability_score": 0.7,
            "lead_time_std": 3.0,
            "lead_time_avg": 7.0
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "supplier_delay_risk_score" in data
    assert "supplier_delay_risk_level" in data
    assert "recommended_buffer_days" in data


def test_api_evaluate_batch_endpoint():
    """POST /api/v1/risk/evaluate-batch endpoint integration test."""
    response = client.post(
        "/api/v1/risk/evaluate-batch",
        json={
            "items": [
                {
                    "product_id": "P1",
                    "warehouse_id": "W1",
                    "current_stock": 10.0,
                    "reorder_point": 50.0,
                    "safety_stock": 20.0,
                    "avg_daily_demand": 5.0
                }
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "items" in data
    assert len(data["items"]) == 1
