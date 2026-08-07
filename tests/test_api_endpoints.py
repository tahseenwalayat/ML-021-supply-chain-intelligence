import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "scenario_simulation" in data["modules"]


def test_api_supplier_delay():
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
    assert "recommended_buffer_days" in data


def test_api_stockout():
    response = client.post(
        "/api/v1/risk/stockout",
        json={
            "current_stock": 0.0,
            "reorder_point": 50.0,
            "safety_stock": 20.0,
            "avg_daily_demand": 5.0,
            "lead_time_days": 7.0
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["stockout_risk_score"] == 1.0


def test_api_overstock():
    response = client.post(
        "/api/v1/risk/overstock",
        json={
            "current_stock": 500.0,
            "reorder_point": 100.0,
            "unit_cost": 20.0,
            "avg_daily_demand": 2.0
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_overstocked"] is True


def test_api_simulation_sku():
    response = client.post(
        "/api/v1/simulation/simulate-sku",
        json={
            "base_daily_demand": 10.0,
            "current_stock": 100.0,
            "reorder_point": 50.0,
            "safety_stock": 20.0,
            "base_lead_time": 7.0,
            "unit_cost": 15.0,
            "scenario_parameters": {
                "scenario_name": "Supplier Delay Shock",
                "supplier_delay_days": 5.0,
                "price_change_pct": 10.0,
                "demand_surge_multiplier": 1.2,
                "simulation_horizon_days": 14
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "summary_metrics" in data
    assert len(data["daily_trajectories"]["scenario"]) == 14


def test_api_mlops_registered_models():
    response = client.get("/api/v1/mlops/models")
    assert response.status_code == 200
    data = response.json()
    assert "registered_models" in data


def test_api_alerts_scan():
    response = client.post(
        "/api/v1/alerts/scan",
        json={
            "items_evaluation": {
                "items": [
                    {
                        "product_id": "P1",
                        "warehouse_id": "W1",
                        "current_stock": 0.0,
                        "reorder_point": 50.0,
                        "safety_stock": 20.0,
                        "avg_daily_demand": 5.0,
                        "avg_lead_time": 7.0,
                        "supplier_reliability_score": 0.5,
                        "unit_cost": 10.0
                    }
                ]
            },
            "capacity_utilization_pct": 92.0,
            "forecast_drift_pct": 25.0
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_alerts"] >= 2
    assert data["critical_count"] >= 1
