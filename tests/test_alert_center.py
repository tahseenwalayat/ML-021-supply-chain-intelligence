import pytest
from src.risk.alert_center import AlertCenterEngine


def test_alert_center_generation():
    engine = AlertCenterEngine()
    evaluated_items = [
        {
            "product_id": "P100",
            "warehouse_id": "W1",
            "current_stock": 0.0,
            "safety_stock": 20.0,
            "risk_components": {
                "overstock_risk": {"is_overstocked": False},
                "supplier_delay_risk": {"supplier_delay_risk_level": "HIGH", "supplier_delay_risk_score": 0.8, "recommended_buffer_days": 4},
                "demand_anomaly_risk": {"is_anomaly": True, "z_score": 3.5}
            }
        }
    ]

    alerts = engine.scan_inventory_and_risk_alerts(
        evaluated_items=evaluated_items,
        capacity_utilization_pct=92.0,
        forecast_drift_pct=25.0
    )

    assert len(alerts) >= 4  # Warehouse Capacity + Forecast Drift + Zero Stock + Supplier Delay + Demand Spike
    severities = [a["severity"] for a in alerts]
    assert "CRITICAL" in severities
    assert "HIGH" in severities
