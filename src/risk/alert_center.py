import datetime
import pandas as pd
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.utils.logging_config import get_logger

logger = get_logger("risk.alert_center")


class AlertItem(BaseModel):
    alert_id: str
    timestamp: str
    severity: str  # CRITICAL, HIGH, MEDIUM, INFO
    category: str  # LOW_INVENTORY, OVERSTOCK, DEMAND_SPIKE, SUPPLIER_DELAY, WAREHOUSE_CAPACITY, FORECAST_DRIFT
    product_id: Optional[str] = None
    warehouse_id: Optional[str] = None
    title: str
    description: str
    recommended_action: str


class AlertCenterEngine:
    """
    Automated Alert Monitoring & Event Center Engine.
    Scans inventory metrics, risk engine evaluation outputs, supplier performance,
    warehouse capacity utilization, and forecast drift metrics to generate prioritized operational alerts.
    """

    def scan_inventory_and_risk_alerts(
        self,
        evaluated_items: List[Dict[str, Any]],
        capacity_utilization_pct: float = 85.0,
        forecast_drift_pct: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Scans batch-evaluated inventory items and system metrics to generate real-time prioritized alerts.
        """
        alerts = []
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        counter = 1

        # 1. Warehouse Capacity Alert
        if capacity_utilization_pct >= 90.0:
            alerts.append({
                "alert_id": f"ALT-{counter:04d}",
                "timestamp": now_str,
                "severity": "CRITICAL" if capacity_utilization_pct >= 95.0 else "HIGH",
                "category": "WAREHOUSE_CAPACITY",
                "product_id": None,
                "warehouse_id": "ALL",
                "title": f"Warehouse Capacity Breach ({capacity_utilization_pct:.1f}%)",
                "description": f"Overall warehouse utilization has reached {capacity_utilization_pct:.1f}%, exceeding 90% threshold.",
                "recommended_action": "Initiate inter-warehouse inventory transfers or clear slow-moving dead stock immediately."
            })
            counter += 1

        # 2. Forecast Drift Alert
        if forecast_drift_pct >= 20.0:
            alerts.append({
                "alert_id": f"ALT-{counter:04d}",
                "timestamp": now_str,
                "severity": "HIGH",
                "category": "FORECAST_DRIFT",
                "product_id": None,
                "warehouse_id": None,
                "title": f"Demand Forecast Model Drift ({forecast_drift_pct:.1f}% WMAPE degradation)",
                "description": f"Forecast WMAPE error has degraded by {forecast_drift_pct:.1f}% relative to baseline.",
                "recommended_action": "Trigger automated model retraining using the latest 30-day feature store snapshot."
            })
            counter += 1

        # 3. Item-level risk scanning
        for item in evaluated_items:
            product_id = item.get("product_id", "UNKNOWN")
            wh_id = item.get("warehouse_id", "ALL")

            # Low Inventory / Stockout Risk
            if item.get("current_stock", 0) == 0:
                alerts.append({
                    "alert_id": f"ALT-{counter:04d}",
                    "timestamp": now_str,
                    "severity": "CRITICAL",
                    "category": "LOW_INVENTORY",
                    "product_id": product_id,
                    "warehouse_id": wh_id,
                    "title": f"Zero Stock Outage for {product_id}",
                    "description": f"Item {product_id} at {wh_id} is completely out of stock (Stock = 0).",
                    "recommended_action": "Place emergency expediting order or fulfill via nearby warehouse transfer."
                })
                counter += 1
            elif item.get("current_stock", 0) < item.get("safety_stock", 0):
                alerts.append({
                    "alert_id": f"ALT-{counter:04d}",
                    "timestamp": now_str,
                    "severity": "HIGH",
                    "category": "LOW_INVENTORY",
                    "product_id": product_id,
                    "warehouse_id": wh_id,
                    "title": f"Safety Stock Breach for {product_id}",
                    "description": f"Current stock ({item['current_stock']}) fell below safety stock ({item['safety_stock']}).",
                    "recommended_action": "Issue purchase order up to target reorder level."
                })
                counter += 1

            # Overstock Alert
            sub_overstock = item.get("risk_components", {}).get("overstock_risk", {})
            if sub_overstock.get("is_overstocked", False):
                excess = sub_overstock.get("excess_units", 0)
                capital = sub_overstock.get("capital_tied_up", 0)
                alerts.append({
                    "alert_id": f"ALT-{counter:04d}",
                    "timestamp": now_str,
                    "severity": "MEDIUM",
                    "category": "OVERSTOCK",
                    "product_id": product_id,
                    "warehouse_id": wh_id,
                    "title": f"Excess Inventory & Tied-Up Capital ({product_id})",
                    "description": f"Excess inventory of {excess} units costing ${capital:,.2f} in working capital.",
                    "recommended_action": "Halt automatic reorders and launch promotional clearance or reallocation."
                })
                counter += 1

            # Supplier Delay Alert
            sub_supplier = item.get("risk_components", {}).get("supplier_delay_risk", {})
            if sub_supplier.get("supplier_delay_risk_level") in ["CRITICAL", "HIGH"]:
                alerts.append({
                    "alert_id": f"ALT-{counter:04d}",
                    "timestamp": now_str,
                    "severity": sub_supplier.get("supplier_delay_risk_level"),
                    "category": "SUPPLIER_DELAY",
                    "product_id": product_id,
                    "warehouse_id": wh_id,
                    "title": f"High Supplier Delay Exposure ({product_id})",
                    "description": f"Supplier delivery reliability is degraded (Score: {sub_supplier.get('supplier_delay_risk_score', 0):.2f}).",
                    "recommended_action": f"Add {sub_supplier.get('recommended_buffer_days', 3):.1f} buffer days to lead time or split purchase orders."
                })
                counter += 1

            # Demand Spike Alert
            sub_anomaly = item.get("risk_components", {}).get("demand_anomaly_risk", {})
            if sub_anomaly.get("is_anomaly", False):
                alerts.append({
                    "alert_id": f"ALT-{counter:04d}",
                    "timestamp": now_str,
                    "severity": "HIGH",
                    "category": "DEMAND_SPIKE",
                    "product_id": product_id,
                    "warehouse_id": wh_id,
                    "title": f"Unusual Demand Surge Anomaly ({product_id})",
                    "description": f"Observed demand spike Z-score is {sub_anomaly.get('z_score', 0):.2f}.",
                    "recommended_action": "Verify if demand is promotional or permanent trend and adjust short-term forecast."
                })
                counter += 1

        logger.info(f"Generated {len(alerts)} prioritized operational alerts.")
        return alerts
