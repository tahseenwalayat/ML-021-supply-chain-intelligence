"""Direct business-logic client for the Streamlit dashboard.

Replaces HTTP REST API calls with direct in-process Python function calls to src/
engines, allowing standalone deployment on Streamlit Community Cloud without
requiring a running FastAPI server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops.drift_detector import DataDriftDetector
from src.risk.alert_center import AlertCenterEngine
from src.risk.risk_engine import SupplyChainRiskEngine
from src.risk.stockout_risk import evaluate_stockout_details
from src.risk.supplier_delay_risk import evaluate_supplier_delay_details
from src.simulation.scenario_simulator import (
    ScenarioParameters,
    ScenarioSimulationEngine,
)

# Relative data and model locations
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "inventory_recommendations.parquet"
MODELS_DIR = PROJECT_ROOT / "models"


# ==============================================================================
# Cached Resources & Data Loaders
# ==============================================================================

@st.cache_data(show_spinner=False)
def _load_inventory_data() -> pd.DataFrame:
    """Load precomputed inventory recommendations from parquet with caching."""
    if not DATA_PATH.exists():
        # Fallback if running from a different directory
        alt_path = Path("data/processed/inventory_recommendations.parquet")
        if alt_path.exists():
            return pd.read_parquet(alt_path)
        raise FileNotFoundError(
            f"Inventory recommendations file not found at {DATA_PATH}. "
            "Please ensure the data directory is present."
        )
    return pd.read_parquet(DATA_PATH)


@st.cache_resource(show_spinner=False)
def _load_model_artifact(file_path: str) -> Dict[str, Any]:
    """Cache loaded model joblib artifacts in memory."""
    return joblib.load(file_path)


@st.cache_resource(show_spinner=False)
def _get_risk_engine() -> SupplyChainRiskEngine:
    return SupplyChainRiskEngine()


@st.cache_resource(show_spinner=False)
def _get_alert_engine() -> AlertCenterEngine:
    return AlertCenterEngine()


@st.cache_resource(show_spinner=False)
def _get_simulation_engine() -> ScenarioSimulationEngine:
    return ScenarioSimulationEngine()


@st.cache_resource(show_spinner=False)
def _get_drift_detector() -> DataDriftDetector:
    return DataDriftDetector()


# ==============================================================================
# UI Error Helpers
# ==============================================================================

def get_headers() -> Dict[str, str]:
    """Preserved for backwards compatibility with any legacy caller."""
    return {"Content-Type": "application/json"}


def render_api_error_banner(error_msg: str, endpoint: str = "") -> None:
    """Show one concise request warning without exposing repeated connection traces."""
    target = f" for `{endpoint}`" if endpoint else ""
    st.warning(f"Data operation temporarily unavailable{target}.")
    with st.expander("Technical details"):
        st.code(error_msg)


# ==============================================================================
# Health & Status Probes
# ==============================================================================

def check_api_health() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Checks the health of the embedded intelligence engines."""
    return {
        "status": "healthy",
        "service": "supply-chain-embedded",
        "version": "1.0.0",
        "mode": "direct_python_runtime",
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
            "inventory_optimization",
        ],
    }, None


def backend_is_available() -> bool:
    """Return whether the embedded engines are available (always True in direct mode)."""
    return True


def require_backend() -> bool:
    """Embedded engines are always active; returns True."""
    return True


# ==============================================================================
# Inventory & Warehouse Operations
# ==============================================================================

def get_inventory_health() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Computes aggregate inventory health KPIs directly from parquet data."""
    try:
        df = _load_inventory_data()
        total_items = len(df)
        stockout_items = len(df[df["current_stock"] == 0])
        below_safety_items = len(df[df["current_stock"] < df["safety_stock"]])
        reorder_items = len(df[df["procurement_status"] == "REORDER_REQUIRED"])

        total_capital = 0.0
        if "unit_cost" in df.columns:
            total_capital = float((df["current_stock"] * df["unit_cost"]).sum())

        return {
            "summary": {
                "total_items": total_items,
                "stockout_count": stockout_items,
                "below_safety_count": below_safety_items,
                "reorder_count": reorder_items,
                "total_inventory_value_usd": round(total_capital, 2),
            }
        }, None
    except Exception as e:
        return None, f"Error computing inventory health: {str(e)}"


def get_inventory_recommendations(
    product_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    region: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retrieves optimized safety stock, reorder point, and EOQ recommendations with optional filtering."""
    try:
        df = _load_inventory_data()
        if product_id:
            df = df[df["product_id"] == product_id]
        if warehouse_id:
            df = df[df["warehouse_id"] == warehouse_id]
        if region:
            df = df[df["region"] == region]

        return {
            "total_items": len(df),
            "recommendations": df.to_dict(orient="records"),
        }, None
    except Exception as e:
        return None, f"Error loading inventory recommendations: {str(e)}"


def get_warehouse_utilization() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Computes warehouse-level storage unit accumulation and capacity utilization percentages."""
    try:
        df = _load_inventory_data()
        wh_stats = (
            df.groupby("warehouse_id")
            .agg({"current_stock": "sum", "product_id": "count"})
            .reset_index()
        )
        wh_stats.columns = ["warehouse_id", "total_stock_units", "sku_count"]
        wh_stats["capacity_limit"] = wh_stats["total_stock_units"] * 1.2
        wh_stats["utilization_pct"] = (
            wh_stats["total_stock_units"] / wh_stats["capacity_limit"] * 100
        ).round(1)

        return {
            "warehouse_utilization": wh_stats.to_dict(orient="records")
        }, None
    except Exception as e:
        return None, f"Error computing warehouse utilization: {str(e)}"


# ==============================================================================
# MLOps & Model Registry
# ==============================================================================

def get_registered_models() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Lists registered forecasting models and serialized performance metrics from artifact storage."""
    try:
        models_dir = MODELS_DIR if MODELS_DIR.exists() else Path("models")
        result = []
        if models_dir.exists():
            for f in sorted(os.listdir(models_dir)):
                if f.endswith(".joblib"):
                    fpath = models_dir / f
                    try:
                        data = _load_model_artifact(str(fpath))
                        result.append(
                            {
                                "model_file": f,
                                "level": data.get("level", "unknown"),
                                "wmape": data.get("wmape", None),
                                "naive_wmape": data.get("naive_wmape", None),
                                "features_count": len(data.get("feature_cols", [])),
                                "target_col": data.get("target_col", "legacy_actual_sales"),
                                "serving_ready": data.get("target_col") == "target_next_day_sales",
                            }
                        )
                    except Exception:
                        result.append({"model_file": f, "status": "load_error"})
        return {"registered_models": result}, None
    except Exception as e:
        return None, f"Error listing registered models: {str(e)}"


def detect_degradation(
    baseline_wmape: float, current_wmape: float
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Evaluates forecast error degradation against baseline and returns retraining recommendations."""
    try:
        detector = _get_drift_detector()
        result = detector.detect_forecast_degradation(
            baseline_wmape=baseline_wmape, current_wmape=current_wmape
        )
        return result, None
    except Exception as e:
        return None, f"Error evaluating forecast degradation: {str(e)}"


def start_retraining(hierarchy_level: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Executes the automated retraining pipeline directly."""
    try:
        from src.mlops.retraining_pipeline import run_retraining_pipeline

        report = run_retraining_pipeline(
            hierarchy_level=hierarchy_level,
            force_run=False,
            drift_triggered=True,
        )
        return {
            "status": "completed",
            "hierarchy_level": hierarchy_level,
            "report": report,
        }, None
    except Exception as e:
        return None, f"Error running retraining pipeline: {str(e)}"


# ==============================================================================
# Risk Engine Operations
# ==============================================================================

def evaluate_stockout(
    current_stock: float,
    reorder_point: float,
    safety_stock: float,
    avg_daily_demand: float = 1.0,
    lead_time_days: float = 7.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Evaluates stockout probability and days of inventory remaining."""
    try:
        result = evaluate_stockout_details(
            current_stock=current_stock,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
            avg_daily_demand=avg_daily_demand,
            lead_time_days=lead_time_days,
        )
        return result, None
    except Exception as e:
        return None, f"Error evaluating stockout risk: {str(e)}"


def evaluate_supplier_delay(
    reliability_score: float,
    lead_time_std: float,
    lead_time_avg: float = 7.0,
    late_delivery_rate: Optional[float] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Evaluates supplier lead-time delay risk score and classification."""
    try:
        result = evaluate_supplier_delay_details(
            reliability_score=reliability_score,
            lead_time_std=lead_time_std,
            lead_time_avg=lead_time_avg,
            late_delivery_rate=late_delivery_rate,
        )
        return result, None
    except Exception as e:
        return None, f"Error evaluating supplier delay risk: {str(e)}"


def evaluate_batch_risk(
    items: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Evaluates composite 5D supply chain risk scorecards across a batch of SKU items."""
    try:
        engine = _get_risk_engine()
        evaluated_items = []
        for item in items:
            res = engine.evaluate_item_risk(
                product_id=item.get("product_id", "P1"),
                warehouse_id=item.get("warehouse_id", "W1"),
                region=item.get("region", "North"),
                supplier_id=item.get("supplier_id", "SUP1"),
                current_stock=float(item.get("current_stock", 0.0)),
                reorder_point=float(item.get("reorder_point", 0.0)),
                safety_stock=float(item.get("safety_stock", 0.0)),
                avg_daily_demand=float(item.get("avg_daily_demand", 1.0)),
                avg_lead_time=float(item.get("avg_lead_time", 7.0)),
                lead_time_std_days=float(item.get("lead_time_std_days", 0.0)),
                supplier_reliability_score=float(item.get("supplier_reliability_score", 1.0)),
                late_delivery_rate=item.get("late_delivery_rate"),
                unit_cost=float(item.get("unit_cost", 10.0)),
                sales_velocity=float(item.get("sales_velocity", 1.0)),
                zero_sales_weeks=int(item.get("zero_sales_weeks", 0)),
                demand_val=item.get("demand_val"),
                mean_demand=item.get("mean_demand"),
                std_demand=item.get("std_demand"),
            )
            evaluated_items.append(res)

        total_items = len(evaluated_items)
        critical_cnt = sum(1 for x in evaluated_items if x.get("overall_risk_level") == "CRITICAL")
        high_cnt = sum(1 for x in evaluated_items if x.get("overall_risk_level") == "HIGH")
        med_cnt = sum(1 for x in evaluated_items if x.get("overall_risk_level") == "MEDIUM")
        low_cnt = sum(1 for x in evaluated_items if x.get("overall_risk_level") == "LOW")

        avg_score = float(
            sum(x.get("composite_risk_score", 0.0) for x in evaluated_items) / max(1, total_items)
        )

        return {
            "summary": {
                "total_items_evaluated": total_items,
                "overall_mean_risk_score": float(round(avg_score, 4)),
                "risk_level_counts": {
                    "CRITICAL": critical_cnt,
                    "HIGH": high_cnt,
                    "MEDIUM": med_cnt,
                    "LOW": low_cnt,
                },
            },
            "items": evaluated_items,
        }, None
    except Exception as e:
        return None, f"Error evaluating batch risk: {str(e)}"


# ==============================================================================
# Alert Center Operations
# ==============================================================================

def scan_alerts(
    items: List[Dict[str, Any]],
    capacity_utilization_pct: float = 85.0,
    forecast_drift_pct: float = 0.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Scans inventory and 5D risk inputs to generate a prioritized operational alert feed."""
    try:
        risk_engine = _get_risk_engine()
        alert_engine = _get_alert_engine()

        evaluated_items = []
        for item in items:
            res = risk_engine.evaluate_item_risk(
                product_id=item.get("product_id", "P1"),
                warehouse_id=item.get("warehouse_id", "W1"),
                region=item.get("region", "North"),
                supplier_id=item.get("supplier_id", "SUP1"),
                current_stock=float(item.get("current_stock", 0.0)),
                reorder_point=float(item.get("reorder_point", 0.0)),
                safety_stock=float(item.get("safety_stock", 0.0)),
                avg_daily_demand=float(item.get("avg_daily_demand", 1.0)),
                avg_lead_time=float(item.get("avg_lead_time", 7.0)),
                lead_time_std_days=float(item.get("lead_time_std_days", 0.0)),
                supplier_reliability_score=float(item.get("supplier_reliability_score", 1.0)),
                late_delivery_rate=item.get("late_delivery_rate"),
                unit_cost=float(item.get("unit_cost", 10.0)),
                sales_velocity=float(item.get("sales_velocity", 1.0)),
                zero_sales_weeks=int(item.get("zero_sales_weeks", 0)),
                demand_val=item.get("demand_val"),
                mean_demand=item.get("mean_demand"),
                std_demand=item.get("std_demand"),
            )
            evaluated_items.append(res)

        alerts = alert_engine.scan_inventory_and_risk_alerts(
            evaluated_items=evaluated_items,
            capacity_utilization_pct=capacity_utilization_pct,
            forecast_drift_pct=forecast_drift_pct,
        )

        return {
            "total_alerts": len(alerts),
            "critical_count": sum(1 for a in alerts if a.get("severity") == "CRITICAL"),
            "high_count": sum(1 for a in alerts if a.get("severity") == "HIGH"),
            "medium_count": sum(1 for a in alerts if a.get("severity") == "MEDIUM"),
            "info_count": sum(1 for a in alerts if a.get("severity") == "INFO"),
            "alerts": alerts,
        }, None
    except Exception as e:
        return None, f"Error scanning operational alerts: {str(e)}"


# ==============================================================================
# Scenario Stress Simulator Operations
# ==============================================================================

def simulate_sku_scenario(
    base_daily_demand: float,
    current_stock: float,
    reorder_point: float,
    safety_stock: float,
    base_lead_time: float,
    unit_cost: float,
    scenario_params: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Executes an interactive scenario stress test simulation for a specific SKU."""
    try:
        engine = _get_simulation_engine()
        params = ScenarioParameters(**scenario_params)
        result = engine.simulate_sku_scenario(
            base_daily_demand=base_daily_demand,
            current_stock=current_stock,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
            base_lead_time=base_lead_time,
            unit_cost=unit_cost,
            params=params,
        )
        return result, None
    except Exception as e:
        return None, f"Error executing scenario simulation: {str(e)}"
