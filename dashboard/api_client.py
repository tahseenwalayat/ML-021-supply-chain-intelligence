import os
import requests
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional

# Read environment variables with sensible defaults
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}").rstrip("/")
API_KEY = os.getenv("API_KEY", "sc-key-secret-2026")
REQUEST_TIMEOUT = 10  # seconds


def get_headers() -> Dict[str, str]:
    """Returns headers with X-API-Key for FastAPI authentication."""
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }


def render_api_error_banner(error_msg: str, endpoint: str = ""):
    """Renders a prominent, user-friendly error card when the API is unreachable or returns error."""
    st.error(f"⚠️ **Backend API Error / Connection Issue**")
    st.markdown(f"""
    <div style="background-color: #3b1418; border: 1px solid #7f1d1d; border-radius: 8px; padding: 14px; margin-bottom: 15px;">
        <h4 style="color: #fca5a5; margin-top: 0;">Failed to fetch data from backend</h4>
        <p style="color: #fecaca; margin-bottom: 8px;"><strong>Target Endpoint:</strong> <code>{endpoint}</code></p>
        <p style="color: #fecaca; margin-bottom: 8px;"><strong>Error Details:</strong> {error_msg}</p>
        <hr style="border-color: #991b1b; margin: 10px 0;">
        <p style="color: #cbd5e1; font-size: 0.85rem; margin-bottom: 0;">
            💡 <strong>Troubleshooting Steps:</strong><br>
            1. Ensure the FastAPI backend server is running: <code>uvicorn api.main:app --port 8000</code><br>
            2. Verify the base URL environment variable: <code>API_BASE_URL={API_BASE_URL}</code><br>
            3. Check that the <code>API_KEY</code> environment variable matches the backend secret.
        </p>
    </div>
    """, unsafe_allow_html=True)


def check_api_health() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Checks the health of the FastAPI service."""
    url = f"{API_BASE_URL}/health"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"Health check returned HTTP {resp.status_code}"
    except requests.exceptions.RequestException as e:
        return None, str(e)


def get_inventory_health() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """GET /api/v1/inventory/health"""
    url = f"{API_BASE_URL}/api/v1/inventory/health"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error calling {url}: {str(e)}"


def get_inventory_recommendations(
    product_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
    region: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """GET /api/v1/inventory/recommendation"""
    url = f"{API_BASE_URL}/api/v1/inventory/recommendation"
    params = {}
    if product_id:
        params["product_id"] = product_id
    if warehouse_id:
        params["warehouse_id"] = warehouse_id
    if region:
        params["region"] = region

    try:
        resp = requests.get(url, headers=get_headers(), params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error calling {url}: {str(e)}"


def get_warehouse_utilization() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """GET /api/v1/inventory/utilization"""
    url = f"{API_BASE_URL}/api/v1/inventory/utilization"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error calling {url}: {str(e)}"


def get_registered_models() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """GET /api/v1/mlops/models"""
    url = f"{API_BASE_URL}/api/v1/mlops/models"
    try:
        resp = requests.get(url, headers=get_headers(), timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error calling {url}: {str(e)}"


def evaluate_batch_risk(items: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """POST /api/v1/risk/evaluate-batch"""
    url = f"{API_BASE_URL}/api/v1/risk/evaluate-batch"
    payload = {"items": items}
    try:
        resp = requests.post(url, headers=get_headers(), json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error calling {url}: {str(e)}"


def scan_alerts(
    items: List[Dict[str, Any]],
    capacity_utilization_pct: float = 85.0,
    forecast_drift_pct: float = 0.0
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """POST /api/v1/alerts/scan"""
    url = f"{API_BASE_URL}/api/v1/alerts/scan"
    payload = {
        "items_evaluation": {"items": items},
        "capacity_utilization_pct": capacity_utilization_pct,
        "forecast_drift_pct": forecast_drift_pct
    }
    try:
        resp = requests.post(url, headers=get_headers(), json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error calling {url}: {str(e)}"


def simulate_sku_scenario(
    base_daily_demand: float,
    current_stock: float,
    reorder_point: float,
    safety_stock: float,
    base_lead_time: float,
    unit_cost: float,
    scenario_params: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """POST /api/v1/simulation/simulate-sku"""
    url = f"{API_BASE_URL}/api/v1/simulation/simulate-sku"
    payload = {
        "base_daily_demand": base_daily_demand,
        "current_stock": current_stock,
        "reorder_point": reorder_point,
        "safety_stock": safety_stock,
        "base_lead_time": base_lead_time,
        "unit_cost": unit_cost,
        "scenario_parameters": scenario_params
    }
    try:
        resp = requests.post(url, headers=get_headers(), json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error calling {url}: {str(e)}"


def detect_degradation(baseline_wmape: float, current_wmape: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """POST /api/v1/mlops/detect-degradation"""
    url = f"{API_BASE_URL}/api/v1/mlops/detect-degradation"
    payload = {
        "baseline_wmape": baseline_wmape,
        "current_wmape": current_wmape
    }
    try:
        resp = requests.post(url, headers=get_headers(), json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error calling {url}: {str(e)}"


def evaluate_stockout(
    current_stock: float,
    reorder_point: float,
    safety_stock: float,
    avg_daily_demand: float = 1.0,
    lead_time_days: float = 7.0
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """POST /api/v1/risk/stockout"""
    url = f"{API_BASE_URL}/api/v1/risk/stockout"
    payload = {
        "current_stock": current_stock,
        "reorder_point": reorder_point,
        "safety_stock": safety_stock,
        "avg_daily_demand": avg_daily_demand,
        "lead_time_days": lead_time_days
    }
    try:
        resp = requests.post(url, headers=get_headers(), json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error calling {url}: {str(e)}"


def evaluate_supplier_delay(
    reliability_score: float,
    lead_time_std: float,
    lead_time_avg: float = 7.0,
    late_delivery_rate: Optional[float] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """POST /api/v1/risk/supplier-delay"""
    url = f"{API_BASE_URL}/api/v1/risk/supplier-delay"
    payload = {
        "reliability_score": reliability_score,
        "lead_time_std": lead_time_std,
        "lead_time_avg": lead_time_avg,
        "late_delivery_rate": late_delivery_rate
    }
    try:
        resp = requests.post(url, headers=get_headers(), json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}: {resp.text}"
    except requests.exceptions.RequestException as e:
        return None, f"Connection error calling {url}: {str(e)}"
