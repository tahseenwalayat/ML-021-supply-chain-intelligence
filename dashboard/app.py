import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import api_client

st.set_page_config(
    page_title="Supply Chain Executive Dashboard & Alert Center",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; font-family: 'Inter', sans-serif; }
    .status-badge-healthy {
        background-color: #064e3b; color: #6ee7b7; border: 1px solid #059669;
        padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;
    }
    .status-badge-error {
        background-color: #451a03; color: #fca5a5; border: 1px solid #dc2626;
        padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Header & API Health Check
st.sidebar.image("https://img.icons8.com/isometric/100/supplier.png", width=64)
st.sidebar.title("Supply Chain Platform")
st.sidebar.caption("Enterprise Demand Forecasting & Risk Engine")

# Live Health Check from API
health_data, health_err = api_client.check_api_health()

if health_data and health_data.get("status") == "healthy":
    st.sidebar.markdown('<span class="status-badge-healthy">🟢 REST API Online</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-badge-error">🔴 REST API Offline</span>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Executive Navigation
Use the **sidebar page selector** or the navigation menu above to access all 7 core modules:
1. 📊 **KPI Overview**
2. 🔮 **Forecast Accuracy**
3. 📦 **Inventory Health**
4. 🏭 **Warehouse Utilization**
5. 📈 **Fill Rate & Stockout**
6. 🛒 **Procurement Recommendations**
7. 🚨 **Alert Center (6 Alert Types)**
""")

st.title("📦 Enterprise Supply Chain Executive Dashboard & Alert Center")
st.markdown("""
Welcome to the **Supply Chain Intelligence Platform**. This dashboard is built strictly according to **API-First Architecture** standards — fetching 100% of data dynamically via REST calls to the FastAPI backend.
""")

if health_err:
    api_client.render_api_error_banner(health_err, "/health")
else:
    st.success("✅ **FastAPI Backend Connected Successfully** (`http://localhost:8000`)")
    st.json(health_data)

st.info("👈 **Select a page from the sidebar to explore the platform modules.**")
