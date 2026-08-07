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

# Custom Dark Glassmorphism CSS Theme
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
    .hero-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    .module-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
        transition: transform 0.2s ease-in-out;
    }
    .module-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 6px;
    }
    .module-desc {
        font-size: 0.88rem;
        color: #94a3b8;
    }
    .engine-chip {
        display: inline-block;
        background-color: #1e293b;
        color: #e2e8f0;
        border: 1px solid #475569;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 4px;
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
### 🧭 Executive Navigation
Select a module from the **left sidebar menu**:
1. 📊 **KPI Overview**
2. 🔮 **Forecast Accuracy**
3. 📦 **Inventory Health**
4. 🏭 **Warehouse Utilization**
5. 📈 **Fill Rate & Stockout**
6. 🛒 **Procurement Recommendations**
7. 🚨 **Alert Center (6 Alert Types)**
""")

# Main Landing Page Hero Banner
st.title("📦 Enterprise Supply Chain Executive Dashboard")
st.markdown("### AI-Powered Demand Forecasting, Inventory Optimization & 5D Risk Intelligence")

st.markdown("""
<div class="hero-card">
    <h4 style="color:#f8fafc; margin-top:0;">🌟 Welcome to the Executive Control Hub</h4>
    <p style="color:#cbd5e1; font-size:0.95rem; margin-bottom:0;">
        This platform delivers multi-horizon AI demand forecasting (LightGBM, XGBoost, Prophet), mathematical safety stock & reorder point optimization, 
        5D supply chain risk evaluation, and interactive scenario stress testing. Built strictly adhering to <b>API-First Architecture</b> standards.
    </p>
</div>
""", unsafe_allow_html=True)

# Health Status Summary Cards
if health_err:
    api_client.render_api_error_banner(health_err, "/health")
else:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("System Health", "🟢 Operational", delta="REST API Connected")
    with col2:
        st.metric("Backend Version", f"v{health_data.get('version', '1.0.0')}", delta="FastAPI Framework")
    with col3:
        modules_count = len(health_data.get("modules", []))
        st.metric("Active Decision Engines", f"{modules_count} Engines", delta="100% Online")
    with col4:
        st.metric("Architecture Mode", "API-First REST", delta="High Availability")

    st.markdown("---")
    st.subheader("⚡ Active System Engine Modules")
    
    # Render modules as clean badges instead of raw JSON
    modules = health_data.get("modules", [])
    chips_html = "".join([f'<span class="engine-chip">⚡ {m.replace("_", " ").title()}</span>' for m in modules])
    st.markdown(f'<div style="margin-bottom: 20px;">{chips_html}</div>', unsafe_allow_html=True)

st.subheader("🗺️ Platform Module Overview")
m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown("""
    <div class="module-card">
        <div class="module-title">📊 1. KPI Overview Scorecard</div>
        <div class="module-desc">Executive summary of total inventory valuation, stockout risk counts, and high-risk capital exposure.</div>
    </div>
    <div class="module-card">
        <div class="module-title">🔮 2. Multi-Horizon Forecast Accuracy</div>
        <div class="module-desc">Benchmark forecast accuracy across SKU-Region, Category-Region, and Region-Total hierarchy levels.</div>
    </div>
    <div class="module-card">
        <div class="module-title">📦 3. Inventory Optimization & Health</div>
        <div class="module-desc">Mathematical Safety Stock ($SS$), Reorder Point ($ROP$), and Economic Order Quantity ($EOQ$) recommendations.</div>
    </div>
    <div class="module-card">
        <div class="module-title">🏭 4. Warehouse Utilization</div>
        <div class="module-desc">Storage unit accumulation and capacity utilization percentage indicators per regional facility.</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown("""
    <div class="module-card">
        <div class="module-title">📈 5. Fill Rate & Stockout Exposure</div>
        <div class="module-desc">Service level probability distribution and fill rate curves across all distribution nodes.</div>
    </div>
    <div class="module-card">
        <div class="module-title">🛒 6. Procurement & Reorder Triggers</div>
        <div class="module-desc">Automated purchase order recommendations and EOQ calculations to prevent stockouts.</div>
    </div>
    <div class="module-card">
        <div class="module-title">🚨 7. Operational Alert Center</div>
        <div class="module-desc">Prioritized alert feed covering Low Inventory, Overstock, Demand Spikes, Supplier Delays, Capacity, and Forecast Drift.</div>
    </div>
    """, unsafe_allow_html=True)

st.info("👈 **Select any module page from the left sidebar to begin exploring.**")
