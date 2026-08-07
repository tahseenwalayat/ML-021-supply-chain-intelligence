import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import api_client

st.set_page_config(
    page_title="Inventory Health & Working Capital",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-title { color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; }
    .metric-value { color: #f8fafc; font-size: 1.8rem; font-weight: 700; margin-top: 4px; }
    .metric-delta { font-size: 0.85rem; font-weight: 600; margin-top: 4px; }
    .delta-green { color: #10b981; }
    .delta-amber { color: #f59e0b; }
    .delta-red { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

st.title("📦 Inventory Health & Capital Optimization")
st.caption("Monitoring stockout exposure, safety stock breaches, dead stock risks, and tied-up working capital")

# Fetch inventory health & recommendations from API
health_res, error_health = api_client.get_inventory_health()
rec_res, error_rec = api_client.get_inventory_recommendations()

if error_health or error_rec:
    api_client.render_api_error_banner(
        f"Health error: {error_health or 'None'} | Rec error: {error_rec or 'None'}",
        "/api/v1/inventory/health"
    )
else:
    summary = health_res.get("summary", {})
    recommendations = rec_res.get("recommendations", [])
    df_rec = pd.DataFrame(recommendations)

    # Scorecards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Active SKUs</div>
            <div class="metric-value">{summary.get('total_items', 0)}</div>
            <div class="metric-delta delta-amber">Tracked Inventory SKUs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Stockouts (Zero Stock)</div>
            <div class="metric-value">{summary.get('stockout_count', 0)} SKUs</div>
            <div class="metric-delta delta-red">🚨 Immediate Outage</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Below Safety Stock</div>
            <div class="metric-value">{summary.get('below_safety_count', 0)} SKUs</div>
            <div class="metric-delta delta-amber">Buffer Breached</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Reorders Required</div>
            <div class="metric-value">{summary.get('reorder_count', 0)} SKUs</div>
            <div class="metric-delta delta-amber">Below ROP Level</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        val_usd = summary.get('total_inventory_value_usd', 0.0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Tied-Up Working Capital</div>
            <div class="metric-value">${val_usd/1e3:.1f}k</div>
            <div class="metric-delta delta-amber">Total Stock Valuation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Filter Controls & Interactive Risk Calculation
    f_col1, f_col2 = st.columns([4, 6])
    with f_col1:
        st.subheader("Slow-Moving & Dead Stock API Calculator")
        sales_vel = st.number_input("Sales Velocity (units/day)", min_value=0.0, value=0.5, step=0.1)
        zero_weeks = st.slider("Consecutive Zero-Sales Weeks", 0, 26, 6)
        curr_stk = st.number_input("Current Stock Units", min_value=0.0, value=250.0, step=10.0)
        unit_c = st.number_input("Unit Purchasing Cost ($)", min_value=0.0, value=45.0, step=5.0)

        dead_res, dead_err = api_client.evaluate_stockout(
            current_stock=curr_stk,
            reorder_point=curr_stk * 0.8,
            safety_stock=curr_stk * 0.3,
            avg_daily_demand=sales_vel
        )

        st.markdown("### Evaluation Results from API:")
        if dead_res:
            st.json(dead_res)

    with f_col2:
        st.subheader("Inventory Stock Valuation Distribution by Region")
        if not df_rec.empty:
            df_rec["inventory_value"] = df_rec["current_stock"] * df_rec["unit_cost"]
            fig_bar = px.bar(
                df_rec,
                x="region",
                y="inventory_value",
                color="procurement_status",
                title="Tied-up Capital ($) by Region and Procurement Status",
                color_discrete_map={"REORDER_REQUIRED": "#ef4444", "HEALTHY": "#10b981", "OVERSTOCKED": "#f97316"},
                template="plotly_dark",
                height=380
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No recommendations data available.")

    # Detailed Table
    st.subheader("Inventory Recommendations & Health Status (Live Data)")
    if not df_rec.empty:
        st.dataframe(
            df_rec[["product_id", "warehouse_id", "region", "current_stock", "reorder_point", "safety_stock", "procurement_status", "unit_cost"]],
            use_container_width=True
        )
