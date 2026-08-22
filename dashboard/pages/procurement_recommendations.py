import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import api_client
from ui import configure_page

configure_page("Procurement", "🛒")

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

st.title("🛒 Procurement & Inventory Replenishment Recommendations")
st.caption("Safety Stock (SS), Reorder Points (ROP), Economic Order Quantity (EOQ), and Supplier Delay Risk")

if not api_client.require_backend():
    st.stop()

# Fetch procurement recommendations from API
rec_res, error_rec = api_client.get_inventory_recommendations()

if error_rec:
    api_client.render_api_error_banner(error_rec, "/api/v1/inventory/recommendation")
else:
    recommendations = rec_res.get("recommendations", [])
    df_rec = pd.DataFrame(recommendations)

    # The API provides a replenishment quantity rather than a monetary unit cost.
    df_reorder = (
        df_rec[df_rec["recommended_procurement_qty"] > 0]
        if not df_rec.empty else pd.DataFrame()
    )
    reorder_cnt = len(df_reorder)
    total_replenishment_units = (
        df_reorder["recommended_procurement_qty"].sum() if not df_reorder.empty else 0.0
    )

    # Overview Scorecards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Reorder Items Pending</div>
            <div class="metric-value">{reorder_cnt} SKUs</div>
            <div class="metric-delta delta-red">🚨 Action Required</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Recommended Replenishment</div>
            <div class="metric-value">{total_replenishment_units:,.0f} units</div>
            <div class="metric-delta delta-amber">Unit costs are not supplied by the API</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_lead_time = df_rec["avg_lead_time"].mean() if not df_rec.empty else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average Lead Time</div>
            <div class="metric-value">{avg_lead_time:.1f} days</div>
            <div class="metric-delta delta-green">Supplier planning input</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">EOQ Cost Formula</div>
            <div class="metric-value">sqrt(2DS / H)</div>
            <div class="metric-delta delta-green">Economic Order Quantity</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Calculators & API Evaluators
    c1, c2 = st.columns([5, 5])
    with c1:
        st.subheader("EOQ & Safety Stock Calculator ($Z \\cdot \\sigma_d \\cdot \\sqrt{L}$)")
        d_val = st.number_input("Average Daily Demand (d)", value=30.0, min_value=1.0)
        sigma_d = st.number_input("Demand Std Dev (sigma_d)", value=5.0, min_value=0.1)
        lt_val = st.number_input("Supplier Lead Time (L, Days)", value=7.0, min_value=1.0)
        sl_val = st.select_slider("Target Service Level (Z)", options=[0.90, 0.95, 0.98, 0.99], value=0.95)
        
        z_map = {0.90: 1.28, 0.95: 1.65, 0.98: 2.05, 0.99: 2.33}
        z_score = z_map[sl_val]

        ss_target = z_score * sigma_d * np.sqrt(lt_val)
        rop_target = d_val * lt_val + ss_target

        # EOQ params
        annual_demand = d_val * 365
        order_cost = st.number_input("Fixed Order Cost (S, $)", value=50.0)
        holding_cost = st.number_input("Annual Holding Cost / Unit (H, $)", value=4.0)

        eoq_target = np.sqrt((2 * annual_demand * order_cost) / holding_cost)

        st.success(f"**Safety Stock Target (SS)**: `{ss_target:.1f} Units`")
        st.info(f"**Reorder Point (ROP)**: `{rop_target:.1f} Units`")
        st.warning(f"**Economic Order Quantity (EOQ)**: `{eoq_target:.1f} Units`")

    with c2:
        st.subheader("Supplier Delay Risk API Evaluator")
        rel_score = st.slider("Supplier Reliability Score [0.0, 1.0]", 0.0, 1.0, 0.82, step=0.05)
        lt_std = st.slider("Lead Time Std Dev (Days)", 0.0, 10.0, 2.5, step=0.5)

        supp_res, supp_err = api_client.evaluate_supplier_delay(
            reliability_score=rel_score,
            lead_time_std=lt_std,
            lead_time_avg=lt_val
        )

        if supp_res:
            st.markdown("### Supplier Risk Output from API:")
            st.json(supp_res)

    # Detailed Table
    st.subheader("Prioritized Replenishment Action Items (API Live Data)")
    available_statuses = sorted(df_rec["procurement_status"].dropna().unique()) if not df_rec.empty else []
    default_statuses = ["REORDER_REQUIRED"] if "REORDER_REQUIRED" in available_statuses else available_statuses
    filter_status = st.multiselect("Filter Procurement Status", options=available_statuses, default=default_statuses)
    
    if not df_rec.empty:
        df_show = df_rec[df_rec["procurement_status"].isin(filter_status)]
        st.dataframe(
            df_show[["product_id", "warehouse_id", "allocated_daily_demand", "current_stock", "on_order", "reorder_point", "safety_stock", "eoq", "recommended_procurement_qty", "procurement_status"]],
            use_container_width=True
        )
