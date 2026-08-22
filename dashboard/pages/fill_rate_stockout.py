import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import api_client
from ui import configure_page

configure_page("Fill rate and stockout", "⚠️")

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

st.title("📈 Service Fill Rate & Stockout Probability Engine")
st.caption("Evaluating service fill rate %, stockout probability distributions, and lead time supply shocks")

if not api_client.require_backend():
    st.stop()

# Fetch inventory recommendations to evaluate stockouts
rec_res, error_rec = api_client.get_inventory_recommendations()

if error_rec:
    api_client.render_api_error_banner(error_rec, "/api/v1/inventory/recommendation")
else:
    recommendations = rec_res.get("recommendations", [])
    df_rec = pd.DataFrame(recommendations)

    # Scorecards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Target Service Fill Rate</div>
            <div class="metric-value">95.4%</div>
            <div class="metric-delta delta-green">▲ Exceeding 95.0% Goal</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        zero_cnt = sum(df_rec["current_stock"] == 0) if not df_rec.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Critical Stockouts</div>
            <div class="metric-value">{zero_cnt} SKUs</div>
            <div class="metric-delta delta-red">Immediate Zero-Stock</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        below_ss = sum(df_rec["current_stock"] < df_rec["safety_stock"]) if not df_rec.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Safety Stock Breached</div>
            <div class="metric-value">{below_ss} SKUs</div>
            <div class="metric-delta delta-amber">High Stockout Risk</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Service Level Z-Factor</div>
            <div class="metric-value">1.65 (95%)</div>
            <div class="metric-delta delta-green">Standard Normal Service Level</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Risk Calculator & Stress Simulator
    col1, col2 = st.columns([5, 5])

    with col1:
        st.subheader("SKU Stockout Risk API Calculator")
        curr_stock = st.number_input("Current Stock Units", value=15.0, min_value=0.0)
        rop_val = st.number_input("Reorder Point (ROP)", value=50.0, min_value=0.0)
        ss_val = st.number_input("Safety Stock Buffer", value=20.0, min_value=0.0)
        daily_d = st.number_input("Average Daily Demand", value=5.0, min_value=0.1)
        lead_t = st.number_input("Supplier Lead Time (Days)", value=7.0, min_value=1.0)

        so_res, so_err = api_client.evaluate_stockout(
            current_stock=curr_stock,
            reorder_point=rop_val,
            safety_stock=ss_val,
            avg_daily_demand=daily_d,
            lead_time_days=lead_t
        )

        if so_res:
            st.markdown("### Stockout Risk Output from API:")
            st.json(so_res)

    with col2:
        st.subheader("Scenario Fill Rate Stress Tester (API Driven)")
        supp_delay = st.slider("Supplier Delay Shock (Days)", 0, 14, 4)
        demand_surge = st.slider("Demand Surge Multiplier", 0.5, 2.5, 1.3)

        sim_params = {
            "scenario_name": "Fill Rate Stress Test",
            "supplier_delay_days": float(supp_delay),
            "price_change_pct": 0.0,
            "demand_surge_multiplier": float(demand_surge),
            "simulation_horizon_days": 30
        }

        sim_res, sim_err = api_client.simulate_sku_scenario(
            base_daily_demand=10.0,
            current_stock=100.0,
            reorder_point=50.0,
            safety_stock=20.0,
            base_lead_time=7.0,
            unit_cost=15.0,
            scenario_params=sim_params
        )

        if sim_res:
            metrics = sim_res.get("summary_metrics", {})
            st.metric("Baseline Fill Rate", f"{metrics.get('baseline_fill_rate_pct')}%")
            st.metric("Scenario Fill Rate", f"{metrics.get('scenario_fill_rate_pct')}%", delta=f"{metrics.get('fill_rate_delta_pct')}%")

            # Trajectory Chart
            b_df = pd.DataFrame(sim_res["daily_trajectories"]["baseline"])
            s_df = pd.DataFrame(sim_res["daily_trajectories"]["scenario"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=b_df["day"], y=b_df["stock"], name="Baseline Stock", line=dict(color="#10b981", width=3)))
            fig.add_trace(go.Scatter(x=s_df["day"], y=s_df["stock"], name="Scenario Shocked Stock", line=dict(color="#ef4444", width=3, dash="dash")))
            fig.update_layout(template="plotly_dark", height=320, xaxis_title="Simulation Day", yaxis_title="Stock Units")
            st.plotly_chart(fig, use_container_width=True)

    # Detailed Table
    st.subheader("SKU Stockout Risk Breakdown")
    if not df_rec.empty:
        st.dataframe(
            df_rec[["product_id", "warehouse_id", "current_stock", "reorder_point", "safety_stock", "allocated_daily_demand", "procurement_status"]],
            use_container_width=True
        )
