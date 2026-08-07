import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import api_client

st.set_page_config(
    page_title="Warehouse Utilization",
    page_icon="🏭",
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

st.title("🏭 Warehouse Capacity Utilization & Operational Risk")
st.caption("Monitoring facility fill levels, storage constraints, and capacity breach thresholds")

# Fetch warehouse utilization from API
util_res, error_util = api_client.get_warehouse_utilization()

if error_util:
    api_client.render_api_error_banner(error_util, "/api/v1/inventory/utilization")
else:
    wh_list = util_res.get("warehouse_utilization", [])
    df_wh = pd.DataFrame(wh_list)

    if not df_wh.empty:
        avg_util = df_wh["utilization_pct"].mean()
        high_risk_wh = df_wh[df_wh["utilization_pct"] >= 90.0]
        critical_wh = df_wh[df_wh["utilization_pct"] >= 95.0]

        # Scorecards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Active Facilities</div>
                <div class="metric-value">{len(df_wh)}</div>
                <div class="metric-delta delta-amber">Monitored Warehouses</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Average Utilization</div>
                <div class="metric-value">{avg_util:.1f}%</div>
                <div class="metric-delta delta-green">Network Average</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Capacity Risk (>90%)</div>
                <div class="metric-value">{len(high_risk_wh)} Facilities</div>
                <div class="metric-delta delta-amber">High Storage Load</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Critical Overload (>95%)</div>
                <div class="metric-value">{len(critical_wh)} Facilities</div>
                <div class="metric-delta delta-red">🚨 Immediate Transfer Needed</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Capacity Breach Warnings
        if not high_risk_wh.empty:
            for _, r in high_risk_wh.iterrows():
                st.warning(
                    f"⚠️ **Capacity Risk Alert**: Warehouse **{r['warehouse_id']}** utilization is at **{r['utilization_pct']}%** "
                    f"({r['total_stock_units']} / {r['capacity_limit']} units). Recommended Action: Trigger inter-warehouse transfer or clear dead stock."
                )

        # Visualizations
        col1, col2 = st.columns([6, 4])
        with col1:
            st.subheader("Facility Storage Utilization vs Capacity Limit")
            fig_bar = px.bar(
                df_wh,
                x="warehouse_id",
                y=["total_stock_units", "capacity_limit"],
                barmode="group",
                title="Stock Units vs Storage Capacity Limit",
                labels={"value": "Units", "warehouse_id": "Warehouse Facility"},
                template="plotly_dark",
                height=380
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.subheader("Facility Utilization % Gauges")
            fig_gauge = px.bar(
                df_wh,
                x="warehouse_id",
                y="utilization_pct",
                color="utilization_pct",
                color_continuous_scale=["#10b981", "#eab308", "#ef4444"],
                title="Utilization Rate (%) per Facility",
                labels={"utilization_pct": "Utilization %"},
                template="plotly_dark",
                height=380
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Facility Details Data Table
        st.subheader("Facility Level Detail Table (API Live Data)")
        st.dataframe(df_wh, use_container_width=True)
    else:
        st.info("No warehouse utilization data returned from API.")
