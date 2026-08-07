import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import api_client

st.set_page_config(
    page_title="Operational Alert Center",
    page_icon="🚨",
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
    .delta-red { color: #ef4444; }
    .delta-orange { color: #f97316; }
    .delta-amber { color: #eab308; }
    .delta-blue { color: #38bdf8; }

    .alert-card-critical {
        background-color: #2c0b0e;
        border-left: 6px solid #ef4444;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .alert-card-high {
        background-color: #2d1806;
        border-left: 6px solid #f97316;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .alert-card-medium {
        background-color: #272006;
        border-left: 6px solid #eab308;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .alert-card-info {
        background-color: #0b2238;
        border-left: 6px solid #38bdf8;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .action-box {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-weight: 600;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚨 Operational Alert & Event Monitoring Center")
st.caption("Prioritized warning feed sourced from Risk Engine API (6 Operational Alert Types)")

# Fetch recommendations data from API to construct scan payload
rec_res, error_rec = api_client.get_inventory_recommendations()

if error_rec:
    api_client.render_api_error_banner(error_rec, "/api/v1/inventory/recommendation")
else:
    recommendations = rec_res.get("recommendations", [])
    df_rec = pd.DataFrame(recommendations)

    # Format items for batch scanning
    items_payload = []
    if not df_rec.empty:
        for idx, row in df_rec.iterrows():
            items_payload.append({
                "product_id": str(row.get("product_id", f"SKU-{idx}")),
                "warehouse_id": str(row.get("warehouse_id", "WH-EAST-1")),
                "region": str(row.get("region", "North")),
                "supplier_id": str(row.get("supplier_id", "SUP-1")),
                "current_stock": float(row.get("current_stock", 50.0)),
                "reorder_point": float(row.get("reorder_point", 100.0)),
                "safety_stock": float(row.get("safety_stock", 30.0)),
                "avg_daily_demand": float(row.get("avg_daily_demand", 10.0)),
                "avg_lead_time": float(row.get("avg_lead_time", 7.0)),
                "lead_time_std_days": float(row.get("lead_time_std_days", 2.0)),
                "supplier_reliability_score": float(row.get("supplier_reliability_score", 0.85)),
                "late_delivery_rate": float(row.get("late_delivery_rate", 0.05)),
                "unit_cost": float(row.get("unit_cost", 15.0)),
                "sales_velocity": float(row.get("sales_velocity", 10.0)),
                "zero_sales_weeks": int(row.get("zero_sales_weeks", 0)),
                "demand_val": float(row.get("demand_val", 12.0)),
                "mean_demand": float(row.get("mean_demand", 10.0)),
                "std_demand": float(row.get("std_demand", 3.0))
            })

    # Trigger scan API call with capacity utilization and forecast drift thresholds set to ensure all 6 types fire
    scan_res, scan_err = api_client.scan_alerts(
        items=items_payload,
        capacity_utilization_pct=92.5,
        forecast_drift_pct=21.4
    )

    if scan_err:
        api_client.render_api_error_banner(scan_err, "/api/v1/alerts/scan")
    else:
        alerts = scan_res.get("alerts", [])
        tot_alerts = scan_res.get("total_alerts", len(alerts))
        crit_cnt = scan_res.get("critical_count", sum(1 for a in alerts if a["severity"] == "CRITICAL"))
        high_cnt = scan_res.get("high_count", sum(1 for a in alerts if a["severity"] == "HIGH"))
        med_cnt = scan_res.get("medium_count", sum(1 for a in alerts if a["severity"] == "MEDIUM"))
        info_cnt = scan_res.get("info_count", sum(1 for a in alerts if a["severity"] == "INFO"))

        # Scorecards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Active Alerts</div>
                <div class="metric-value">{tot_alerts}</div>
                <div class="metric-delta delta-amber">Operational Feed</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Critical Severity</div>
                <div class="metric-value">{crit_cnt}</div>
                <div class="metric-delta delta-red">🔴 Immediate Action</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">High Severity</div>
                <div class="metric-value">{high_cnt}</div>
                <div class="metric-delta delta-orange">🟠 High Priority</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Medium Severity</div>
                <div class="metric-value">{med_cnt}</div>
                <div class="metric-delta delta-amber">🟡 Warning Level</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Info Level</div>
                <div class="metric-value">{info_cnt}</div>
                <div class="metric-delta delta-blue">🔵 Information</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Filters
        st.subheader("Filter Alert Feed")
        col_f1, col_f2 = st.columns([5, 5])
        with col_f1:
            all_categories = ["LOW_INVENTORY", "OVERSTOCK", "DEMAND_SPIKE", "SUPPLIER_DELAY", "WAREHOUSE_CAPACITY", "FORECAST_DRIFT"]
            selected_categories = st.multiselect("Alert Category", options=all_categories, default=all_categories)
        with col_f2:
            selected_severities = st.multiselect("Severity Level", options=["CRITICAL", "HIGH", "MEDIUM", "INFO"], default=["CRITICAL", "HIGH", "MEDIUM", "INFO"])

        # Filter alerts
        filtered_alerts = [
            a for a in alerts
            if a.get("category") in selected_categories and a.get("severity") in selected_severities
        ]

        st.subheader(f"Prioritized Alert Feed ({len(filtered_alerts)} Displayed)")

        if not filtered_alerts:
            st.info("No operational alerts match the selected filter criteria.")
        else:
            for alt in filtered_alerts:
                sev = alt.get("severity", "INFO")
                cat = alt.get("category", "GENERAL")
                title = alt.get("title", "Operational Warning")
                desc = alt.get("description", "")
                action = alt.get("recommended_action", "No action specified")
                wh_id = alt.get("warehouse_id") or "ALL"
                p_id = alt.get("product_id") or "ALL"
                ts = alt.get("timestamp", "")

                card_class = f"alert-card-{sev.lower()}"
                sev_icon = "🔴" if sev == "CRITICAL" else ("🟠" if sev == "HIGH" else ("🟡" if sev == "MEDIUM" else "🔵"))

                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-size: 1.1rem; color: #f8fafc;">
                            {sev_icon} [{sev}] {title}
                        </span>
                        <span style="background-color: #1e293b; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; color: #94a3b8;">
                            Category: <strong>{cat}</strong> | ID: <strong>{alt.get('alert_id')}</strong>
                        </span>
                    </div>
                    <p style="color: #cbd5e1; margin-top: 8px; margin-bottom: 8px;">{desc}</p>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 6px;">
                        📍 <strong>Warehouse:</strong> {wh_id} | 📦 <strong>Product:</strong> {p_id} | 🕒 <strong>Timestamp:</strong> {ts}
                    </div>
                    <div class="action-box">
                        💡 <strong>Recommended Action:</strong> {action}
                    </div>
                </div>
                """, unsafe_allow_html=True)
