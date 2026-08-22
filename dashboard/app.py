"""Landing page for the Supply Chain Control Hub."""

import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import api_client
from ui import configure_page


configure_page("Executive home", "🏠")

health_data, health_error = api_client.check_api_health()
is_online = bool(health_data and health_data.get("status") == "healthy")
status_label = "API connected" if is_online else "API needs attention"
status_class = "#0f766e" if is_online else "#b45309"

st.markdown(
    f"""
    <div class="hero-panel">
        <div class="eyebrow">Start here</div>
        <h1 style="margin:0.35rem 0 0.55rem;">Make the next supply-chain decision with confidence.</h1>
        <p style="color:#52677d; max-width:760px; font-size:1.02rem; margin-bottom:1rem;">
            Move from an executive summary to forecast quality, inventory actions, warehouse capacity,
            service risk, procurement priorities, or alerts using the menu above.
        </p>
        <span style="display:inline-block; border:1px solid {status_class}; color:{status_class}; border-radius:999px; padding:0.35rem 0.7rem; font-size:0.86rem; font-weight:700;">{status_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if health_error:
    st.warning("The dashboard navigation is available, but live data is unavailable right now.")

metrics = health_data or {}
metric_columns = st.columns(4)
with metric_columns[0]:
    st.metric("Platform status", "Online" if is_online else "Offline")
with metric_columns[1]:
    st.metric("API version", f"v{metrics.get('version', '—')}")
with metric_columns[2]:
    st.metric("Decision engines", len(metrics.get("modules", [])))
with metric_columns[3]:
    st.metric("Data access", "Secure API")

st.markdown("### Choose a workspace")
workspace_columns = st.columns(2)
with workspace_columns[0]:
    st.markdown(
        """
        <div class="help-card">
            <strong>For daily operations</strong><br>
            Start with <b>Overview</b>, then review <b>Inventory</b>, <b>Service risk</b>, and <b>Alerts</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
with workspace_columns[1]:
    st.markdown(
        """
        <div class="help-card">
            <strong>For planning and leadership</strong><br>
            Use <b>Forecasts</b>, <b>Warehouses</b>, and <b>Procurement</b> to plan capacity and replenishment.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### What each page helps you do")
overview, planning, action = st.columns(3)
with overview:
    st.markdown("**Monitor**\n\nOverview, inventory health, and warehouse utilization show the current position.")
with planning:
    st.markdown("**Plan**\n\nForecast accuracy and service-risk analysis reveal where plans need adjustment.")
with action:
    st.markdown("**Act**\n\nProcurement recommendations and prioritized alerts turn insight into next steps.")

st.caption("Tip: the complete page menu stays at the top of every dashboard view, so you can move between tasks without losing your place.")
