"""Shared presentation and navigation utilities for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st


NAV_ITEMS = (
    ("Home", "app.py"),
    ("Overview", "pages/kpi_overview.py"),
    ("Forecasts", "pages/forecast_accuracy.py"),
    ("Inventory", "pages/inventory_health.py"),
    ("Warehouses", "pages/warehouse_utilization.py"),
    ("Service risk", "pages/fill_rate_stockout.py"),
    ("Procurement", "pages/procurement_recommendations.py"),
    ("Alerts", "pages/alert_center.py"),
)
PRIMARY_NAV_ITEMS = NAV_ITEMS[:4]
SECONDARY_NAV_ITEMS = NAV_ITEMS[4:]


def configure_page(title: str, icon: str = "🏠") -> None:
    """Configure a page and render the shared, accessible dashboard header."""
    st.set_page_config(
        page_title=f"{title} | Supply Chain Control Hub",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_global_styles()
    render_header(title)


def _inject_global_styles() -> None:
    """Apply one high-contrast visual system across all dashboard pages."""
    st.markdown(
        """
        <style>
            :root {
                --ink: #102a43;
                --muted: #52677d;
                --surface: #ffffff;
                --canvas: #f4f7fb;
                --line: #d9e2ec;
                --brand: #0b7285;
                --brand-deep: #064e5b;
                --accent: #f59f00;
            }
            .stApp {
                background:
                    radial-gradient(circle at 8% 0%, rgba(19, 148, 165, 0.10), transparent 22rem),
                    var(--canvas);
                color: var(--ink);
            }
            [data-testid="stHeader"] {
                display: none;
            }
            .block-container {
                max-width: 1440px;
                padding-top: 1.75rem;
                padding-bottom: 3rem;
            }
            h1, h2, h3 { color: var(--ink) !important; letter-spacing: -0.025em; }
            h1 { font-weight: 750 !important; margin-bottom: 0.25rem !important; }
            h2, h3 { font-weight: 700 !important; }
            [data-testid="stMetric"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 1rem 1.1rem;
                box-shadow: 0 8px 20px rgba(16, 42, 67, 0.06);
            }
            [data-testid="stMetricLabel"] { color: var(--muted); font-weight: 650; }
            [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
            [data-testid="stSidebar"] { background: var(--surface); }
            [data-testid="stVerticalBlockBorderWrapper"] {
                border-color: var(--line);
                border-radius: 16px;
                box-shadow: 0 8px 22px rgba(16, 42, 67, 0.04);
            }
            .metric-card {
                background: var(--surface) !important;
                border: 1px solid var(--line) !important;
                border-radius: 16px !important;
                box-shadow: 0 4px 14px rgba(16, 42, 67, 0.05) !important;
            }
            .metric-title { color: var(--muted) !important; }
            .metric-value { color: var(--ink) !important; }
            .topbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                background: linear-gradient(115deg, #073b4c 0%, #0b7285 60%, #1394a5 100%);
                border-radius: 18px;
                color: white;
                margin-bottom: 0.6rem;
                padding: 0.95rem 1.25rem;
                box-shadow: 0 12px 28px rgba(7, 59, 76, 0.18);
            }
            .topbar h1 { color: white !important; font-size: 1.2rem !important; margin: 0 !important; }
            .topbar p { color: #d8f3f5; font-size: 0.86rem; margin: 0.15rem 0 0; }
            .topbar .tag { background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.32); border-radius: 999px; font-size: 0.78rem; padding: 0.35rem 0.65rem; white-space: nowrap; }
            .topbar .workspace { color: #9ee8ef; font-size: 0.76rem; font-weight: 750; letter-spacing: 0.08em; text-transform: uppercase; }
            .nav-label {
                color: var(--muted);
                font-size: 0.72rem;
                font-weight: 750;
                letter-spacing: 0.08em;
                margin: 0.1rem 0 0.45rem;
                text-transform: uppercase;
            }
            .hero-panel {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 1.5rem;
                box-shadow: 0 10px 24px rgba(16, 42, 67, 0.06);
            }
            .eyebrow { color: var(--brand); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
            .help-card { background: #e6fcf5; border: 1px solid #b2f2e5; border-radius: 12px; padding: 1rem; color: #064e3b; }
            .stPageLink a {
                border-radius: 10px;
                color: var(--muted);
                font-size: 0.84rem;
                font-weight: 700;
                min-height: 2.35rem;
                padding: 0.45rem 0.6rem;
            }
            .stPageLink a:hover { background: #dff4f6; color: var(--brand-deep); }
            [data-testid="stAlert"] { border-radius: 12px; }
            .stButton > button { border-radius: 10px; font-weight: 700; }
            @media (max-width: 800px) {
                .block-container { padding-left: 1rem; padding-right: 1rem; }
                .topbar { align-items: flex-start; flex-direction: column; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(page_title: str) -> None:
    """Render a compact brand bar and responsive page navigation."""
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <h1>Supply Chain Control Hub</h1>
                <p><span class="workspace">{page_title}</span><br>Clear decisions for demand, inventory, suppliers, and risk.</p>
            </div>
            <span class="tag">Decision support workspace</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="nav-label">Operations</p>', unsafe_allow_html=True)
    with st.container(border=True):
        columns = st.columns(4, gap="small")
        for column, (label, page) in zip(columns, PRIMARY_NAV_ITEMS):
            with column:
                st.page_link(page, label=label, use_container_width=True)

    st.markdown('<p class="nav-label">Planning and risk</p>', unsafe_allow_html=True)
    with st.container(border=True):
        columns = st.columns(4, gap="small")
        for column, (label, page) in zip(columns, SECONDARY_NAV_ITEMS):
            with column:
                st.page_link(page, label=label, use_container_width=True)

    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
