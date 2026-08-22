import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import api_client
from ui import configure_page

configure_page("Forecast quality", "📈")
st.title("Multi-Horizon Demand Forecast Accuracy")
st.caption("Live model-artifact quality metrics and one-step-ahead forecast-serving readiness.")

if not api_client.require_backend():
    st.stop()


def model_metric(models: list[dict], model_prefix: str) -> dict | None:
    candidates = [
        model for model in models
        if model.get("model_file", "").lower().startswith(model_prefix)
        and isinstance(model.get("wmape"), (int, float))
    ]
    return min(candidates, key=lambda model: model["wmape"]) if candidates else None


models_res, error_models = api_client.get_registered_models()
if error_models:
    api_client.render_api_error_banner(error_models, "/api/v1/mlops/models")
    st.stop()

registered_models = models_res.get("registered_models", [])
if not registered_models:
    st.warning("No model artifacts are registered. Run the retraining workflow before using forecasts.")
    st.stop()

lightgbm = model_metric(registered_models, "lightgbm")
xgboost = model_metric(registered_models, "xgboost")
prophet = model_metric(registered_models, "prophet")
champion = min(
    [model for model in registered_models if isinstance(model.get("wmape"), (int, float))],
    key=lambda model: model["wmape"],
    default=None,
)

metric_columns = st.columns(4)
for column, label, model in zip(
    metric_columns,
    ["Best LightGBM WMAPE", "Best XGBoost WMAPE", "Best Prophet WMAPE", "Current Champion"],
    [lightgbm, xgboost, prophet, champion],
):
    with column:
        if model:
            st.metric(label, f"{model['wmape']:.2f}%", model.get("model_file", ""))
        else:
            st.metric(label, "Not available")

serving_ready = [model for model in registered_models if model.get("serving_ready")]
if serving_ready:
    st.success(f"{len(serving_ready)} model artifact(s) are ready for the authenticated forecast API.")
else:
    st.warning(
        "The installed artifacts use the legacy same-day target and cannot serve forecasts. "
        "Start retraining after restarting the backend."
    )

metric_rows = [
    {
        "model_file": model["model_file"],
        "hierarchy_level": model.get("level", "unknown"),
        "wmape": model.get("wmape"),
        "naive_wmape": model.get("naive_wmape"),
        "serving_ready": model.get("serving_ready", False),
    }
    for model in registered_models
    if isinstance(model.get("wmape"), (int, float))
]
if metric_rows:
    st.subheader("Validation WMAPE by Registered Model")
    metrics_df = pd.DataFrame(metric_rows)
    figure = px.bar(
        metrics_df,
        x="model_file",
        y="wmape",
        color="hierarchy_level",
        hover_data=["naive_wmape", "serving_ready"],
        labels={"wmape": "Validation WMAPE (%)", "model_file": "Model artifact"},
    )
    st.plotly_chart(figure, use_container_width=True)

st.subheader("Drift and Retraining")
baseline = float(champion["wmape"]) if champion else 0.0
current_wmape = st.number_input(
    "Current observed WMAPE (%)",
    min_value=0.0,
    max_value=1000.0,
    value=baseline,
    step=0.5,
)
if champion:
    degradation, degradation_error = api_client.detect_degradation(baseline, current_wmape)
    if degradation_error:
        api_client.render_api_error_banner(degradation_error, "/api/v1/mlops/detect-degradation")
    else:
        st.write(
            f"Relative degradation: **{degradation['relative_degradation_pct']:.2f}%** · "
            f"Retraining recommended: **{degradation['model_retrain_recommended']}**"
        )

retraining_level = st.selectbox(
    "Retraining hierarchy",
    ["sku_region", "category_region", "region_total"],
)
if st.button(f"Start retraining for {retraining_level}"):
    job, job_error = api_client.start_retraining(retraining_level)
    if job_error:
        api_client.render_api_error_banner(job_error, "/api/v1/mlops/retrain")
    else:
        st.success(f"Retraining job queued: {job['job_id']}")

st.subheader("Registered Models & Artifact Metrics")
st.dataframe(pd.DataFrame(registered_models), use_container_width=True)
