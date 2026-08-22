"""Authenticated real-time demand forecasting endpoints."""

import os
from typing import Any, Dict, Literal, Optional

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.forecasting.dataset_split import load_feature_store, prepare_hierarchy_data


router = APIRouter(prefix="/api/v1/forecast", tags=["Demand Forecasting"])

MODELS_DIR = "models"
TARGET_COLUMN = "target_next_day_sales"


class ForecastRequest(BaseModel):
    """Request a one-step-ahead forecast from a registered model artifact."""

    region: str = Field(..., min_length=1)
    product_id: Optional[str] = Field(None, min_length=1)
    hierarchy_level: Literal["sku_region", "category_region", "region_total"] = "sku_region"
    model_type: Literal["lightgbm", "xgboost"] = "lightgbm"
    feature_overrides: Dict[str, Any] = Field(default_factory=dict)


def _load_artifact(model_type: str, hierarchy_level: str) -> Dict[str, Any]:
    path = os.path.join(MODELS_DIR, f"{model_type}_{hierarchy_level}.joblib")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Model artifact not found: {os.path.basename(path)}")

    artifact = joblib.load(path)
    if artifact.get("target_col") != TARGET_COLUMN:
        raise HTTPException(
            status_code=409,
            detail=(
                "This model was trained before the one-step-ahead target correction. "
                "Run the authenticated MLOps retraining workflow before serving forecasts."
            ),
        )
    if "model" not in artifact or "feature_cols" not in artifact:
        raise HTTPException(status_code=500, detail="Model artifact is incomplete.")
    return artifact


def _latest_context(request: ForecastRequest) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_store, product_dim = load_feature_store()
    level_df = prepare_hierarchy_data(feature_store, product_dim, level=request.hierarchy_level)

    if request.hierarchy_level == "sku_region":
        if not request.product_id:
            raise HTTPException(status_code=422, detail="product_id is required for sku_region forecasts.")
        filtered = level_df[
            (level_df["product_id"] == request.product_id) & (level_df["region"] == request.region)
        ]
    elif request.hierarchy_level == "category_region":
        if not request.product_id or product_dim is None or "category" not in product_dim.columns:
            raise HTTPException(
                status_code=422,
                detail="product_id with a matching product category is required for category_region forecasts.",
            )
        category_rows = product_dim.loc[product_dim["product_id"] == request.product_id, "category"]
        if category_rows.empty:
            raise HTTPException(status_code=404, detail="product_id was not found in product_dim.")
        filtered = level_df[
            (level_df["category"] == category_rows.iloc[0]) & (level_df["region"] == request.region)
        ]
    else:
        filtered = level_df[level_df["region"] == request.region]

    if filtered.empty:
        raise HTTPException(status_code=404, detail="No feature context was found for the supplied forecast request.")
    return filtered.sort_values("date").tail(1).copy(), level_df


@router.post("/predict", summary="Generate an authenticated one-step-ahead demand forecast")
def predict_demand(request: ForecastRequest) -> Dict[str, Any]:
    """Predict the next day's demand from the latest compatible feature-store row."""
    artifact = _load_artifact(request.model_type, request.hierarchy_level)
    context, full_level_df = _latest_context(request)
    feature_cols = artifact["feature_cols"]
    unknown_overrides = sorted(set(request.feature_overrides) - set(feature_cols))
    if unknown_overrides:
        raise HTTPException(status_code=422, detail=f"Unknown forecast features: {unknown_overrides}")

    context = context.copy()
    for feature_name, value in request.feature_overrides.items():
        context.loc[:, feature_name] = value

    missing_features = [name for name in feature_cols if name not in context.columns]
    if missing_features:
        raise HTTPException(status_code=500, detail=f"Feature store is missing model features: {missing_features}")

    model_input = context[feature_cols].copy()
    for column in artifact.get("cat_cols", []):
        if column in model_input.columns:
            categories = full_level_df[column].astype("category").cat.categories
            model_input[column] = pd.Categorical(model_input[column], categories=categories)

    prediction = max(0.0, float(artifact["model"].predict(model_input)[0]))
    context_date = pd.to_datetime(context.iloc[0]["date"])
    return {
        "model_file": f"{request.model_type}_{request.hierarchy_level}.joblib",
        "hierarchy_level": request.hierarchy_level,
        "context_date": context_date.date().isoformat(),
        "forecast_date": (context_date + pd.Timedelta(days=1)).date().isoformat(),
        "daily_demand_forecast": round(prediction, 2),
        "weekly_demand_estimate": round(prediction * 7, 2),
        "monthly_demand_estimate": round(prediction * 30, 2),
        "note": "Weekly and monthly estimates extrapolate the one-step daily forecast; they are not separate horizon models.",
    }
