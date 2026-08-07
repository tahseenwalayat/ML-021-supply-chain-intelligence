import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger("features.lead_time")


def compute_lead_time_features(
    supplier_reliability_df: pd.DataFrame,
    velocity_df: pd.DataFrame,
    spine_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes replenishment lead time demand estimates and safety buffers.
    Keys: (product_id, region, date)
    """
    logger.info("Computing Lead Time features...")

    df = spine_df[["product_id", "region", "date"]].merge(
        supplier_reliability_df[["product_id", "region", "date", "supplier_lead_time_days"]],
        on=["product_id", "region", "date"],
        how="left"
    ).merge(
        velocity_df[["product_id", "region", "date", "sales_velocity_7d"]],
        on=["product_id", "region", "date"],
        how="left"
    )

    df["lead_time_days"] = df["supplier_lead_time_days"].fillna(4).astype(float)
    df["sales_velocity_7d"] = df["sales_velocity_7d"].fillna(0.0).astype(float)

    # Lead Time Demand = Lead Time (days) * Daily Sales Velocity
    df["lead_time_demand_estimate"] = df["lead_time_days"] * df["sales_velocity_7d"]

    # Lead Time Variability Estimate (15% of nominal lead time)
    df["lead_time_std_days"] = df["lead_time_days"] * 0.15

    # Lead Time Safety Buffer (95% service factor 1.65 * lead_time_std)
    df["lead_time_safety_buffer"] = 1.65 * df["lead_time_std_days"]

    cols = [
        "product_id", "region", "date",
        "lead_time_days", "lead_time_demand_estimate",
        "lead_time_std_days", "lead_time_safety_buffer"
    ]
    logger.info(f"Lead Time features complete: shape={df[cols].shape}")
    return df[cols]
