import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger("features.velocity")


def compute_velocity_features(
    sales_fact: pd.DataFrame,
    spine_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes strictly causal sales velocity and acceleration features using fast vectorized rolling.
    Keys: (product_id, region, date)
    All rolling windows use .shift(1) to avoid lookahead leakage.
    """
    logger.info("Computing Velocity features...")

    daily = sales_fact.groupby(["product_id", "region", "date"]).agg(
        daily_qty=("quantity", "sum"),
        daily_revenue=("total_sales", "sum")
    ).reset_index()

    df = spine_df[["product_id", "region", "date"]].merge(
        daily, on=["product_id", "region", "date"], how="left"
    ).fillna({"daily_qty": 0.0, "daily_revenue": 0.0})

    df = df.sort_values(by=["product_id", "region", "date"]).reset_index(drop=True)

    # Shift(1) guarantees today's features rely ONLY on historical data up to yesterday (t-1)
    df["qty_shift1"] = df.groupby(["product_id", "region"])["daily_qty"].shift(1)
    df["rev_shift1"] = df.groupby(["product_id", "region"])["daily_revenue"].shift(1)

    grouped_qty = df.groupby(["product_id", "region"])["qty_shift1"]
    grouped_rev = df.groupby(["product_id", "region"])["rev_shift1"]

    df["sales_velocity_7d"] = grouped_qty.rolling(7, min_periods=1).mean().reset_index(level=[0, 1], drop=True).fillna(0.0)
    df["sales_velocity_14d"] = grouped_qty.rolling(14, min_periods=1).mean().reset_index(level=[0, 1], drop=True).fillna(0.0)
    df["sales_velocity_30d"] = grouped_qty.rolling(30, min_periods=1).mean().reset_index(level=[0, 1], drop=True).fillna(0.0)

    df["revenue_velocity_7d"] = grouped_rev.rolling(7, min_periods=1).mean().reset_index(level=[0, 1], drop=True).fillna(0.0)
    df["revenue_velocity_30d"] = grouped_rev.rolling(30, min_periods=1).mean().reset_index(level=[0, 1], drop=True).fillna(0.0)

    df["sales_acceleration_7d"] = np.where(
        df["sales_velocity_30d"] > 0,
        df["sales_velocity_7d"] / df["sales_velocity_30d"],
        1.0
    )

    cols = [
        "product_id", "region", "date",
        "sales_velocity_7d", "sales_velocity_14d", "sales_velocity_30d",
        "revenue_velocity_7d", "revenue_velocity_30d", "sales_acceleration_7d"
    ]
    logger.info(f"Velocity features complete: shape={df[cols].shape}")
    return df[cols]
