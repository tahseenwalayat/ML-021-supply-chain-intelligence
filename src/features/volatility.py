import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger("features.volatility")


def compute_volatility_features(
    sales_fact: pd.DataFrame,
    velocity_df: pd.DataFrame,
    spine_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes demand volatility, rolling standard deviations, and CV metrics using fast vectorized rolling.
    Keys: (product_id, region, date)
    """
    logger.info("Computing Volatility features...")

    daily = sales_fact.groupby(["product_id", "region", "date"]).agg(
        daily_qty=("quantity", "sum")
    ).reset_index()

    df = spine_df[["product_id", "region", "date"]].merge(
        daily, on=["product_id", "region", "date"], how="left"
    ).fillna({"daily_qty": 0.0})

    df = df.sort_values(by=["product_id", "region", "date"]).reset_index(drop=True)

    df["qty_shift1"] = df.groupby(["product_id", "region"])["daily_qty"].shift(1)
    grouped = df.groupby(["product_id", "region"])["qty_shift1"]

    df["sales_std_7d"] = grouped.rolling(7, min_periods=2).std().reset_index(level=[0, 1], drop=True).fillna(0.0)
    df["sales_std_30d"] = grouped.rolling(30, min_periods=2).std().reset_index(level=[0, 1], drop=True).fillna(0.0)

    df = df.merge(
        velocity_df[["product_id", "region", "date", "sales_velocity_30d"]],
        on=["product_id", "region", "date"],
        how="left"
    )

    df["sales_cv_30d"] = np.where(
        df["sales_velocity_30d"] > 0,
        df["sales_std_30d"] / (df["sales_velocity_30d"] + 1e-5),
        0.0
    )

    conditions = [
        df["sales_cv_30d"] <= 0.5,
        (df["sales_cv_30d"] > 0.5) & (df["sales_cv_30d"] <= 1.0),
        df["sales_cv_30d"] > 1.0
    ]
    choices = [1, 2, 3]
    df["demand_volatility_tier"] = np.select(conditions, choices, default=2)

    df["sales_iqr_30d"] = 1.35 * df["sales_std_30d"]

    cols = [
        "product_id", "region", "date",
        "sales_std_7d", "sales_std_30d", "sales_cv_30d",
        "demand_volatility_tier", "sales_iqr_30d"
    ]
    logger.info(f"Volatility features complete: shape={df[cols].shape}")
    return df[cols]
