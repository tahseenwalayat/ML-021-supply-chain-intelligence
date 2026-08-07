import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger("features.promotion_impact")


def compute_promotion_impact_features(
    sales_fact: pd.DataFrame,
    promotion_dim: pd.DataFrame,
    spine_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes promotion impact, discount percentage, and causal promo activity windows.
    Keys: (product_id, region, date)
    """
    logger.info("Computing Promotion Impact features...")

    sf = sales_fact.copy()
    sf["is_promo"] = (sf["promotion_id"] != "promo_none").astype(int)
    daily_promo = sf.groupby(["product_id", "region", "date"]).agg(
        is_promo_active=("is_promo", "max"),
        discount_amount=("discount_amount", "max")
    ).reset_index()

    df = spine_df[["product_id", "region", "date"]].merge(
        daily_promo, on=["product_id", "region", "date"], how="left"
    ).fillna({"is_promo_active": 0, "discount_amount": 0.0})

    df["is_promo_active"] = df["is_promo_active"].astype(int)
    df["discount_percent"] = np.where(df["is_promo_active"] == 1, 0.15, 0.0)

    df = df.sort_values(by=["product_id", "region", "date"]).reset_index(drop=True)
    df["promo_shift1"] = df.groupby(["product_id", "region"])["is_promo_active"].shift(1)
    grouped = df.groupby(["product_id", "region"])["promo_shift1"]

    df["promo_days_in_last_7d"] = grouped.rolling(7, min_periods=1).sum().reset_index(level=[0, 1], drop=True).fillna(0.0)
    df["promo_days_in_last_14d"] = grouped.rolling(14, min_periods=1).sum().reset_index(level=[0, 1], drop=True).fillna(0.0)

    df["promo_sales_lift_factor"] = np.where(df["is_promo_active"] == 1, 1.285, 1.0)

    cols = [
        "product_id", "region", "date",
        "is_promo_active", "discount_percent", "discount_amount",
        "promo_days_in_last_7d", "promo_days_in_last_14d", "promo_sales_lift_factor"
    ]
    logger.info(f"Promotion Impact features complete: shape={df[cols].shape}")
    return df[cols]
