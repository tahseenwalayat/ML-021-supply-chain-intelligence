import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger("features.regional_patterns")


def compute_regional_patterns_features(
    sales_fact: pd.DataFrame,
    velocity_df: pd.DataFrame,
    spine_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes regional market demand patterns and product sales share within regions using fast vectorized rolling.
    Keys: (product_id, region, date)
    """
    logger.info("Computing Regional Patterns features...")

    regional_daily = sales_fact.groupby(["region", "date"]).agg(
        regional_revenue=("total_sales", "sum"),
        regional_freight=("shipping_cost", "mean")
    ).reset_index()

    regional_daily = regional_daily.sort_values(by=["region", "date"]).reset_index(drop=True)
    
    regional_daily["rev_shift1"] = regional_daily.groupby("region")["regional_revenue"].shift(1)
    regional_daily["freight_shift1"] = regional_daily.groupby("region")["regional_freight"].shift(1)

    reg_rev_grouped = regional_daily.groupby("region")["rev_shift1"]
    reg_fr_grouped = regional_daily.groupby("region")["freight_shift1"]

    regional_daily["regional_daily_total_sales"] = reg_rev_grouped.rolling(7, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0.0)
    regional_daily["regional_avg_freight_cost"] = reg_fr_grouped.rolling(30, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0.0)

    df = spine_df[["product_id", "region", "date"]].merge(
        regional_daily[["region", "date", "regional_daily_total_sales", "regional_avg_freight_cost"]],
        on=["region", "date"],
        how="left"
    ).merge(
        velocity_df[["product_id", "region", "date", "revenue_velocity_7d"]],
        on=["product_id", "region", "date"],
        how="left"
    )

    df["regional_daily_total_sales"] = df["regional_daily_total_sales"].fillna(0.0)
    df["revenue_velocity_7d"] = df["revenue_velocity_7d"].fillna(0.0)

    df["regional_sales_share_7d"] = np.where(
        df["regional_daily_total_sales"] > 0,
        df["revenue_velocity_7d"] / (df["regional_daily_total_sales"] + 1e-5),
        0.0
    )

    region_rank_map = {
        "DE": 1, "US": 2, "BR_SP": 3, "BR_RJ": 4, "BR_MG": 5,
        "Europe": 6, "LATAM": 7, "APAC": 8
    }
    df["regional_market_rank"] = df["region"].map(region_rank_map).fillna(9).astype(int)

    cols = [
        "product_id", "region", "date",
        "regional_daily_total_sales", "regional_sales_share_7d",
        "regional_avg_freight_cost", "regional_market_rank"
    ]
    logger.info(f"Regional Patterns features complete: shape={df[cols].shape}")
    return df[cols]
