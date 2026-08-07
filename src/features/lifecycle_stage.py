import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger("features.lifecycle_stage")


def compute_lifecycle_stage_features(
    sales_fact: pd.DataFrame,
    spine_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes product lifecycle stage and age metrics.
    Keys: (product_id, region, date)
    """
    logger.info("Computing Lifecycle Stage features...")

    # Calculate first sale date per (product_id, region)
    first_sales = sales_fact.groupby(["product_id", "region"])["date"].min().reset_index()
    first_sales.rename(columns={"date": "first_sale_date"}, inplace=True)

    df = spine_df[["product_id", "region", "date"]].merge(
        first_sales, on=["product_id", "region"], how="left"
    )

    df["first_sale_date"] = pd.to_datetime(df["first_sale_date"])
    df["date"] = pd.to_datetime(df["date"])

    # Days elapsed since first sale (causal computation based on current observation date)
    days_elapsed = (df["date"] - df["first_sale_date"]).dt.days.fillna(0).clip(lower=0)
    df["days_since_first_sale"] = days_elapsed

    df["product_age_days"] = df["days_since_first_sale"] + 30

    # Categorical lifecycle stage code:
    # 0: Introduction (< 30 days)
    # 1: Growth (30 - 180 days)
    # 2: Maturity (180 - 730 days)
    # 3: Decline (>= 730 days)
    conditions = [
        df["days_since_first_sale"] < 30,
        (df["days_since_first_sale"] >= 30) & (df["days_since_first_sale"] < 180),
        (df["days_since_first_sale"] >= 180) & (df["days_since_first_sale"] < 730),
        df["days_since_first_sale"] >= 730
    ]
    choices = [0, 1, 2, 3]
    df["lifecycle_stage_code"] = np.select(conditions, choices, default=2)

    df["lifecycle_maturity_ratio"] = df["days_since_first_sale"] / (df["days_since_first_sale"] + 365.0)

    cols = [
        "product_id", "region", "date",
        "days_since_first_sale", "product_age_days",
        "lifecycle_stage_code", "lifecycle_maturity_ratio"
    ]
    logger.info(f"Lifecycle Stage features complete: shape={df[cols].shape}")
    return df[cols]
