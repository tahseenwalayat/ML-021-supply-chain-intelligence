import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger("features.supplier_reliability")


def compute_supplier_reliability_features(
    sales_fact: pd.DataFrame,
    supplier_dim: pd.DataFrame,
    spine_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes supplier reliability, ratings, and order volume metrics.
    Keys: (product_id, region, date)
    """
    logger.info("Computing Supplier Reliability features...")

    # Map supplier metadata via sales_fact
    prod_supp = sales_fact.groupby(["product_id", "region"]).agg(
        supplier_id=("supplier_id", "first")
    ).reset_index()

    supp_merged = prod_supp.merge(supplier_dim, on="supplier_id", how="left", suffixes=("", "_supp"))

    df = spine_df[["product_id", "region", "date"]].merge(
        supp_merged[["product_id", "region", "supplier_id", "reliability_score", "lead_time_days"]],
        on=["product_id", "region"],
        how="left"
    )

    df["supplier_reliability_score"] = df["reliability_score"].fillna(4.5).astype(float)
    df["supplier_lead_time_days"] = df["lead_time_days"].fillna(4).astype(int)

    # Fulfillment rate estimate derived from reliability score
    df["supplier_fulfillment_rate"] = np.clip(df["supplier_reliability_score"] / 5.0, 0.70, 0.99)

    cols = [
        "product_id", "region", "date",
        "supplier_id", "supplier_reliability_score",
        "supplier_lead_time_days", "supplier_fulfillment_rate"
    ]
    logger.info(f"Supplier Reliability features complete: shape={df[cols].shape}")
    return df[cols]
