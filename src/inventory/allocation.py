import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from src.utils.logging_config import get_logger

logger = get_logger("inventory.allocation")


def calculate_warehouse_fulfillment_shares(
    sales_fact: pd.DataFrame,
    warehouse_dim: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Computes historical fulfillment shares per (product_id, warehouse_id) and (region, warehouse_id).
    
    Returns a DataFrame with columns:
    ['product_id', 'region', 'warehouse_id', 'fulfillment_share']
    """
    df = sales_fact.copy()
    
    # Merge region if not present in sales_fact
    if "region" not in df.columns and warehouse_dim is not None and "region" in warehouse_dim.columns:
        df = df.merge(warehouse_dim[["warehouse_id", "region"]], on="warehouse_id", how="left")

    if "quantity" in df.columns:
        vol_col = "quantity"
    elif "actual_sales" in df.columns:
        vol_col = "actual_sales"
    else:
        vol_col = df.select_dtypes(include=[np.number]).columns[0]

    # Aggregate total volume per (product_id, region, warehouse_id)
    group_cols = ["product_id", "region", "warehouse_id"]
    group_cols = [c for c in group_cols if c in df.columns]

    vol_by_wh = df.groupby(group_cols)[vol_col].sum().reset_index()

    # Total volume per product (and region)
    total_group_cols = [c for c in ["product_id", "region"] if c in vol_by_wh.columns]
    vol_total = vol_by_wh.groupby(total_group_cols)[vol_col].transform("sum")

    vol_by_wh["fulfillment_share"] = np.where(vol_total > 0, vol_by_wh[vol_col] / vol_total, 0.0)

    logger.info(f"Calculated historical fulfillment shares across {len(vol_by_wh)} product-warehouse pairs.")
    return vol_by_wh


def allocate_demand_to_warehouses(
    forecast_df: pd.DataFrame,
    sales_fact: pd.DataFrame,
    warehouse_dim: pd.DataFrame,
    forecast_demand_col: str = "actual_sales"
) -> pd.DataFrame:
    """
    Allocates regional or product-level forecasted demand across individual warehouses
    proportional to historical fulfillment share.

    If a product has no historical sales in a region, demand is allocated equally
    across all active warehouses in that region.

    Returns DataFrame with:
    ['product_id', 'region', 'warehouse_id', 'forecasted_daily_demand', 'allocated_daily_demand', 'fulfillment_share']
    """
    logger.info("Allocating forecasted demand across warehouses...")
    f_df = forecast_df.copy()
    
    if forecast_demand_col not in f_df.columns:
        # Fallback demand column
        num_cols = f_df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            forecast_demand_col = num_cols[0]
        else:
            raise ValueError(f"Forecast demand column '{forecast_demand_col}' not found in forecast DataFrame.")

    # 1. Compute historical fulfillment shares
    shares_df = calculate_warehouse_fulfillment_shares(sales_fact, warehouse_dim)

    # 2. Get active warehouses per region
    wh_per_region = warehouse_dim.groupby("region")["warehouse_id"].unique().to_dict()

    # 3. Merge forecast with fulfillment shares
    # Match on product_id and region if available
    merge_keys = [k for k in ["product_id", "region"] if k in f_df.columns and k in shares_df.columns]
    
    if merge_keys:
        merged = f_df.merge(shares_df, on=merge_keys, how="left")
    else:
        # Cross join fallback
        merged = f_df.merge(shares_df, how="cross")

    # For rows where warehouse_id is NaN (e.g. cold-start or new product with no historical shares),
    # expand across all active warehouses in the product's region with equal share
    missing_mask = merged["warehouse_id"].isna()
    if missing_mask.any():
        logger.info(f"Found {missing_mask.sum()} forecast rows with no historical warehouse share. Applying equal regional fallback allocation.")
        
        missing_rows = merged[missing_mask].drop(columns=["warehouse_id", "fulfillment_share"], errors="ignore")
        
        expanded_missing = []
        for _, row in missing_rows.iterrows():
            reg = row.get("region", None)
            wh_list = wh_per_region.get(reg, warehouse_dim["warehouse_id"].unique())
            if len(wh_list) == 0:
                wh_list = warehouse_dim["warehouse_id"].unique()
            
            eq_share = 1.0 / len(wh_list)
            for wh_id in wh_list:
                r_dict = row.to_dict()
                r_dict["warehouse_id"] = wh_id
                r_dict["fulfillment_share"] = eq_share
                expanded_missing.append(r_dict)
                
        valid_merged = merged[~missing_mask].copy()
        if expanded_missing:
            expanded_df = pd.DataFrame(expanded_missing)
            merged = pd.concat([valid_merged, expanded_df], ignore_index=True)

    # Re-normalize fulfillment shares per product-region to ensure sum(shares) == 1.0
    norm_group_cols = [c for c in ["product_id", "region"] if c in merged.columns]
    share_sum = merged.groupby(norm_group_cols)["fulfillment_share"].transform("sum")
    merged["fulfillment_share"] = np.where(share_sum > 0, merged["fulfillment_share"] / share_sum, 0.0)

    # Calculate allocated daily demand
    merged["forecasted_daily_demand"] = merged[forecast_demand_col].fillna(0.0).clip(lower=0.0)
    merged["allocated_daily_demand"] = np.round(merged["forecasted_daily_demand"] * merged["fulfillment_share"], 4)

    # Ensure required columns are present and no NaNs
    cols_to_keep = ["product_id", "region", "warehouse_id", "forecasted_daily_demand", "allocated_daily_demand", "fulfillment_share"]
    final_cols = [c for c in cols_to_keep if c in merged.columns]
    
    result_df = merged[final_cols].drop_duplicates(subset=["product_id", "warehouse_id"]).reset_index(drop=True)
    result_df["allocated_daily_demand"] = result_df["allocated_daily_demand"].fillna(0.0).clip(lower=0.0)
    result_df["fulfillment_share"] = result_df["fulfillment_share"].fillna(0.0).clip(lower=0.0)
    
    logger.info(f"Completed demand allocation across {len(result_df)} product-warehouse records.")
    return result_df
