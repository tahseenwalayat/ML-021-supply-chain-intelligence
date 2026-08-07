import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logging_config import get_logger

logger = get_logger("inventory.transfers")


def calculate_surplus_and_deficit(
    inventory_df: pd.DataFrame,
    stock_col: str = "current_stock",
    rop_col: str = "reorder_point",
    eoq_col: str = "eoq",
    safety_stock_col: str = "safety_stock"
) -> pd.DataFrame:
    """
    Computes surplus inventory and stockout deficit per product-warehouse:
    
    Surplus = max(0, current_stock - (rop + safety_stock))
    Deficit = max(0, rop - current_stock)
    """
    df = inventory_df.copy()
    
    stock = df[stock_col].fillna(0.0).clip(lower=0.0) if stock_col in df.columns else pd.Series(0.0, index=df.index)
    rop = df[rop_col].fillna(0.0).clip(lower=0.0) if rop_col in df.columns else pd.Series(0.0, index=df.index)
    ss = df[safety_stock_col].fillna(0.0).clip(lower=0.0) if safety_stock_col in df.columns else pd.Series(0.0, index=df.index)

    # Surplus is stock above ROP + safety_stock buffer
    target_stock = rop + ss
    df["surplus_qty"] = np.maximum(0.0, stock - target_stock)
    df["deficit_qty"] = np.maximum(0.0, rop - stock)

    return df


def recommend_inter_warehouse_transfers(
    inventory_df: pd.DataFrame,
    product_col: str = "product_id",
    warehouse_col: str = "warehouse_id",
    region_col: str = "region",
    stock_col: str = "current_stock",
    rop_col: str = "reorder_point",
    eoq_col: str = "eoq",
    safety_stock_col: str = "safety_stock"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identifies surplus and deficit warehouses per product and generates inter-warehouse transfer recommendations.
    
    Prefers intra-region transfers first to minimize shipping costs, followed by cross-region transfers.

    Returns:
    1. recommendations_df: List of explicit transfer orders ['from_warehouse_id', 'to_warehouse_id', 'product_id', 'transfer_qty', 'reason']
    2. updated_inventory_df: Input inventory_df with added ['inward_transfer_qty', 'outward_transfer_qty', 'net_transfer_qty']
    """
    logger.info("Computing inter-warehouse transfer recommendations...")
    df = calculate_surplus_and_deficit(inventory_df, stock_col, rop_col, eoq_col, safety_stock_col)

    # Prepare tracking maps
    df["inward_transfer_qty"] = 0.0
    df["outward_transfer_qty"] = 0.0

    transfer_records = []

    # Group by product
    products = df[product_col].unique()

    for pid in products:
        p_mask = df[product_col] == pid
        p_df = df[p_mask].copy()

        surplus_nodes = p_df[p_df["surplus_qty"] > 0].copy()
        deficit_nodes = p_df[p_df["deficit_qty"] > 0].copy()

        if surplus_nodes.empty or deficit_nodes.empty:
            continue

        # 1. Intra-region matching first
        for d_idx, d_row in deficit_nodes.iterrows():
            d_wh = d_row[warehouse_col]
            d_reg = d_row.get(region_col, None)
            needed = d_row["deficit_qty"] - df.loc[d_idx, "inward_transfer_qty"]

            if needed <= 1e-4:
                continue

            # Find surplus warehouses in same region
            same_reg_surplus = surplus_nodes[surplus_nodes[region_col] == d_reg] if region_col in surplus_nodes.columns else pd.DataFrame()
            
            for s_idx in same_reg_surplus.index:
                s_wh = surplus_nodes.loc[s_idx, warehouse_col]
                avail = surplus_nodes.loc[s_idx, "surplus_qty"] - df.loc[s_idx, "outward_transfer_qty"]

                if avail <= 1e-4:
                    continue

                transfer_amt = float(np.round(min(needed, avail), 4))
                if transfer_amt > 0:
                    df.loc[d_idx, "inward_transfer_qty"] += transfer_amt
                    df.loc[s_idx, "outward_transfer_qty"] += transfer_amt
                    needed -= transfer_amt

                    transfer_records.append({
                        "from_warehouse_id": s_wh,
                        "to_warehouse_id": d_wh,
                        "product_id": pid,
                        "transfer_qty": transfer_amt,
                        "reason": f"Intra-region deficit mitigation (Region: {d_reg})"
                    })

                if needed <= 1e-4:
                    break

            # 2. Inter-region matching if deficit remains
            if needed > 1e-4:
                other_reg_surplus = surplus_nodes[surplus_nodes[region_col] != d_reg] if region_col in surplus_nodes.columns else surplus_nodes
                
                for s_idx in other_reg_surplus.index:
                    s_wh = surplus_nodes.loc[s_idx, warehouse_col]
                    s_reg = surplus_nodes.loc[s_idx].get(region_col, "Unknown")
                    avail = surplus_nodes.loc[s_idx, "surplus_qty"] - df.loc[s_idx, "outward_transfer_qty"]

                    if avail <= 1e-4:
                        continue

                    transfer_amt = float(np.round(min(needed, avail), 4))
                    if transfer_amt > 0:
                        df.loc[d_idx, "inward_transfer_qty"] += transfer_amt
                        df.loc[s_idx, "outward_transfer_qty"] += transfer_amt
                        needed -= transfer_amt

                        transfer_records.append({
                            "from_warehouse_id": s_wh,
                            "to_warehouse_id": d_wh,
                            "product_id": pid,
                            "transfer_qty": transfer_amt,
                            "reason": f"Inter-region surplus rebalancing ({s_reg} -> {d_reg})"
                        })

                    if needed <= 1e-4:
                        break

    transfers_df = pd.DataFrame(transfer_records)
    if transfers_df.empty:
        transfers_df = pd.DataFrame(columns=["from_warehouse_id", "to_warehouse_id", "product_id", "transfer_qty", "reason"])

    df["net_transfer_qty"] = df["inward_transfer_qty"] - df["outward_transfer_qty"]

    logger.info(f"Generated {len(transfers_df)} inter-warehouse transfer recommendations.")
    return transfers_df, df
