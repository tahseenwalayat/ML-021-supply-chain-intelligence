import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.utils.logging_config import get_logger

logger = get_logger("inventory.procurement")


def calculate_procurement_quantity(
    current_stock: float,
    reorder_point: float,
    eoq: float,
    on_order: float = 0.0,
    inward_transfers: float = 0.0,
    outward_transfers: float = 0.0
) -> Tuple[float, str]:
    """
    Calculates final procurement order quantity for a single product-warehouse item:

    Inventory Position = current_stock + on_order + inward_transfers - outward_transfers

    If Inventory Position <= reorder_point:
        net_deficit = (reorder_point + eoq) - Inventory Position
        procurement_qty = max(eoq, net_deficit)
        status = "REORDER_REQUIRED"
    Else:
        procurement_qty = 0.0
        status = "STOCK_ADEQUATE"

    Enforces procurement_qty >= 0.0 constraint.
    """
    # 1. Clean & validate inputs
    c_stock = max(0.0, float(current_stock)) if current_stock is not None and not np.isnan(current_stock) else 0.0
    rop = max(0.0, float(reorder_point)) if reorder_point is not None and not np.isnan(reorder_point) else 0.0
    eoq_val = max(0.0, float(eoq)) if eoq is not None and not np.isnan(eoq) else 0.0
    o_order = max(0.0, float(on_order)) if on_order is not None and not np.isnan(on_order) else 0.0
    in_trans = max(0.0, float(inward_transfers)) if inward_transfers is not None and not np.isnan(inward_transfers) else 0.0
    out_trans = max(0.0, float(outward_transfers)) if outward_transfers is not None and not np.isnan(outward_transfers) else 0.0

    # 2. Compute Effective Inventory Position
    inv_position = float(c_stock + o_order + in_trans - out_trans)

    # 3. Order trigger evaluation
    if inv_position <= rop:
        net_deficit = (rop + eoq_val) - inv_position
        order_qty = max(eoq_val, net_deficit)
        status = "REORDER_REQUIRED"
    else:
        order_qty = 0.0
        status = "STOCK_ADEQUATE"

    final_procurement_qty = max(0.0, float(np.round(order_qty, 4)))
    return final_procurement_qty, status


def compute_procurement_recommendations_df(
    df: pd.DataFrame,
    stock_col: str = "current_stock",
    rop_col: str = "reorder_point",
    eoq_col: str = "eoq",
    on_order_col: str = "on_order",
    inward_transfer_col: str = "inward_transfer_qty",
    outward_transfer_col: str = "outward_transfer_qty",
    output_qty_col: str = "recommended_procurement_qty",
    output_status_col: str = "procurement_status"
) -> pd.DataFrame:
    """
    Vectorized calculation of net procurement recommendations for a pandas DataFrame.
    Guarantees no NaNs/nulls and enforces non-negative procurement values.
    """
    df_copy = df.copy()

    c_stock = df_copy[stock_col].fillna(0.0).clip(lower=0.0) if stock_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    rop = df_copy[rop_col].fillna(0.0).clip(lower=0.0) if rop_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    eoq_val = df_copy[eoq_col].fillna(0.0).clip(lower=0.0) if eoq_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    o_order = df_copy[on_order_col].fillna(0.0).clip(lower=0.0) if on_order_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    in_trans = df_copy[inward_transfer_col].fillna(0.0).clip(lower=0.0) if inward_transfer_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    out_trans = df_copy[outward_transfer_col].fillna(0.0).clip(lower=0.0) if outward_transfer_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)

    # Effective Inventory Position
    inv_pos = c_stock + o_order + in_trans - out_trans

    # Deficit calculation
    deficit = (rop + eoq_val) - inv_pos
    needed_qty = np.maximum(eoq_val, deficit)

    reorder_mask = inv_pos <= rop
    rec_qty = np.where(reorder_mask, np.maximum(0.0, needed_qty), 0.0)
    rec_status = np.where(reorder_mask, "REORDER_REQUIRED", "STOCK_ADEQUATE")

    df_copy[output_qty_col] = np.round(rec_qty, 4)
    df_copy[output_status_col] = rec_status

    logger.info(
        f"Procurement analysis complete. {np.sum(reorder_mask)} out of {len(df_copy)} "
        f"product-warehouse items require reorder."
    )
    return df_copy
