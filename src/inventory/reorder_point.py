import numpy as np
import pandas as pd
from typing import Union, Optional

from src.utils.logging_config import get_logger

logger = get_logger("inventory.reorder_point")


def calculate_reorder_point(
    avg_daily_demand: float,
    avg_lead_time: float,
    safety_stock: float,
    default_lead_time: float = 7.0
) -> float:
    """
    Calculates the Reorder Point (ROP):
    ROP = (avg_daily_demand * avg_lead_time) + safety_stock

    Handles invalid/missing inputs and enforces ROP >= 0.0 constraint.
    """
    # 1. Clean & validate inputs
    if avg_daily_demand is None or np.isnan(avg_daily_demand) or avg_daily_demand < 0:
        avg_daily_demand = 0.0

    if avg_lead_time is None or np.isnan(avg_lead_time) or avg_lead_time <= 0:
        logger.warning(
            f"Invalid or missing avg_lead_time ({avg_lead_time}). "
            f"Falling back to default lead time of {default_lead_time} days."
        )
        avg_lead_time = default_lead_time

    if safety_stock is None or np.isnan(safety_stock) or safety_stock < 0:
        safety_stock = 0.0

    # 2. ROP calculation & non-negativity constraint
    lead_time_demand = avg_daily_demand * avg_lead_time
    rop = float(lead_time_demand + safety_stock)
    return max(0.0, float(np.round(rop, 4)))


def compute_reorder_point_df(
    df: pd.DataFrame,
    avg_demand_col: str = "avg_daily_demand",
    lead_time_col: str = "avg_lead_time",
    safety_stock_col: str = "safety_stock",
    default_lead_time: float = 7.0,
    output_col: str = "reorder_point"
) -> pd.DataFrame:
    """
    Vectorized computation of Reorder Point for a pandas DataFrame.
    Guarantees no NaN/nulls and enforces non-negative values.
    """
    df_copy = df.copy()

    d_avg = df_copy[avg_demand_col].fillna(0.0).clip(lower=0.0) if avg_demand_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    ss = df_copy[safety_stock_col].fillna(0.0).clip(lower=0.0) if safety_stock_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)

    if lead_time_col in df_copy.columns:
        lt_avg = df_copy[lead_time_col].apply(lambda x: default_lead_time if pd.isna(x) or x <= 0 else float(x))
    else:
        lt_avg = pd.Series(default_lead_time, index=df_copy.index)

    rop_values = np.maximum(0.0, (d_avg * lt_avg) + ss)
    df_copy[output_col] = np.round(rop_values, 4)
    return df_copy
