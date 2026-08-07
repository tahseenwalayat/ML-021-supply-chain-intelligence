import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy.stats import norm

from src.utils.logging_config import get_logger

logger = get_logger("inventory.safety_stock")


def get_z_score(service_level: float = 0.95) -> float:
    """
    Computes the z-score corresponding to the target service level.
    If service_level is outside (0, 1), logs a warning and defaults to 0.95 (Z ≈ 1.645).
    """
    if service_level <= 0.0 or service_level >= 1.0:
        logger.warning(
            f"Invalid service level '{service_level}' specified. Must be strictly between 0 and 1. "
            f"Falling back to default service level of 0.95."
        )
        service_level = 0.95
    return float(norm.ppf(service_level))


def calculate_safety_stock(
    avg_daily_demand: float,
    std_daily_demand: float,
    avg_lead_time: float,
    std_lead_time: float = 0.0,
    service_level: float = 0.95,
    default_lead_time: float = 7.0
) -> float:
    """
    Calculates safety stock using the standard supply chain formula:
    Safety Stock = Z * std_demand_during_lead_time
    
    where:
    std_demand_during_lead_time = sqrt( avg_lead_time * std_daily_demand^2 + avg_daily_demand^2 * std_lead_time^2 )

    Handles missing/invalid lead times, zero-variance demand, and negative inputs gracefully.
    Guarantees Safety Stock >= 0.0.
    """
    # 1. Clean & validate inputs
    if avg_daily_demand is None or np.isnan(avg_daily_demand) or avg_daily_demand < 0:
        avg_daily_demand = 0.0

    if std_daily_demand is None or np.isnan(std_daily_demand) or std_daily_demand < 0:
        std_daily_demand = 0.0

    if avg_lead_time is None or np.isnan(avg_lead_time) or avg_lead_time <= 0:
        logger.warning(
            f"Invalid or missing avg_lead_time ({avg_lead_time}). "
            f"Falling back to default lead time of {default_lead_time} days."
        )
        avg_lead_time = default_lead_time

    if std_lead_time is None or np.isnan(std_lead_time) or std_lead_time < 0:
        std_lead_time = 0.0

    # 2. Zero demand variance case
    if std_daily_demand == 0.0 and std_lead_time == 0.0:
        return 0.0

    # 3. Z-score calculation
    z = get_z_score(service_level)

    # 4. Standard deviation during lead time
    variance_lt = (avg_lead_time * (std_daily_demand ** 2)) + ((avg_daily_demand ** 2) * (std_lead_time ** 2))
    std_lt = np.sqrt(max(0.0, variance_lt))

    # 5. Safety Stock calculation & non-negativity constraint
    safety_stock = float(z * std_lt)
    return max(0.0, float(np.round(safety_stock, 4)))


def compute_safety_stock_df(
    df: pd.DataFrame,
    avg_demand_col: str = "avg_daily_demand",
    std_demand_col: str = "std_daily_demand",
    lead_time_col: str = "avg_lead_time",
    std_lead_time_col: str = "std_lead_time",
    service_level: float = 0.95,
    default_lead_time: float = 7.0,
    output_col: str = "safety_stock"
) -> pd.DataFrame:
    """
    Vectorized computation of safety stock for a pandas DataFrame.
    Guarantees no NaN/nulls and enforces non-negative values.
    """
    df_copy = df.copy()
    
    d_avg = df_copy[avg_demand_col].fillna(0.0).clip(lower=0.0) if avg_demand_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    d_std = df_copy[std_demand_col].fillna(0.0).clip(lower=0.0) if std_demand_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    
    if lead_time_col in df_copy.columns:
        lt_avg = df_copy[lead_time_col].apply(lambda x: default_lead_time if pd.isna(x) or x <= 0 else float(x))
    else:
        lt_avg = pd.Series(default_lead_time, index=df_copy.index)

    if std_lead_time_col in df_copy.columns:
        lt_std = df_copy[std_lead_time_col].fillna(0.0).clip(lower=0.0)
    else:
        lt_std = pd.Series(0.0, index=df_copy.index)

    z = get_z_score(service_level)
    var_lt = (lt_avg * (d_std ** 2)) + ((d_avg ** 2) * (lt_std ** 2))
    std_lt = np.sqrt(np.maximum(0.0, var_lt))
    
    ss_values = np.maximum(0.0, z * std_lt)
    df_copy[output_col] = np.round(ss_values, 4)
    return df_copy
