import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.utils.logging_config import get_logger
from src.risk.supplier_delay_risk import classify_risk_level, load_risk_config

logger = get_logger("risk.slow_dead_inventory")


def classify_slow_dead_inventory(
    sales_velocity: float = 1.5,
    zero_sales_weeks: int = 0,
    current_stock: float = 100.0,
    unit_cost: float = 10.0,
    config_path: str = "configs/config.yaml"
) -> Dict[str, Any]:
    """
    Classifies products as HEALTHY, SLOW_MOVING, or DEAD_STOCK based on configurable
    velocity and zero-sales-week thresholds loaded from configs/config.yaml.

    Config thresholds:
    - low_velocity_threshold: default 1.0 unit/day
    - slow_moving_zero_sales_weeks: default 4 weeks
    - dead_inventory_zero_sales_weeks: default 12 weeks

    Returns:
    - Dict with classification status, bounded risk score [0.0, 1.0], risk level, aging days,
      holding value, and turnover rate.
    """
    cfg = load_risk_config(config_path)
    low_vel_thresh = cfg.get("low_velocity_threshold", 1.0)
    slow_weeks_thresh = cfg.get("slow_moving_zero_sales_weeks", 4)
    dead_weeks_thresh = cfg.get("dead_inventory_zero_sales_weeks", 12)

    v_sales = max(0.0, sales_velocity if sales_velocity is not None and not np.isnan(sales_velocity) else 0.0)
    z_weeks = max(0, zero_sales_weeks if zero_sales_weeks is not None and not np.isnan(zero_sales_weeks) else 0)
    c_stock = max(0.0, current_stock if current_stock is not None and not np.isnan(current_stock) else 0.0)
    u_cost = max(0.0, unit_cost if unit_cost is not None and not np.isnan(unit_cost) else 0.0)

    aging_days = z_weeks * 7
    holding_value = c_stock * u_cost

    # Classification logic
    if z_weeks >= dead_weeks_thresh:
        status = "DEAD_STOCK"
        extra_weeks = z_weeks - dead_weeks_thresh
        risk_score = 0.75 + 0.25 * min(1.0, extra_weeks / 12.0)
    elif z_weeks >= slow_weeks_thresh or v_sales < low_vel_thresh:
        status = "SLOW_MOVING"
        if z_weeks >= slow_weeks_thresh:
            progress = (z_weeks - slow_weeks_thresh) / max(1, dead_weeks_thresh - slow_weeks_thresh)
            risk_score = 0.25 + 0.50 * min(1.0, progress)
        else:
            vel_ratio = (low_vel_thresh - v_sales) / max(1e-6, low_vel_thresh)
            risk_score = 0.25 + 0.25 * min(1.0, vel_ratio)
    else:
        status = "HEALTHY"
        risk_score = 0.0

    score = float(np.round(np.clip(risk_score, 0.0, 1.0), 4))
    level = classify_risk_level(score)

    annual_demand = v_sales * 365.0
    turnover_rate = float(np.round(annual_demand / max(1.0, c_stock), 2))

    return {
        "inventory_health_status": status,
        "slow_dead_risk_score": score,
        "slow_dead_risk_level": level,
        "aging_days": aging_days,
        "holding_value": float(np.round(holding_value, 2)),
        "inventory_turnover_rate": turnover_rate,
        "is_dead_stock": status == "DEAD_STOCK",
        "is_slow_moving": status == "SLOW_MOVING"
    }


def compute_slow_dead_inventory_df(
    df: pd.DataFrame,
    velocity_col: str = "sales_velocity",
    zero_weeks_col: str = "zero_sales_weeks",
    stock_col: str = "current_stock",
    cost_col: str = "unit_cost",
    config_path: str = "configs/config.yaml",
    output_col: str = "slow_dead_risk_score"
) -> pd.DataFrame:
    """
    Vectorized classification of slow-moving and dead inventory for a pandas DataFrame.
    """
    df_copy = df.copy()

    vel = df_copy[velocity_col].fillna(1.0).clip(lower=0.0) if velocity_col in df_copy.columns else (
        df_copy["avg_daily_demand"].fillna(1.0).clip(lower=0.0) if "avg_daily_demand" in df_copy.columns else pd.Series(1.0, index=df_copy.index)
    )
    z_weeks = df_copy[zero_weeks_col].fillna(0).astype(int).clip(lower=0) if zero_weeks_col in df_copy.columns else pd.Series(0, index=df_copy.index)
    c_stock = df_copy[stock_col].fillna(0.0).clip(lower=0.0) if stock_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    cost = df_copy[cost_col].fillna(10.0).clip(lower=0.0) if cost_col in df_copy.columns else pd.Series(10.0, index=df_copy.index)

    scores, statuses, levels, aging_list, turnover_list = [], [], [], [], []
    for v, z, s, c in zip(vel, z_weeks, c_stock, cost):
        res = classify_slow_dead_inventory(v, z, s, c, config_path=config_path)
        scores.append(res["slow_dead_risk_score"])
        statuses.append(res["inventory_health_status"])
        levels.append(res["slow_dead_risk_level"])
        aging_list.append(res["aging_days"])
        turnover_list.append(res["inventory_turnover_rate"])

    df_copy[output_col] = scores
    df_copy["inventory_health_status"] = statuses
    df_copy["slow_dead_risk_level"] = levels
    df_copy["aging_days"] = aging_list
    df_copy["inventory_turnover_rate"] = turnover_list

    return df_copy
