import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.utils.logging_config import get_logger
from src.risk.supplier_delay_risk import classify_risk_level, load_risk_config

logger = get_logger("risk.overstock_risk")


def calculate_overstock_risk(
    current_stock: float,
    reorder_point: float,
    unit_cost: float = 10.0,
    sales_velocity: float = 1.0,
    overstock_rop_multiplier: Optional[float] = None,
    low_velocity_threshold: Optional[float] = None,
    config_path: str = "configs/config.yaml"
) -> Tuple[float, float, float, bool]:
    """
    Calculates overstock risk score bounded in [0.0, 1.0], excess units, tied-up capital,
    and overstock flag.

    Inputs:
    - current_stock: Available inventory on hand
    - reorder_point: Reorder Point (ROP) threshold
    - unit_cost: Cost per unit ($)
    - sales_velocity: Sales velocity (units per day)
    - overstock_rop_multiplier: Multiplier threshold N above ROP (loaded from config.yaml, default 3.0)
    - low_velocity_threshold: Sales velocity threshold (loaded from config.yaml, default 1.0)

    Logic:
    - Overstock condition flagged when current_stock > N * reorder_point and sales_velocity < low_velocity_threshold (or low turn).
    - Returns Tuple[overstock_risk_score, excess_inventory_units, tied_up_capital, is_overstocked_flag]
    """
    cfg = load_risk_config(config_path)
    if overstock_rop_multiplier is None:
        overstock_rop_multiplier = cfg.get("overstock_rop_multiplier", 3.0)
    if low_velocity_threshold is None:
        low_velocity_threshold = cfg.get("low_velocity_threshold", 1.0)

    if current_stock is None or np.isnan(current_stock) or current_stock < 0:
        current_stock = 0.0
    c_stock = float(current_stock)

    if sales_velocity is None or np.isnan(sales_velocity) or sales_velocity < 0:
        sales_velocity = 0.0
    s_vel = float(sales_velocity)

    if reorder_point is None or np.isnan(reorder_point) or reorder_point <= 0:
        reorder_point = max(1.0, s_vel * 14.0)
    rop = float(reorder_point)

    if unit_cost is None or np.isnan(unit_cost) or unit_cost < 0:
        unit_cost = 0.0
    u_cost = float(unit_cost)

    threshold = overstock_rop_multiplier * rop

    # Check if stock exceeds N * ROP
    is_exceeding_threshold = c_stock > threshold
    is_low_velocity = s_vel <= low_velocity_threshold

    # Flag condition: stock exceeds N * ROP with low velocity or high excess ratio
    is_overstocked = is_exceeding_threshold or (c_stock > (2.0 * rop) and is_low_velocity)

    if not is_overstocked:
        if c_stock > rop:
            ratio = (c_stock - rop) / max(1e-6, threshold - rop)
            risk = 0.25 * min(1.0, ratio)
        else:
            risk = 0.0
        return float(np.round(np.clip(risk, 0.0, 0.25), 4)), 0.0, 0.0, False

    excess_units = max(0.0, c_stock - threshold) if is_exceeding_threshold else max(0.0, c_stock - (2.0 * rop))
    tied_up_capital = excess_units * u_cost

    # Risk score scaling
    overstock_ratio = (c_stock - rop) / max(1.0, rop)
    velocity_penalty = 1.0 if is_low_velocity else 0.7
    risk = 0.25 + 0.75 * min(1.0, (overstock_ratio / max(1.0, overstock_rop_multiplier)) * velocity_penalty)

    return float(np.round(np.clip(risk, 0.0, 1.0), 4)), float(np.round(excess_units, 2)), float(np.round(tied_up_capital, 2)), True


def evaluate_overstock_details(
    current_stock: float,
    reorder_point: float,
    unit_cost: float = 10.0,
    sales_velocity: float = 1.0,
    avg_daily_demand: Optional[float] = None,
    overstock_rop_multiplier: Optional[float] = None,
    config_path: str = "configs/config.yaml"
) -> Dict[str, Any]:
    """
    Returns detailed overstock risk metrics.
    """
    s_vel = sales_velocity if sales_velocity is not None else (avg_daily_demand or 1.0)
    score, excess_units, tied_cap, is_over = calculate_overstock_risk(
        current_stock=current_stock,
        reorder_point=reorder_point,
        unit_cost=unit_cost,
        sales_velocity=s_vel,
        overstock_rop_multiplier=overstock_rop_multiplier,
        config_path=config_path
    )
    level = classify_risk_level(score)

    return {
        "overstock_risk_score": score,
        "overstock_risk_level": level,
        "excess_inventory_units": excess_units,
        "tied_up_capital": tied_cap,
        "is_overstocked": is_over
    }


def compute_overstock_risk_df(
    df: pd.DataFrame,
    stock_col: str = "current_stock",
    rop_col: str = "reorder_point",
    cost_col: str = "unit_cost",
    velocity_col: str = "sales_velocity",
    demand_col: Optional[str] = None,
    config_path: str = "configs/config.yaml",
    output_col: str = "overstock_risk_score"
) -> pd.DataFrame:
    """
    Vectorized calculation of overstock risk for a pandas DataFrame.
    """
    df_copy = df.copy()

    v_col = velocity_col if velocity_col in df_copy.columns else (demand_col if demand_col and demand_col in df_copy.columns else "avg_daily_demand")

    c_stock = df_copy[stock_col].fillna(0.0).clip(lower=0.0) if stock_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    rop = df_copy[rop_col].fillna(10.0).clip(lower=1.0) if rop_col in df_copy.columns else pd.Series(10.0, index=df_copy.index)
    cost = df_copy[cost_col].fillna(10.0).clip(lower=0.0) if cost_col in df_copy.columns else pd.Series(10.0, index=df_copy.index)
    vel = df_copy[v_col].fillna(1.0).clip(lower=0.0) if v_col in df_copy.columns else pd.Series(1.0, index=df_copy.index)

    scores, excess_list, capital_list, flags = [], [], [], []
    for s, r, c, v in zip(c_stock, rop, cost, vel):
        score, excess, cap, is_over = calculate_overstock_risk(s, r, c, v, config_path=config_path)
        scores.append(score)
        excess_list.append(excess)
        capital_list.append(cap)
        flags.append(is_over)

    df_copy[output_col] = scores
    df_copy["overstock_risk_level"] = [classify_risk_level(v) for v in scores]
    df_copy["excess_inventory_units"] = excess_list
    df_copy["tied_up_capital"] = capital_list
    df_copy["is_overstocked"] = flags

    return df_copy
