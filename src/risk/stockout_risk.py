import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from scipy.stats import norm

from src.utils.logging_config import get_logger
from src.risk.supplier_delay_risk import classify_risk_level, load_risk_config

logger = get_logger("risk.stockout_risk")


def calculate_stockout_risk(
    current_stock: float,
    reorder_point: float,
    safety_stock: float,
    avg_daily_demand: float = 1.0,
    std_daily_demand: float = 0.2,
    lead_time_days: float = 7.0,
    lead_time_std: float = 1.0
) -> float:
    """
    Calculates stockout probability & risk score bounded strictly in [0.0, 1.0].
    Evaluates probability that on-hand inventory falls below forecasted demand before next delivery.

    Formulation:
    - Expected Lead Time Demand = lead_time_days * avg_daily_demand
    - Lead Time Demand Variance = (lead_time_days * std_daily_demand^2) + (avg_daily_demand^2 * lead_time_std^2)
    - Z = (current_stock - Expected Lead Time Demand) / std_lt_demand
    - Stockout Probability = 1.0 - norm.cdf(Z)
    """
    # 1. Input sanitization
    if current_stock is None or np.isnan(current_stock):
        current_stock = 0.0
    c_stock = float(current_stock)

    if reorder_point is None or np.isnan(reorder_point) or reorder_point < 0:
        reorder_point = 0.0
    rop = float(reorder_point)

    if safety_stock is None or np.isnan(safety_stock) or safety_stock < 0:
        safety_stock = 0.0
    ss = float(safety_stock)

    if avg_daily_demand is None or np.isnan(avg_daily_demand) or avg_daily_demand < 0:
        avg_daily_demand = 0.0
    mu_d = float(avg_daily_demand)

    if std_daily_demand is None or np.isnan(std_daily_demand) or std_daily_demand < 0:
        std_daily_demand = max(0.1, mu_d * 0.2)
    sigma_d = float(std_daily_demand)

    if lead_time_days is None or np.isnan(lead_time_days) or lead_time_days <= 0:
        lead_time_days = 7.0
    lt_days = float(lead_time_days)

    if lead_time_std is None or np.isnan(lead_time_std) or lead_time_std < 0:
        lead_time_std = 0.0
    sigma_l = float(lead_time_std)

    # Out of stock condition
    if c_stock <= 0:
        return 1.0

    # 2. Compute Lead Time Demand Distribution Parameters
    expected_lt_demand = lt_days * mu_d
    lt_variance = (lt_days * (sigma_d ** 2)) + ((mu_d ** 2) * (sigma_l ** 2))
    sigma_lt = np.sqrt(max(1e-6, lt_variance))

    # 3. Calculate Z-score and Stockout Probability
    z_score = (c_stock - expected_lt_demand) / sigma_lt
    prob_stockout = float(1.0 - norm.cdf(z_score))

    # 4. Integrate Safety Stock and ROP Breaches
    if ss > 0 and c_stock <= ss:
        prob_stockout = max(prob_stockout, 0.75 + 0.25 * (1.0 - c_stock / ss))
    elif rop > ss and c_stock <= rop:
        range_rop = rop - ss
        pos_in_range = (c_stock - ss) / max(1e-6, range_rop)
        prob_stockout = max(prob_stockout, 0.25 + 0.50 * (1.0 - pos_in_range))

    return float(np.round(np.clip(prob_stockout, 0.0, 1.0), 4))


def evaluate_stockout_details(
    current_stock: float,
    reorder_point: float,
    safety_stock: float,
    avg_daily_demand: float = 1.0,
    std_daily_demand: float = 0.2,
    lead_time_days: float = 7.0,
    lead_time_std: float = 1.0
) -> Dict[str, Any]:
    """
    Returns detailed stockout risk analysis.
    """
    score = calculate_stockout_risk(
        current_stock=current_stock,
        reorder_point=reorder_point,
        safety_stock=safety_stock,
        avg_daily_demand=avg_daily_demand,
        std_daily_demand=std_daily_demand,
        lead_time_days=lead_time_days,
        lead_time_std=lead_time_std
    )
    level = classify_risk_level(score)

    c_stock = max(0.0, current_stock or 0.0)
    a_demand = max(0.001, avg_daily_demand or 1.0)
    days_of_supply = float(np.round(c_stock / a_demand, 2))

    return {
        "stockout_risk_score": score,
        "stockout_risk_level": level,
        "days_of_supply": days_of_supply,
        "is_safety_stock_breached": bool(c_stock <= (safety_stock or 0.0)),
        "is_reorder_point_breached": bool(c_stock <= (reorder_point or 0.0)),
        "is_out_of_stock": bool(c_stock <= 0)
    }


def compute_stockout_risk_df(
    df: pd.DataFrame,
    stock_col: str = "current_stock",
    rop_col: str = "reorder_point",
    ss_col: str = "safety_stock",
    demand_col: str = "avg_daily_demand",
    std_demand_col: str = "std_daily_demand",
    lead_time_col: str = "avg_lead_time",
    std_lead_time_col: str = "lead_time_std_days",
    output_col: str = "stockout_risk_score"
) -> pd.DataFrame:
    """
    Vectorized calculation of stockout risk for a pandas DataFrame.
    """
    df_copy = df.copy()

    c_stock = df_copy[stock_col].fillna(0.0) if stock_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    rop = df_copy[rop_col].fillna(0.0).clip(lower=0.0) if rop_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    ss = df_copy[ss_col].fillna(0.0).clip(lower=0.0) if ss_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    demand = df_copy[demand_col].fillna(1.0).clip(lower=0.001) if demand_col in df_copy.columns else pd.Series(1.0, index=df_copy.index)
    std_demand = df_copy[std_demand_col].fillna(0.2).clip(lower=0.0) if std_demand_col in df_copy.columns else (demand * 0.2)
    lt = df_copy[lead_time_col].fillna(7.0).clip(lower=1.0) if lead_time_col in df_copy.columns else pd.Series(7.0, index=df_copy.index)
    std_lt = df_copy[std_lead_time_col].fillna(1.0).clip(lower=0.0) if std_lead_time_col in df_copy.columns else pd.Series(1.0, index=df_copy.index)

    scores = []
    for s, r, ss_v, d, sd, l, sl in zip(c_stock, rop, ss, demand, std_demand, lt, std_lt):
        scores.append(calculate_stockout_risk(s, r, ss_v, d, sd, l, sl))

    df_copy[output_col] = scores
    df_copy["stockout_risk_level"] = [classify_risk_level(v) for v in scores]
    df_copy["days_of_supply"] = np.round(c_stock / demand, 2)

    return df_copy
