import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.utils.logging_config import get_logger

logger = get_logger("inventory.eoq")


def load_inventory_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads inventory configuration parameters from config.yaml.
    Falls back to safe defaults if config file is missing or invalid.
    """
    defaults = {
        "service_level": 0.95,
        "default_lead_time_days": 7.0,
        "ordering_cost": 50.0,
        "holding_cost_rate": 0.20,
        "holding_cost_fixed": 2.0
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg and "inventory" in cfg:
                    defaults.update(cfg["inventory"])
        except Exception as e:
            logger.warning(f"Failed to read inventory configuration from '{config_path}' ({e}). Using default values.")
    else:
        logger.info(f"Config file '{config_path}' not found. Using default inventory settings.")

    return defaults


def calculate_eoq(
    annual_demand: float,
    ordering_cost: float = 50.0,
    holding_cost: float = 2.0,
    unit_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None
) -> float:
    """
    Calculates Economic Order Quantity (EOQ):
    EOQ = sqrt( (2 * D * S) / H )

    where:
    - D: Annual demand (units per year)
    - S: Ordering cost per order ($)
    - H: Holding cost per unit per year ($). If unit_cost & holding_cost_rate are provided, H = unit_cost * holding_cost_rate.

    Handles zero demand, invalid holding costs, and negative inputs gracefully.
    Guarantees EOQ >= 0.0.
    """
    # 1. Clean & validate inputs
    if annual_demand is None or np.isnan(annual_demand) or annual_demand <= 0:
        return 0.0

    if ordering_cost is None or np.isnan(ordering_cost) or ordering_cost < 0:
        logger.warning(f"Invalid ordering_cost ({ordering_cost}). Defaulting to 50.0.")
        ordering_cost = 50.0

    # Determine effective holding cost H
    if unit_cost is not None and not np.isnan(unit_cost) and unit_cost > 0 and holding_cost_rate is not None and holding_cost_rate > 0:
        h_eff = float(unit_cost * holding_cost_rate)
    else:
        h_eff = float(holding_cost) if holding_cost is not None and not np.isnan(holding_cost) else 2.0

    if h_eff <= 0:
        logger.warning(f"Calculated holding cost H={h_eff} is non-positive. Falling back to default holding cost 2.0.")
        h_eff = 2.0

    # 2. Compute EOQ & non-negativity constraint
    eoq_val = np.sqrt((2.0 * annual_demand * ordering_cost) / h_eff)
    return max(0.0, float(np.round(eoq_val, 4)))


def compute_eoq_df(
    df: pd.DataFrame,
    daily_demand_col: str = "avg_daily_demand",
    annual_demand_col: Optional[str] = None,
    unit_cost_col: Optional[str] = "unit_cost",
    config_path: str = "configs/config.yaml",
    output_col: str = "eoq"
) -> pd.DataFrame:
    """
    Vectorized computation of EOQ for a pandas DataFrame.
    Guarantees no NaN/nulls and enforces non-negative values.
    """
    df_copy = df.copy()
    cfg = load_inventory_config(config_path)
    
    s = cfg.get("ordering_cost", 50.0)
    h_rate = cfg.get("holding_cost_rate", 0.20)
    h_fixed = cfg.get("holding_cost_fixed", 2.0)

    # 1. Determine annual demand D
    if annual_demand_col and annual_demand_col in df_copy.columns:
        d_annual = df_copy[annual_demand_col].fillna(0.0).clip(lower=0.0)
    elif daily_demand_col in df_copy.columns:
        d_annual = (df_copy[daily_demand_col].fillna(0.0).clip(lower=0.0)) * 365.0
    else:
        d_annual = pd.Series(0.0, index=df_copy.index)

    # 2. Determine effective holding cost H per row
    if unit_cost_col and unit_cost_col in df_copy.columns:
        u_cost = df_copy[unit_cost_col].fillna(0.0).clip(lower=0.0)
        h_eff = np.where(u_cost > 0, u_cost * h_rate, h_fixed)
    else:
        h_eff = pd.Series(h_fixed, index=df_copy.index)

    h_eff = np.maximum(1e-4, h_eff)  # Prevent division by zero

    # 3. Vectorized EOQ calculation
    eoq_vals = np.where(d_annual > 0, np.sqrt((2.0 * d_annual * s) / h_eff), 0.0)
    df_copy[output_col] = np.round(np.maximum(0.0, eoq_vals), 4)
    return df_copy
