import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.utils.logging_config import get_logger

logger = get_logger("risk.supplier_delay_risk")


def load_risk_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads risk configuration thresholds and weights from configs/config.yaml.
    """
    defaults = {
        "overstock_rop_multiplier": 3.0,
        "low_velocity_threshold": 1.0,
        "slow_moving_zero_sales_weeks": 4,
        "dead_inventory_zero_sales_weeks": 12,
        "anomaly_z_score_threshold": 3.0,
        "supplier_weight_late_rate": 0.6,
        "supplier_weight_delay_var": 0.4
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg and "risk" in cfg:
                    defaults.update(cfg["risk"])
        except Exception as e:
            logger.warning(f"Failed to read config from '{config_path}': {e}. Falling back to defaults.")

    return defaults


def classify_risk_level(score: float) -> str:
    """
    Maps a risk score in [0.0, 1.0] to an interpretable risk category:
    - score >= 0.75: CRITICAL
    - score >= 0.50: HIGH
    - score >= 0.25: MEDIUM
    - score < 0.25: LOW
    """
    if score is None or np.isnan(score):
        return "LOW"
    s = float(np.clip(score, 0.0, 1.0))
    if s >= 0.75:
        return "CRITICAL"
    elif s >= 0.50:
        return "HIGH"
    elif s >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_supplier_delay_risk(
    reliability_score: float = 1.0,
    lead_time_std: float = 0.0,
    lead_time_avg: float = 7.0,
    late_delivery_rate: Optional[float] = None,
    w_late: Optional[float] = None,
    w_var: Optional[float] = None,
    config_path: str = "configs/config.yaml"
) -> float:
    """
    Calculates supplier delay risk score bounded in [0.0, 1.0].

    Inputs:
    - reliability_score: On-time reliability ratio [0.0, 1.0] (1.0 = 100% on-time)
    - lead_time_std: Standard deviation of delivery lead time in days
    - lead_time_avg: Mean delivery lead time in days
    - late_delivery_rate: Late delivery rate [0.0, 1.0]. Defaults to 1.0 - reliability_score.
    - w_late / w_var: Weights loaded dynamically from config file.

    Returns:
    - Bounded float score in [0.0, 1.0]. A perfect on-time record returns 0.0.
    """
    cfg = load_risk_config(config_path)
    if w_late is None:
        w_late = cfg.get("supplier_weight_late_rate", 0.6)
    if w_var is None:
        w_var = cfg.get("supplier_weight_delay_var", 0.4)

    # 1. Clean inputs
    if reliability_score is None or np.isnan(reliability_score):
        reliability_score = 1.0
    reliability_score = float(np.clip(reliability_score, 0.0, 1.0))

    if late_delivery_rate is None or np.isnan(late_delivery_rate):
        late_rate = 1.0 - reliability_score
    else:
        late_rate = float(np.clip(late_delivery_rate, 0.0, 1.0))

    if lead_time_std is None or np.isnan(lead_time_std) or lead_time_std < 0:
        lead_time_std = 0.0

    if lead_time_avg is None or np.isnan(lead_time_avg) or lead_time_avg <= 0:
        lead_time_avg = 7.0

    # 2. Variance Penalty Component
    var_component = float(np.clip(lead_time_std / max(1.0, lead_time_avg), 0.0, 1.0))

    # 3. Weighted Composite Score
    total_w = w_late + w_var
    if total_w <= 0:
        w_late_norm, w_var_norm = 0.6, 0.4
    else:
        w_late_norm = w_late / total_w
        w_var_norm = w_var / total_w

    risk_score = (w_late_norm * late_rate) + (w_var_norm * var_component)
    return float(np.round(np.clip(risk_score, 0.0, 1.0), 4))


def evaluate_supplier_delay_details(
    reliability_score: float = 1.0,
    lead_time_std: float = 0.0,
    lead_time_avg: float = 7.0,
    late_delivery_rate: Optional[float] = None,
    w_late: Optional[float] = None,
    w_var: Optional[float] = None,
    config_path: str = "configs/config.yaml"
) -> Dict[str, Any]:
    """
    Returns detailed supplier delay risk breakdown.
    """
    score = calculate_supplier_delay_risk(
        reliability_score=reliability_score,
        lead_time_std=lead_time_std,
        lead_time_avg=lead_time_avg,
        late_delivery_rate=late_delivery_rate,
        w_late=w_late,
        w_var=w_var,
        config_path=config_path
    )
    level = classify_risk_level(score)

    l_rate = max(0.0, late_delivery_rate if late_delivery_rate is not None else 1.0 - (reliability_score or 1.0))
    lt_std = max(0.0, lead_time_std or 0.0)
    lt_avg = max(1.0, lead_time_avg or 7.0)

    recommended_buffer = float(np.round(lt_std * 1.65 + (l_rate * lt_avg * 0.5), 2))

    return {
        "supplier_delay_risk_score": score,
        "supplier_delay_risk_level": level,
        "late_delivery_rate": float(np.round(l_rate, 4)),
        "lead_time_std_days": float(np.round(lt_std, 2)),
        "recommended_buffer_days": recommended_buffer,
        "is_high_risk": score >= 0.50
    }


def compute_supplier_delay_risk_df(
    df: pd.DataFrame,
    reliability_col: str = "supplier_reliability_score",
    std_lead_time_col: str = "lead_time_std_days",
    avg_lead_time_col: str = "avg_lead_time",
    config_path: str = "configs/config.yaml",
    output_col: str = "supplier_delay_risk_score"
) -> pd.DataFrame:
    """
    Vectorized calculation of supplier delay risk for a pandas DataFrame.
    """
    df_copy = df.copy()
    cfg = load_risk_config(config_path)
    w_late = cfg.get("supplier_weight_late_rate", 0.6)
    w_var = cfg.get("supplier_weight_delay_var", 0.4)

    rel = df_copy[reliability_col].fillna(1.0).clip(lower=0.0, upper=1.0) if reliability_col in df_copy.columns else pd.Series(1.0, index=df_copy.index)
    late_rate = 1.0 - rel

    std_lt = df_copy[std_lead_time_col].fillna(0.0).clip(lower=0.0) if std_lead_time_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    avg_lt = df_copy[avg_lead_time_col].fillna(7.0).clip(lower=1.0) if avg_lead_time_col in df_copy.columns else pd.Series(7.0, index=df_copy.index)

    var_component = np.clip(std_lt / np.maximum(1.0, avg_lt), 0.0, 1.0)

    total_w = w_late + w_var
    w_l_n = w_late / total_w if total_w > 0 else 0.6
    w_v_n = w_var / total_w if total_w > 0 else 0.4

    risk_vals = np.clip((w_l_n * late_rate) + (w_v_n * var_component), 0.0, 1.0)
    df_copy[output_col] = np.round(risk_vals, 4)
    df_copy["supplier_delay_risk_level"] = [classify_risk_level(v) for v in df_copy[output_col]]
    df_copy["recommended_buffer_days"] = np.round(std_lt * 1.65 + (late_rate * avg_lt * 0.5), 2)

    return df_copy
