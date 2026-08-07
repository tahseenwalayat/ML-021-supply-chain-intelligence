import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.utils.logging_config import get_logger
from src.risk.supplier_delay_risk import classify_risk_level, load_risk_config

logger = get_logger("risk.demand_anomaly_risk")


def calculate_demand_anomaly_risk(
    demand_val: float,
    mean_demand: float = 100.0,
    std_demand: float = 15.0,
    anomaly_z_score_threshold: Optional[float] = None,
    config_path: str = "configs/config.yaml"
) -> Dict[str, Any]:
    """
    Evaluates demand value or forecast residual against historical statistics to detect demand anomalies (spikes/drops).

    Inputs:
    - demand_val: Observed or forecasted daily demand
    - mean_demand: Historical mean daily demand
    - std_demand: Historical standard deviation of daily demand
    - anomaly_z_score_threshold: Z-score cutoff for anomaly detection (default: 3.0)

    Returns:
    - Dict with z_score, is_anomaly, anomaly_type (DEMAND_SPIKE, DEMAND_DROP, NORMAL), demand_anomaly_risk_score, risk_level.
    """
    if anomaly_z_score_threshold is None:
        cfg = load_risk_config(config_path)
        anomaly_z_score_threshold = cfg.get("anomaly_z_score_threshold", 3.0)

    d_val = max(0.0, demand_val if demand_val is not None and not np.isnan(demand_val) else 0.0)
    m_val = max(0.0, mean_demand if mean_demand is not None and not np.isnan(mean_demand) else 100.0)
    s_val = max(1e-6, std_demand if std_demand is not None and not np.isnan(std_demand) else 15.0)

    z_score = (d_val - m_val) / s_val
    abs_z = abs(z_score)

    is_anomaly = abs_z >= anomaly_z_score_threshold

    if z_score >= anomaly_z_score_threshold:
        anomaly_type = "DEMAND_SPIKE"
    elif z_score <= -anomaly_z_score_threshold:
        anomaly_type = "DEMAND_DROP"
    else:
        anomaly_type = "NORMAL"

    # Risk score calculation
    if abs_z < anomaly_z_score_threshold:
        risk_score = 0.50 * (abs_z / anomaly_z_score_threshold)
    else:
        excess_z = abs_z - anomaly_z_score_threshold
        risk_score = 0.50 + 0.50 * min(1.0, excess_z / max(1.0, anomaly_z_score_threshold))

    score = float(np.round(np.clip(risk_score, 0.0, 1.0), 4))
    level = classify_risk_level(score)

    return {
        "demand_anomaly_risk_score": score,
        "demand_anomaly_risk_level": level,
        "z_score": float(np.round(z_score, 4)),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "excess_demand_spike": float(np.round(max(0.0, d_val - (m_val + anomaly_z_score_threshold * s_val)), 2))
    }


def compute_demand_anomaly_risk_df(
    df: pd.DataFrame,
    demand_col: str = "demand",
    mean_col: str = "mean_demand",
    std_col: str = "std_demand",
    config_path: str = "configs/config.yaml",
    output_col: str = "demand_anomaly_risk_score"
) -> pd.DataFrame:
    """
    Vectorized computation of demand anomaly risk for a pandas DataFrame.
    """
    df_copy = df.copy()

    d_series = df_copy[demand_col].fillna(0.0).clip(lower=0.0) if demand_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    m_series = df_copy[mean_col].fillna(100.0).clip(lower=0.0) if mean_col in df_copy.columns else pd.Series(100.0, index=df_copy.index)
    s_series = df_copy[std_col].fillna(15.0).clip(lower=1e-6) if std_col in df_copy.columns else pd.Series(15.0, index=df_copy.index)

    scores, levels, z_scores, is_anom_list, types_list = [], [], [], [], []
    for d, m, s in zip(d_series, m_series, s_series):
        res = calculate_demand_anomaly_risk(d, m, s, config_path=config_path)
        scores.append(res["demand_anomaly_risk_score"])
        levels.append(res["demand_anomaly_risk_level"])
        z_scores.append(res["z_score"])
        is_anom_list.append(res["is_anomaly"])
        types_list.append(res["anomaly_type"])

    df_copy[output_col] = scores
    df_copy["demand_anomaly_risk_level"] = levels
    df_copy["demand_z_score"] = z_scores
    df_copy["is_demand_anomaly"] = is_anom_list
    df_copy["demand_anomaly_type"] = types_list

    return df_copy
