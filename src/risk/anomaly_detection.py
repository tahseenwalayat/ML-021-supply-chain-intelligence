import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.utils.logging_config import get_logger
from src.risk.supplier_delay_risk import classify_risk_level, load_risk_config

logger = get_logger("risk.anomaly_detection")


def detect_demand_anomaly(
    actual_demand: float,
    forecasted_demand: float,
    std_residual: float = 10.0,
    anomaly_z_score_threshold: Optional[float] = None,
    config_path: str = "configs/config.yaml"
) -> Dict[str, Any]:
    """
    Residual Z-Score Anomaly Detection Method.
    Evaluates forecast residuals (actual - forecasted) against residual standard deviation
    to flag demand spikes (positive residual) and demand dips (negative residual).

    Formulation:
    - Residual residual = actual_demand - forecasted_demand
    - Z-Score = residual / max(1e-6, std_residual)
    - Threshold loaded dynamically from configs/config.yaml (default: 3.0)

    Returns:
    - Dict with z_score, is_anomaly (bool), anomaly_type (SPIKE, DIP, NORMAL), risk_score [0.0, 1.0].
    """
    cfg = load_risk_config(config_path)
    if anomaly_z_score_threshold is None:
        anomaly_z_score_threshold = cfg.get("anomaly_z_score_threshold", 3.0)

    act = max(0.0, actual_demand if actual_demand is not None and not np.isnan(actual_demand) else 0.0)
    fcst = max(0.0, forecasted_demand if forecasted_demand is not None and not np.isnan(forecasted_demand) else 0.0)
    std_res = max(1e-6, std_residual if std_residual is not None and not np.isnan(std_residual) else 10.0)

    residual = act - fcst
    z_score = residual / std_res
    abs_z = abs(z_score)

    is_anomaly = abs_z >= anomaly_z_score_threshold

    if z_score >= anomaly_z_score_threshold:
        anomaly_type = "SPIKE"
    elif z_score <= -anomaly_z_score_threshold:
        anomaly_type = "DIP"
    else:
        anomaly_type = "NORMAL"

    # Risk score calculation bounded in [0.0, 1.0]
    if abs_z < anomaly_z_score_threshold:
        risk_score = 0.50 * (abs_z / anomaly_z_score_threshold)
    else:
        excess_z = abs_z - anomaly_z_score_threshold
        risk_score = 0.50 + 0.50 * min(1.0, excess_z / max(1.0, anomaly_z_score_threshold))

    score = float(np.round(np.clip(risk_score, 0.0, 1.0), 4))
    level = classify_risk_level(score)

    return {
        "anomaly_risk_score": score,
        "anomaly_risk_level": level,
        "residual": float(np.round(residual, 2)),
        "z_score": float(np.round(z_score, 4)),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
        "excess_spike_units": float(np.round(max(0.0, residual - (anomaly_z_score_threshold * std_res)), 2))
    }


def compute_anomaly_detection_df(
    df: pd.DataFrame,
    actual_col: str = "actual_demand",
    forecast_col: str = "forecasted_demand",
    std_residual_col: str = "std_residual",
    config_path: str = "configs/config.yaml",
    output_col: str = "anomaly_risk_score"
) -> pd.DataFrame:
    """
    Vectorized residual Z-score anomaly detection for a pandas DataFrame.
    """
    df_copy = df.copy()

    # Fallback column mapping
    act_col = actual_col if actual_col in df_copy.columns else ("demand" if "demand" in df_copy.columns else "actual_sales")
    fcst_col = forecast_col if forecast_col in df_copy.columns else ("mean_demand" if "mean_demand" in df_copy.columns else "avg_daily_demand")

    act_s = df_copy[act_col].fillna(0.0).clip(lower=0.0) if act_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    fcst_s = df_copy[fcst_col].fillna(0.0).clip(lower=0.0) if fcst_col in df_copy.columns else pd.Series(0.0, index=df_copy.index)
    std_res_s = df_copy[std_residual_col].fillna(10.0).clip(lower=1e-6) if std_residual_col in df_copy.columns else (
        df_copy["std_daily_demand"].fillna(10.0).clip(lower=1e-6) if "std_daily_demand" in df_copy.columns else pd.Series(10.0, index=df_copy.index)
    )

    scores, levels, z_scores, is_anom_list, types_list = [], [], [], [], []
    for a, f, s in zip(act_s, fcst_s, std_res_s):
        res = detect_demand_anomaly(a, f, s, config_path=config_path)
        scores.append(res["anomaly_risk_score"])
        levels.append(res["anomaly_risk_level"])
        z_scores.append(res["z_score"])
        is_anom_list.append(res["is_anomaly"])
        types_list.append(res["anomaly_type"])

    df_copy[output_col] = scores
    df_copy["anomaly_risk_level"] = levels
    df_copy["demand_z_score"] = z_scores
    df_copy["is_demand_anomaly"] = is_anom_list
    df_copy["anomaly_type"] = types_list

    return df_copy
