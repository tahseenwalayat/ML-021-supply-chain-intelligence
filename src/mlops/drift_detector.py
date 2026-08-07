import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats

from src.utils.logging_config import get_logger

logger = get_logger("mlops.drift_detector")


def compute_ks_test(baseline_data: np.ndarray, current_data: np.ndarray, p_val_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Computes 2-sample Kolmogorov-Smirnov (KS) Test between baseline feature distribution and current batch data.
    Reject H0 if p-value < threshold -> drift detected.
    """
    base = np.asarray(baseline_data, dtype=float)
    curr = np.asarray(current_data, dtype=float)

    # Remove NaNs
    base = base[~np.isnan(base)]
    curr = curr[~np.isnan(curr)]

    if len(base) == 0 or len(curr) == 0:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    ks_stat, p_val = stats.ks_2samp(base, curr)
    drift_detected = bool(p_val < p_val_threshold)

    return {
        "ks_statistic": float(round(ks_stat, 4)),
        "p_value": float(round(p_val, 4)),
        "drift_detected": drift_detected
    }


def compute_psi(baseline_data: np.ndarray, current_data: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and current distributions.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.25: Moderate shift
    PSI >= 0.25: Significant data drift!
    """
    base = np.asarray(baseline_data, dtype=float)
    curr = np.asarray(current_data, dtype=float)

    base = base[~np.isnan(base)]
    curr = curr[~np.isnan(curr)]

    if len(base) == 0 or len(curr) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(base, percentiles)
    # Handle duplicates in bin edges
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    base_counts, _ = np.histogram(base, bins=bin_edges)
    curr_counts, _ = np.histogram(curr, bins=bin_edges)

    base_pct = base_counts / max(1, len(base))
    curr_pct = curr_counts / max(1, len(curr))

    # Add small epsilon to avoid log(0) or division by zero
    eps = 1e-4
    base_pct = np.where(base_pct == 0, eps, base_pct)
    curr_pct = np.where(curr_pct == 0, eps, curr_pct)

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(round(psi, 4))


class DataDriftDetector:
    """
    Data Drift & Concept Drift Detector.
    Monitors input feature distribution shifts and forecast performance degradation.
    """

    def __init__(self, psi_threshold: float = 0.25, wmape_degradation_threshold_pct: float = 20.0):
        self.psi_threshold = psi_threshold
        self.wmape_degradation_threshold_pct = wmape_degradation_threshold_pct

    def detect_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates KS-Test and PSI across all numeric features in feature_cols.
        """
        drift_results = {}
        features_drifted = 0

        for col in feature_cols:
            if col not in baseline_df.columns or col not in current_df.columns:
                continue

            b_vals = baseline_df[col].values
            c_vals = current_df[col].values

            ks_res = compute_ks_test(b_vals, c_vals)
            psi_val = compute_psi(b_vals, c_vals)

            drift_level = "NO_DRIFT"
            if psi_val >= self.psi_threshold or ks_res["drift_detected"]:
                drift_level = "CRITICAL_DRIFT" if psi_val >= 0.25 else "MODERATE_DRIFT"
                features_drifted += 1

            drift_results[col] = {
                "ks_statistic": ks_res["ks_statistic"],
                "p_value": ks_res["p_value"],
                "psi": psi_val,
                "drift_level": drift_level,
                "drift_detected": (drift_level != "NO_DRIFT")
            }

        overall_drift = bool(features_drifted > 0)
        return {
            "overall_drift_detected": overall_drift,
            "total_features_evaluated": len(feature_cols),
            "features_drifted_count": features_drifted,
            "feature_details": drift_results
        }

    def detect_forecast_degradation(
        self,
        baseline_wmape: float,
        current_wmape: float
    ) -> Dict[str, Any]:
        """
        Detects model performance degradation if current WMAPE increases significantly over baseline WMAPE.
        """
        wmape_diff = current_wmape - baseline_wmape
        rel_degradation_pct = (wmape_diff / max(1e-5, baseline_wmape)) * 100.0

        is_degraded = bool(rel_degradation_pct >= self.wmape_degradation_threshold_pct)

        return {
            "baseline_wmape": round(baseline_wmape, 2),
            "current_wmape": round(current_wmape, 2),
            "wmape_increase": round(wmape_diff, 2),
            "relative_degradation_pct": round(rel_degradation_pct, 2),
            "degradation_threshold_pct": self.wmape_degradation_threshold_pct,
            "model_retrain_recommended": is_degraded
        }
