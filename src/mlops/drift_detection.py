import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from src.utils.logging_config import get_logger

logger = get_logger("mlops.drift_detection")


def compute_ks_test(
    baseline_data: np.ndarray,
    current_data: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Computes two-sample Kolmogorov-Smirnov (KS) Test.
    H0: Baseline and Current data originate from the same statistical distribution.
    If p-value < alpha, reject H0 -> Statistical drift detected.
    """
    base = np.asarray(baseline_data, dtype=float)
    curr = np.asarray(current_data, dtype=float)

    base = base[~np.isnan(base)]
    curr = curr[~np.isnan(curr)]

    if len(base) == 0 or len(curr) == 0:
        return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    ks_stat, p_val = stats.ks_2samp(base, curr)
    drift_detected = bool(p_val < alpha)

    return {
        "ks_statistic": float(round(ks_stat, 4)),
        "p_value": float(round(p_val, 4)),
        "drift_detected": drift_detected
    }


def compute_psi(
    baseline_data: np.ndarray,
    current_data: np.ndarray,
    num_bins: int = 10
) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and current distributions.
    PSI interpretation:
      PSI < 0.10: No significant distribution change
      0.10 <= PSI < 0.25: Moderate distribution shift
      PSI >= 0.25: Significant data drift (Requires investigation/retraining)
    """
    base = np.asarray(baseline_data, dtype=float)
    curr = np.asarray(current_data, dtype=float)

    base = base[~np.isnan(base)]
    curr = curr[~np.isnan(curr)]

    if len(base) == 0 or len(curr) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_bins + 1)
    try:
        bin_edges = np.percentile(base, percentiles)
    except Exception:
        bin_edges = np.linspace(np.min(base), np.max(base), num_bins + 1)

    # Handle duplicate bin boundaries
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    base_counts, _ = np.histogram(base, bins=bin_edges)
    curr_counts, _ = np.histogram(curr, bins=bin_edges)

    base_pct = base_counts / max(1, len(base))
    curr_pct = curr_counts / max(1, len(curr))

    # Add epsilon to prevent log(0) or division by zero
    eps = 1e-4
    base_pct = np.where(base_pct == 0, eps, base_pct)
    curr_pct = np.where(curr_pct == 0, eps, curr_pct)

    psi_val = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(round(psi_val, 4))


class FeatureDriftDetector:
    """
    Feature Data Drift Detector.
    Compares recent inference/batch data feature distributions against training-time baseline distributions.
    """

    def __init__(self, psi_threshold: float = 0.25, ks_alpha: float = 0.05):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha

    def detect_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluates PSI and KS Test across specified feature columns.
        Flags overall_drift_detected if any feature exceeds psi_threshold or rejects KS H0.
        """
        drift_results = {}
        drifted_count = 0

        for col in feature_cols:
            if col not in baseline_df.columns or col not in current_df.columns:
                continue

            # Ensure numeric type
            b_vals = pd.to_numeric(baseline_df[col], errors="coerce").dropna().values
            c_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values

            if len(b_vals) == 0 or len(c_vals) == 0:
                continue

            psi_val = compute_psi(b_vals, c_vals)
            ks_res = compute_ks_test(b_vals, c_vals, alpha=self.ks_alpha)

            drift_flagged = bool(psi_val >= self.psi_threshold or ks_res["drift_detected"])
            
            if psi_val >= self.psi_threshold:
                drift_severity = "CRITICAL"
            elif psi_val >= 0.10:
                drift_severity = "MODERATE"
            else:
                drift_severity = "NONE"

            if drift_flagged:
                drifted_count += 1

            drift_results[col] = {
                "psi": psi_val,
                "ks_statistic": ks_res["ks_statistic"],
                "p_value": ks_res["p_value"],
                "drift_severity": drift_severity,
                "drift_flagged": drift_flagged
            }

        overall_drift = bool(drifted_count > 0)
        logger.info(
            f"Evaluated {len(feature_cols)} features for data drift. "
            f"Drifted features: {drifted_count}/{len(feature_cols)} | Overall Drift: {overall_drift}"
        )

        return {
            "overall_drift_detected": overall_drift,
            "total_features_evaluated": len(feature_cols),
            "features_drifted_count": drifted_count,
            "psi_threshold": self.psi_threshold,
            "feature_details": drift_results
        }


def inject_synthetic_drift(
    df: pd.DataFrame,
    feature_cols: List[str],
    shift_factor: float = 2.5,
    noise_std: float = 1.5
) -> pd.DataFrame:
    """
    Injects synthetic drift into specified columns of a DataFrame by applying a mean shift and noise scaling.
    Used to test and demonstrate automated drift detection flagging.
    """
    df_drifted = df.copy()
    for col in feature_cols:
        if col in df_drifted.columns and pd.api.types.is_numeric_dtype(df_drifted[col]):
            mean_val = df_drifted[col].mean()
            std_val = df_drifted[col].std() if df_drifted[col].std() > 0 else 1.0
            # Apply mean shift + additional noise
            df_drifted[col] = df_drifted[col] + (shift_factor * std_val) + np.random.normal(0, noise_std, len(df_drifted))
    return df_drifted
