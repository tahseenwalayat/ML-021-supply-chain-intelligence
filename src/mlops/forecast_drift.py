import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.utils.logging_config import get_logger

logger = get_logger("mlops.forecast_drift")


def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Weighted Absolute Percentage Error (WMAPE)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    total_actual = np.sum(np.abs(yt))
    if total_actual < 1e-5:
        return 0.0
    return float(np.sum(np.abs(yt - yp)) / total_actual * 100.0)


class ForecastDriftTracker:
    """
    Tracks rolling actual-vs-predicted demand forecast errors over time and flags worsening performance trends.
    """

    def __init__(self, degradation_threshold_pct: float = 20.0, rolling_window_days: int = 14):
        self.degradation_threshold_pct = degradation_threshold_pct
        self.rolling_window_days = rolling_window_days
        self.history: List[Dict[str, Any]] = []

    def record_forecast_performance(
        self,
        date: str,
        y_true: List[float],
        y_pred: List[float],
        model_id: str = "production_lgbm"
    ) -> Dict[str, Any]:
        """
        Records actual vs predicted values for a given evaluation date.
        """
        yt = np.array(y_true)
        yp = np.array(y_pred)
        
        wmape = calculate_wmape(yt, yp)
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

        entry = {
            "date": date,
            "model_id": model_id,
            "sample_count": len(yt),
            "wmape": round(wmape, 2),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2)
        }
        self.history.append(entry)
        logger.info(f"Recorded forecast error for {date} [{model_id}]: WMAPE = {wmape:.2f}% | MAE = {mae:.2f}")
        return entry

    def evaluate_forecast_drift(
        self,
        baseline_wmape: float,
        recent_evaluations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates rolling actual-vs-predicted WMAPE against baseline WMAPE and flags worsening performance trends.
        """
        eval_list = recent_evaluations if recent_evaluations is not None else self.history

        if not eval_list:
            return {
                "baseline_wmape": round(baseline_wmape, 2),
                "current_rolling_wmape": round(baseline_wmape, 2),
                "relative_degradation_pct": 0.0,
                "worsening_trend_flagged": False,
                "retrain_recommended": False,
                "message": "No historical evaluation data recorded."
            }

        # Focus on the most recent window
        recent_window = eval_list[-self.rolling_window_days:]
        current_wmape = float(np.mean([e["wmape"] for e in recent_window]))
        
        wmape_diff = current_wmape - baseline_wmape
        rel_degradation_pct = (wmape_diff / max(1e-5, baseline_wmape)) * 100.0

        # Check linear trend over recent window
        worsening_trend = False
        if len(recent_window) >= 3:
            wmapes = [e["wmape"] for e in recent_window]
            x = np.arange(len(wmapes))
            slope, _ = np.polyfit(x, wmapes, 1)
            # Positive slope indicates increasing error over time
            if slope > 0.1 and rel_degradation_pct >= (self.degradation_threshold_pct / 2.0):
                worsening_trend = True

        retrain_recommended = bool(
            rel_degradation_pct >= self.degradation_threshold_pct or worsening_trend
        )

        msg = (
            f"Forecast WMAPE degraded by {rel_degradation_pct:.1f}% "
            f"(Baseline: {baseline_wmape:.2f}%, Current: {current_wmape:.2f}%). "
            f"Automated retraining recommended."
            if retrain_recommended
            else "Forecast performance is within acceptable stability thresholds."
        )

        logger.info(f"Forecast Drift Status: {msg}")

        return {
            "baseline_wmape": round(baseline_wmape, 2),
            "current_rolling_wmape": round(current_wmape, 2),
            "wmape_increase": round(wmape_diff, 2),
            "relative_degradation_pct": round(rel_degradation_pct, 2),
            "degradation_threshold_pct": self.degradation_threshold_pct,
            "worsening_trend_flagged": worsening_trend,
            "retrain_recommended": retrain_recommended,
            "message": msg
        }
