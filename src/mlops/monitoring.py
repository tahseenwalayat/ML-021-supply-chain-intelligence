import os
import json
import time
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.utils.logging_config import get_logger

logger = get_logger("mlops.monitoring")

MONITORING_LOG_FILE = "models/prediction_monitoring_logs.json"


def _load_monitoring_logs() -> List[Dict[str, Any]]:
    """Loads prediction monitoring log store."""
    if os.path.exists(MONITORING_LOG_FILE):
        try:
            with open(MONITORING_LOG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading monitoring log file: {e}")
    return []


def _save_monitoring_logs(logs: List[Dict[str, Any]]):
    """Saves prediction monitoring log store."""
    os.makedirs(os.path.dirname(MONITORING_LOG_FILE), exist_ok=True)
    try:
        # Retain last 1000 events to prevent unbounded growth
        logs_to_keep = logs[-1000:]
        with open(MONITORING_LOG_FILE, "w") as f:
            json.dump(logs_to_keep, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save monitoring log file: {e}")


def check_schema_violations(
    input_df: pd.DataFrame,
    expected_schema: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Validates input DataFrame against expected schema rules.
    Detects missing columns, null count spikes, and data type mismatches.
    """
    violations = []
    
    # 1. Null check across critical columns
    null_counts = input_df.isnull().sum().to_dict()
    for col, count in null_counts.items():
        if count > 0:
            violations.append({
                "type": "NULL_VALUES_DETECTED",
                "column": col,
                "null_count": int(count),
                "null_percentage": float(round(count / max(1, len(input_df)) * 100.0, 2))
            })

    # 2. Schema type validation if expected schema provided
    if expected_schema:
        for col, expected_type in expected_schema.items():
            if col not in input_df.columns:
                violations.append({
                    "type": "MISSING_REQUIRED_COLUMN",
                    "column": col,
                    "expected_type": expected_type
                })
            else:
                actual_type = str(input_df[col].dtype)
                if expected_type in ["float", "int", "numeric"] and not pd.api.types.is_numeric_dtype(input_df[col]):
                    violations.append({
                        "type": "DATA_TYPE_MISMATCH",
                        "column": col,
                        "expected_type": expected_type,
                        "actual_type": actual_type
                    })

    return violations


def compute_prediction_stats(predictions: np.ndarray) -> Dict[str, float]:
    """Computes distribution statistics for model predictions."""
    preds = np.asarray(predictions, dtype=float)
    preds_valid = preds[~np.isnan(preds)]

    if len(preds_valid) == 0:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "zero_count": 0
        }

    return {
        "count": int(len(preds_valid)),
        "min": float(round(np.min(preds_valid), 4)),
        "max": float(round(np.max(preds_valid), 4)),
        "mean": float(round(np.mean(preds_valid), 4)),
        "std": float(round(np.std(preds_valid), 4)),
        "median": float(round(np.median(preds_valid), 4)),
        "zero_count": int(np.sum(preds_valid == 0.0))
    }


class ModelMonitoringService:
    """
    Real-Time Model Performance & Data Quality Monitoring Service.
    Logs prediction latency ms, schema violations, and prediction distribution stats to a monitoring table.
    """

    def log_prediction_event(
        self,
        model_id: str,
        input_df: pd.DataFrame,
        predictions: np.ndarray,
        latency_ms: float,
        expected_schema: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Logs a single inference prediction batch event with latency, schema checks, and distribution stats.
        """
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Check schema violations
        violations = check_schema_violations(input_df, expected_schema)
        
        # 2. Compute prediction distribution stats
        dist_stats = compute_prediction_stats(predictions)

        event_record = {
            "timestamp": now_str,
            "model_id": model_id,
            "latency_ms": float(round(latency_ms, 2)),
            "sample_count": len(input_df),
            "schema_violation_count": len(violations),
            "schema_violations": violations,
            "prediction_stats": dist_stats
        }

        logs = _load_monitoring_logs()
        logs.append(event_record)
        _save_monitoring_logs(logs)

        logger.info(
            f"Monitored prediction event [{model_id}]: Latency = {latency_ms:.2f}ms | "
            f"Samples = {len(input_df)} | Violations = {len(violations)} | Pred Mean = {dist_stats['mean']:.2f}"
        )

        return event_record

    def get_monitoring_summary(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates aggregate monitoring summary across stored inference events.
        """
        logs = _load_monitoring_logs()
        if model_id:
            logs = [l for l in logs if l.get("model_id") == model_id]

        if not logs:
            return {
                "total_events": 0,
                "avg_latency_ms": 0.0,
                "total_schema_violations": 0,
                "message": "No monitoring events recorded."
            }

        latencies = [l["latency_ms"] for l in logs]
        total_violations = sum(l["schema_violation_count"] for l in logs)
        total_samples = sum(l["sample_count"] for l in logs)

        avg_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))

        return {
            "total_events": len(logs),
            "total_samples_scored": total_samples,
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "total_schema_violations": total_violations,
            "recent_event_timestamp": logs[-1]["timestamp"]
        }
