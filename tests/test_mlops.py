import os
import pytest
import numpy as np
import pandas as pd
from src.mlops.registry import promote_model, demote_model, get_current_production_model, get_model_history
from src.mlops.drift_detection import FeatureDriftDetector, compute_ks_test, compute_psi, inject_synthetic_drift
from src.mlops.forecast_drift import ForecastDriftTracker, calculate_wmape
from src.mlops.retraining_pipeline import run_retraining_pipeline
from src.mlops.monitoring import ModelMonitoringService, check_schema_violations, compute_prediction_stats


def test_ks_test_no_drift():
    base = np.random.normal(10, 2, 100)
    curr = np.random.normal(10, 2, 100)
    res = compute_ks_test(base, curr)
    assert "p_value" in res
    assert "ks_statistic" in res


def test_ks_test_with_drift():
    base = np.random.normal(10, 2, 100)
    curr = np.random.normal(25, 2, 100)
    res = compute_ks_test(base, curr)
    assert res["drift_detected"] is True
    assert res["p_value"] < 0.05


def test_psi_drift_detection():
    base_df = pd.DataFrame({"feat1": np.random.normal(10, 2, 200)})
    curr_df = pd.DataFrame({"feat1": np.random.normal(25, 5, 200)})

    detector = FeatureDriftDetector(psi_threshold=0.25)
    res = detector.detect_feature_drift(base_df, curr_df, feature_cols=["feat1"])

    assert res["overall_drift_detected"] is True
    assert res["features_drifted_count"] == 1
    assert res["feature_details"]["feat1"]["drift_flagged"] is True


def test_synthetic_drift_injection_detection():
    """Validates success criteria: synthetic drift injection is correctly flagged by drift_detection.py"""
    np.random.seed(42)
    base_df = pd.DataFrame({
        "sales_velocity_7d": np.random.normal(15.0, 3.0, 300),
        "lead_time_days": np.random.normal(7.0, 1.5, 300)
    })
    
    # Inject synthetic mean shift and noise drift
    drifted_df = inject_synthetic_drift(base_df, feature_cols=["sales_velocity_7d"], shift_factor=3.0)

    detector = FeatureDriftDetector(psi_threshold=0.25)
    res = detector.detect_feature_drift(base_df, drifted_df, feature_cols=["sales_velocity_7d", "lead_time_days"])

    assert res["overall_drift_detected"] is True
    assert res["features_drifted_count"] >= 1
    assert res["feature_details"]["sales_velocity_7d"]["drift_flagged"] is True


def test_registry_promotion_and_demotion():
    """Validates success criteria: promotion decisions are logged with justification."""
    model_name = "test_model_sku"
    version = "20260807_test"
    justification = "Test candidate WMAPE 10.2% outperformed baseline 18.5%"

    # Promote to Production
    promo_res = promote_model(model_name, version, target_stage="Production", reason=justification, metrics={"wmape": 10.2})
    assert promo_res["event"] == "PROMOTE"
    assert promo_res["to_stage"] == "Production"
    assert promo_res["reason"] == justification

    # Check active production model
    current_prod = get_current_production_model(model_name)
    assert current_prod is not None
    assert current_prod["version"] == version

    # Demote model
    demote_res = demote_model(model_name, version, target_stage="Staging", reason="Manual rollback test")
    assert demote_res["event"] == "DEMOTE"
    assert demote_res["to_stage"] == "Staging"

    # Check audit history
    history = get_model_history(model_name)
    assert len(history) >= 2


def test_forecast_drift_tracking():
    tracker = ForecastDriftTracker(degradation_threshold_pct=20.0, rolling_window_days=7)
    
    # Record baseline & worsening predictions
    tracker.record_forecast_performance("2026-08-01", [10, 20, 30], [10, 20, 30])
    tracker.record_forecast_performance("2026-08-02", [10, 20, 30], [15, 30, 45])
    
    res = tracker.evaluate_forecast_drift(baseline_wmape=10.0)
    assert "relative_degradation_pct" in res
    assert "retrain_recommended" in res


def test_monitoring_service():
    service = ModelMonitoringService()
    input_df = pd.DataFrame({"feat1": [1.0, 2.0, None], "feat2": [10, 20, 30]})
    preds = np.array([12.5, 22.1, 31.8])

    event = service.log_prediction_event(
        model_id="lightgbm_sku_region",
        input_df=input_df,
        predictions=preds,
        latency_ms=18.4,
        expected_schema={"feat1": "float", "feat2": "int"}
    )

    assert event["latency_ms"] == 18.4
    assert event["schema_violation_count"] == 1  # null value in feat1
    assert event["prediction_stats"]["mean"] > 0

    summary = service.get_monitoring_summary("lightgbm_sku_region")
    assert summary["total_events"] >= 1
    assert summary["avg_latency_ms"] > 0


def test_retraining_pipeline_end_to_end():
    """Validates success criteria: retraining_pipeline.py runs end-to-end unattended."""
    report = run_retraining_pipeline(hierarchy_level="sku_region", force_run=False)
    
    assert report["status"] == "SUCCESS"
    assert "candidate_version" in report
    assert "candidate_wmape" in report
    assert "decision_status" in report
    assert "justification" in report
    assert report["decision_status"] in ["PROMOTED_TO_PRODUCTION", "REJECTED_STAGING_ONLY"]
