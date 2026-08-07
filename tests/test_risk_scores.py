import os
import pytest
import numpy as np
import pandas as pd

from src.risk.supplier_delay_risk import calculate_supplier_delay_risk, classify_risk_level
from src.risk.stockout_risk import calculate_stockout_risk
from src.risk.overstock_risk import calculate_overstock_risk
from src.risk.slow_dead_inventory import classify_slow_dead_inventory
from src.risk.anomaly_detection import detect_demand_anomaly, compute_anomaly_detection_df
from src.risk.run_risk_engine import run_risk_engine


def test_perfect_supplier_returns_zero_risk():
    """
    Test supplier with perfect on-time record (reliability = 1.0, std = 0.0)
    returns 0.0 / near-zero risk.
    """
    score = calculate_supplier_delay_risk(
        reliability_score=1.0,
        lead_time_std=0.0,
        lead_time_avg=7.0,
        late_delivery_rate=0.0,
        config_path="configs/config.yaml"
    )
    assert score == 0.0
    assert classify_risk_level(score) == "LOW"


def test_synthetic_demand_spike_flagged_by_anomaly_detection():
    """
    Inject a synthetic demand spike (e.g. actual = 250, forecasted = 50, std_residual = 10)
    and confirm anomaly_detection.py flags it as a SPIKE anomaly.
    """
    actual_demand_spike = 250.0
    forecasted_demand = 50.0
    std_residual = 10.0

    result = detect_demand_anomaly(
        actual_demand=actual_demand_spike,
        forecasted_demand=forecasted_demand,
        std_residual=std_residual,
        config_path="configs/config.yaml"
    )

    # Z-score = (250 - 50) / 10 = 20.0 (well above threshold 3.0)
    assert result["is_anomaly"] is True
    assert result["anomaly_type"] == "SPIKE"
    assert result["z_score"] == pytest.approx(20.0, rel=1e-3)
    assert result["anomaly_risk_score"] >= 0.50
    assert result["anomaly_risk_level"] in ["HIGH", "CRITICAL"]


def test_stockout_risk_bounded():
    """Verify stockout risk is bounded in [0.0, 1.0]."""
    score_out_of_stock = calculate_stockout_risk(
        current_stock=0.0,
        reorder_point=50.0,
        safety_stock=20.0
    )
    assert score_out_of_stock == 1.0

    score_healthy = calculate_stockout_risk(
        current_stock=200.0,
        reorder_point=50.0,
        safety_stock=20.0
    )
    assert 0.0 <= score_healthy <= 0.25


def test_overstock_risk_flag():
    """Verify overstock risk flags stock > N * ROP with low velocity."""
    score, excess, capital, is_over = calculate_overstock_risk(
        current_stock=500.0,
        reorder_point=50.0,
        unit_cost=10.0,
        sales_velocity=0.5,
        config_path="configs/config.yaml"
    )
    assert is_over is True
    assert excess > 0.0
    assert capital > 0.0
    assert 0.0 <= score <= 1.0


def test_slow_dead_inventory_classification():
    """Verify dead stock classification for 12+ zero-sales weeks."""
    res_dead = classify_slow_dead_inventory(
        sales_velocity=0.0,
        zero_sales_weeks=14,
        current_stock=100.0,
        config_path="configs/config.yaml"
    )
    assert res_dead["inventory_health_status"] == "DEAD_STOCK"
    assert res_dead["is_dead_stock"] is True

    res_slow = classify_slow_dead_inventory(
        sales_velocity=0.5,
        zero_sales_weeks=2,
        current_stock=100.0,
        config_path="configs/config.yaml"
    )
    assert res_slow["inventory_health_status"] == "SLOW_MOVING"


def test_run_risk_engine_produces_parquet():
    """
    Test that run_risk_engine produces data/processed/risk_scores.parquet
    for all product-warehouse pairs without NaNs.
    """
    df_eval = run_risk_engine(
        processed_dir="data/processed",
        config_path="configs/config.yaml",
        output_filename="risk_scores.parquet"
    )

    parquet_path = os.path.join("data/processed", "risk_scores.parquet")
    assert os.path.exists(parquet_path)
    assert len(df_eval) > 0
    assert "composite_risk_score" in df_eval.columns
    assert "overall_risk_level" in df_eval.columns
    assert df_eval["composite_risk_score"].isna().sum() == 0


def test_brand_new_product_edge_case():
    """
    Brand-new product with no delay/velocity history (None/NaN inputs)
    must return neutral/default risk categories without raising errors or NaNs.
    """
    score_delay = calculate_supplier_delay_risk(
        reliability_score=None,
        lead_time_std=None,
        lead_time_avg=None,
        late_delivery_rate=None
    )
    assert 0.0 <= score_delay <= 1.0
    assert not np.isnan(score_delay)
    assert classify_risk_level(score_delay) in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    overstock_score, excess, cap, is_over = calculate_overstock_risk(
        current_stock=None,
        reorder_point=None,
        unit_cost=None,
        sales_velocity=None
    )
    assert 0.0 <= overstock_score <= 1.0
    assert not np.isnan(overstock_score)

    health_res = classify_slow_dead_inventory(
        sales_velocity=None,
        zero_sales_weeks=None,
        current_stock=None,
        unit_cost=None
    )
    assert health_res["inventory_health_status"] in ["HEALTHY", "SLOW_MOVING", "DEAD_STOCK"]
    assert 0.0 <= health_res["slow_dead_risk_score"] <= 1.0

    anomaly_res = detect_demand_anomaly(
        actual_demand=None,
        forecasted_demand=None,
        std_residual=None
    )
    assert 0.0 <= anomaly_res["anomaly_risk_score"] <= 1.0
    assert anomaly_res["anomaly_type"] in ["NORMAL", "SPIKE", "DIP"]

