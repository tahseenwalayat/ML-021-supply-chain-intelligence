import pytest
import numpy as np
import pandas as pd

from src.risk.supplier_delay_risk import (
    calculate_supplier_delay_risk,
    evaluate_supplier_delay_details,
    compute_supplier_delay_risk_df,
    classify_risk_level
)


def test_perfect_supplier_zero_risk():
    """100% reliable supplier with 0 lead time variance must yield 0.0 risk score."""
    score = calculate_supplier_delay_risk(
        reliability_score=1.0,
        lead_time_std=0.0,
        lead_time_avg=7.0,
        late_delivery_rate=0.0
    )
    assert score == 0.0
    assert classify_risk_level(score) == "LOW"


def test_unreliable_supplier_high_risk():
    """Unreliable supplier with high late delivery rate and high variance must yield high risk score."""
    score = calculate_supplier_delay_risk(
        reliability_score=0.2,
        lead_time_std=7.0,
        lead_time_avg=7.0,
        late_delivery_rate=0.8,
        w_late=0.6,
        w_var=0.4
    )
    # late component = 0.8, var component = min(1.0, 7.0/7.0) = 1.0
    # composite = 0.6 * 0.8 + 0.4 * 1.0 = 0.48 + 0.40 = 0.88
    assert score == pytest.approx(0.88, rel=1e-3)
    assert classify_risk_level(score) == "CRITICAL"


def test_supplier_delay_risk_bounds():
    """Risk score must remain strictly bounded in [0.0, 1.0] for unexpected or invalid inputs."""
    # Negative lead time std or NaN values
    score_nan = calculate_supplier_delay_risk(
        reliability_score=np.nan,
        lead_time_std=-5.0,
        lead_time_avg=0.0
    )
    assert 0.0 <= score_nan <= 1.0

    # Oversized values
    score_over = calculate_supplier_delay_risk(
        reliability_score=-2.0,
        lead_time_std=100.0,
        lead_time_avg=5.0,
        late_delivery_rate=1.5
    )
    assert score_over == 1.0


def test_evaluate_supplier_delay_details():
    """Detailed evaluation dict must contain all required key metrics."""
    details = evaluate_supplier_delay_details(
        reliability_score=0.70,
        lead_time_std=2.0,
        lead_time_avg=10.0,
        w_late=0.6,
        w_var=0.4
    )
    assert "supplier_delay_risk_score" in details
    assert "supplier_delay_risk_level" in details
    assert "late_delivery_rate" in details
    assert "recommended_buffer_days" in details
    assert details["late_delivery_rate"] == pytest.approx(0.30, rel=1e-3)
    assert details["recommended_buffer_days"] >= 0.0


def test_compute_supplier_delay_risk_df():
    """Vectorized dataframe computation must output valid columns without NaNs."""
    df = pd.DataFrame([
        {"supplier_reliability_score": 1.0, "lead_time_std_days": 0.0, "avg_lead_time": 7.0},
        {"supplier_reliability_score": 0.5, "lead_time_std_days": 3.5, "avg_lead_time": 7.0},
        {"supplier_reliability_score": np.nan, "lead_time_std_days": np.nan, "avg_lead_time": np.nan},
    ])

    result_df = compute_supplier_delay_risk_df(df)

    assert "supplier_delay_risk_score" in result_df.columns
    assert "supplier_delay_risk_level" in result_df.columns
    assert "recommended_buffer_days" in result_df.columns
    assert result_df["supplier_delay_risk_score"].isna().sum() == 0
    assert result_df.iloc[0]["supplier_delay_risk_score"] == 0.0
    assert result_df.iloc[1]["supplier_delay_risk_score"] > 0.0
