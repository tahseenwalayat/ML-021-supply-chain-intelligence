import pytest
import numpy as np
import pandas as pd

from src.inventory.safety_stock import calculate_safety_stock, compute_safety_stock_df, get_z_score
from src.inventory.reorder_point import calculate_reorder_point, compute_reorder_point_df
from src.inventory.eoq import calculate_eoq, compute_eoq_df, load_inventory_config
from src.inventory.allocation import calculate_warehouse_fulfillment_shares, allocate_demand_to_warehouses
from src.inventory.transfers import calculate_surplus_and_deficit, recommend_inter_warehouse_transfers
from src.inventory.procurement import calculate_procurement_quantity, compute_procurement_recommendations_df


def test_safety_stock_constant_lead_time():
    """
    Hand-calculated Example 1:
    - avg_daily_demand = 100.0
    - std_daily_demand = 10.0
    - avg_lead_time = 9.0 (sqrt(9) = 3.0)
    - service_level = 0.95 (Z ≈ 1.6448536)

    std_lt = 10.0 * 3.0 = 30.0
    Safety Stock = 1.6448536 * 30.0 = 49.3456
    """
    ss = calculate_safety_stock(
        avg_daily_demand=100.0,
        std_daily_demand=10.0,
        avg_lead_time=9.0,
        std_lead_time=0.0,
        service_level=0.95
    )
    assert ss == pytest.approx(49.3456, rel=1e-3)
    assert ss >= 0.0


def test_safety_stock_variable_lead_time():
    """
    Hand-calculated Example 2:
    - avg_daily_demand = 50.0
    - std_daily_demand = 5.0
    - avg_lead_time = 4.0
    - std_lead_time = 1.0
    - service_level = 0.95 (Z ≈ 1.6448536)

    variance_lt = (4.0 * 5.0^2) + (50.0^2 * 1.0^2) = 100.0 + 2500.0 = 2600.0
    std_lt = sqrt(2600.0) ≈ 50.990195
    Safety Stock = 1.6448536 * 50.990195 ≈ 83.8714
    """
    ss = calculate_safety_stock(
        avg_daily_demand=50.0,
        std_daily_demand=5.0,
        avg_lead_time=4.0,
        std_lead_time=1.0,
        service_level=0.95
    )
    assert ss == pytest.approx(83.8714, rel=1e-3)
    assert ss >= 0.0


def test_safety_stock_zero_variance():
    """Safety stock must equal 0.0 when demand and lead time have zero variance."""
    ss = calculate_safety_stock(
        avg_daily_demand=50.0,
        std_daily_demand=0.0,
        avg_lead_time=5.0,
        std_lead_time=0.0,
        service_level=0.95
    )
    assert ss == 0.0


def test_safety_stock_missing_lead_time_fallback():
    """Invalid lead time (<= 0 or NaN) must trigger default lead time fallback gracefully."""
    ss = calculate_safety_stock(
        avg_daily_demand=10.0,
        std_daily_demand=2.0,
        avg_lead_time=-1.0,
        service_level=0.95,
        default_lead_time=9.0
    )
    # Uses default_lead_time = 9.0 -> std_lt = 2.0 * 3.0 = 6.0 -> SS = 1.64485 * 6 = 9.8691
    assert ss == pytest.approx(9.8691, rel=1e-3)


def test_reorder_point_normal():
    """
    Hand-calculated ROP Example:
    - avg_daily_demand = 20.0
    - avg_lead_time = 5.0
    - safety_stock = 15.0

    ROP = (20.0 * 5.0) + 15.0 = 115.0
    """
    rop = calculate_reorder_point(
        avg_daily_demand=20.0,
        avg_lead_time=5.0,
        safety_stock=15.0
    )
    assert rop == 115.0


def test_reorder_point_no_negative():
    """ROP must never be negative even with zero/invalid demand or negative inputs."""
    rop = calculate_reorder_point(
        avg_daily_demand=-10.0,
        avg_lead_time=5.0,
        safety_stock=-5.0
    )
    assert rop >= 0.0


def test_eoq_normal():
    """
    Hand-calculated EOQ Example:
    - annual_demand = 1000.0
    - ordering_cost = 50.0
    - holding_cost = 2.5

    EOQ = sqrt( (2 * 1000 * 50) / 2.5 ) = sqrt( 100000 / 2.5 ) = sqrt( 40000 ) = 200.0
    """
    eoq_val = calculate_eoq(
        annual_demand=1000.0,
        ordering_cost=50.0,
        holding_cost=2.5
    )
    assert eoq_val == 200.0


def test_eoq_zero_demand():
    """EOQ must equal 0.0 when annual demand is 0.0."""
    eoq_val = calculate_eoq(
        annual_demand=0.0,
        ordering_cost=50.0,
        holding_cost=2.0
    )
    assert eoq_val == 0.0


def test_allocation_proportional_share():
    """Demand allocation must divide forecast proportionally to historical warehouse sales."""
    sales_fact = pd.DataFrame([
        {"product_id": "P1", "region": "North", "warehouse_id": "W1", "quantity": 300},
        {"product_id": "P1", "region": "North", "warehouse_id": "W2", "quantity": 700},
    ])
    warehouse_dim = pd.DataFrame([
        {"warehouse_id": "W1", "region": "North"},
        {"warehouse_id": "W2", "region": "North"},
    ])
    forecast_df = pd.DataFrame([
        {"product_id": "P1", "region": "North", "actual_sales": 100.0}
    ])

    allocated = allocate_demand_to_warehouses(forecast_df, sales_fact, warehouse_dim)
    
    w1_alloc = allocated[allocated["warehouse_id"] == "W1"]["allocated_daily_demand"].values[0]
    w2_alloc = allocated[allocated["warehouse_id"] == "W2"]["allocated_daily_demand"].values[0]

    assert w1_alloc == pytest.approx(30.0, rel=1e-3)
    assert w2_alloc == pytest.approx(70.0, rel=1e-3)
    assert (w1_alloc + w2_alloc) == pytest.approx(100.0, rel=1e-3)


def test_allocation_equal_fallback_zero_history():
    """Products with no historical sales must be allocated equally across active regional warehouses."""
    sales_fact = pd.DataFrame(columns=["product_id", "region", "warehouse_id", "quantity"])
    warehouse_dim = pd.DataFrame([
        {"warehouse_id": "W1", "region": "North"},
        {"warehouse_id": "W2", "region": "North"},
    ])
    forecast_df = pd.DataFrame([
        {"product_id": "P_NEW", "region": "North", "actual_sales": 60.0}
    ])

    allocated = allocate_demand_to_warehouses(forecast_df, sales_fact, warehouse_dim)
    
    assert len(allocated) == 2
    for val in allocated["allocated_daily_demand"].values:
        assert val == pytest.approx(30.0, rel=1e-3)


def test_transfers_surplus_deficit_matching():
    """Surplus warehouse must transfer stock to deficit warehouse up to the required deficit."""
    inv_df = pd.DataFrame([
        {"product_id": "P1", "warehouse_id": "W1", "region": "North", "current_stock": 500.0, "reorder_point": 200.0, "safety_stock": 100.0, "eoq": 100.0},
        {"product_id": "P1", "warehouse_id": "W2", "region": "North", "current_stock": 50.0, "reorder_point": 150.0, "safety_stock": 50.0, "eoq": 100.0},
    ])

    transfers_df, updated_inv = recommend_inter_warehouse_transfers(inv_df)

    assert len(transfers_df) == 1
    row = transfers_df.iloc[0]
    assert row["from_warehouse_id"] == "W1"
    assert row["to_warehouse_id"] == "W2"
    assert row["product_id"] == "P1"
    assert row["transfer_qty"] == 100.0  # Deficit of W2 is 150 - 50 = 100


def test_procurement_quantity_reorder_required():
    """
    Hand-calculated Procurement Example (Reorder Required):
    - current_stock = 30.0
    - reorder_point = 50.0
    - eoq = 100.0
    - on_order = 0.0

    Inventory Position = 30.0 <= 50.0 ROP -> Trigger Reorder
    Net Deficit = (50.0 + 100.0) - 30.0 = 120.0
    Recommended Order Qty = max(100.0, 120.0) = 120.0
    """
    qty, status = calculate_procurement_quantity(
        current_stock=30.0,
        reorder_point=50.0,
        eoq=100.0,
        on_order=0.0
    )
    assert qty == 120.0
    assert status == "REORDER_REQUIRED"


def test_procurement_quantity_stock_adequate():
    """
    Hand-calculated Procurement Example (Stock Adequate):
    - current_stock = 80.0
    - reorder_point = 50.0
    - eoq = 100.0

    Inventory Position = 80.0 > 50.0 ROP -> Stock Adequate
    Recommended Order Qty = 0.0
    """
    qty, status = calculate_procurement_quantity(
        current_stock=80.0,
        reorder_point=50.0,
        eoq=100.0
    )
    assert qty == 0.0
    assert status == "STOCK_ADEQUATE"
