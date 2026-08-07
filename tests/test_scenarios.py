import os
import pytest
import pandas as pd
import numpy as np

from src.simulation.base import BaseScenario
from src.simulation.supplier_failure import SupplierFailureScenario
from src.simulation.price_increase import PriceIncreaseScenario
from src.simulation.holiday_sales import HolidaySalesScenario
from src.simulation.new_product_launch import NewProductLaunchScenario
from src.simulation.transport_delay import TransportDelayScenario
from src.simulation.demand_surge import DemandSurgeScenario
from src.simulation.orchestrator import ScenarioOrchestrator, run_simulation_pipeline


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "product_id": "SKU_0001",
            "warehouse_id": "WH_NORTH",
            "supplier_id": "SUP_SINGLE",
            "region": "North",
            "category": "Electronics",
            "avg_daily_demand": 10.0,
            "std_daily_demand": 2.0,
            "current_stock": 100.0,
            "reorder_point": 50.0,
            "safety_stock": 20.0,
            "avg_lead_time": 7.0,
            "lead_time_std_days": 1.0,
            "unit_cost": 20.0,
            "sales_velocity": 10.0,
            "zero_sales_weeks": 0,
            "supplier_reliability_score": 0.90
        },
        {
            "product_id": "SKU_0002",
            "warehouse_id": "WH_SOUTH",
            "supplier_id": "SUP_MULTI",
            "region": "South",
            "category": "Apparel",
            "avg_daily_demand": 5.0,
            "std_daily_demand": 1.0,
            "current_stock": 30.0,
            "reorder_point": 40.0,
            "safety_stock": 15.0,
            "avg_lead_time": 5.0,
            "lead_time_std_days": 0.5,
            "unit_cost": 10.0,
            "sales_velocity": 5.0,
            "zero_sales_weeks": 0,
            "supplier_reliability_score": 0.95
        }
    ])


def test_supplier_failure_scenario(sample_df):
    scenario = SupplierFailureScenario(delay_days=10.0)
    res = scenario.run(sample_df)

    assert res["scenario_type"] == "supplier_failure"
    assert "baseline_metrics" in res
    assert "scenario_metrics" in res
    assert res["scenario_metrics"]["mean_stockout_risk_score"] >= res["baseline_metrics"]["mean_stockout_risk_score"]


def test_supplier_failure_single_product_supplier(sample_df):
    """Edge Case: Supplier failure on a supplier associated with only one product (SUP_SINGLE)."""
    scenario = SupplierFailureScenario(supplier_id="SUP_SINGLE", delay_days=14.0)
    res = scenario.run(sample_df)

    assert res["scenario_type"] == "supplier_failure"
    assert res["impact_deltas"]["stockout_risk_score_delta"] >= 0.0


def test_price_increase_zero_percent_equals_baseline(sample_df):
    """Edge Case: Price increase of 0% should equal baseline metrics (sanity check)."""
    scenario = PriceIncreaseScenario(price_change_pct=0.0, price_elasticity=-1.2)
    res = scenario.run(sample_df)

    assert res["scenario_type"] == "price_increase"
    assert res["scenario_metrics"]["mean_fill_rate_pct"] == res["baseline_metrics"]["mean_fill_rate_pct"]
    assert res["impact_deltas"]["fill_rate_delta_pct"] == 0.0
    assert res["impact_deltas"]["stockout_risk_score_delta"] == 0.0


def test_holiday_sales_scenario(sample_df):
    scenario = HolidaySalesScenario(demand_multiplier=1.8)
    res = scenario.run(sample_df)

    assert res["scenario_type"] == "holiday_sales"
    assert res["scenario_metrics"]["total_stockout_days"] >= res["baseline_metrics"]["total_stockout_days"]


def test_new_product_launch_scenario(sample_df):
    scenario = NewProductLaunchScenario(new_sku_id="NEW_SKU_TEST", category="Electronics")
    res = scenario.run(sample_df)

    assert res["scenario_type"] == "new_product_launch"
    assert res["scenario_metrics"] is not None


def test_transport_delay_scenario(sample_df):
    scenario = TransportDelayScenario(region="North", transit_delay_days=5.0)
    res = scenario.run(sample_df)

    assert res["scenario_type"] == "transport_delay"
    assert res["impact_deltas"]["extra_capital_required_usd"] >= 0.0


def test_demand_surge_scenario(sample_df):
    scenario = DemandSurgeScenario(demand_surge_multiplier=2.0)
    res = scenario.run(sample_df)

    assert res["scenario_type"] == "demand_surge"
    assert res["scenario_metrics"]["mean_fill_rate_pct"] <= res["baseline_metrics"]["mean_fill_rate_pct"]


def test_scenario_orchestrator_all_configs(sample_df):
    """Runs Orchestrator with configs/scenarios directory to confirm all 6 scenario types execute."""
    orchestrator = ScenarioOrchestrator(scenarios_dir="configs/scenarios")
    results = orchestrator.run_all_scenarios(sample_df)

    assert len(results) == 6
    types_found = {r["scenario_type"] for r in results}
    expected_types = {
        "supplier_failure", "price_increase", "holiday_sales",
        "new_product_launch", "transport_delay", "demand_surge"
    }
    assert types_found == expected_types


def test_run_simulation_pipeline_parquet_output():
    """Runs full simulation pipeline script and verifies scenario_comparison_report.parquet output."""
    df_report = run_simulation_pipeline(
        processed_dir="data/processed",
        scenarios_dir="configs/scenarios",
        output_filename="scenario_comparison_report.parquet"
    )

    output_path = os.path.join("data/processed", "scenario_comparison_report.parquet")
    assert os.path.exists(output_path)
    assert len(df_report) == 6
    assert "baseline_fill_rate_pct" in df_report.columns
    assert "scenario_fill_rate_pct" in df_report.columns
    assert "fill_rate_delta_pct" in df_report.columns
