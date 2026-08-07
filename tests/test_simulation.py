import pytest
from src.simulation.scenario_simulator import ScenarioSimulationEngine, ScenarioParameters


def test_scenario_simulation_supplier_delay():
    engine = ScenarioSimulationEngine()
    params = ScenarioParameters(
        scenario_name="Supplier Delay Test",
        supplier_delay_days=5.0,
        demand_surge_multiplier=1.2,
        simulation_horizon_days=30
    )

    res = engine.simulate_sku_scenario(
        base_daily_demand=10.0,
        current_stock=100.0,
        reorder_point=50.0,
        safety_stock=20.0,
        base_lead_time=7.0,
        unit_cost=15.0,
        params=params
    )

    assert res["scenario_name"] == "Supplier Delay Test"
    assert "summary_metrics" in res
    assert res["summary_metrics"]["scenario_stockout_days"] >= res["summary_metrics"]["baseline_stockout_days"]
    assert res["summary_metrics"]["extra_safety_stock_needed_units"] >= 0.0
    assert len(res["daily_trajectories"]["baseline"]) == 30
    assert len(res["daily_trajectories"]["scenario"]) == 30


def test_price_elasticity_demand_drop():
    engine = ScenarioSimulationEngine()
    # 20% price increase with -1.5 elasticity -> 30% demand drop
    params = ScenarioParameters(
        price_change_pct=20.0,
        price_elasticity=-1.5,
        simulation_horizon_days=14
    )

    res = engine.simulate_sku_scenario(
        base_daily_demand=100.0,
        current_stock=1000.0,
        reorder_point=200.0,
        safety_stock=50.0,
        base_lead_time=5.0,
        unit_cost=10.0,
        params=params
    )

    assert res["inputs"]["scenario_daily_demand"] == 70.0  # 100 * (1 - 0.3)
