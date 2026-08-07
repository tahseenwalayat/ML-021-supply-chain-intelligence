import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from src.utils.logging_config import get_logger
from src.simulation.scenario_simulator import ScenarioSimulationEngine, ScenarioParameters

logger = get_logger("simulation.run_simulation")


def run_batch_scenario_simulation(
    processed_dir: str = "data/processed",
    output_filename: str = "scenario_simulation_results.parquet",
    scenario_params: Optional[ScenarioParameters] = None
) -> pd.DataFrame:
    """
    Runs batch operational scenario simulation across all product-warehouse pairs.
    Reads inventory_recommendations.parquet and risk_scores.parquet, applies stress-test
    scenarios (supplier delay, demand surge, price elasticity), and generates projected metrics.
    """
    logger.info("=== Starting Batch Scenario Simulation Pipeline ===")
    engine = ScenarioSimulationEngine()

    if scenario_params is None:
        scenario_params = ScenarioParameters(
            scenario_name="Stress Test: 5-Day Supplier Delay + 20% Demand Surge",
            supplier_delay_days=5.0,
            demand_surge_multiplier=1.20,
            price_change_pct=0.0,
            simulation_horizon_days=30
        )

    # 1. Load inputs
    inv_path = os.path.join(processed_dir, "inventory_recommendations.parquet")
    risk_path = os.path.join(processed_dir, "risk_scores.parquet")

    if os.path.exists(inv_path):
        df_base = pd.read_parquet(inv_path)
    elif os.path.exists(risk_path):
        df_base = pd.read_parquet(risk_path)
    else:
        logger.info("Generating synthetic SKU data for scenario simulation demonstration...")
        df_base = pd.DataFrame([
            {
                "product_id": f"SKU_{i:04d}",
                "warehouse_id": "WH_NORTH",
                "allocated_daily_demand": 10.0,
                "current_stock": 50.0,
                "reorder_point": 100.0,
                "safety_stock": 30.0,
                "avg_lead_time": 7.0,
                "unit_cost": 20.0
            } for i in range(1, 21)
        ])

    # Standardize column mappings
    demand_col = "allocated_daily_demand" if "allocated_daily_demand" in df_base.columns else "avg_daily_demand"
    d_series = df_base[demand_col].fillna(5.0).clip(lower=0.1) if demand_col in df_base.columns else pd.Series(5.0, index=df_base.index)
    c_stock_series = df_base["current_stock"].fillna(50.0).clip(lower=0.0) if "current_stock" in df_base.columns else pd.Series(50.0, index=df_base.index)
    rop_series = df_base["reorder_point"].fillna(100.0).clip(lower=1.0) if "reorder_point" in df_base.columns else pd.Series(100.0, index=df_base.index)
    ss_series = df_base["safety_stock"].fillna(30.0).clip(lower=0.0) if "safety_stock" in df_base.columns else pd.Series(30.0, index=df_base.index)
    lt_series = df_base["avg_lead_time"].fillna(7.0).clip(lower=1.0) if "avg_lead_time" in df_base.columns else pd.Series(7.0, index=df_base.index)
    cost_series = df_base["unit_cost"].fillna(15.0).clip(lower=0.0) if "unit_cost" in df_base.columns else pd.Series(15.0, index=df_base.index)

    results = []
    logger.info(f"Simulating scenario '{scenario_params.scenario_name}' across {len(df_base)} items...")

    for i in range(len(df_base)):
        res = engine.simulate_sku_scenario(
            base_daily_demand=float(d_series.iloc[i]),
            current_stock=float(c_stock_series.iloc[i]),
            reorder_point=float(rop_series.iloc[i]),
            safety_stock=float(ss_series.iloc[i]),
            base_lead_time=float(lt_series.iloc[i]),
            unit_cost=float(cost_series.iloc[i]),
            params=scenario_params
        )

        row_dict = {
            "product_id": df_base["product_id"].iloc[i] if "product_id" in df_base.columns else f"P_{i}",
            "warehouse_id": df_base["warehouse_id"].iloc[i] if "warehouse_id" in df_base.columns else "W1",
            "scenario_name": scenario_params.scenario_name,
            "baseline_fill_rate_pct": res["summary_metrics"]["baseline_fill_rate_pct"],
            "scenario_fill_rate_pct": res["summary_metrics"]["scenario_fill_rate_pct"],
            "fill_rate_delta_pct": res["summary_metrics"]["fill_rate_delta_pct"],
            "baseline_stockout_days": res["summary_metrics"]["baseline_stockout_days"],
            "scenario_stockout_days": res["summary_metrics"]["scenario_stockout_days"],
            "stockout_days_increase": res["summary_metrics"]["stockout_days_increase"],
            "extra_safety_stock_needed_units": res["summary_metrics"]["extra_safety_stock_needed_units"],
            "extra_capital_required_usd": res["summary_metrics"]["extra_capital_required_usd"]
        }
        results.append(row_dict)

    df_res = pd.DataFrame(results)

    # Save artifact
    output_path = os.path.join(processed_dir, output_filename)
    os.makedirs(processed_dir, exist_ok=True)
    df_res.to_parquet(output_path, index=False)

    logger.info(
        f"Batch simulation complete. Saved output to '{output_path}'. "
        f"Average Fill Rate Delta: {df_res['fill_rate_delta_pct'].mean():.2f}%. "
        f"Total Extra Capital Required: ${df_res['extra_capital_required_usd'].sum():,.2f}"
    )

    return df_res


if __name__ == "__main__":
    run_batch_scenario_simulation()
