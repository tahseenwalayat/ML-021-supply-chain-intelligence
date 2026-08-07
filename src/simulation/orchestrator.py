import os
import glob
import yaml
import pandas as pd
from typing import Dict, Any, List, Optional

from src.utils.logging_config import get_logger
from src.simulation.base import BaseScenario
from src.simulation.supplier_failure import SupplierFailureScenario
from src.simulation.price_increase import PriceIncreaseScenario
from src.simulation.holiday_sales import HolidaySalesScenario
from src.simulation.new_product_launch import NewProductLaunchScenario
from src.simulation.transport_delay import TransportDelayScenario
from src.simulation.demand_surge import DemandSurgeScenario

logger = get_logger("simulation.orchestrator")


SCENARIO_REGISTRY = {
    "supplier_failure": SupplierFailureScenario,
    "price_increase": PriceIncreaseScenario,
    "holiday_sales": HolidaySalesScenario,
    "new_product_launch": NewProductLaunchScenario,
    "transport_delay": TransportDelayScenario,
    "demand_surge": DemandSurgeScenario
}


class ScenarioOrchestrator:
    """
    Scenario Simulation Orchestrator.
    Loads scenario configurations from YAML files or programmatic objects,
    instantiates scenario classes, executes stress tests against baseline data,
    and formats before/after comparison summaries.
    """

    def __init__(self, scenarios_dir: str = "configs/scenarios"):
        self.scenarios_dir = scenarios_dir

    def load_scenario_from_config(self, yaml_path: str) -> BaseScenario:
        """Loads a YAML scenario config and instantiates the matching scenario type."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        s_type = cfg.get("scenario_type")
        if s_type not in SCENARIO_REGISTRY:
            raise ValueError(f"Unknown scenario type '{s_type}' in config '{yaml_path}'. Registered types: {list(SCENARIO_REGISTRY.keys())}")

        scenario_cls = SCENARIO_REGISTRY[s_type]
        instance = scenario_cls(config_path=yaml_path)
        return instance

    def run_all_scenarios(
        self,
        df_baseline: pd.DataFrame,
        scenarios_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Loads all scenario YAML configs from scenarios_dir and executes simulations.
        """
        s_dir = scenarios_dir or self.scenarios_dir
        yaml_files = sorted(glob.glob(os.path.join(s_dir, "*.yaml")))

        if not yaml_files:
            logger.warning(f"No scenario YAML configs found in '{s_dir}'. Using built-in default scenario instances.")
            scenarios = [
                SupplierFailureScenario(),
                PriceIncreaseScenario(),
                HolidaySalesScenario(),
                NewProductLaunchScenario(),
                TransportDelayScenario(),
                DemandSurgeScenario()
            ]
        else:
            scenarios = [self.load_scenario_from_config(f) for f in yaml_files]

        results = []
        for scenario in scenarios:
            res = scenario.run(df_baseline)
            results.append(res)

        return results


def run_simulation_pipeline(
    processed_dir: str = "data/processed",
    scenarios_dir: str = "configs/scenarios",
    output_filename: str = "scenario_comparison_report.parquet"
) -> pd.DataFrame:
    """
    Runs full batch simulation pipeline across all 6 scenario types.
    Exports before/after scenario comparison report to parquet.
    """
    logger.info("=== Starting Scenario Simulation Orchestrator Pipeline ===")

    inv_path = os.path.join(processed_dir, "inventory_recommendations.parquet")
    risk_path = os.path.join(processed_dir, "risk_scores.parquet")

    if os.path.exists(inv_path):
        df_base = pd.read_parquet(inv_path)
    elif os.path.exists(risk_path):
        df_base = pd.read_parquet(risk_path)
    else:
        df_base = pd.DataFrame([
            {
                "product_id": f"SKU_{i:04d}",
                "warehouse_id": "WH_NORTH",
                "region": "North",
                "category": "Electronics",
                "avg_daily_demand": 10.0,
                "current_stock": 50.0,
                "reorder_point": 100.0,
                "safety_stock": 30.0,
                "avg_lead_time": 7.0,
                "unit_cost": 20.0,
                "sales_velocity": 10.0,
                "zero_sales_weeks": 0
            } for i in range(1, 21)
        ])

    orchestrator = ScenarioOrchestrator(scenarios_dir=scenarios_dir)
    sim_results = orchestrator.run_all_scenarios(df_base)

    summary_rows = []
    for r in sim_results:
        summary_rows.append({
            "scenario_name": r["scenario_name"],
            "scenario_type": r["scenario_type"],
            "horizon_days": r["horizon_days"],
            "baseline_fill_rate_pct": r["baseline_metrics"]["mean_fill_rate_pct"],
            "scenario_fill_rate_pct": r["scenario_metrics"]["mean_fill_rate_pct"],
            "fill_rate_delta_pct": r["impact_deltas"]["fill_rate_delta_pct"],
            "baseline_stockout_risk": r["baseline_metrics"]["mean_stockout_risk_score"],
            "scenario_stockout_risk": r["scenario_metrics"]["mean_stockout_risk_score"],
            "stockout_risk_delta": r["impact_deltas"]["stockout_risk_score_delta"],
            "stockout_days_increase": r["impact_deltas"]["stockout_days_increase"],
            "extra_capital_required_usd": r["impact_deltas"]["extra_capital_required_usd"]
        })

    df_report = pd.DataFrame(summary_rows)

    output_path = os.path.join(processed_dir, output_filename)
    os.makedirs(processed_dir, exist_ok=True)
    df_report.to_parquet(output_path, index=False)

    logger.info(f"Successfully generated scenario comparison report '{output_path}' with {len(df_report)} scenario runs.")
    return df_report


if __name__ == "__main__":
    run_simulation_pipeline()
