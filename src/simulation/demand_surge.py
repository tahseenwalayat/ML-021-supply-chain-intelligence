import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.simulation.base import BaseScenario


class DemandSurgeScenario(BaseScenario):
    """
    Scenario 6: Demand Surge / Macroeconomic Demand Shock.
    Applies a multiplicative demand shock across all or specific SKUs/warehouses,
    rerunning inventory fill rates and stockout risk calculations.
    """

    def __init__(
        self,
        scenario_name: str = "Macroeconomic Demand Surge Shock",
        horizon_days: int = 30,
        demand_surge_multiplier: float = 1.50,
        warehouse_id: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        super().__init__(
            scenario_name=scenario_name,
            scenario_type="demand_surge",
            horizon_days=horizon_days,
            affected_entities={"warehouse_id": warehouse_id} if warehouse_id else {"all_warehouses": True},
            config_path=config_path
        )
        self.demand_surge_multiplier = self.parameters.get("demand_surge_multiplier", demand_surge_multiplier)
        self.warehouse_id = self.parameters.get("warehouse_id", warehouse_id)

    def apply_perturbation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies multiplicative demand surge perturbation to demand parameters."""
        df_pert = df.copy()

        if self.warehouse_id and "warehouse_id" in df_pert.columns:
            mask = df_pert["warehouse_id"] == self.warehouse_id
        else:
            mask = pd.Series(True, index=df_pert.index)

        if not mask.any():
            return df_pert

        demand_col = "avg_daily_demand" if "avg_daily_demand" in df_pert.columns else "demand"
        std_col = "std_daily_demand" if "std_daily_demand" in df_pert.columns else "std_demand"

        if demand_col in df_pert.columns:
            df_pert.loc[mask, demand_col] = df_pert.loc[mask, demand_col] * self.demand_surge_multiplier

        if std_col in df_pert.columns:
            df_pert.loc[mask, std_col] = df_pert.loc[mask, std_col] * np.sqrt(self.demand_surge_multiplier)

        if "sales_velocity" in df_pert.columns:
            df_pert.loc[mask, "sales_velocity"] = df_pert.loc[mask, "sales_velocity"] * self.demand_surge_multiplier

        return df_pert
