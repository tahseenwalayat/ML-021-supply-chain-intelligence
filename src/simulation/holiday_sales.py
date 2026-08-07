import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.simulation.base import BaseScenario


class HolidaySalesScenario(BaseScenario):
    """
    Scenario 3: Holiday & Peak Seasonal Sales Surge.
    Applies seasonal multipliers (e.g., Black Friday / Cyber Monday surge) to demand
    for specified product categories or regions over the holiday period.
    """

    def __init__(
        self,
        scenario_name: str = "Holiday / Cyber Week Demand Spike",
        horizon_days: int = 30,
        demand_multiplier: float = 1.75,
        category: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        super().__init__(
            scenario_name=scenario_name,
            scenario_type="holiday_sales",
            horizon_days=horizon_days,
            affected_entities={"category": category} if category else {"all_categories": True},
            config_path=config_path
        )
        self.demand_multiplier = self.parameters.get("demand_multiplier", demand_multiplier)
        self.category = self.parameters.get("category", category)

    def apply_perturbation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies holiday seasonal demand multiplier perturbation to specified product categories."""
        df_pert = df.copy()

        if self.category and "category" in df_pert.columns:
            mask = df_pert["category"] == self.category
        else:
            mask = pd.Series(True, index=df_pert.index)

        if not mask.any():
            return df_pert

        demand_col = "avg_daily_demand" if "avg_daily_demand" in df_pert.columns else "demand"
        std_col = "std_daily_demand" if "std_daily_demand" in df_pert.columns else "std_demand"

        if demand_col in df_pert.columns:
            df_pert.loc[mask, demand_col] = df_pert.loc[mask, demand_col] * self.demand_multiplier

        if std_col in df_pert.columns:
            df_pert.loc[mask, std_col] = df_pert.loc[mask, std_col] * np.sqrt(self.demand_multiplier)

        return df_pert
