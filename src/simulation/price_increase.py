import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.simulation.base import BaseScenario


class PriceIncreaseScenario(BaseScenario):
    """
    Scenario 2: Price Increase & Price Elasticity.
    Applies a price change percentage and price elasticity assumption to calculate
    demand reduction (% change demand = price_elasticity * % change price).
    """

    def __init__(
        self,
        scenario_name: str = "Unit Price Hike with Elasticity Adjustment",
        horizon_days: int = 30,
        price_change_pct: float = 15.0,
        price_elasticity: float = -1.2,
        product_id: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        super().__init__(
            scenario_name=scenario_name,
            scenario_type="price_increase",
            horizon_days=horizon_days,
            affected_entities={"product_id": product_id} if product_id else {"all_products": True},
            config_path=config_path
        )
        self.price_change_pct = self.parameters.get("price_change_pct", price_change_pct)
        self.price_elasticity = self.parameters.get("price_elasticity", price_elasticity)
        self.product_id = self.parameters.get("product_id", product_id)

    def apply_perturbation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies price hike percentage and price elasticity demand reduction perturbation."""
        df_pert = df.copy()

        if self.product_id and "product_id" in df_pert.columns:
            mask = df_pert["product_id"] == self.product_id
        else:
            mask = pd.Series(True, index=df_pert.index)

        if not mask.any():
            return df_pert

        cost_col = "unit_cost" if "unit_cost" in df_pert.columns else "price"
        demand_col = "avg_daily_demand" if "avg_daily_demand" in df_pert.columns else "demand"

        # % Change in demand = price_elasticity * (% change in price)
        demand_change_pct = self.price_elasticity * (self.price_change_pct / 100.0)
        demand_multiplier = max(0.0, 1.0 + demand_change_pct)

        if cost_col in df_pert.columns:
            df_pert.loc[mask, cost_col] = df_pert.loc[mask, cost_col] * (1.0 + self.price_change_pct / 100.0)

        if demand_col in df_pert.columns:
            df_pert.loc[mask, demand_col] = df_pert.loc[mask, demand_col] * demand_multiplier

        if "sales_velocity" in df_pert.columns:
            df_pert.loc[mask, "sales_velocity"] = df_pert.loc[mask, "sales_velocity"] * demand_multiplier

        return df_pert
