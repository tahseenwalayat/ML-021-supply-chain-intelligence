import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.simulation.base import BaseScenario


class NewProductLaunchScenario(BaseScenario):
    """
    Scenario 4: New Product Launch (Cold-Start Proxy).
    Uses category-average demand profiles as a cold-start proxy for a new SKU,
    simulating demand ramp-up over a launch period.
    """

    def __init__(
        self,
        scenario_name: str = "New SKU Product Launch Cold-Start Ramp",
        horizon_days: int = 30,
        new_sku_id: str = "NEW_SKU_2026",
        category: str = "Electronics",
        initial_stock: float = 200.0,
        launch_ramp_days: int = 14,
        target_daily_demand: float = 15.0,
        config_path: Optional[str] = None
    ):
        super().__init__(
            scenario_name=scenario_name,
            scenario_type="new_product_launch",
            horizon_days=horizon_days,
            affected_entities={"new_sku_id": new_sku_id, "category": category},
            config_path=config_path
        )
        self.new_sku_id = self.parameters.get("new_sku_id", new_sku_id)
        self.category = self.parameters.get("category", category)
        self.initial_stock = self.parameters.get("initial_stock", initial_stock)
        self.launch_ramp_days = self.parameters.get("launch_ramp_days", launch_ramp_days)
        self.target_daily_demand = self.parameters.get("target_daily_demand", target_daily_demand)

    def apply_perturbation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Injects a new product launch cold-start profile into the dataframe."""
        df_pert = df.copy()

        # Compute category average daily demand as proxy if target not specified
        if "category" in df_pert.columns and "avg_daily_demand" in df_pert.columns:
            cat_df = df_pert[df_pert["category"] == self.category]
            if len(cat_df) > 0:
                proxy_demand = float(cat_df["avg_daily_demand"].mean())
            else:
                proxy_demand = float(df_pert["avg_daily_demand"].mean())
        else:
            proxy_demand = self.target_daily_demand

        # Ramp up factor over launch horizon
        avg_ramped_demand = proxy_demand * 0.70

        new_row = {
            "product_id": self.new_sku_id,
            "warehouse_id": "WH_MAIN",
            "region": "North",
            "category": self.category,
            "current_stock": self.initial_stock,
            "reorder_point": proxy_demand * 10,
            "safety_stock": proxy_demand * 3,
            "avg_daily_demand": avg_ramped_demand,
            "std_daily_demand": proxy_demand * 0.25,
            "avg_lead_time": 7.0,
            "lead_time_std_days": 1.5,
            "supplier_reliability_score": 0.90,
            "unit_cost": 25.0,
            "sales_velocity": avg_ramped_demand,
            "zero_sales_weeks": 0
        }

        # Append new SKU row to DataFrame
        return pd.concat([df_pert, pd.DataFrame([new_row])], ignore_index=True)
