import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.simulation.base import BaseScenario


class TransportDelayScenario(BaseScenario):
    """
    Scenario 5: Regional Transport & Transit Delay.
    Increases delivery lead time by N days across an entire geographical region,
    recomputing reorder points and stockout exposure.
    """

    def __init__(
        self,
        scenario_name: str = "Regional Transit Disruptions & Delay",
        horizon_days: int = 30,
        region: str = "North",
        transit_delay_days: float = 6.0,
        config_path: Optional[str] = None
    ):
        super().__init__(
            scenario_name=scenario_name,
            scenario_type="transport_delay",
            horizon_days=horizon_days,
            affected_entities={"region": region},
            config_path=config_path
        )
        self.region = self.parameters.get("region", region)
        self.transit_delay_days = self.parameters.get("transit_delay_days", transit_delay_days)

    def apply_perturbation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies regional transit disruption lead time delay perturbation across target regions."""
        df_pert = df.copy()

        if self.region and "region" in df_pert.columns:
            mask = df_pert["region"] == self.region
        else:
            mask = pd.Series(True, index=df_pert.index)

        if not mask.any():
            return df_pert

        lt_col = "avg_lead_time" if "avg_lead_time" in df_pert.columns else "lead_time"
        std_col = "lead_time_std_days" if "lead_time_std_days" in df_pert.columns else "lead_time_std"

        if lt_col in df_pert.columns:
            df_pert.loc[mask, lt_col] = df_pert.loc[mask, lt_col] + self.transit_delay_days

        if std_col in df_pert.columns:
            df_pert.loc[mask, std_col] = df_pert.loc[mask, std_col] + (self.transit_delay_days * 0.3)

        return df_pert
