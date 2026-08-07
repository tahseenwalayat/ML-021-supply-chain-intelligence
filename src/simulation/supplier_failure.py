import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.simulation.base import BaseScenario


class SupplierFailureScenario(BaseScenario):
    """
    Scenario 1: Supplier Failure / Severe Delivery Delay.
    Increases delivery lead time by N days (or sets to high delay) for a target supplier or all suppliers.
    Recomputes lead time variance, stockout risk, and safety buffer requirements.
    """

    def __init__(
        self,
        scenario_name: str = "Supplier Failure / Severe Lead Time Delay",
        horizon_days: int = 30,
        supplier_id: Optional[str] = None,
        delay_days: float = 14.0,
        lead_time_std_multiplier: float = 2.5,
        config_path: Optional[str] = None
    ):
        super().__init__(
            scenario_name=scenario_name,
            scenario_type="supplier_failure",
            horizon_days=horizon_days,
            affected_entities={"supplier_id": supplier_id} if supplier_id else {"all_suppliers": True},
            config_path=config_path
        )
        self.delay_days = self.parameters.get("delay_days", delay_days)
        self.std_mult = self.parameters.get("lead_time_std_multiplier", lead_time_std_multiplier)
        self.supplier_id = self.parameters.get("supplier_id", supplier_id)

    def apply_perturbation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies supplier delivery lead time delay and variance inflation perturbation."""
        df_pert = df.copy()

        # Determine target mask
        if self.supplier_id and "supplier_id" in df_pert.columns:
            mask = df_pert["supplier_id"] == self.supplier_id
        else:
            mask = pd.Series(True, index=df_pert.index)

        if not mask.any():
            return df_pert

        lt_col = "avg_lead_time" if "avg_lead_time" in df_pert.columns else "lead_time"
        std_col = "lead_time_std_days" if "lead_time_std_days" in df_pert.columns else "lead_time_std"

        if lt_col in df_pert.columns:
            df_pert.loc[mask, lt_col] = df_pert.loc[mask, lt_col] + self.delay_days
        else:
            df_pert[lt_col] = 7.0 + self.delay_days

        if std_col in df_pert.columns:
            df_pert.loc[mask, std_col] = df_pert.loc[mask, std_col] * self.std_mult
        else:
            df_pert[std_col] = 2.0 * self.std_mult

        if "supplier_reliability_score" in df_pert.columns:
            df_pert.loc[mask, "supplier_reliability_score"] = df_pert.loc[mask, "supplier_reliability_score"] * 0.5

        return df_pert
