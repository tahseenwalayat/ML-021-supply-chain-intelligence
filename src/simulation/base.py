import yaml
import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from src.utils.logging_config import get_logger
from src.risk.stockout_risk import compute_stockout_risk_df
from src.risk.overstock_risk import compute_overstock_risk_df
from src.risk.risk_engine import classify_risk_level

logger = get_logger("simulation.base")


class BaseScenario(ABC):
    """
    Abstract Base Class for all Supply Chain Stress-Test Scenarios.
    Defines generic interface: input perturbation + affected entities + simulation horizon.
    """

    def __init__(
        self,
        scenario_name: str,
        scenario_type: str,
        horizon_days: int = 30,
        affected_entities: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        self.scenario_name = scenario_name
        self.scenario_type = scenario_type
        self.horizon_days = horizon_days
        self.affected_entities = affected_entities or {}
        self.config_path = config_path
        self.parameters: Dict[str, Any] = {}

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """Loads scenario configuration from a YAML file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    self.scenario_name = cfg.get("scenario_name", self.scenario_name)
                    self.scenario_type = cfg.get("scenario_type", self.scenario_type)
                    self.horizon_days = cfg.get("horizon_days", self.horizon_days)
                    self.affected_entities = cfg.get("affected_entities", self.affected_entities)
                    self.parameters = cfg.get("parameters", {})
        except Exception as e:
            logger.warning(f"Failed to load scenario config '{config_path}': {e}")

    @abstractmethod
    def apply_perturbation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies scenario-specific perturbation to the baseline inventory/demand DataFrame.
        Must return a copy of df with perturbed parameters (e.g. lead_time, demand, price).
        """
        pass

    def evaluate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Re-evaluates inventory and risk metrics using the existing risk/inventory engines.
        """
        df_eval = df.copy()

        # Re-compute stockout & overstock risks using existing risk engine functions
        df_eval = compute_stockout_risk_df(df_eval)
        df_eval = compute_overstock_risk_df(df_eval)

        c_stock = df_eval["current_stock"].fillna(0.0) if "current_stock" in df_eval.columns else pd.Series(0.0, index=df_eval.index)
        d_mean = df_eval["avg_daily_demand"].fillna(1.0).clip(lower=0.001) if "avg_daily_demand" in df_eval.columns else pd.Series(1.0, index=df_eval.index)
        rop = df_eval["reorder_point"].fillna(10.0) if "reorder_point" in df_eval.columns else pd.Series(10.0, index=df_eval.index)
        unit_cost = df_eval["unit_cost"].fillna(15.0) if "unit_cost" in df_eval.columns else pd.Series(15.0, index=df_eval.index)
        lt = df_eval["avg_lead_time"].fillna(7.0) if "avg_lead_time" in df_eval.columns else pd.Series(7.0, index=df_eval.index)

        # Days of supply & stockout exposure days over horizon
        days_supply = c_stock / d_mean
        stockout_days = np.clip(np.ceil(np.maximum(0.0, self.horizon_days - days_supply)), 0, self.horizon_days)

        # Expected fill rate over horizon
        total_horizon_demand = d_mean * self.horizon_days
        fulfilled_demand = np.minimum(total_horizon_demand, c_stock + (c_stock > 0) * (d_mean * np.maximum(0.0, self.horizon_days - lt)))
        fill_rate_pct = np.clip((fulfilled_demand / np.maximum(1e-5, total_horizon_demand)) * 100.0, 0.0, 100.0)

        # Additional safety stock needed
        required_safety_stock = d_mean * np.sqrt(lt) * 1.65
        extra_capital = np.maximum(0.0, required_safety_stock - c_stock) * unit_cost

        return {
            "mean_stockout_risk_score": float(np.round(df_eval["stockout_risk_score"].mean(), 4)),
            "mean_overstock_risk_score": float(np.round(df_eval["overstock_risk_score"].mean(), 4)),
            "mean_fill_rate_pct": float(np.round(fill_rate_pct.mean(), 2)),
            "total_stockout_days": int(stockout_days.sum()),
            "total_tied_up_capital_usd": float(np.round(df_eval["tied_up_capital"].sum(), 2)),
            "total_extra_capital_required_usd": float(np.round(extra_capital.sum(), 2)),
            "critical_risk_items_count": int((df_eval["stockout_risk_score"] >= 0.75).sum())
        }

    def run(self, df_baseline: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes full simulation comparison: baseline vs scenario.
        Returns a structured dictionary with before/after comparison metrics and daily trajectories.
        """
        logger.info(f"Running scenario '{self.scenario_name}' ({self.scenario_type})...")

        # 1. Baseline Evaluation
        baseline_metrics = self.evaluate_metrics(df_baseline)

        # 2. Apply Perturbation
        df_scenario = self.apply_perturbation(df_baseline)

        # 3. Scenario Evaluation
        scenario_metrics = self.evaluate_metrics(df_scenario)

        # 4. Before/After Deltas
        deltas = {
            "stockout_risk_score_delta": round(scenario_metrics["mean_stockout_risk_score"] - baseline_metrics["mean_stockout_risk_score"], 4),
            "fill_rate_delta_pct": round(scenario_metrics["mean_fill_rate_pct"] - baseline_metrics["mean_fill_rate_pct"], 2),
            "stockout_days_increase": scenario_metrics["total_stockout_days"] - baseline_metrics["total_stockout_days"],
            "tied_up_capital_delta_usd": round(scenario_metrics["total_tied_up_capital_usd"] - baseline_metrics["total_tied_up_capital_usd"], 2),
            "extra_capital_required_usd": scenario_metrics["total_extra_capital_required_usd"]
        }

        return {
            "scenario_name": self.scenario_name,
            "scenario_type": self.scenario_type,
            "horizon_days": self.horizon_days,
            "affected_entities": self.affected_entities,
            "parameters": self.parameters,
            "baseline_metrics": baseline_metrics,
            "scenario_metrics": scenario_metrics,
            "impact_deltas": deltas
        }
