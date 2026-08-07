import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.utils.logging_config import get_logger

logger = get_logger("simulation.scenario_simulator")


class ScenarioParameters(BaseModel):
    """Parameter configuration for supply chain scenario simulation."""
    scenario_name: str = Field("Supplier Delay & Demand Surge", description="Name of simulation scenario")
    supplier_delay_days: float = Field(0.0, ge=0.0, description="Additional lead time delay in days")
    price_change_pct: float = Field(0.0, description="Percentage change in unit price (e.g. +10.0 for 10% hike)")
    price_elasticity: float = Field(-1.2, description="Price elasticity of demand (typically negative)")
    demand_surge_multiplier: float = Field(1.0, ge=0.0, description="Multiplier for baseline demand (e.g. 1.25 for +25%)")
    holiday_spike_multiplier: float = Field(1.0, ge=0.0, description="Additional multiplier for holiday period")
    transport_delay_days: float = Field(0.0, ge=0.0, description="Additional transit delay days")
    new_product_ramp_days: int = Field(14, ge=1, description="Days required for new product launch demand ramp-up")
    simulation_horizon_days: int = Field(30, ge=1, le=180, description="Simulation forecast horizon in days")


class ScenarioSimulationEngine:
    """
    Scenario Simulation Engine.
    Simulates operational stress tests across supply chain nodes including:
    - Supplier delays & transit disruptions
    - Price changes with price elasticity effects
    - Demand surges & holiday peaks
    - New product launch ramp-up curves
    Recalculates fill rate, stockout exposure, safety stock requirements, and tied-up capital.
    """

    def __init__(self, default_holding_cost_rate: float = 0.20):
        self.holding_cost_rate = default_holding_cost_rate

    def simulate_sku_scenario(
        self,
        base_daily_demand: float,
        current_stock: float,
        reorder_point: float,
        safety_stock: float,
        base_lead_time: float,
        unit_cost: float,
        params: ScenarioParameters
    ) -> Dict[str, Any]:
        """
        Simulates daily inventory trajectories under baseline vs scenario stress parameters over the simulation horizon.
        """
        horizon = params.simulation_horizon_days
        
        # 1. Price elasticity effect on daily demand
        # % Change in Demand = Price Elasticity * % Change in Price
        demand_price_adj = 1.0 + (params.price_elasticity * (params.price_change_pct / 100.0))
        demand_price_adj = max(0.0, demand_price_adj)

        # Adjusted daily demand under scenario
        scenario_daily_demand = (
            base_daily_demand
            * demand_price_adj
            * params.demand_surge_multiplier
            * params.holiday_spike_multiplier
        )

        # 2. Total adjusted lead time
        total_scenario_lead_time = base_lead_time + params.supplier_delay_days + params.transport_delay_days

        # 3. Simulate day-by-day inventory trajectory
        baseline_stock = current_stock
        scenario_stock = current_stock

        baseline_unmet_demand = 0.0
        scenario_unmet_demand = 0.0

        baseline_fulfilled_demand = 0.0
        scenario_fulfilled_demand = 0.0

        baseline_stockout_days = 0
        scenario_stockout_days = 0

        baseline_trajectory = []
        scenario_trajectory = []

        # Replenishment pipeline (day_arriving -> quantity)
        baseline_pending_orders = {}
        scenario_pending_orders = {}

        for day in range(1, horizon + 1):
            # Process arrivals
            if day in baseline_pending_orders:
                baseline_stock += baseline_pending_orders.pop(day)
            if day in scenario_pending_orders:
                scenario_stock += scenario_pending_orders.pop(day)

            # Baseline demand simulation
            b_demand = base_daily_demand
            if baseline_stock >= b_demand:
                baseline_stock -= b_demand
                baseline_fulfilled_demand += b_demand
            else:
                baseline_fulfilled_demand += max(0.0, baseline_stock)
                baseline_unmet_demand += (b_demand - max(0.0, baseline_stock))
                baseline_stock = 0.0
                baseline_stockout_days += 1

            # Scenario demand simulation
            s_demand = scenario_daily_demand
            if scenario_stock >= s_demand:
                scenario_stock -= s_demand
                scenario_fulfilled_demand += s_demand
            else:
                scenario_fulfilled_demand += max(0.0, scenario_stock)
                scenario_unmet_demand += (s_demand - max(0.0, scenario_stock))
                scenario_stock = 0.0
                scenario_stockout_days += 1

            # Trigger reorder if stock falls below ROP and no pending order
            if baseline_stock <= reorder_point and not baseline_pending_orders:
                arrival_day = day + int(np.ceil(base_lead_time))
                order_qty = base_daily_demand * (base_lead_time + 7)  # standard target stock order
                baseline_pending_orders[arrival_day] = order_qty

            if scenario_stock <= reorder_point and not scenario_pending_orders:
                arrival_day = day + int(np.ceil(total_scenario_lead_time))
                order_qty = scenario_daily_demand * (total_scenario_lead_time + 7)
                scenario_pending_orders[arrival_day] = order_qty

            baseline_trajectory.append({
                "day": day,
                "demand": round(b_demand, 2),
                "stock": round(baseline_stock, 2)
            })
            scenario_trajectory.append({
                "day": day,
                "demand": round(s_demand, 2),
                "stock": round(scenario_stock, 2)
            })

        # Calculate metrics
        tot_b_demand = base_daily_demand * horizon
        tot_s_demand = scenario_daily_demand * horizon

        baseline_fill_rate = float(np.clip(baseline_fulfilled_demand / max(1e-5, tot_b_demand) * 100.0, 0.0, 100.0))
        scenario_fill_rate = float(np.clip(scenario_fulfilled_demand / max(1e-5, tot_s_demand) * 100.0, 0.0, 100.0))

        # Additional required safety stock under scenario lead time and demand
        # SS = Z * sqrt(L * sigma_d^2 + d^2 * sigma_l^2) -> approximated delta
        req_scenario_safety_stock = safety_stock * np.sqrt(total_scenario_lead_time / max(1.0, base_lead_time)) * (scenario_daily_demand / max(1.0, base_daily_demand))
        extra_safety_stock_needed = max(0.0, req_scenario_safety_stock - safety_stock)
        extra_capital_required = extra_safety_stock_needed * unit_cost

        return {
            "scenario_name": params.scenario_name,
            "horizon_days": horizon,
            "inputs": {
                "base_daily_demand": base_daily_demand,
                "scenario_daily_demand": round(scenario_daily_demand, 2),
                "base_lead_time": base_lead_time,
                "scenario_lead_time": total_scenario_lead_time,
                "unit_cost": unit_cost
            },
            "summary_metrics": {
                "baseline_fill_rate_pct": round(baseline_fill_rate, 2),
                "scenario_fill_rate_pct": round(scenario_fill_rate, 2),
                "fill_rate_delta_pct": round(scenario_fill_rate - baseline_fill_rate, 2),
                "baseline_stockout_days": baseline_stockout_days,
                "scenario_stockout_days": scenario_stockout_days,
                "stockout_days_increase": scenario_stockout_days - baseline_stockout_days,
                "extra_safety_stock_needed_units": round(extra_safety_stock_needed, 2),
                "extra_capital_required_usd": round(extra_capital_required, 2)
            },
            "daily_trajectories": {
                "baseline": baseline_trajectory,
                "scenario": scenario_trajectory
            }
        }
