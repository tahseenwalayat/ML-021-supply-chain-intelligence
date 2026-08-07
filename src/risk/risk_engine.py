import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from src.utils.logging_config import get_logger
from src.risk.supplier_delay_risk import (
    calculate_supplier_delay_risk,
    evaluate_supplier_delay_details,
    compute_supplier_delay_risk_df,
    classify_risk_level,
    load_risk_config
)
from src.risk.stockout_risk import (
    calculate_stockout_risk,
    evaluate_stockout_details,
    compute_stockout_risk_df
)
from src.risk.overstock_risk import (
    calculate_overstock_risk,
    evaluate_overstock_details,
    compute_overstock_risk_df
)
from src.risk.inventory_health_risk import (
    calculate_inventory_health_risk,
    compute_inventory_health_risk_df
)
from src.risk.demand_anomaly_risk import (
    calculate_demand_anomaly_risk,
    compute_demand_anomaly_risk_df
)

logger = get_logger("risk.risk_engine")


class SupplyChainRiskEngine:
    """
    Enterprise Supply Chain Risk Engine.
    Evaluates multi-dimensional operational supply chain risks across inventory nodes,
    SKUs, and suppliers, computing composite risk scores, severity levels, actionable
    mitigation recommendations, and prioritized alert feeds.
    """

    DEFAULT_WEIGHTS = {
        "supplier_delay": 0.25,
        "stockout": 0.30,
        "overstock": 0.15,
        "inventory_health": 0.15,
        "demand_anomaly": 0.15
    }

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        weights: Optional[Dict[str, float]] = None
    ):
        self.config_path = config_path
        self.config = load_risk_config(config_path)

        # Set risk dimension weights
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if weights:
            self.weights.update(weights)

        # Normalize weights to sum to 1.0
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}

    def evaluate_item_risk(
        self,
        product_id: str = "P1",
        warehouse_id: str = "W1",
        region: str = "North",
        supplier_id: str = "SUP1",
        current_stock: float = 50.0,
        reorder_point: float = 100.0,
        safety_stock: float = 30.0,
        avg_daily_demand: float = 10.0,
        avg_lead_time: float = 7.0,
        lead_time_std_days: float = 2.0,
        supplier_reliability_score: float = 0.85,
        late_delivery_rate: Optional[float] = None,
        unit_cost: float = 15.0,
        sales_velocity: float = 10.0,
        zero_sales_weeks: int = 0,
        demand_val: Optional[float] = None,
        mean_demand: Optional[float] = None,
        std_demand: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates complete 5-dimensional supply chain risk for a single item/node.
        """
        if demand_val is None:
            demand_val = avg_daily_demand
        if mean_demand is None:
            mean_demand = avg_daily_demand
        if std_demand is None:
            std_demand = max(1.0, avg_daily_demand * 0.2)

        # 1. Supplier Delay Risk
        sup_delay = evaluate_supplier_delay_details(
            reliability_score=supplier_reliability_score,
            lead_time_std=lead_time_std_days,
            lead_time_avg=avg_lead_time,
            late_delivery_rate=late_delivery_rate,
            w_late=self.config.get("supplier_weight_late_rate", 0.6),
            w_var=self.config.get("supplier_weight_delay_var", 0.4)
        )

        # 2. Stockout Risk
        stockout = evaluate_stockout_details(
            current_stock=current_stock,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
            avg_daily_demand=avg_daily_demand,
            lead_time_days=avg_lead_time
        )

        # 3. Overstock Risk
        overstock = evaluate_overstock_details(
            current_stock=current_stock,
            reorder_point=reorder_point,
            unit_cost=unit_cost,
            avg_daily_demand=avg_daily_demand,
            config_path=self.config_path
        )

        # 4. Inventory Health Risk
        health = calculate_inventory_health_risk(
            sales_velocity=sales_velocity,
            zero_sales_weeks=zero_sales_weeks,
            current_stock=current_stock,
            unit_cost=unit_cost,
            config_path=self.config_path
        )

        # 5. Demand Anomaly Risk
        anomaly = calculate_demand_anomaly_risk(
            demand_val=demand_val,
            mean_demand=mean_demand,
            std_demand=std_demand,
            config_path=self.config_path
        )

        # Composite Risk Score Calculation
        comp_score = (
            self.weights["supplier_delay"] * sup_delay["supplier_delay_risk_score"] +
            self.weights["stockout"] * stockout["stockout_risk_score"] +
            self.weights["overstock"] * overstock["overstock_risk_score"] +
            self.weights["inventory_health"] * health["inventory_health_risk_score"] +
            self.weights["demand_anomaly"] * anomaly["demand_anomaly_risk_score"]
        )

        comp_score = float(np.round(np.clip(comp_score, 0.0, 1.0), 4))
        overall_level = classify_risk_level(comp_score)

        # Generate Actionable Recommendations
        recommendations = self._generate_recommendations(
            sup_delay=sup_delay,
            stockout=stockout,
            overstock=overstock,
            health=health,
            anomaly=anomaly,
            overall_level=overall_level
        )

        return {
            "product_id": product_id,
            "warehouse_id": warehouse_id,
            "region": region,
            "supplier_id": supplier_id,
            "composite_risk_score": comp_score,
            "overall_risk_level": overall_level,
            "dimension_scores": {
                "supplier_delay_risk_score": sup_delay["supplier_delay_risk_score"],
                "stockout_risk_score": stockout["stockout_risk_score"],
                "overstock_risk_score": overstock["overstock_risk_score"],
                "inventory_health_risk_score": health["inventory_health_risk_score"],
                "demand_anomaly_risk_score": anomaly["demand_anomaly_risk_score"]
            },
            "dimension_levels": {
                "supplier_delay_risk_level": sup_delay["supplier_delay_risk_level"],
                "stockout_risk_level": stockout["stockout_risk_level"],
                "overstock_risk_level": overstock["overstock_risk_level"],
                "inventory_health_risk_level": health["inventory_health_risk_level"],
                "demand_anomaly_risk_level": anomaly["demand_anomaly_risk_level"]
            },
            "details": {
                "supplier_delay": sup_delay,
                "stockout": stockout,
                "overstock": overstock,
                "inventory_health": health,
                "demand_anomaly": anomaly
            },
            "recommendations": recommendations,
            "is_critical": overall_level == "CRITICAL"
        }

    def _generate_recommendations(
        self,
        sup_delay: Dict[str, Any],
        stockout: Dict[str, Any],
        overstock: Dict[str, Any],
        health: Dict[str, Any],
        anomaly: Dict[str, Any],
        overall_level: str
    ) -> List[str]:
        recs = []

        if stockout["is_out_of_stock"]:
            recs.append("CRITICAL: SKU is OUT OF STOCK. Issue emergency purchase order & execute inter-warehouse transfer immediately.")
        elif stockout["is_safety_stock_breached"]:
            recs.append("HIGH STOCKOUT RISK: Safety stock breached. Expedite open purchase orders and reallocate stock from surplus nodes.")
        elif stockout["is_reorder_point_breached"]:
            recs.append("REORDER TRIGGER: Current inventory below Reorder Point. Place replenishment purchase order.")

        if sup_delay["is_high_risk"]:
            buf = sup_delay.get("recommended_buffer_days", 3.0)
            recs.append(f"SUPPLIER RISK: High supplier lead-time variance. Increase safety stock buffer by {buf} days and evaluate secondary backup suppliers.")

        if overstock["is_overstocked"]:
            cap = overstock.get("tied_up_capital", 0.0)
            recs.append(f"OVERSTOCK EXPOSURE: ${cap:,.2f} tied up in excess stock. Pause purchase orders and trigger inter-warehouse redistribution.")

        if health["is_dead_stock"]:
            val = health.get("holding_value", 0.0)
            recs.append(f"DEAD STOCK: 12+ weeks of zero sales (${val:,.2f} valuation). Initiate clearance liquidation or inventory write-down.")
        elif health["is_slow_moving"]:
            recs.append("SLOW MOVING: Low sales velocity. Reduce reorder batch sizes and implement promotional discounting.")

        if anomaly["is_anomaly"]:
            atype = anomaly["anomaly_type"]
            recs.append(f"DEMAND ANOMALY DETECTED ({atype}): Significant statistical demand shift. Adjust baseline forecast models and verify regional promotional events.")

        if not recs and overall_level == "LOW":
            recs.append("OPERATIONAL HEALTHY: Inventory and supplier performance metrics are within normal parameters.")

        return recs

    def evaluate_supply_chain_risk_df(
        self,
        df: pd.DataFrame,
        stock_col: str = "current_stock",
        rop_col: str = "reorder_point",
        ss_col: str = "safety_stock",
        demand_col: str = "avg_daily_demand",
        avg_lead_time_col: str = "avg_lead_time",
        std_lead_time_col: str = "lead_time_std_days",
        reliability_col: str = "supplier_reliability_score",
        late_rate_col: str = "late_delivery_rate",
        cost_col: str = "unit_cost",
        velocity_col: str = "sales_velocity",
        zero_weeks_col: str = "zero_sales_weeks",
        observed_demand_col: str = "demand",
        mean_demand_col: str = "mean_demand",
        std_demand_col: str = "std_demand"
    ) -> pd.DataFrame:
        """
        Runs full vectorized risk evaluation across a DataFrame of SKUs/warehouses/suppliers.
        Adds composite risk scores, dimension scores, overall risk levels, and top recommendations.
        """
        df_res = df.copy()

        # Run dimension models
        df_res = compute_supplier_delay_risk_df(
            df_res,
            reliability_col=reliability_col,
            std_lead_time_col=std_lead_time_col,
            avg_lead_time_col=avg_lead_time_col,
            config_path=self.config_path,
            output_col="supplier_delay_risk_score"
        )

        df_res = compute_stockout_risk_df(
            df_res,
            stock_col=stock_col,
            rop_col=rop_col,
            ss_col=ss_col,
            demand_col=demand_col,
            lead_time_col=avg_lead_time_col,
            output_col="stockout_risk_score"
        )

        df_res = compute_overstock_risk_df(
            df_res,
            stock_col=stock_col,
            rop_col=rop_col,
            cost_col=cost_col,
            demand_col=demand_col,
            config_path=self.config_path,
            output_col="overstock_risk_score"
        )

        df_res = compute_inventory_health_risk_df(
            df_res,
            velocity_col=velocity_col if velocity_col in df_res.columns else demand_col,
            zero_weeks_col=zero_weeks_col,
            stock_col=stock_col,
            cost_col=cost_col,
            config_path=self.config_path,
            output_col="inventory_health_risk_score"
        )

        df_res = compute_demand_anomaly_risk_df(
            df_res,
            demand_col=observed_demand_col if observed_demand_col in df_res.columns else demand_col,
            mean_col=mean_demand_col if mean_demand_col in df_res.columns else demand_col,
            std_col=std_demand_col,
            config_path=self.config_path,
            output_col="demand_anomaly_risk_score"
        )

        # Composite score calculation
        comp = (
            self.weights["supplier_delay"] * df_res["supplier_delay_risk_score"] +
            self.weights["stockout"] * df_res["stockout_risk_score"] +
            self.weights["overstock"] * df_res["overstock_risk_score"] +
            self.weights["inventory_health"] * df_res["inventory_health_risk_score"] +
            self.weights["demand_anomaly"] * df_res["demand_anomaly_risk_score"]
        )

        df_res["composite_risk_score"] = np.round(np.clip(comp, 0.0, 1.0), 4)
        df_res["overall_risk_level"] = [classify_risk_level(v) for v in df_res["composite_risk_score"]]

        return df_res

    def generate_risk_summary(self, df_evaluated: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes summary statistics across evaluated supply chain risk dataset.
        """
        total_items = len(df_evaluated)
        if total_items == 0:
            return {"total_items": 0}

        risk_counts = df_evaluated["overall_risk_level"].value_counts().to_dict()
        critical_count = risk_counts.get("CRITICAL", 0)
        high_count = risk_counts.get("HIGH", 0)
        medium_count = risk_counts.get("MEDIUM", 0)
        low_count = risk_counts.get("LOW", 0)

        mean_comp_score = float(np.round(df_evaluated["composite_risk_score"].mean(), 4))

        out_of_stock_count = int((df_evaluated.get("current_stock", pd.Series(1)) <= 0).sum())
        overstocked_count = int((df_evaluated.get("excess_inventory_units", pd.Series(0)) > 0).sum())
        tied_up_capital_sum = float(np.round(df_evaluated.get("tied_up_capital", pd.Series(0)).sum(), 2))

        return {
            "total_items_evaluated": total_items,
            "overall_mean_risk_score": mean_mean_comp_score if 'mean_mean_comp_score' in locals() else mean_comp_score,
            "risk_level_counts": {
                "CRITICAL": critical_count,
                "HIGH": high_count,
                "MEDIUM": medium_count,
                "LOW": low_count
            },
            "risk_level_percentages": {
                "CRITICAL": float(np.round(critical_count / total_items * 100, 2)),
                "HIGH": float(np.round(high_count / total_items * 100, 2)),
                "MEDIUM": float(np.round(medium_count / total_items * 100, 2)),
                "LOW": float(np.round(low_count / total_items * 100, 2))
            },
            "operational_alerts": {
                "out_of_stock_items": out_of_stock_count,
                "overstocked_items": overstocked_count,
                "total_tied_up_capital": tied_up_capital_sum
            }
        }


def evaluate_full_supply_chain_risk(
    df: pd.DataFrame,
    config_path: str = "configs/config.yaml",
    weights: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience wrapper function to evaluate supply chain risk on a DataFrame and get summary.
    """
    engine = SupplyChainRiskEngine(config_path=config_path, weights=weights)
    df_eval = engine.evaluate_supply_chain_risk_df(df)
    summary = engine.generate_risk_summary(df_eval)
    return df_eval, summary
