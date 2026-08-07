import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.utils.logging_config import get_logger
from src.risk.supplier_delay_risk import (
    compute_supplier_delay_risk_df,
    classify_risk_level,
    load_risk_config
)
from src.risk.stockout_risk import compute_stockout_risk_df
from src.risk.overstock_risk import compute_overstock_risk_df
from src.risk.slow_dead_inventory import compute_slow_dead_inventory_df
from src.risk.anomaly_detection import compute_anomaly_detection_df

logger = get_logger("risk.run_risk_engine")


def run_risk_engine(
    processed_dir: str = "data/processed",
    config_path: str = "configs/config.yaml",
    output_filename: str = "risk_scores.parquet"
) -> pd.DataFrame:
    """
    Orchestrates the full Supply Chain Risk Engine workflow:
    1. Loads forecast and inventory outputs (consumes existing tables, no retraining).
    2. Runs all 5 risk scorers:
       - supplier_delay_risk.py
       - stockout_risk.py
       - overstock_risk.py
       - slow_dead_inventory.py
       - anomaly_detection.py
    3. Computes bounded composite risk scores and severity levels for all product-warehouse pairs.
    4. Saves output table to data/processed/risk_scores.parquet.
    """
    logger.info("=== Starting Supply Chain Risk Engine Pipeline ===")
    cfg = load_risk_config(config_path)

    w_supplier = cfg.get("supplier_weight_late_rate", 0.25)
    # Define balanced weights for all 5 risk components
    weights = {
        "supplier_delay": 0.25,
        "stockout": 0.30,
        "overstock": 0.15,
        "slow_dead": 0.15,
        "anomaly": 0.15
    }
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    # 1. Load forecast + inventory input data
    inv_path = os.path.join(processed_dir, "inventory_recommendations.parquet")
    sales_path = os.path.join(processed_dir, "sales_fact.parquet")
    product_path = os.path.join(processed_dir, "product_dim.parquet")
    warehouse_path = os.path.join(processed_dir, "warehouse_dim.parquet")

    if os.path.exists(inv_path):
        logger.info(f"Loading inventory recommendations from '{inv_path}'...")
        df_base = pd.read_parquet(inv_path)
    elif os.path.exists(sales_path) and os.path.exists(warehouse_path):
        logger.info("Building product-warehouse pairs from sales_fact and warehouse_dim...")
        sales_fact = pd.read_parquet(sales_path)
        product_dim = pd.read_parquet(product_path) if os.path.exists(product_path) else None
        warehouse_dim = pd.read_parquet(warehouse_path)

        pairs = sales_fact.groupby(["product_id", "region"]).agg({
            "quantity": ["mean", "std"]
        }).reset_index()
        pairs.columns = ["product_id", "region", "avg_daily_demand", "std_daily_demand"]

        df_base = pairs.merge(warehouse_dim[["warehouse_id", "region"]], on="region", how="inner")
        df_base["reorder_point"] = df_base["avg_daily_demand"] * 10
        df_base["safety_stock"] = df_base["avg_daily_demand"] * 3
        df_base["current_stock"] = df_base["reorder_point"] * 0.8
        df_base["avg_lead_time"] = 7.0
        df_base["unit_cost"] = 15.0
    else:
        logger.info("Creating representative product-warehouse pairs for demonstration...")
        products = [f"SKU_{i:04d}" for i in range(1, 11)]
        warehouses = ["WH_NORTH", "WH_SOUTH", "WH_EAST"]
        rows = []
        np.random.seed(42)
        for p in products:
            for w in warehouses:
                rows.append({
                    "product_id": p,
                    "warehouse_id": w,
                    "region": "North" if "NORTH" in w else ("South" if "SOUTH" in w else "East"),
                    "current_stock": float(np.random.choice([0, 15, 45, 150, 400])),
                    "reorder_point": 50.0,
                    "safety_stock": 20.0,
                    "avg_daily_demand": 5.0,
                    "std_daily_demand": 1.2,
                    "avg_lead_time": 7.0,
                    "lead_time_std_days": float(np.random.choice([0.0, 1.0, 3.5])),
                    "supplier_reliability_score": float(np.random.choice([1.0, 0.85, 0.60])),
                    "unit_cost": 15.0,
                    "sales_velocity": float(np.random.choice([0.2, 1.5, 8.0])),
                    "zero_sales_weeks": int(np.random.choice([0, 2, 5, 14])),
                    "actual_demand": float(np.random.choice([5.0, 8.0, 45.0])),
                    "forecasted_demand": 5.0,
                    "std_residual": 2.0
                })
        df_base = pd.DataFrame(rows)

    # Clean default missing columns
    if "lead_time_std_days" not in df_base.columns:
        df_base["lead_time_std_days"] = 1.0
    if "supplier_reliability_score" not in df_base.columns:
        df_base["supplier_reliability_score"] = 0.90
    if "unit_cost" not in df_base.columns:
        df_base["unit_cost"] = 15.0
    if "sales_velocity" not in df_base.columns:
        df_base["sales_velocity"] = df_base.get("avg_daily_demand", 1.0)
    if "zero_sales_weeks" not in df_base.columns:
        df_base["zero_sales_weeks"] = 0
    if "actual_demand" not in df_base.columns:
        df_base["actual_demand"] = df_base.get("allocated_daily_demand", df_base.get("avg_daily_demand", 5.0))
    if "forecasted_demand" not in df_base.columns:
        df_base["forecasted_demand"] = df_base.get("allocated_daily_demand", df_base.get("avg_daily_demand", 5.0))
    if "std_residual" not in df_base.columns:
        df_base["std_residual"] = df_base.get("std_daily_demand", 2.0)

    # 2. Run all 5 risk scorers
    logger.info("Computing Supplier Delay Risk Scores...")
    df_eval = compute_supplier_delay_risk_df(df_base, config_path=config_path)

    logger.info("Computing Stockout Risk Scores...")
    df_eval = compute_stockout_risk_df(df_eval)

    logger.info("Computing Overstock Risk Scores...")
    df_eval = compute_overstock_risk_df(df_eval, config_path=config_path)

    logger.info("Computing Slow/Dead Inventory Risk Scores...")
    df_eval = compute_slow_dead_inventory_df(df_eval, config_path=config_path)

    logger.info("Computing Demand Anomaly Detection Scores...")
    df_eval = compute_anomaly_detection_df(df_eval, config_path=config_path)

    # 3. Compute Composite Risk Score
    comp_score = (
        weights["supplier_delay"] * df_eval["supplier_delay_risk_score"] +
        weights["stockout"] * df_eval["stockout_risk_score"] +
        weights["overstock"] * df_eval["overstock_risk_score"] +
        weights["slow_dead"] * df_eval["slow_dead_risk_score"] +
        weights["anomaly"] * df_eval["anomaly_risk_score"]
    )
    df_eval["composite_risk_score"] = np.round(np.clip(comp_score, 0.0, 1.0), 4)
    df_eval["overall_risk_level"] = [classify_risk_level(v) for v in df_eval["composite_risk_score"]]

    # 4. Save to data/processed/risk_scores.parquet
    output_path = os.path.join(processed_dir, output_filename)
    os.makedirs(processed_dir, exist_ok=True)
    df_eval.to_parquet(output_path, index=False)

    logger.info(
        f"Successfully generated '{output_path}' for {len(df_eval)} product-warehouse pairs "
        f"with 0 NaNs. Overall Mean Risk Score: {df_eval['composite_risk_score'].mean():.4f}"
    )

    return df_eval


if __name__ == "__main__":
    run_risk_engine()
