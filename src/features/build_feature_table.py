import os
import time
import pandas as pd
import numpy as np
from typing import Dict

from src.utils.logging_config import get_logger
from src.features.velocity import compute_velocity_features
from src.features.seasonality import compute_seasonality_features
from src.features.promotion_impact import compute_promotion_impact_features
from src.features.supplier_reliability import compute_supplier_reliability_features
from src.features.lead_time import compute_lead_time_features
from src.features.lifecycle_stage import compute_lifecycle_stage_features
from src.features.volatility import compute_volatility_features
from src.features.regional_patterns import compute_regional_patterns_features
from src.features.holiday_effects import compute_holiday_effects_features

logger = get_logger("build_feature_table")


def build_feature_store(
    processed_dir: str = "data/processed",
    output_file: str = "data/processed/feature_store.parquet"
) -> pd.DataFrame:
    """
    Builds the unified feature store by merging all 9 engineered feature frames.
    Saves output to data/processed/feature_store.parquet.
    """
    start_time = time.time()
    logger.info("Starting Unified Feature Store Assembly Pipeline...")

    # 1. Load Processed Parquet Tables
    sales_fact = pd.read_parquet(os.path.join(processed_dir, "sales_fact.parquet"))
    product_dim = pd.read_parquet(os.path.join(processed_dir, "product_dim.parquet"))
    supplier_dim = pd.read_parquet(os.path.join(processed_dir, "supplier_dim.parquet"))
    promotion_dim = pd.read_parquet(os.path.join(processed_dir, "promotion_dim.parquet"))
    calendar_dim = pd.read_parquet(os.path.join(processed_dir, "calendar_dim.parquet"))
    weather_dim = pd.read_parquet(os.path.join(processed_dir, "weather_dim.parquet"))
    event_dim = pd.read_parquet(os.path.join(processed_dir, "event_dim.parquet"))

    sales_fact["date"] = pd.to_datetime(sales_fact["date"])

    # 2. Build Base Spine: Daily product-region-date aggregation
    logger.info("Building base Spine DataFrame...")
    daily_spine = sales_fact.groupby(["product_id", "region", "date"]).agg(
        actual_sales=("quantity", "sum")
    ).reset_index()

    daily_spine = daily_spine.sort_values(by=["product_id", "region", "date"]).reset_index(drop=True)

    # Compute target_next_day_sales (sales on date t + 1 day for same product & region)
    grouped = daily_spine.groupby(["product_id", "region"])
    daily_spine["target_next_day_sales"] = grouped["actual_sales"].shift(-1).fillna(0.0)

    # 3. Compute All 9 Feature Sets
    logger.info("--- Feature Set 1: Velocity ---")
    df_vel = compute_velocity_features(sales_fact, daily_spine)

    logger.info("--- Feature Set 2: Seasonality ---")
    df_seas = compute_seasonality_features(calendar_dim, daily_spine)

    logger.info("--- Feature Set 3: Promotion Impact ---")
    df_promo = compute_promotion_impact_features(sales_fact, promotion_dim, daily_spine)

    logger.info("--- Feature Set 4: Supplier Reliability ---")
    df_supp = compute_supplier_reliability_features(sales_fact, supplier_dim, daily_spine)

    logger.info("--- Feature Set 5: Lead Time ---")
    df_lead = compute_lead_time_features(df_supp, df_vel, daily_spine)

    logger.info("--- Feature Set 6: Lifecycle Stage ---")
    df_life = compute_lifecycle_stage_features(sales_fact, daily_spine)

    logger.info("--- Feature Set 7: Volatility ---")
    df_vol = compute_volatility_features(sales_fact, df_vel, daily_spine)

    logger.info("--- Feature Set 8: Regional Patterns ---")
    df_reg = compute_regional_patterns_features(sales_fact, df_vel, daily_spine)

    logger.info("--- Feature Set 9: Holiday Effects & Context ---")
    df_hol = compute_holiday_effects_features(calendar_dim, weather_dim, event_dim, daily_spine)

    # 4. Merge All Feature Frames
    logger.info("Joining all feature frames into unified Feature Store...")
    keys = ["product_id", "region", "date"]

    feature_store = daily_spine[keys + ["actual_sales", "target_next_day_sales"]].copy()

    feature_dfs = [df_vel, df_seas, df_promo, df_supp, df_lead, df_life, df_vol, df_reg, df_hol]

    for f_df in feature_dfs:
        feature_store = feature_store.merge(f_df, on=keys, how="left")

    # Clean nulls from initial window periods
    numeric_cols = feature_store.select_dtypes(include=[np.number]).columns
    feature_store[numeric_cols] = feature_store[numeric_cols].fillna(0.0)

    # 5. Export to Parquet
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    feature_store.to_parquet(output_file, index=False, engine="pyarrow")

    elapsed = time.time() - start_time
    logger.info(
        f"Feature Store successfully created! Rows: {len(feature_store):,}, "
        f"Columns: {feature_store.shape[1]}, Saved -> {output_file} ({elapsed:.2f}s)"
    )

    return feature_store


if __name__ == "__main__":
    build_feature_store()
