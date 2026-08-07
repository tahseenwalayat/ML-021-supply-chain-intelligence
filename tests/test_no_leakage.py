import pytest
import pandas as pd
import numpy as np
from src.features.velocity import compute_velocity_features
from src.features.volatility import compute_volatility_features
from src.features.regional_patterns import compute_regional_patterns_features


def test_no_future_data_leakage():
    """
    Proves that modifying sales data on a future date T_future does NOT alter
    computed rolling/window feature values on any past date t < T_future.
    """
    # 1. Construct synthetic 10-day time series for a single product/region
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    base_qty = [10.0, 12.0, 15.0, 8.0, 20.0, 14.0, 18.0, 25.0, 30.0, 22.0]
    base_rev = [100.0, 120.0, 150.0, 80.0, 200.0, 140.0, 180.0, 250.0, 300.0, 220.0]

    sales_fact_orig = pd.DataFrame({
        "product_id": "prod_test",
        "region": "US",
        "date": dates,
        "quantity": base_qty,
        "total_sales": base_rev,
        "promotion_id": "promo_none",
        "discount_amount": 0.0,
        "shipping_cost": 5.0
    })

    spine = pd.DataFrame({
        "product_id": "prod_test",
        "region": "US",
        "date": dates
    })

    # 2. Compute baseline features on original dataset
    df_vel_orig = compute_velocity_features(sales_fact_orig, spine)
    df_vol_orig = compute_volatility_features(sales_fact_orig, df_vel_orig, spine)
    df_reg_orig = compute_regional_patterns_features(sales_fact_orig, df_vel_orig, spine)

    # Pick evaluation date: Day 5 (2023-01-05)
    eval_date = pd.to_datetime("2023-01-05")

    vel_day5_orig = df_vel_orig[df_vel_orig["date"] == eval_date].iloc[0].to_dict()
    vol_day5_orig = df_vol_orig[df_vol_orig["date"] == eval_date].iloc[0].to_dict()
    reg_day5_orig = df_reg_orig[df_reg_orig["date"] == eval_date].iloc[0].to_dict()

    # 3. Mutate FUTURE data on Day 8 (2023-01-08) — massive 100x spike
    sales_fact_mutated = sales_fact_orig.copy()
    sales_fact_mutated.loc[sales_fact_mutated["date"] == pd.to_datetime("2023-01-08"), "quantity"] = 5000.0
    sales_fact_mutated.loc[sales_fact_mutated["date"] == pd.to_datetime("2023-01-08"), "total_sales"] = 50000.0

    # 4. Re-compute features on mutated dataset
    df_vel_mut = compute_velocity_features(sales_fact_mutated, spine)
    df_vol_mut = compute_volatility_features(sales_fact_mutated, df_vel_mut, spine)
    df_reg_mut = compute_regional_patterns_features(sales_fact_mutated, df_vel_mut, spine)

    vel_day5_mut = df_vel_mut[df_vel_mut["date"] == eval_date].iloc[0].to_dict()
    vol_day5_mut = df_vol_mut[df_vol_mut["date"] == eval_date].iloc[0].to_dict()
    reg_day5_mut = df_reg_mut[df_reg_mut["date"] == eval_date].iloc[0].to_dict()

    # 5. Assert 100% equality for Day 5 features before the Day 8 mutation
    # Test Velocity Features
    for col in ["sales_velocity_7d", "sales_velocity_14d", "revenue_velocity_7d", "sales_acceleration_7d"]:
        assert np.isclose(vel_day5_orig[col], vel_day5_mut[col], atol=1e-6), (
            f"Leakage Detected in Velocity feature '{col}' on Day 5! "
            f"Original={vel_day5_orig[col]}, Mutated={vel_day5_mut[col]}"
        )

    # Test Volatility Features
    for col in ["sales_std_7d", "sales_cv_30d", "demand_volatility_tier"]:
        assert np.isclose(vol_day5_orig[col], vol_day5_mut[col], atol=1e-6), (
            f"Leakage Detected in Volatility feature '{col}' on Day 5! "
            f"Original={vol_day5_orig[col]}, Mutated={vol_day5_mut[col]}"
        )

    # Test Regional Pattern Features
    for col in ["regional_daily_total_sales", "regional_sales_share_7d"]:
        assert np.isclose(reg_day5_orig[col], reg_day5_mut[col], atol=1e-6), (
            f"Leakage Detected in Regional feature '{col}' on Day 5! "
            f"Original={reg_day5_orig[col]}, Mutated={reg_day5_mut[col]}"
        )

    print("SUCCESS: Zero data leakage confirmed across all rolling feature windows!")


if __name__ == "__main__":
    test_no_future_data_leakage()
