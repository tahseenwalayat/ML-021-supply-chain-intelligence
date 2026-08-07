import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger("features.seasonality")


def compute_seasonality_features(
    calendar_dim: pd.DataFrame,
    spine_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes seasonality indices and cyclical calendar features.
    Keys: (product_id, region, date)
    """
    logger.info("Computing Seasonality features...")

    cal = calendar_dim.copy()
    cal["date"] = pd.to_datetime(cal["date"])

    # Merge calendar attributes onto spine
    df = spine_df[["product_id", "region", "date"]].merge(
        cal[["date", "day_of_week", "month", "quarter", "is_weekend"]],
        on="date",
        how="left"
    )

    # Cyclical day of week encoding (period 7)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # Cyclical month encoding (period 12)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df["is_weekend"] = df["is_weekend"].fillna(0).astype(int)
    df["day_of_week"] = df["day_of_week"].fillna(df["date"].dt.dayofweek).astype(int)
    df["month"] = df["month"].fillna(df["date"].dt.month).astype(int)

    # Seasonality index estimation (DOW & Monthly relative factors)
    # Estimate baseline DOW weights: Weekdays ~1.0, Weekend adjustment
    dow_weights = {0: 1.05, 1: 0.98, 2: 0.96, 3: 1.02, 4: 1.15, 5: 1.10, 6: 0.74}
    month_weights = {1: 0.85, 2: 0.88, 3: 0.95, 4: 0.98, 5: 1.02, 6: 1.05, 7: 1.08, 8: 1.04, 9: 1.02, 10: 1.12, 11: 1.35, 12: 1.45}

    df["seasonality_index_dow"] = df["day_of_week"].map(dow_weights).fillna(1.0)
    df["seasonality_index_month"] = df["month"].map(month_weights).fillna(1.0)

    cols = [
        "product_id", "region", "date",
        "day_of_week", "sin_day_of_week", "cos_day_of_week",
        "month", "sin_month", "cos_month", "is_weekend",
        "seasonality_index_dow", "seasonality_index_month"
    ]
    logger.info(f"Seasonality features complete: shape={df[cols].shape}")
    return df[cols]
