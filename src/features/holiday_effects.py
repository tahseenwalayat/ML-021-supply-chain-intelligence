import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger("features.holiday_effects")


def compute_holiday_effects_features(
    calendar_dim: pd.DataFrame,
    weather_dim: pd.DataFrame,
    event_dim: pd.DataFrame,
    spine_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes holiday effects, proximity to upcoming/past holidays, and weather/event parameters.
    Keys: (product_id, region, date)
    """
    logger.info("Computing Holiday Effects & Contextual features...")

    cal = calendar_dim.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["is_holiday"] = cal["is_holiday"].fillna(0).astype(int)

    # Calculate days since last holiday and days until next holiday
    holiday_dates = cal[cal["is_holiday"] == 1]["date"].sort_values().tolist()

    if holiday_dates:
        # Distance calculation
        def get_days_since(d):
            """Calculates elapsed days since the most recent past holiday."""
            past = [h for h in holiday_dates if h <= d]
            return (d - past[-1]).days if past else 999

        def get_days_until(d):
            """Calculates remaining days until the next upcoming holiday."""
            future = [h for h in holiday_dates if h >= d]
            return (future[0] - d).days if future else 999

        cal["days_since_last_holiday"] = cal["date"].apply(get_days_since)
        cal["days_until_next_holiday"] = cal["date"].apply(get_days_until)
    else:
        cal["days_since_last_holiday"] = 999
        cal["days_until_next_holiday"] = 999

    # Merge calendar attributes
    df = spine_df[["product_id", "region", "date"]].merge(
        cal[["date", "is_holiday", "days_since_last_holiday", "days_until_next_holiday"]],
        on="date",
        how="left"
    )
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    df["days_since_last_holiday"] = df["days_since_last_holiday"].fillna(999).astype(int)
    df["days_until_next_holiday"] = df["days_until_next_holiday"].fillna(999).astype(int)

    # Merge weather_dim
    w = weather_dim.copy()
    w["date"] = pd.to_datetime(w["date"])
    df = df.merge(
        w[["region", "date", "temperature_c", "precipitation_mm"]],
        on=["region", "date"],
        how="left"
    ).fillna({"temperature_c": 18.0, "precipitation_mm": 0.0})

    # Merge event_dim
    if not event_dim.empty:
        e = event_dim.copy()
        e["date"] = pd.to_datetime(e["date"])
        df = df.merge(
            e[["region", "date", "impact_score"]],
            on=["region", "date"],
            how="left"
        )
        df["is_event_day"] = np.where(df["impact_score"].notnull(), 1, 0)
        df["event_impact_score"] = df["impact_score"].fillna(1.0)
        df.drop(columns=["impact_score"], inplace=True)
    else:
        df["is_event_day"] = 0
        df["event_impact_score"] = 1.0

    cols = [
        "product_id", "region", "date",
        "is_holiday", "days_since_last_holiday", "days_until_next_holiday",
        "is_event_day", "event_impact_score", "weather_temperature_c", "weather_precipitation_mm"
    ]
    df.rename(columns={
        "temperature_c": "weather_temperature_c",
        "precipitation_mm": "weather_precipitation_mm"
    }, inplace=True)

    logger.info(f"Holiday Effects features complete: shape={df[cols].shape}")
    return df[cols]
