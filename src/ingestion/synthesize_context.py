import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.utils.logging_config import get_logger

logger = get_logger("synthesize_context")


def generate_synthetic_weather_and_events(
    sales_fact: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates plausible, synthetic weather_dim and event_dim dataframes
    keyed by region + date. All generated rows have is_synthetic=True.
    """
    logger.info("Generating synthetic weather_dim and event_dim...")

    if sales_fact.empty or "region" not in sales_fact.columns or "date" not in sales_fact.columns:
        logger.warning("sales_fact is empty or missing region/date. Returning empty synthetic dimensions.")
        empty_weather = pd.DataFrame(columns=[
            "weather_id", "date", "region", "temperature_c", "precipitation_mm",
            "humidity_pct", "weather_condition", "is_synthetic"
        ])
        empty_event = pd.DataFrame(columns=[
            "event_id", "date", "region", "event_name", "event_type",
            "impact_score", "is_synthetic"
        ])
        return empty_weather, empty_event

    # 1. Unique (region, date) combinations
    region_dates = sales_fact[["region", "date"]].drop_duplicates().copy()
    region_dates["date"] = pd.to_datetime(region_dates["date"])
    region_dates = region_dates.sort_values(by=["region", "date"]).reset_index(drop=True)

    # 2. Build weather_dim
    dates = region_dates["date"]
    months = dates.dt.month
    days = dates.dt.day

    # Set seed for reproducible synthetic generation
    np.random.seed(42)

    # Calculate seasonal temperature (sine curve simulation) + noise
    base_temp = 18.0 + 10.0 * np.sin((months - 4) * np.pi / 6.0)
    temp_noise = np.random.normal(0, 3.5, size=len(region_dates))
    temperatures = np.round(base_temp + temp_noise, 1)

    # Precipitation simulation (70% dry days, 30% wet)
    precip_raw = np.random.exponential(scale=5.0, size=len(region_dates))
    precip_mask = np.random.rand(len(region_dates)) > 0.70
    precipitations = np.round(np.where(precip_mask, precip_raw, 0.0), 1)

    # Humidity simulation
    humidity = np.round(np.clip(55.0 + 0.8 * precipitations + np.random.normal(0, 10, size=len(region_dates)), 30.0, 99.0), 1)

    # Condition mapping
    conditions = []
    for temp, precip in zip(temperatures, precipitations):
        if precip > 15.0 and temp <= 0:
            conditions.append("Snowy")
        elif precip > 2.0:
            conditions.append("Rainy")
        elif precip > 0.0:
            conditions.append("Cloudy")
        else:
            conditions.append("Sunny")

    date_strs = dates.dt.strftime("%Y-%m-%d")
    weather_ids = region_dates["region"].astype(str) + "_" + date_strs

    weather_dim = pd.DataFrame({
        "weather_id": weather_ids,
        "date": dates,
        "region": region_dates["region"].astype(str),
        "temperature_c": temperatures,
        "precipitation_mm": precipitations,
        "humidity_pct": humidity,
        "weather_condition": conditions,
        "is_synthetic": True
    })

    # 3. Build event_dim
    # Filter specific event dates (New Year, Black Friday, Christmas, Mid-Year Sale, Cyber Monday)
    event_rows = []
    for idx, row in region_dates.iterrows():
        dt = row["date"]
        reg = str(row["region"])
        m, d, dw = dt.month, dt.day, dt.dayofweek

        date_str = dt.strftime("%Y-%m-%d")

        if m == 1 and d == 1:
            event_rows.append({
                "event_id": f"{reg}_{date_str}_New_Year",
                "date": dt,
                "region": reg,
                "event_name": "New Year's Day",
                "event_type": "Holiday",
                "impact_score": 1.3,
                "is_synthetic": True
            })
        elif m == 12 and d == 25:
            event_rows.append({
                "event_id": f"{reg}_{date_str}_Christmas",
                "date": dt,
                "region": reg,
                "event_name": "Christmas Day",
                "event_type": "Holiday",
                "impact_score": 2.2,
                "is_synthetic": True
            })
        elif m == 11 and dw == 4 and 23 <= d <= 29:
            event_rows.append({
                "event_id": f"{reg}_{date_str}_Black_Friday",
                "date": dt,
                "region": reg,
                "event_name": "Black Friday",
                "event_type": "Commercial",
                "impact_score": 2.5,
                "is_synthetic": True
            })
        elif m == 11 and dw == 0 and 26 <= d <= 30:
            event_rows.append({
                "event_id": f"{reg}_{date_str}_Cyber_Monday",
                "date": dt,
                "region": reg,
                "event_name": "Cyber Monday",
                "event_type": "Commercial",
                "impact_score": 2.1,
                "is_synthetic": True
            })
        elif m == 7 and d == 4:
            event_rows.append({
                "event_id": f"{reg}_{date_str}_Summer_Promo",
                "date": dt,
                "region": reg,
                "event_name": "Mid-Year Summer Special",
                "event_type": "Commercial",
                "impact_score": 1.4,
                "is_synthetic": True
            })

    if event_rows:
        event_dim = pd.DataFrame(event_rows)
    else:
        event_dim = pd.DataFrame(columns=[
            "event_id", "date", "region", "event_name", "event_type",
            "impact_score", "is_synthetic"
        ])

    logger.info(
        f"Synthetic context generation complete: weather_dim={len(weather_dim)} rows, "
        f"event_dim={len(event_dim)} rows"
    )

    return weather_dim, event_dim
