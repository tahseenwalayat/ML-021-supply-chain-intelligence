import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

from src.utils.logging_config import get_logger

logger = get_logger("clean_utils")


def parse_dates(
    df: pd.DataFrame,
    date_cols: List[str],
    date_format: Optional[str] = None,
    dayfirst: bool = False
) -> pd.DataFrame:
    """Safely parse specified columns to datetime64[ns]."""
    df = df.copy()
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                format=date_format,
                dayfirst=dayfirst,
                errors="coerce"
            )
    return df


def coerce_dtypes(df: pd.DataFrame, dtype_map: Dict[str, str]) -> pd.DataFrame:
    """Coerce DataFrame column types according to dtype_map safely."""
    df = df.copy()
    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                if dtype.startswith("int") or dtype.startswith("INT"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)
                elif dtype.startswith("float") or dtype.startswith("FLOAT"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float64)
                elif dtype == "bool" or dtype == "BOOLEAN":
                    df[col] = df[col].astype(bool)
                elif dtype.startswith("str") or dtype.startswith("VARCHAR"):
                    df[col] = df[col].astype(str).fillna("UNKNOWN")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Error coercing column {col} to {dtype}: {e}")
    return df


def handle_nulls(df: pd.DataFrame, fill_defaults: Dict[str, Union[str, int, float]]) -> pd.DataFrame:
    """Fill missing values in columns using specified defaults."""
    df = df.copy()
    for col, default_val in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)
    return df


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first"
) -> (pd.DataFrame, int):
    """Remove duplicate rows from DataFrame and return cleaned DF and removed count."""
    initial_count = len(df)
    cleaned_df = df.drop_duplicates(subset=subset, keep=keep).copy()
    removed_count = initial_count - len(cleaned_df)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate rows (subset={subset})")
    return cleaned_df, removed_count


def winsorize_sales(
    df: pd.DataFrame,
    sales_col: str = "total_sales",
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99
) -> pd.DataFrame:
    """Winsorize sales column by capping values at 1st and 99th percentiles."""
    if sales_col not in df.columns or df.empty:
        return df

    df = df.copy()
    # Filter negative sales or handle zero bounds
    sales_series = df[sales_col].clip(lower=0.0)

    lower_bound = sales_series.quantile(lower_quantile)
    upper_bound = sales_series.quantile(upper_quantile)

    logger.info(
        f"Winsorizing '{sales_col}' bounds: [{lower_bound:.2f}, {upper_bound:.2f}] "
        f"for {len(df)} rows"
    )
    df[sales_col] = sales_series.clip(lower=lower_bound, upper=upper_bound)
    return df


def generate_calendar_dim(start_date: str = "2010-01-01", end_date: str = "2020-12-31") -> pd.DataFrame:
    """Generate standardized calendar_dim DataFrame spanning start_date to end_date."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df = pd.DataFrame({"date": dates})
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_holiday"] = False

    # Mark major fixed holidays (e.g. New Year, Christmas, July 4, Thanksgiving estimate)
    is_jan1 = (df["month"] == 1) & (df["day"] == 1)
    is_jul4 = (df["month"] == 7) & (df["day"] == 4)
    is_dec25 = (df["month"] == 12) & (df["day"] == 25)
    is_black_friday = (df["month"] == 11) & (df["day_of_week"] == 4) & (df["day"] >= 23) & (df["day"] <= 29)

    df["is_holiday"] = is_jan1 | is_jul4 | is_dec25 | is_black_friday

    return df
