import os
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Tuple, Dict, Any, Optional

from src.utils.logging_config import get_logger

logger = get_logger("forecasting.dataset_split")


def compute_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Weighted Mean Absolute Percentage Error (WMAPE).
    WMAPE = sum(|y_true - y_pred|) / sum(y_true) * 100%
    Avoids division by zero on zero-demand periods.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sum_true = np.sum(y_true)
    if sum_true <= 1e-8:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / sum_true * 100.0)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Percentage Error (MAPE), handling zero division safely."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero_mask = y_true > 1e-5
    if not np.any(nonzero_mask):
        return 0.0
    return float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)


def compute_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Forecast Bias (Percentage Mean Error).
    Bias = sum(y_pred - y_true) / sum(y_true) * 100%
    Positive value indicates over-forecasting, negative indicates under-forecasting.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sum_true = np.sum(y_true)
    if sum_true <= 1e-8:
        return 0.0
    return float(np.sum(y_pred - y_true) / sum_true * 100.0)


def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes all standard forecasting evaluation metrics."""
    return {
        "wmape": compute_wmape(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "bias": compute_bias(y_true, y_pred)
    }


def generate_expanding_window_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    n_splits: int = 5,
    val_horizon_days: int = 28,
    embargo_days: int = 1
) -> List[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
    """
    Generates 5-fold expanding window time-series train/validation splits.
    
    Guarantees:
    - Training fold strictly uses historical data up to cutoff_date (no future data leakage).
    - Validation fold covers fixed horizon (28 days).
    - Optional embargo window between training cutoff and validation start.
    
    Returns:
        List of tuples: (train_df, val_df, fold_meta_dict)
    """
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame.")

    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    max_date = df_copy[date_col].max()
    min_date = df_copy[date_col].min()

    logger.info(
        f"Generating {n_splits} expanding-window CV folds for date range "
        f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}..."
    )

    splits = []
    for fold_idx in range(1, n_splits + 1):
        # fold_idx 1 is oldest fold, fold_idx n_splits is latest fold
        offset_folds = n_splits - fold_idx
        val_end_date = max_date - timedelta(days=offset_folds * val_horizon_days)
        val_start_date = val_end_date - timedelta(days=val_horizon_days - 1)
        cutoff_date = val_start_date - timedelta(days=embargo_days)

        train_mask = df_copy[date_col] <= cutoff_date
        val_mask = (df_copy[date_col] >= val_start_date) & (df_copy[date_col] <= val_end_date)

        train_df = df_copy[train_mask].copy()
        val_df = df_copy[val_mask].copy()

        meta = {
            "fold": fold_idx,
            "train_start": train_df[date_col].min(),
            "train_cutoff": cutoff_date,
            "val_start": val_start_date,
            "val_end": val_end_date,
            "train_size": len(train_df),
            "val_size": len(val_df),
        }

        logger.info(
            f"Fold {fold_idx}/{n_splits} | Train: [{meta['train_start'].strftime('%Y-%m-%d')} -> "
            f"{cutoff_date.strftime('%Y-%m-%d')}] ({len(train_df)} rows) | "
            f"Val: [{val_start_date.strftime('%Y-%m-%d')} -> {val_end_date.strftime('%Y-%m-%d')}] ({len(val_df)} rows)"
        )

        splits.append((train_df, val_df, meta))

    return splits


def load_feature_store(
    feature_store_path: str = "data/processed/feature_store.parquet",
    product_dim_path: str = "data/processed/product_dim.parquet"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Loads feature store parquet file and product dimension table."""
    if not os.path.exists(feature_store_path):
        raise FileNotFoundError(f"Feature store file not found at {feature_store_path}. Please run build_feature_table.py first.")
    
    logger.info(f"Loading Feature Store from {feature_store_path}...")
    feature_df = pd.read_parquet(feature_store_path)
    feature_df["date"] = pd.to_datetime(feature_df["date"])

    product_dim = None
    if os.path.exists(product_dim_path):
        product_dim = pd.read_parquet(product_dim_path)

    return feature_df, product_dim


def prepare_hierarchy_data(
    feature_df: pd.DataFrame,
    product_dim: Optional[pd.DataFrame] = None,
    level: str = "sku_region"
) -> pd.DataFrame:
    """
    Prepares dataset for specified hierarchy level:
    - 'sku_region': Level 1 (product_id, region, date)
    - 'category_region': Level 2 (category, region, date)
    - 'region_total': Level 3 (region, date)
    """
    logger.info(f"Preparing hierarchy dataset for level: {level}")
    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if level == "sku_region":
        if product_dim is not None and "category" in product_dim.columns and "category" not in df.columns:
            df = df.merge(product_dim[["product_id", "category"]], on="product_id", how="left")
            df["category"] = df["category"].fillna("unknown")
        return df

    if level == "category_region":
        if "category" not in df.columns:
            if product_dim is not None and "category" in product_dim.columns:
                df = df.merge(product_dim[["product_id", "category"]], on="product_id", how="left")
                df["category"] = df["category"].fillna("unknown")
            else:
                raise ValueError("Category column missing and product_dim not available.")

        group_cols = ["category", "region", "date"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Targets sum up across category
        agg_dict = {}
        for c in numeric_cols:
            if c in ["actual_sales", "target_next_day_sales"]:
                agg_dict[c] = "sum"
            else:
                agg_dict[c] = "mean"

        agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
        return agg_df

    if level == "region_total":
        group_cols = ["region", "date"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        agg_dict = {}
        for c in numeric_cols:
            if c in ["actual_sales", "target_next_day_sales"]:
                agg_dict[c] = "sum"
            else:
                agg_dict[c] = "mean"

        agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
        return agg_df

    raise ValueError(f"Unknown hierarchy level: {level}. Options are 'sku_region', 'category_region', 'region_total'.")


def compute_seasonal_naive_baseline(
    val_df: pd.DataFrame,
    full_df: pd.DataFrame,
    group_cols: List[str],
    target_col: str = "actual_sales",
    date_col: str = "date",
    lag_days: int = 7
) -> Dict[str, float]:
    """
    Computes Seasonal Naive Baseline metrics for validation set using lag_days (default 7 days).
    y_pred(t) = y(t - lag_days)
    """
    val = val_df.copy()
    val[date_col] = pd.to_datetime(val[date_col])

    # Construct lookup from full_df
    lookup = full_df[group_cols + [date_col, target_col]].copy()
    lookup[date_col] = pd.to_datetime(lookup[date_col])
    lookup["lagged_date"] = lookup[date_col] + timedelta(days=lag_days)
    lookup.rename(columns={target_col: "naive_pred"}, inplace=True)

    merged = val.merge(
        lookup[group_cols + ["lagged_date", "naive_pred"]],
        left_on=group_cols + [date_col],
        right_on=group_cols + ["lagged_date"],
        how="left"
    )

    merged["naive_pred"] = merged["naive_pred"].fillna(0.0)
    return evaluate_forecasts(merged[target_col].values, merged["naive_pred"].values)
