import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMRegressor

from src.utils.logging_config import get_logger
from src.forecasting.dataset_split import (
    load_feature_store,
    prepare_hierarchy_data,
    generate_expanding_window_splits,
    evaluate_forecasts,
    compute_seasonal_naive_baseline
)

logger = get_logger("forecasting.train_lightgbm")

# Default LightGBM hyperparameters per docs/model_plan.md
DEFAULT_LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1
}


def get_feature_and_target_cols(df: pd.DataFrame, level: str) -> Tuple[List[str], List[str], str]:
    """Identifies feature columns, categorical columns, and target column."""
    # The feature row at date t predicts demand at t + 1.  Using actual_sales
    # here would train on the same day's observed demand rather than a forecast.
    target_col = "target_next_day_sales"
    exclude_cols = {"date", "actual_sales", "target_next_day_sales"}
    
    cat_cols = []
    if "product_id" in df.columns and level == "sku_region":
        cat_cols.append("product_id")
    if "category" in df.columns:
        cat_cols.append("category")
    if "region" in df.columns:
        cat_cols.append("region")

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols, cat_cols, target_col


def preprocess_features(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Preprocesses DataFrame by converting categorical/object columns to category dtype."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object" or col in cat_cols:
            df_copy[col] = df_copy[col].astype("category")
    return df_copy


def train_and_eval_lightgbm_level(
    df: pd.DataFrame,
    product_dim: pd.DataFrame,
    level: str = "sku_region",
    params: Dict[str, Any] = DEFAULT_LGBM_PARAMS,
    models_dir: str = "models",
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Trains LightGBM model across 5 expanding window CV folds for a specific hierarchy level.
    Logs metrics, parameters, and model to MLflow. Saves best model to models_dir.
    """
    logger.info(f"=== Starting LightGBM Training for Hierarchy Level: {level} ===")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Demand_Forecasting_LightGBM")
    mlflow.lightgbm.autolog(log_models=False)
    
    # 1. Prepare hierarchy level dataset
    level_df = prepare_hierarchy_data(df, product_dim, level=level)
    feature_cols, cat_cols, target_col = get_feature_and_target_cols(level_df, level)
    level_df = preprocess_features(level_df, cat_cols)

    # Grouping columns for seasonal naive baseline
    group_cols = [c for c in cat_cols if c in level_df.columns]
    if not group_cols and "region" in level_df.columns:
        group_cols = ["region"]

    # 2. Time-series CV splits
    splits = generate_expanding_window_splits(level_df, date_col="date", n_splits=n_splits)

    fold_metrics_list = []
    naive_metrics_list = []

    with mlflow.start_run(run_name=f"LightGBM_{level}") as run:
        mlflow.log_params(params)
        mlflow.log_param("hierarchy_level", level)
        mlflow.log_param("num_features", len(feature_cols))

        for train_df, val_df, meta in splits:
            X_train, y_train = train_df[feature_cols], train_df[target_col].values
            X_val, y_val = val_df[feature_cols], val_df[target_col].values

            model = LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[]
            )

            preds = model.predict(X_val)
            preds = np.clip(preds, 0.0, None)  # Sales cannot be negative

            fold_metrics = evaluate_forecasts(y_val, preds)
            fold_metrics_list.append(fold_metrics)

            # Seasonal-naive baseline for comparison
            naive_metrics = compute_seasonal_naive_baseline(
                val_df, level_df, group_cols=group_cols, target_col=target_col, lag_days=7
            )
            naive_metrics_list.append(naive_metrics)

            fold_idx = meta["fold"]
            logger.info(
                f"Fold {fold_idx} | WMAPE: {fold_metrics['wmape']:.2f}% (Naive: {naive_metrics['wmape']:.2f}%) | "
                f"RMSE: {fold_metrics['rmse']:.2f} | Bias: {fold_metrics['bias']:.2f}%"
            )
            mlflow.log_metrics({
                f"fold_{fold_idx}_wmape": fold_metrics["wmape"],
                f"fold_{fold_idx}_rmse": fold_metrics["rmse"],
                f"fold_{fold_idx}_mape": fold_metrics["mape"],
                f"fold_{fold_idx}_bias": fold_metrics["bias"],
            })

        # Calculate average metrics across CV folds
        avg_wmape = float(np.mean([m["wmape"] for m in fold_metrics_list]))
        avg_rmse = float(np.mean([m["rmse"] for m in fold_metrics_list]))
        avg_mape = float(np.mean([m["mape"] for m in fold_metrics_list]))
        avg_bias = float(np.mean([m["bias"] for m in fold_metrics_list]))

        avg_naive_wmape = float(np.mean([m["wmape"] for m in naive_metrics_list]))

        mlflow.log_metrics({
            "val_wmape_avg": avg_wmape,
            "val_rmse_avg": avg_rmse,
            "val_mape_avg": avg_mape,
            "val_bias_avg": avg_bias,
            "seasonal_naive_wmape_avg": avg_naive_wmape,
            "wmape_improvement_vs_naive": avg_naive_wmape - avg_wmape
        })

        logger.info(
            f"Summary [{level}] | Avg Validation WMAPE: {avg_wmape:.2f}% vs Seasonal-Naive: {avg_naive_wmape:.2f}% "
            f"(Beat naive by {avg_naive_wmape - avg_wmape:.2f} percentage points!)"
        )

        # 3. Train final model on full dataset
        X_full, y_full = level_df[feature_cols], level_df[target_col].values
        final_model = LGBMRegressor(**params)
        final_model.fit(X_full, y_full)

        # 4. Save model artifact
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"lightgbm_{level}.joblib")
        joblib.dump({
            "model": final_model,
            "feature_cols": feature_cols,
            "cat_cols": cat_cols,
            "level": level,
            "params": params,
            "wmape": avg_wmape,
            "naive_wmape": avg_naive_wmape,
            "target_col": target_col,
        }, model_path)
        
        mlflow.log_artifact(model_path)
        logger.info(f"Saved LightGBM model artifact to {model_path}")

    return {
        "level": level,
        "wmape": avg_wmape,
        "naive_wmape": avg_naive_wmape,
        "model_path": model_path
    }


def train_lightgbm_all_levels(
    feature_store_path: str = "data/processed/feature_store.parquet",
    product_dim_path: str = "data/processed/product_dim.parquet",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """Trains LightGBM models for all three hierarchy levels."""
    df, product_dim = load_feature_store(feature_store_path, product_dim_path)
    levels = ["sku_region", "category_region", "region_total"]
    results = {}
    for level in levels:
        res = train_and_eval_lightgbm_level(df, product_dim, level=level, models_dir=models_dir)
        results[level] = res
    return results


if __name__ == "__main__":
    train_lightgbm_all_levels()
