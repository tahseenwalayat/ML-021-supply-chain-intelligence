import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import mlflow
from lightgbm import LGBMRegressor

from src.utils.logging_config import get_logger
from src.forecasting.dataset_split import (
    load_feature_store,
    prepare_hierarchy_data,
    generate_expanding_window_splits,
    evaluate_forecasts
)

logger = get_logger("forecasting.feature_selection")


def get_feature_and_target_cols(df: pd.DataFrame, level: str) -> Tuple[List[str], List[str], str]:
    """Identifies feature columns, categorical columns, and target column."""
    target_col = "actual_sales"
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


def run_feature_selection(
    level: str = "sku_region",
    importance_threshold: float = 0.005,
    max_degradation_pct: float = 2.0,
    feature_store_path: str = "data/processed/feature_store.parquet",
    product_dim_path: str = "data/processed/product_dim.parquet",
    models_dir: str = "models",
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Identifies and drops low-importance features below importance_threshold,
    then re-validates to ensure WMAPE does not degrade by more than max_degradation_pct (2%).
    Logs results and saves final pruned model artifact.
    """
    logger.info(
        f"=== Starting Feature Selection Pipeline | Level: {level} | "
        f"Threshold: {importance_threshold} | Max Allowed Degradation: {max_degradation_pct}% ==="
    )

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Feature_Selection")

    df, product_dim = load_feature_store(feature_store_path, product_dim_path)
    level_df = prepare_hierarchy_data(df, product_dim, level=level)
    full_feature_cols, cat_cols, target_col = get_feature_and_target_cols(level_df, level)
    level_df = preprocess_features(level_df, cat_cols)

    splits = generate_expanding_window_splits(level_df, date_col="date", n_splits=n_splits)

    # 1. Baseline training on full feature set
    params = {
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

    baseline_wmapes = []
    feature_importances_sum = np.zeros(len(full_feature_cols))

    for train_df, val_df, meta in splits:
        X_train, y_train = train_df[full_feature_cols], train_df[target_col].values
        X_val, y_val = val_df[full_feature_cols], val_df[target_col].values

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        preds = np.clip(preds, 0.0, None)

        metrics = evaluate_forecasts(y_val, preds)
        baseline_wmapes.append(metrics["wmape"])

        # Accumulate feature importances
        feature_importances_sum += model.feature_importances_

    baseline_wmape = float(np.mean(baseline_wmapes))

    # Normalize feature importances
    total_imp = np.sum(feature_importances_sum)
    if total_imp > 0:
        norm_importances = feature_importances_sum / total_imp
    else:
        norm_importances = feature_importances_sum

    importance_df = pd.DataFrame({
        "feature": full_feature_cols,
        "importance": norm_importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    logger.info(f"Baseline [{level}] | Full Features Count: {len(full_feature_cols)} | Baseline WMAPE: {baseline_wmape:.2f}%")

    # 2. Filter features by importance threshold
    selected_features = importance_df[importance_df["importance"] >= importance_threshold]["feature"].tolist()
    dropped_features = importance_df[importance_df["importance"] < importance_threshold]["feature"].tolist()

    # Safety check: ensure at least 5 top features remain
    if len(selected_features) < 5:
        logger.warning(
            f"Selected features count ({len(selected_features)}) is too small. "
            f"Falling back to top 10 features."
        )
        selected_features = importance_df.head(10)["feature"].tolist()
        dropped_features = [f for f in full_feature_cols if f not in selected_features]

    logger.info(
        f"Filtered features: Kept {len(selected_features)} features, "
        f"Dropped {len(dropped_features)} low-importance features."
    )

    # 3. Re-validate on reduced feature set
    reduced_cat_cols = [c for c in cat_cols if c in selected_features]
    reduced_wmapes = []

    for train_df, val_df, meta in splits:
        X_train, y_train = train_df[selected_features], train_df[target_col].values
        X_val, y_val = val_df[selected_features], val_df[target_col].values

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        preds = np.clip(preds, 0.0, None)

        metrics = evaluate_forecasts(y_val, preds)
        reduced_wmapes.append(metrics["wmape"])

    reduced_wmape = float(np.mean(reduced_wmapes))

    # 4. Check metric degradation constraint (<= 2% relative)
    if baseline_wmape > 0:
        degradation_pct = float(((reduced_wmape - baseline_wmape) / baseline_wmape) * 100.0)
    else:
        degradation_pct = 0.0

    passed_constraint = degradation_pct <= max_degradation_pct

    logger.info(
        f"Validation Result | Baseline WMAPE: {baseline_wmape:.2f}% -> Reduced WMAPE: {reduced_wmape:.2f}% | "
        f"Degradation: {degradation_pct:+.2f}% | Constraint Passed: {passed_constraint}"
    )

    # If degradation failed, revert to full feature set
    if not passed_constraint:
        logger.warning(
            f"Degradation ({degradation_pct:.2f}%) exceeded maximum allowed threshold ({max_degradation_pct}%). "
            f"Reverting to full feature set for model export."
        )
        final_feature_cols = full_feature_cols
        final_wmape = baseline_wmape
    else:
        final_feature_cols = selected_features
        final_wmape = reduced_wmape

    # 5. Log to MLflow & Save final artifact
    with mlflow.start_run(run_name=f"Feature_Selection_{level}") as run:
        mlflow.log_param("hierarchy_level", level)
        mlflow.log_param("importance_threshold", importance_threshold)
        mlflow.log_param("max_degradation_pct", max_degradation_pct)
        mlflow.log_param("full_features_count", len(full_feature_cols))
        mlflow.log_param("selected_features_count", len(final_feature_cols))
        mlflow.log_param("dropped_features_count", len(full_feature_cols) - len(final_feature_cols))

        mlflow.log_metric("baseline_wmape", baseline_wmape)
        mlflow.log_metric("reduced_wmape", reduced_wmape)
        mlflow.log_metric("degradation_pct", degradation_pct)

        # Fit final model on full dataset
        X_full, y_full = level_df[final_feature_cols], level_df[target_col].values
        final_model = LGBMRegressor(**params)
        final_model.fit(X_full, y_full)

        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"feature_selected_lightgbm_{level}.joblib")
        joblib.dump({
            "model": final_model,
            "selected_features": final_feature_cols,
            "dropped_features": dropped_features,
            "importance_df": importance_df,
            "level": level,
            "baseline_wmape": baseline_wmape,
            "final_wmape": final_wmape,
            "degradation_pct": degradation_pct
        }, model_path)

        mlflow.log_artifact(model_path)
        logger.info(f"Saved feature-selected model artifact to {model_path}")

    return {
        "level": level,
        "baseline_wmape": baseline_wmape,
        "reduced_wmape": reduced_wmape,
        "degradation_pct": degradation_pct,
        "passed_constraint": passed_constraint,
        "selected_features_count": len(final_feature_cols),
        "model_path": model_path
    }


if __name__ == "__main__":
    run_feature_selection()
