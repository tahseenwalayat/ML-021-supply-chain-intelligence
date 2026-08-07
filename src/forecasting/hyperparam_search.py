import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import mlflow
import optuna
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.utils.logging_config import get_logger
from src.forecasting.dataset_split import (
    load_feature_store,
    prepare_hierarchy_data,
    generate_expanding_window_splits,
    evaluate_forecasts
)

logger = get_logger("forecasting.hyperparam_search")

# Suppress Optuna verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)


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
    """Preprocesses DataFrame by converting categorical columns to category dtype."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object" or col in cat_cols:
            df_copy[col] = df_copy[col].astype("category")
    return df_copy


def run_hyperparameter_search(
    model_type: str = "lightgbm",
    level: str = "sku_region",
    n_trials: int = 15,
    n_splits: int = 5,
    feature_store_path: str = "data/processed/feature_store.parquet",
    product_dim_path: str = "data/processed/product_dim.parquet",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Executes Optuna TPE hyperparameter optimization over LightGBM or XGBoost.
    Logs every trial run to MLflow and selects the best parameters by validation WMAPE.
    Saves best model to models_dir.
    """
    logger.info(
        f"=== Starting Optuna Hyperparameter Search | Model: {model_type.upper()} | "
        f"Level: {level} | Budget: {n_trials} Trials ==="
    )

    experiment_name = f"Hyperparameter_Search_{model_type.upper()}"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)
    if model_type.lower() == "lightgbm":
        mlflow.lightgbm.autolog(log_models=False)
    elif model_type.lower() == "xgboost":
        mlflow.xgboost.autolog(log_models=False)

    df, product_dim = load_feature_store(feature_store_path, product_dim_path)
    level_df = prepare_hierarchy_data(df, product_dim, level=level)
    feature_cols, cat_cols, target_col = get_feature_and_target_cols(level_df, level)
    level_df = preprocess_features(level_df, cat_cols)

    splits = generate_expanding_window_splits(level_df, date_col="date", n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective evaluation function for hyperparameter optimization trial."""
        # Search space specifications per docs/model_plan.md Section 6
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1
        }

        if model_type.lower() == "lightgbm":
            params["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
            params["verbose"] = -1
        elif model_type.lower() == "xgboost":
            params["enable_categorical"] = True
            params["tree_method"] = "hist"

        fold_wmapes = []
        for train_df, val_df, meta in splits:
            X_train, y_train = train_df[feature_cols], train_df[target_col].values
            X_val, y_val = val_df[feature_cols], val_df[target_col].values

            if model_type.lower() == "lightgbm":
                model = LGBMRegressor(**params)
            else:
                model = XGBRegressor(**params)

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds = np.clip(preds, 0.0, None)

            metrics = evaluate_forecasts(y_val, preds)
            fold_wmapes.append(metrics["wmape"])

        avg_val_wmape = float(np.mean(fold_wmapes))

        # Log trial to MLflow as nested run
        with mlflow.start_run(run_name=f"Trial_{trial.number}_{model_type}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("val_wmape_avg", avg_val_wmape)
            mlflow.log_param("trial_number", trial.number)

        return avg_val_wmape

    with mlflow.start_run(run_name=f"Optuna_Parent_{model_type}_{level}") as parent_run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("hierarchy_level", level)
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_wmape = study.best_value

        if model_type.lower() == "lightgbm":
            best_params["verbose"] = -1
        elif model_type.lower() == "xgboost":
            best_params["enable_categorical"] = True
            best_params["tree_method"] = "hist"

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_wmape", best_wmape)

        logger.info(
            f"Optuna Search Complete! Best Trial #{study.best_trial.number} | "
            f"Best Validation WMAPE: {best_wmape:.2f}% | Best Params: {best_params}"
        )

        # Train final model on full dataset using best params
        X_full, y_full = level_df[feature_cols], level_df[target_col].values
        if model_type.lower() == "lightgbm":
            best_model = LGBMRegressor(**best_params)
        else:
            best_model = XGBRegressor(**best_params)

        best_model.fit(X_full, y_full)

        # Save best model artifact
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"best_{model_type}_{level}_optuna.joblib")
        joblib.dump({
            "model": best_model,
            "best_params": best_params,
            "feature_cols": feature_cols,
            "cat_cols": cat_cols,
            "level": level,
            "model_type": model_type,
            "best_val_wmape": best_wmape
        }, model_path)

        mlflow.log_artifact(model_path)
        logger.info(f"Saved best Optuna-tuned model to {model_path}")

    return {
        "model_type": model_type,
        "level": level,
        "best_wmape": best_wmape,
        "best_params": best_params,
        "model_path": model_path
    }


if __name__ == "__main__":
    run_hyperparameter_search(model_type="lightgbm", n_trials=10)
    run_hyperparameter_search(model_type="xgboost", n_trials=10)
