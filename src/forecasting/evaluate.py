import os
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Any, List, Tuple
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logging_config import get_logger
from src.forecasting.dataset_split import (
    load_feature_store,
    prepare_hierarchy_data,
    evaluate_forecasts
)

logger = get_logger("forecasting.evaluate")


def define_holdout_window(df: pd.DataFrame, date_col: str = "date", holdout_days: int = 28) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Extracts a strictly held-out time window of `holdout_days` (28 days) from the dataset.
    This period was never seen during expanding-window CV or hyperparameter search.
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    max_date = df_copy[date_col].max()
    holdout_start = max_date - timedelta(days=holdout_days - 1)
    
    holdout_mask = (df_copy[date_col] >= holdout_start) & (df_copy[date_col] <= max_date)
    holdout_df = df_copy[holdout_mask].copy()
    
    logger.info(
        f"Extracted Holdout Period: [{holdout_start.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}] "
        f"({len(holdout_df)} rows, {holdout_days} days)"
    )
    return holdout_df, holdout_start, max_date


def get_segment_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Returns boolean masks for evaluation segment breakdowns:
    - overall: All holdout rows
    - cold_start: Products with product_age_days <= 30 or days_since_first_sale <= 30
    - promo_periods: Active promotion or discount > 0
    - holiday_periods: Holiday or event day
    """
    masks = {"overall": pd.Series(True, index=df.index)}

    # Cold-start products
    if "product_age_days" in df.columns:
        masks["cold_start"] = df["product_age_days"] <= 30
    elif "days_since_first_sale" in df.columns:
        masks["cold_start"] = df["days_since_first_sale"] <= 30
    else:
        masks["cold_start"] = pd.Series(False, index=df.index)

    # Promo periods
    promo_cond = pd.Series(False, index=df.index)
    if "is_promo_active" in df.columns:
        promo_cond = promo_cond | (df["is_promo_active"] == 1)
    if "discount_percent" in df.columns:
        promo_cond = promo_cond | (df["discount_percent"] > 0)
    if "discount_amount" in df.columns:
        promo_cond = promo_cond | (df["discount_amount"] > 0)
    masks["promo_periods"] = promo_cond

    # Holiday & Event periods
    holiday_cond = pd.Series(False, index=df.index)
    if "is_holiday" in df.columns:
        holiday_cond = holiday_cond | (df["is_holiday"] == 1)
    if "is_event_day" in df.columns:
        holiday_cond = holiday_cond | (df["is_event_day"] == 1)
    masks["holiday_periods"] = holiday_cond

    return masks


def evaluate_model_file(
    model_path: str,
    feature_df: pd.DataFrame,
    product_dim: pd.DataFrame
) -> Dict[str, Any]:
    """
    Loads a saved model joblib file, prepares hierarchy data on holdout set,
    and computes overall and segmented evaluation metrics.
    """
    model_name = os.path.basename(model_path).replace(".joblib", "")
    logger.info(f"Evaluating Model: {model_name} from {model_path}")
    
    artifact = joblib.load(model_path)
    level = artifact.get("level", "sku_region")
    
    # 1. Prepare hierarchy data & extract holdout window
    level_df = prepare_hierarchy_data(feature_df, product_dim, level=level)
    holdout_df, h_start, h_end = define_holdout_window(level_df, date_col="date", holdout_days=28)
    
    # 2. Check if model is Prophet dictionary or GBDT regressor
    if "top_n_models" in artifact:
        # Prophet series-level model dictionary
        top_models = artifact["top_n_models"]
        preds_list = []
        y_true_list = []
        
        # Prophet predictions on holdout per top series
        for series_id, p_model in top_models.items():
            if "_" in series_id:
                pid, reg = series_id.rsplit("_", 1)
            else:
                pid, reg = series_id, ""
            s_mask = (holdout_df["product_id"] == pid) & (holdout_df["region"] == reg)
            s_holdout = holdout_df[s_mask].copy()
            if len(s_holdout) == 0:
                continue
            
            p_val_dates = pd.DataFrame({"ds": s_holdout["date"]})
            forecast = p_model.predict(p_val_dates)
            yhat = np.clip(forecast["yhat"].values, 0.0, None)
            
            s_holdout["y_pred"] = yhat
            preds_list.append(s_holdout)
            
        if not preds_list:
            logger.warning(f"Prophet evaluation: No holdout rows matched for top series.")
            return {"model_name": model_name, "level": level, "overall_metrics": {}, "segment_metrics": {}}
            
        eval_df = pd.concat(preds_list, ignore_index=True)
    else:
        # LightGBM or XGBoost Regressor
        model = artifact["model"]
        feature_cols = artifact["feature_cols"]
        cat_cols = artifact.get("cat_cols", [])
        
        eval_df = holdout_df.copy()
        for col in eval_df.columns:
            if eval_df[col].dtype == "object" or col in cat_cols:
                eval_df[col] = eval_df[col].astype("category")
                
        X_holdout = eval_df[feature_cols]
        preds = model.predict(X_holdout)
        eval_df["y_pred"] = np.clip(preds, 0.0, None)

    # 3. Compute overall and segment metrics
    y_true = eval_df["actual_sales"].values
    y_pred = eval_df["y_pred"].values
    
    overall_metrics = evaluate_forecasts(y_true, y_pred)
    
    segment_masks = get_segment_masks(eval_df)
    segment_results = {}
    for seg_name, mask in segment_masks.items():
        sub_true = eval_df.loc[mask, "actual_sales"].values
        sub_pred = eval_df.loc[mask, "y_pred"].values
        if len(sub_true) > 0:
            segment_results[seg_name] = evaluate_forecasts(sub_true, sub_pred)
            segment_results[seg_name]["count"] = len(sub_true)
        else:
            segment_results[seg_name] = {"wmape": np.nan, "rmse": np.nan, "mape": np.nan, "bias": np.nan, "count": 0}

    logger.info(
        f"Evaluation Result [{model_name}] ({level}) | Overall WMAPE: {overall_metrics['wmape']:.2f}% | "
        f"RMSE: {overall_metrics['rmse']:.2f} | Bias: {overall_metrics['bias']:.2f}%"
    )

    return {
        "model_name": model_name,
        "level": level,
        "model_path": model_path,
        "artifact": artifact,
        "overall_metrics": overall_metrics,
        "segment_metrics": segment_results,
        "sample_count": len(eval_df)
    }


def register_staging_models(best_models_by_level: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Registers the best model per hierarchy level in the MLflow Model Registry
    and sets the 'staging' alias.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Model_Registry_Evaluation")
    client = MlflowClient()
    registered_info = []

    for level, result in best_models_by_level.items():
        model_name = f"Demand_Forecasting_{level.upper()}"
        model_path = result["model_path"]
        artifact = result["artifact"]
        overall_metrics = result["overall_metrics"]

        logger.info(f"Registering Best Model for level '{level}' -> MLflow Registry Name: '{model_name}'")

        with mlflow.start_run(run_name=f"Register_{level}_Staging") as run:
            mlflow.log_params({
                "hierarchy_level": level,
                "selected_model_file": os.path.basename(model_path),
                "holdout_wmape": overall_metrics["wmape"],
                "holdout_rmse": overall_metrics["rmse"],
                "holdout_bias": overall_metrics["bias"]
            })
            mlflow.log_metrics(overall_metrics)
            mlflow.log_artifact(model_path)
            
            # Save scikit-learn compatible or LightGBM/XGBoost model object
            py_model = artifact.get("model", None)
            if py_model is not None:
                if "lightgbm" in result["model_name"]:
                    model_info = mlflow.lightgbm.log_model(py_model, artifact_path="model")
                elif "xgboost" in result["model_name"]:
                    model_info = mlflow.xgboost.log_model(py_model, artifact_path="model")
                else:
                    model_info = mlflow.sklearn.log_model(py_model, artifact_path="model")
                
                model_uri = model_info.model_uri
                
                # Register model in MLflow Registry
                reg_model = mlflow.register_model(model_uri=model_uri, name=model_name)
                
                # Assign "staging" alias to registered model version
                client.set_registered_model_alias(name=model_name, alias="staging", version=reg_model.version)
                
                logger.info(f"Successfully registered '{model_name}' v{reg_model.version} with alias 'staging'")
                registered_info.append({
                    "level": level,
                    "model_name": model_name,
                    "version": reg_model.version,
                    "alias": "staging",
                    "model_uri": model_uri,
                    "holdout_wmape": overall_metrics["wmape"]
                })

    return registered_info


def run_evaluation(
    models_dir: str = "models",
    feature_store_path: str = "data/processed/feature_store.parquet",
    product_dim_path: str = "data/processed/product_dim.parquet"
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Runs full holdout evaluation for all models in models_dir and registers the best model per level.
    """
    logger.info("=== Starting Holdout Evaluation & Model Registration ===")
    
    feature_df, product_dim = load_feature_store(feature_store_path, product_dim_path)
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' does not exist.")
        
    model_files = [
        os.path.join(models_dir, f) for f in os.listdir(models_dir)
        if f.endswith(".joblib")
    ]
    
    if not model_files:
        raise FileNotFoundError(f"No .joblib model files found in '{models_dir}'.")
        
    all_eval_results = []
    for m_path in model_files:
        try:
            res = evaluate_model_file(m_path, feature_df, product_dim)
            if res["overall_metrics"]:
                all_eval_results.append(res)
        except Exception as e:
            logger.error(f"Error evaluating {m_path}: {e}")

    # Determine best model per hierarchy level (lowest holdout WMAPE)
    best_models_by_level = {}
    for res in all_eval_results:
        level = res["level"]
        wmape = res["overall_metrics"]["wmape"]
        if level not in best_models_by_level or wmape < best_models_by_level[level]["overall_metrics"]["wmape"]:
            best_models_by_level[level] = res

    logger.info("=== Best Models Selected per Hierarchy Level ===")
    for lvl, best_res in best_models_by_level.items():
        logger.info(f"Level: {lvl:15s} | Best Model: {best_res['model_name']} | Holdout WMAPE: {best_res['overall_metrics']['wmape']:.2f}%")

    # Register best models to MLflow Registry as 'staging'
    registered_models = register_staging_models(best_models_by_level)

    return all_eval_results, best_models_by_level


if __name__ == "__main__":
    run_evaluation()
