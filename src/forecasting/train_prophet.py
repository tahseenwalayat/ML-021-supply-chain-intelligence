import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import mlflow

from src.utils.logging_config import get_logger
from src.forecasting.dataset_split import (
    load_feature_store,
    prepare_hierarchy_data,
    generate_expanding_window_splits,
    evaluate_forecasts,
    compute_seasonal_naive_baseline
)

logger = get_logger("forecasting.train_prophet")


def train_and_eval_prophet_series(
    series_df: pd.DataFrame,
    series_id: str,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Trains Prophet model across 5 expanding window CV folds for a single time series.
    """
    try:
        from prophet import Prophet
    except ImportError as e:
        logger.error("Prophet package is not installed. Please install prophet via pip.")
        raise e

    df_copy = series_df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy = df_copy.sort_values(by="date").reset_index(drop=True)

    splits = generate_expanding_window_splits(df_copy, date_col="date", n_splits=n_splits)

    fold_metrics_list = []
    naive_metrics_list = []

    for train_df, val_df, meta in splits:
        # Prepare Prophet format: ds (date), y (target)
        p_train = pd.DataFrame({"ds": train_df["date"], "y": train_df["actual_sales"]})
        p_val_dates = pd.DataFrame({"ds": val_df["date"]})
        y_val = val_df["actual_sales"].values

        # Suppress Prophet internal logs
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95
        )
        model.fit(p_train)

        forecast = model.predict(p_val_dates)
        preds = np.clip(forecast["yhat"].values, 0.0, None)

        fold_metrics = evaluate_forecasts(y_val, preds)
        fold_metrics_list.append(fold_metrics)

        # Seasonal naive baseline (7-day lag)
        group_cols = [c for c in ["product_id", "region"] if c in df_copy.columns]
        if not group_cols:
            group_cols = ["region"] if "region" in df_copy.columns else []

        if group_cols:
            naive_metrics = compute_seasonal_naive_baseline(
                val_df, df_copy, group_cols=group_cols, target_col="actual_sales", lag_days=7
            )
        else:
            # Fallback naive if no group cols
            y_naive = np.roll(val_df["actual_sales"].values, 7)
            naive_metrics = evaluate_forecasts(y_val, y_naive)

        naive_metrics_list.append(naive_metrics)

    avg_wmape = float(np.mean([m["wmape"] for m in fold_metrics_list]))
    avg_rmse = float(np.mean([m["rmse"] for m in fold_metrics_list]))
    avg_mape = float(np.mean([m["mape"] for m in fold_metrics_list]))
    avg_bias = float(np.mean([m["bias"] for m in fold_metrics_list]))
    avg_naive_wmape = float(np.mean([m["wmape"] for m in naive_metrics_list]))

    # Fit final model on full history
    p_full = pd.DataFrame({"ds": df_copy["date"], "y": df_copy["actual_sales"]})
    final_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
    )
    final_model.fit(p_full)

    return {
        "series_id": series_id,
        "wmape": avg_wmape,
        "rmse": avg_rmse,
        "mape": avg_mape,
        "bias": avg_bias,
        "naive_wmape": avg_naive_wmape,
        "model": final_model
    }


def train_prophet_top_series(
    feature_store_path: str = "data/processed/feature_store.parquet",
    product_dim_path: str = "data/processed/product_dim.parquet",
    models_dir: str = "models",
    top_n: int = 10,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Trains Prophet baseline models on the top-N SKU-region series by historical sales volume.
    Logs parameters, per-series metrics, and aggregate metrics to MLflow.
    """
    logger.info(f"=== Starting Prophet Training on Top-{top_n} SKU-Region Series ===")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Demand_Forecasting_Prophet")

    df, product_dim = load_feature_store(feature_store_path, product_dim_path)
    sku_df = prepare_hierarchy_data(df, product_dim, level="sku_region")

    # Find top N SKU-region pairs by volume
    volume_by_series = (
        sku_df.groupby(["product_id", "region"])["actual_sales"]
        .sum()
        .reset_index()
        .sort_values(by="actual_sales", ascending=False)
    )
    top_series_list = volume_by_series.head(top_n)[["product_id", "region"]].to_dict(orient="records")

    logger.info(f"Identified top {len(top_series_list)} series for Prophet modeling:")
    for idx, s in enumerate(top_series_list, 1):
        logger.info(f"  {idx}. SKU: {s['product_id']} | Region: {s['region']}")

    results = []
    with mlflow.start_run(run_name=f"Prophet_Top_{top_n}_Series") as run:
        mlflow.log_param("top_n", top_n)
        mlflow.log_param("n_splits", n_splits)

        for s in top_series_list:
            pid, reg = s["product_id"], s["region"]
            series_id = f"{pid}_{reg}"
            s_mask = (sku_df["product_id"] == pid) & (sku_df["region"] == reg)
            s_df = sku_df[s_mask].copy()

            if len(s_df) < 60:
                logger.warning(f"Series {series_id} has insufficient rows ({len(s_df)}). Skipping.")
                continue

            res = train_and_eval_prophet_series(s_df, series_id=series_id, n_splits=n_splits)
            results.append(res)

            logger.info(
                f"Prophet [{series_id}] | Avg Val WMAPE: {res['wmape']:.2f}% vs Naive: {res['naive_wmape']:.2f}%"
            )
            mlflow.log_metric(f"wmape_{series_id}", res["wmape"])

        # Aggregate metrics across top N series
        avg_top_wmape = float(np.mean([r["wmape"] for r in results]))
        avg_top_rmse = float(np.mean([r["rmse"] for r in results]))
        avg_top_naive = float(np.mean([r["naive_wmape"] for r in results]))

        mlflow.log_metrics({
            "val_wmape_top_n_avg": avg_top_wmape,
            "val_rmse_top_n_avg": avg_top_rmse,
            "seasonal_naive_wmape_avg": avg_top_naive,
            "wmape_improvement_vs_naive": avg_top_naive - avg_top_wmape
        })

        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"prophet_top_{top_n}_series.joblib")
        # Save dictionary of top series models
        saved_models = {r["series_id"]: r["model"] for r in results}
        joblib.dump({
            "top_n_models": saved_models,
            "wmape_avg": avg_top_wmape,
            "naive_wmape_avg": avg_top_naive,
            "top_n": top_n
        }, model_path)

        mlflow.log_artifact(model_path)
        logger.info(
            f"Summary [Prophet Top {top_n}] | Overall Avg WMAPE: {avg_top_wmape:.2f}% vs Naive: {avg_top_naive:.2f}% | "
            f"Saved artifact to {model_path}"
        )

    return {
        "avg_wmape": avg_top_wmape,
        "avg_naive_wmape": avg_top_naive,
        "model_path": model_path,
        "num_series_trained": len(results)
    }


if __name__ == "__main__":
    train_prophet_top_series()
