import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.forecasting.dataset_split import (
    compute_wmape,
    compute_rmse,
    compute_mape,
    compute_bias,
    evaluate_forecasts,
    generate_expanding_window_splits,
    prepare_hierarchy_data,
    compute_seasonal_naive_baseline
)
from src.forecasting.evaluate import evaluate_model_file


def test_metrics_calculation():
    y_true = np.array([100.0, 0.0, 50.0, 200.0])
    y_pred = np.array([110.0, 10.0, 40.0, 190.0])

    wmape = compute_wmape(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    bias = compute_bias(y_true, y_pred)
    all_metrics = evaluate_forecasts(y_true, y_pred)

    assert wmape > 0.0
    assert rmse > 0.0
    assert "wmape" in all_metrics
    assert "rmse" in all_metrics
    assert "bias" in all_metrics


def test_expanding_window_cv_splits_no_leakage():
    # Construct synthetic 200-day daily dataset
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "product_id": "P1",
        "region": "US",
        "date": dates,
        "actual_sales": np.random.uniform(10, 100, size=len(dates))
    })

    splits = generate_expanding_window_splits(df, date_col="date", n_splits=5, val_horizon_days=28, embargo_days=1)

    assert len(splits) == 5

    prev_train_max = None
    for fold_idx, (train_df, val_df, meta) in enumerate(splits, 1):
        train_max_date = train_df["date"].max()
        val_min_date = val_df["date"].min()
        val_max_date = val_df["date"].max()

        # Rule 1: Cutoff date is strictly prior to validation start date (No future leakage)
        assert train_max_date < val_min_date, f"Fold {fold_idx}: Train max {train_max_date} >= Val min {val_min_date}"

        # Rule 2: Validation horizon is exactly 28 days
        val_days = (val_max_date - val_min_date).days + 1
        assert val_days == 28, f"Fold {fold_idx}: Validation horizon expected 28 days, got {val_days}"

        # Rule 3: Expanding window (each successive fold training set expands)
        if prev_train_max is not None:
            assert train_max_date > prev_train_max, f"Fold {fold_idx}: Training set did not expand"
        prev_train_max = train_max_date


def test_hierarchy_data_preparation():
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "product_id": ["P1", "P2"] * 10,
        "region": ["US"] * 20,
        "date": np.repeat(dates, 2),
        "actual_sales": np.ones(20) * 10.0,
        "sales_velocity_7d": np.ones(20) * 5.0
    })

    product_dim = pd.DataFrame({
        "product_id": ["P1", "P2"],
        "category": ["Electronics", "Electronics"]
    })

    # Level 1: SKU-region
    sku_df = prepare_hierarchy_data(df, product_dim, level="sku_region")
    assert len(sku_df) == 20
    assert "category" in sku_df.columns

    # Level 2: Category-region
    cat_df = prepare_hierarchy_data(df, product_dim, level="category_region")
    assert len(cat_df) == 10
    assert cat_df["actual_sales"].iloc[0] == 20.0

    # Level 3: Region-total
    reg_df = prepare_hierarchy_data(df, product_dim, level="region_total")
    assert len(reg_df) == 10
    assert reg_df["actual_sales"].iloc[0] == 20.0


def test_evaluation_uses_artifact_target_label(tmp_path, monkeypatch):
    """Forecast metrics must compare predictions with the training target."""
    dates = pd.date_range(start="2023-01-01", periods=35, freq="D")
    feature_df = pd.DataFrame({
        "product_id": "P1",
        "region": "US",
        "date": dates,
        "actual_sales": np.zeros(len(dates)),
        "target_next_day_sales": np.ones(len(dates)) * 10.0,
        "feat1": np.ones(len(dates)),
    })
    product_dim = pd.DataFrame({"product_id": ["P1"], "category": ["CatA"]})

    class ConstantModel:
        def predict(self, rows):
            return np.ones(len(rows)) * 10.0

    artifact_path = tmp_path / "lightgbm_sku_region.joblib"
    artifact_path.touch()
    monkeypatch.setattr(
        "src.forecasting.evaluate.joblib.load",
        lambda _: {
            "model": ConstantModel(),
            "feature_cols": ["product_id", "region", "feat1"],
            "cat_cols": ["product_id", "region"],
            "level": "sku_region",
            "target_col": "target_next_day_sales",
        },
    )

    result = evaluate_model_file(str(artifact_path), feature_df, product_dim)
    assert result["overall_metrics"]["wmape"] == 0.0


def test_train_lightgbm_level(tmp_path):
    pytest.importorskip("lightgbm")
    from src.forecasting.train_lightgbm import train_and_eval_lightgbm_level

    dates = pd.date_range(start="2023-01-01", periods=160, freq="D")
    df = pd.DataFrame({
        "product_id": "P1",
        "region": "US",
        "date": dates,
        "actual_sales": np.random.uniform(10, 100, size=160),
        "target_next_day_sales": np.random.uniform(10, 100, size=160),
        "feat1": np.random.uniform(0, 1, size=160),
        "feat2": np.random.uniform(0, 1, size=160)
    })
    product_dim = pd.DataFrame({"product_id": ["P1"], "category": ["CatA"]})
    models_dir = str(tmp_path / "models")

    res = train_and_eval_lightgbm_level(df, product_dim, level="sku_region", models_dir=models_dir, n_splits=2)
    assert res["level"] == "sku_region"
    assert "wmape" in res
    assert os.path.exists(res["model_path"])


def test_train_xgboost_level(tmp_path):
    pytest.importorskip("xgboost")
    from src.forecasting.train_xgboost import train_and_eval_xgboost_level

    dates = pd.date_range(start="2023-01-01", periods=160, freq="D")
    df = pd.DataFrame({
        "product_id": "P1",
        "region": "US",
        "date": dates,
        "actual_sales": np.random.uniform(10, 100, size=160),
        "target_next_day_sales": np.random.uniform(10, 100, size=160),
        "feat1": np.random.uniform(0, 1, size=160),
        "feat2": np.random.uniform(0, 1, size=160)
    })
    product_dim = pd.DataFrame({"product_id": ["P1"], "category": ["CatA"]})
    models_dir = str(tmp_path / "models")

    res = train_and_eval_xgboost_level(df, product_dim, level="sku_region", models_dir=models_dir, n_splits=2)
    assert res["level"] == "sku_region"
    assert "wmape" in res
    assert os.path.exists(res["model_path"])


def test_train_prophet_series():
    pytest.importorskip("prophet")
    from src.forecasting.train_prophet import train_and_eval_prophet_series

    dates = pd.date_range(start="2023-01-01", periods=160, freq="D")
    s_df = pd.DataFrame({
        "product_id": "P1",
        "region": "US",
        "date": dates,
        "actual_sales": np.random.uniform(10, 100, size=160)
    })

    res = train_and_eval_prophet_series(s_df, series_id="P1_US", n_splits=2)
    assert res["series_id"] == "P1_US"
    assert "wmape" in res
    assert res["model"] is not None
