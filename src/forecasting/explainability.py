import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

from src.utils.logging_config import get_logger
from src.forecasting.dataset_split import load_feature_store, prepare_hierarchy_data

logger = get_logger("forecasting.explainability")

# Safely import SHAP if available and functional (without DLL policy block)
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    logger.warning(f"SHAP package import failed ({e}). Falling back to native tree feature contribution engine.")


def compute_and_plot_shap(
    model_path: str,
    feature_df: pd.DataFrame,
    product_dim: pd.DataFrame,
    output_dir: str = "docs/shap_summary",
    sample_size: int = 500
) -> Dict[str, Any]:
    """
    Computes SHAP values or native tree feature importances for the given model,
    saves global summary plot in output_dir, and outputs 3 local explanation examples.
    """
    model_name = os.path.basename(model_path).replace(".joblib", "")
    logger.info(f"=== Running Explainability Analysis for {model_name} ===")

    artifact = joblib.load(model_path)
    level = artifact.get("level", "sku_region")
    
    if "model" not in artifact:
        logger.warning(f"Model file {model_name} does not contain a tree-based 'model' object. Skipping explainability.")
        return {}

    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    cat_cols = artifact.get("cat_cols", [])

    # 1. Prepare hierarchy dataset & sample rows
    level_df = prepare_hierarchy_data(feature_df, product_dim, level=level)
    
    eval_df = level_df.copy()
    for col in eval_df.columns:
        if eval_df[col].dtype == "object" or col in cat_cols:
            eval_df[col] = eval_df[col].astype("category")

    if len(eval_df) > sample_size:
        sample_df = eval_df.sample(n=sample_size, random_state=42).copy()
    else:
        sample_df = eval_df.copy()

    X_sample = sample_df[feature_cols]
    os.makedirs(output_dir, exist_ok=True)
    summary_plot_path = os.path.join(output_dir, f"shap_summary_{level}.png")

    shap_values = None
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f"Global SHAP Feature Importance - Hierarchy Level: {level.upper()}", fontsize=12, fontweight="bold", pad=15)
            plt.tight_layout()
            plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved SHAP summary plot for level '{level}' to {summary_plot_path}")
        except Exception as err:
            logger.warning(f"SHAP TreeExplainer execution failed ({err}). Switching to native tree feature importance engine.")
            shap_values = None

    # Fallback / Native Tree Feature Importance Plot
    if shap_values is None:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.ones(len(feature_cols)) / len(feature_cols)

        # Normalize importances
        total_imp = np.sum(importances)
        norm_imp = importances / total_imp if total_imp > 0 else importances

        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": norm_imp
        }).sort_values(by="importance", ascending=False).head(20)

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.4, 0.85, len(imp_df)))
        plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=colors[::-1], edgecolor="none")
        plt.xlabel("Normalized Feature Importance (Global Impact)", fontsize=11, fontweight="bold")
        plt.ylabel("Feature", fontsize=11, fontweight="bold")
        plt.title(f"Global Feature Importance & Impact - Level: {level.upper()}", fontsize=12, fontweight="bold", pad=15)
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved global feature importance summary plot for level '{level}' to {summary_plot_path}")

    # 2. Extract 3 local explanation examples
    # Example 1: High Demand Instance (max actual_sales in sample)
    high_idx = sample_df["actual_sales"].idxmax()

    # Example 2: Active Promo Instance
    if "is_promo_active" in sample_df.columns and (sample_df["is_promo_active"] == 1).any():
        promo_idx = sample_df[sample_df["is_promo_active"] == 1].index[0]
    elif "discount_percent" in sample_df.columns and (sample_df["discount_percent"] > 0).any():
        promo_idx = sample_df[sample_df["discount_percent"] > 0].index[0]
    else:
        promo_idx = sample_df.index[len(sample_df) // 2]

    # Example 3: Low/Zero Demand Instance (min actual_sales in sample)
    low_idx = sample_df["actual_sales"].idxmin()

    local_indices = [
        ("High-Demand Scenario", high_idx),
        ("Promotion Period Scenario", promo_idx),
        ("Low/Zero Demand Scenario", low_idx)
    ]

    base_value = float(sample_df["actual_sales"].mean())

    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
        norm_imp = raw_imp / np.sum(raw_imp) if np.sum(raw_imp) > 0 else raw_imp
    else:
        norm_imp = np.ones(len(feature_cols)) / len(feature_cols)

    # Compute numerical feature z-scores across sample to evaluate local driver direction
    feat_means = X_sample.select_dtypes(include=[np.number]).mean()
    feat_stds = X_sample.select_dtypes(include=[np.number]).std().replace(0, 1.0)

    local_explanations = []
    print(f"\n=======================================================")
    print(f"   LOCAL EXPLANATION EXAMPLES | Level: {level.upper()}")
    print(f"=======================================================")

    for scenario_name, idx in local_indices:
        row_loc = sample_df.index.get_loc(idx)
        actual_val = sample_df.loc[idx, "actual_sales"]
        pred_val = float(model.predict(X_sample.iloc[[row_loc]])[0])

        if shap_values is not None:
            row_shap = shap_values.values[row_loc]
            b_val = shap_values.base_values[row_loc] if isinstance(shap_values.base_values, (np.ndarray, list)) else shap_values.base_values
            feature_impacts = sorted(
                zip(feature_cols, X_sample.iloc[row_loc].values, row_shap),
                key=lambda x: abs(x[2]),
                reverse=True
            )[:5]
            inst_base_val = float(b_val)
        else:
            inst_base_val = base_value
            # Calculate local directional feature contribution
            impacts = []
            for col_i, feat in enumerate(feature_cols):
                val = X_sample.iloc[row_loc][feat]
                if feat in feat_means and feat in feat_stds:
                    z = (float(val) - feat_means[feat]) / feat_stds[feat]
                    impact = float(norm_imp[col_i] * z * (pred_val - inst_base_val))
                else:
                    impact = float(norm_imp[col_i] * (pred_val - inst_base_val))
                impacts.append((feat, val, impact))

            feature_impacts = sorted(impacts, key=lambda x: abs(x[2]), reverse=True)[:5]

        ex_dict = {
            "scenario": scenario_name,
            "actual_sales": float(actual_val),
            "predicted_sales": float(pred_val),
            "base_value": float(inst_base_val),
            "top_features": feature_impacts
        }
        local_explanations.append(ex_dict)

        print(f"\nScenario: {scenario_name}")
        print(f"  - Actual Demand   : {actual_val:.2f}")
        print(f"  - Predicted Demand: {pred_val:.2f}")
        print(f"  - Base Value (Mean): {inst_base_val:.2f}")
        print(f"  - Top 5 Feature Drivers:")
        for feat, val, impact in feature_impacts:
            print(f"      * {feat:30s} = {str(val):10s} | Feature Impact: {impact:+.4f}")

    return {
        "level": level,
        "summary_plot_path": summary_plot_path,
        "local_explanations": local_explanations
    }


def run_explainability_pipeline(
    models_dir: str = "models",
    output_dir: str = "docs/shap_summary",
    feature_store_path: str = "data/processed/feature_store.parquet",
    product_dim_path: str = "data/processed/product_dim.parquet"
) -> Dict[str, Any]:
    """
    Runs explainability analysis for the best model per hierarchy level.
    """
    logger.info("=== Starting Explainability Pipeline ===")
    feature_df, product_dim = load_feature_store(feature_store_path, product_dim_path)

    levels = ["sku_region", "category_region", "region_total"]
    results = {}

    for level in levels:
        candidates = [
            f"lightgbm_{level}.joblib",
            f"xgboost_{level}.joblib",
            f"feature_selected_lightgbm_{level}.joblib",
            f"best_lightgbm_{level}_optuna.joblib"
        ]
        
        selected_model_path = None
        for cand in candidates:
            cand_path = os.path.join(models_dir, cand)
            if os.path.exists(cand_path):
                selected_model_path = cand_path
                break
                
        if selected_model_path is None:
            logger.warning(f"No suitable trained model found for level '{level}' in '{models_dir}'. Skipping.")
            continue

        res = compute_and_plot_shap(
            selected_model_path,
            feature_df,
            product_dim,
            output_dir=output_dir
        )
        results[level] = res

    return results


if __name__ == "__main__":
    run_explainability_pipeline()
