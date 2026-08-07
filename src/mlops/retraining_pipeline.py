import os
import sys
import datetime
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.utils.logging_config import get_logger
from src.features.build_feature_table import build_feature_store
from src.forecasting.dataset_split import (
    load_feature_store,
    prepare_hierarchy_data,
    generate_expanding_window_splits,
    evaluate_forecasts,
    compute_seasonal_naive_baseline
)
from src.forecasting.train_lightgbm import train_and_eval_lightgbm_level
from src.mlops.registry import promote_model, demote_model, get_current_production_model

logger = get_logger("mlops.retraining_pipeline")


def pull_fresh_data() -> bool:
    """Simulates pulling fresh ingestion batch or verifying data presence."""
    feature_store_path = "data/processed/feature_store.parquet"
    product_dim_path = "data/processed/product_dim.parquet"
    if os.path.exists(feature_store_path) and os.path.exists(product_dim_path):
        logger.info("Fresh data verified in processed feature store directory.")
        return True
    logger.warning("Processed feature store not found. Triggering rebuild...")
    return False


def rebuild_features() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuilds feature table using src.features pipeline."""
    logger.info("=== Step 2: Rebuilding Feature Engineering Pipeline ===")
    df_features = build_feature_store()
    df_product_dim = pd.read_parquet("data/processed/product_dim.parquet")
    logger.info(f"Rebuilt feature store with {len(df_features)} rows and {len(df_features.columns)} features.")
    return df_features, df_product_dim



def retrain_and_evaluate_candidate(
    level: str = "sku_region",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Retrains LightGBM candidate model on fresh feature data and computes holdout validation metrics.
    """
    logger.info(f"=== Step 3 & 4: Retraining & Evaluating Candidate Model [{level}] ===")
    res = train_lightgbm_level_pipeline(level=level, models_dir=models_dir)
    return res


def train_lightgbm_level_pipeline(level: str = "sku_region", models_dir: str = "models") -> Dict[str, Any]:
    """Internal helper to train candidate LightGBM model."""
    df, product_dim = load_feature_store()
    return train_and_eval_lightgbm_level(df, product_dim, level=level, models_dir=models_dir)


def run_retraining_pipeline(
    hierarchy_level: str = "sku_region",
    force_run: bool = False,
    drift_triggered: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates unattended end-to-end automated retraining pipeline:
    1. Pull fresh data
    2. Rebuild features
    3. Retrain candidate model
    4. Evaluate candidate on holdout validation set
    5. Compare against current production model (STRICT PROMOTION GATE)
    6. Promote to Production ONLY IF candidate outperforms current production model.
    """
    start_time = datetime.datetime.now()
    logger.info(f"🚀 Starting Automated Retraining Pipeline for level '{hierarchy_level}' (Drift Triggered: {drift_triggered})")

    # Step 1 & 2: Data & Feature Pipeline
    data_ok = pull_fresh_data()
    if not data_ok or force_run:
        df_features, product_dim = rebuild_features()

    # Step 3 & 4: Retrain & Evaluate Candidate Model
    candidate_res = retrain_and_evaluate_candidate(level=hierarchy_level)
    candidate_wmape = candidate_res.get("wmape", 999.0)
    candidate_naive_wmape = candidate_res.get("naive_wmape", 18.5)
    model_name = f"lightgbm_{hierarchy_level}"
    new_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 5: Fetch Current Production Model for Gate Comparison
    current_prod = get_current_production_model(model_name)
    
    if current_prod:
        production_wmape = current_prod.get("metrics", {}).get("wmape", candidate_naive_wmape)
        prod_version = current_prod.get("version", "unknown")
    else:
        # Default baseline if no active production model recorded
        production_wmape = candidate_naive_wmape
        prod_version = "baseline_naive"

    logger.info(f"📊 Promotion Gate Evaluation: Candidate WMAPE = {candidate_wmape:.2f}% | Current Production WMAPE = {production_wmape:.2f}%")

    # Step 6: STRICT PROMOTION GATE
    # Retraining pipeline MUST NEVER promote a model that performs worse than current production model
    is_better = bool(candidate_wmape < production_wmape)

    if is_better:
        justification = (
            f"Candidate model v{new_version} WMAPE ({candidate_wmape:.2f}%) outperformed "
            f"Production model v{prod_version} WMAPE ({production_wmape:.2f}%) on holdout validation set."
        )
        promotion_audit = promote_model(
            model_name=model_name,
            version=new_version,
            target_stage="Production",
            reason=justification,
            metrics={"wmape": candidate_wmape, "naive_wmape": candidate_naive_wmape}
        )
        promoted = True
        decision_status = "PROMOTED_TO_PRODUCTION"
    else:
        justification = (
            f"REJECTED: Candidate model v{new_version} WMAPE ({candidate_wmape:.2f}%) did NOT outperform "
            f"Current Production model v{prod_version} WMAPE ({production_wmape:.2f}%)."
        )
        promotion_audit = promote_model(
            model_name=model_name,
            version=new_version,
            target_stage="Staging",
            reason=justification,
            metrics={"wmape": candidate_wmape, "naive_wmape": candidate_naive_wmape}
        )
        promoted = False
        decision_status = "REJECTED_STAGING_ONLY"

    duration_sec = (datetime.datetime.now() - start_time).total_seconds()

    summary_report = {
        "status": "SUCCESS",
        "hierarchy_level": hierarchy_level,
        "execution_duration_sec": round(duration_sec, 2),
        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_version": new_version,
        "candidate_wmape": round(candidate_wmape, 2),
        "production_baseline_wmape": round(production_wmape, 2),
        "promoted_to_production": promoted,
        "decision_status": decision_status,
        "justification": justification,
        "audit_record": promotion_audit
    }

    logger.info(f"🏁 Automated Retraining Pipeline Finished in {duration_sec:.2f}s | Result: {decision_status}")
    return summary_report


if __name__ == "__main__":
    report = run_retraining_pipeline(force_run=True)
    print("\n--- RETRAINING PIPELINE REPORT ---")
    print(report)
