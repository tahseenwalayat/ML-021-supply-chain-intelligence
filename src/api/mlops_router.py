import os
import joblib
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from src.mlops.drift_detector import DataDriftDetector

router = APIRouter(prefix="/api/v1/mlops", tags=["MLOps & Data Drift Engine"])
drift_detector = DataDriftDetector()
retraining_jobs: Dict[str, Dict[str, Any]] = {}


class ForecastDegradationRequest(BaseModel):
    baseline_wmape: float = Field(..., ge=0.0)
    current_wmape: float = Field(..., ge=0.0)


class RetrainingRequest(BaseModel):
    hierarchy_level: str = Field("sku_region", pattern="^(sku_region|category_region|region_total)$")
    force_run: bool = False


def _run_retraining_job(job_id: str, request: RetrainingRequest) -> None:
    """Run retraining outside the request lifecycle and retain a queryable result."""
    retraining_jobs[job_id].update({"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    try:
        from src.mlops.retraining_pipeline import run_retraining_pipeline

        report = run_retraining_pipeline(
            hierarchy_level=request.hierarchy_level,
            force_run=request.force_run,
            drift_triggered=True,
        )
        retraining_jobs[job_id].update({"status": "completed", "report": report})
    except Exception as exc:
        retraining_jobs[job_id].update({"status": "failed", "error": str(exc)})
    finally:
        retraining_jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()


@router.get("/models", summary="List Registered Models & Artifact Metrics")
def list_registered_models() -> Dict[str, Any]:
    """Lists registered forecasting models and serialized performance metrics from artifact storage."""
    models_dir = "models"
    result = []
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".joblib"):
                fpath = os.path.join(models_dir, f)
                try:
                    data = joblib.load(fpath)
                    result.append({
                        "model_file": f,
                        "level": data.get("level", "unknown"),
                        "wmape": data.get("wmape", None),
                        "naive_wmape": data.get("naive_wmape", None),
                        "features_count": len(data.get("feature_cols", [])),
                        "target_col": data.get("target_col", "legacy_actual_sales"),
                        "serving_ready": data.get("target_col") == "target_next_day_sales",
                    })
                except Exception:
                    result.append({"model_file": f, "status": "load_error"})
    return {"registered_models": result}


@router.post("/detect-degradation", summary="Detect Model WMAPE Degradation & Retrain Recommendation")
def detect_degradation(req: ForecastDegradationRequest) -> Dict[str, Any]:
    """Evaluates forecast error degradation against baseline and returns retraining recommendations."""
    try:
        return drift_detector.detect_forecast_degradation(
            baseline_wmape=req.baseline_wmape,
            current_wmape=req.current_wmape
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/retrain", status_code=202, summary="Start an audited background retraining job")
def start_retraining(req: RetrainingRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Queue retraining; poll the returned job URL rather than holding an API request open."""
    job_id = str(uuid.uuid4())
    retraining_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "hierarchy_level": req.hierarchy_level,
    }
    background_tasks.add_task(_run_retraining_job, job_id, req)
    return {**retraining_jobs[job_id], "status_url": f"/api/v1/mlops/retrain/{job_id}"}


@router.get("/retrain/{job_id}", summary="Get the status of a retraining job")
def get_retraining_status(job_id: str) -> Dict[str, Any]:
    if job_id not in retraining_jobs:
        raise HTTPException(status_code=404, detail="Retraining job not found.")
    return retraining_jobs[job_id]
