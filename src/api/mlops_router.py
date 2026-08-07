import os
import joblib
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.mlops.drift_detector import DataDriftDetector

router = APIRouter(prefix="/api/v1/mlops", tags=["MLOps & Data Drift Engine"])
drift_detector = DataDriftDetector()


class ForecastDegradationRequest(BaseModel):
    baseline_wmape: float = Field(..., ge=0.0)
    current_wmape: float = Field(..., ge=0.0)


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
                        "features_count": len(data.get("feature_cols", []))
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
