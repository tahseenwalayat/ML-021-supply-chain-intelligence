import os
from typing import Dict, Any, Optional, List

from src.utils.logging_config import get_logger

logger = get_logger("mlops.mlflow_tracker")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class MLflowTracker:
    """
    MLflow Experiment Tracker & Model Registry interface.
    Handles tracking URI initialization, experiment setup, parameter/metric logging, and model registration.
    """

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", experiment_name: str = "Supply_Chain_MLOps"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.has_mlflow = HAS_MLFLOW
        if self.has_mlflow:
            self._setup_mlflow()
        else:
            logger.warning("MLflow package not installed; running in mock tracking mode.")

    def _setup_mlflow(self):
        if not HAS_MLFLOW:
            return
        try:
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Initialized MLflow Tracker at {self.tracking_uri} for experiment '{self.experiment_name}'")
        except Exception as e:
            logger.warning(f"Failed to set MLflow tracking URI: {e}")

    def log_run_metrics(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Logs hyperparams, metrics, artifacts, and tags for a single experiment run."""
        if not HAS_MLFLOW:
            logger.info(f"Mock MLflow Tracker: Logged run '{run_name}' with metrics {metrics}")
            return "mock_run_id_12345"

        try:
            with mlflow.start_run(run_name=run_name) as run:
                if params:
                    mlflow.log_params(params)
                if metrics:
                    mlflow.log_metrics(metrics)
                if tags:
                    mlflow.set_tags(tags)
                if artifacts:
                    for art in artifacts:
                        if os.path.exists(art):
                            mlflow.log_artifact(art)

                run_id = run.info.run_id
                logger.info(f"Logged MLflow run '{run_name}' with Run ID: {run_id}")
                return run_id
        except Exception as e:
            logger.warning(f"Error logging to MLflow: {e}")
            return "mock_run_id_fallback"
