import os
import json
import datetime
from typing import Dict, Any, List, Optional
from src.utils.logging_config import get_logger

logger = get_logger("mlops.registry")

REGISTRY_FILE = "models/model_registry.json"

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def _load_registry_store() -> Dict[str, Any]:
    """Loads local persistent registry store."""
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading registry file {REGISTRY_FILE}: {e}")
    return {"models": {}, "audit_history": []}


def _save_registry_store(data: Dict[str, Any]):
    """Saves persistent registry store."""
    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    try:
        with open(REGISTRY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save registry file: {e}")


def promote_model(
    model_name: str,
    version: str,
    target_stage: str = "Production",
    reason: str = "Model metric improvement verified",
    metrics: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Promotes a model version to a target stage (e.g., Staging -> Production) with a logged justification reason.
    Interfaces with MLflow Model Registry when available, with fallback to local persistent store.
    """
    target_stage_clean = target_stage.capitalize()
    if target_stage_clean not in ["Staging", "Production", "Archived"]:
        raise ValueError(f"Invalid target_stage '{target_stage}'. Must be one of Staging, Production, Archived.")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registry_data = _load_registry_store()
    
    models = registry_data.get("models", {})
    history = registry_data.get("audit_history", [])

    current_prod = models.get(model_name, {}).get("Production")
    previous_stage = "Staging"
    if current_prod and current_prod.get("version") == version:
        previous_stage = "Production"

    # If promoting to Production, demote existing Production version to Archived/Staging
    if target_stage_clean == "Production" and current_prod and current_prod.get("version") != version:
        old_version = current_prod.get("version")
        history.append({
            "timestamp": now_str,
            "event": "DEMOTE",
            "model_name": model_name,
            "version": old_version,
            "from_stage": "Production",
            "to_stage": "Archived",
            "reason": f"Superceded by new Production model version {version}"
        })

    # Log MLflow alias / stage if MLflow is available
    if HAS_MLFLOW:
        try:
            client = MlflowClient()
            # Try setting alias
            alias = target_stage_clean.lower()
            try:
                client.set_registered_model_alias(model_name, alias, str(version))
                logger.info(f"MLflow: Set alias '{alias}' for model '{model_name}' version {version}")
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"MLflow client transition note: {e}")

    # Record model entry in local registry
    if model_name not in models:
        models[model_name] = {}
    
    record = {
        "model_name": model_name,
        "version": str(version),
        "stage": target_stage_clean,
        "promoted_at": now_str,
        "reason": reason,
        "metrics": metrics or {}
    }
    models[model_name][target_stage_clean] = record

    audit_entry = {
        "timestamp": now_str,
        "event": "PROMOTE",
        "model_name": model_name,
        "version": str(version),
        "from_stage": previous_stage,
        "to_stage": target_stage_clean,
        "reason": reason,
        "metrics": metrics or {}
    }
    history.append(audit_entry)

    registry_data["models"] = models
    registry_data["audit_history"] = history
    _save_registry_store(registry_data)

    logger.info(
        f"✅ PROMOTED model '{model_name}' v{version} to [{target_stage_clean}]. "
        f"Reason: {reason}"
    )

    return audit_entry


def demote_model(
    model_name: str,
    version: str,
    target_stage: str = "Staging",
    reason: str = "Model performance degradation or manual rollback"
) -> Dict[str, Any]:
    """
    Demotes a model version from Production to Staging or Archived with a logged justification reason.
    """
    target_stage_clean = target_stage.capitalize()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    registry_data = _load_registry_store()

    models = registry_data.get("models", {})
    history = registry_data.get("audit_history", [])

    if model_name in models and "Production" in models[model_name]:
        if models[model_name]["Production"].get("version") == str(version):
            del models[model_name]["Production"]

    audit_entry = {
        "timestamp": now_str,
        "event": "DEMOTE",
        "model_name": model_name,
        "version": str(version),
        "from_stage": "Production",
        "to_stage": target_stage_clean,
        "reason": reason
    }
    history.append(audit_entry)

    registry_data["models"] = models
    registry_data["audit_history"] = history
    _save_registry_store(registry_data)

    logger.info(f"🔻 DEMOTED model '{model_name}' v{version} to [{target_stage_clean}]. Reason: {reason}")
    return audit_entry


def get_current_production_model(model_name: str = "lightgbm_sku_region") -> Optional[Dict[str, Any]]:
    """Retrieves current active Production model details for a given model name."""
    registry_data = _load_registry_store()
    return registry_data.get("models", {}).get(model_name, {}).get("Production")


def get_model_history(model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves audit trail log of model stage transitions."""
    registry_data = _load_registry_store()
    history = registry_data.get("audit_history", [])
    if model_name:
        return [h for h in history if h.get("model_name") == model_name]
    return history
