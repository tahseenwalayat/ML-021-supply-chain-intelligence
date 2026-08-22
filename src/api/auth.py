import os
from fastapi import Header, HTTPException, Depends

# The fallback is restricted to local development so production deployments fail
# closed if an operator forgets to configure a secret.
DEFAULT_DEVELOPMENT_API_KEY = "sc-key-secret-2026"
API_KEY_SECRET = os.getenv("API_KEY", DEFAULT_DEVELOPMENT_API_KEY)
_is_placeholder_key = API_KEY_SECRET == DEFAULT_DEVELOPMENT_API_KEY or API_KEY_SECRET.lower().startswith("replace_")
if os.getenv("ENVIRONMENT", "development").lower() == "production" and _is_placeholder_key:
    raise RuntimeError("API_KEY must be set to a non-default secret in production.")

def verify_api_key(x_api_key: str = Header(...)):
    """
    Dependency to verify the X-API-Key header.
    """
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Could not validate API Key. Access Denied."
        )
    return x_api_key
