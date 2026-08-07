import os
from fastapi import Header, HTTPException, Depends

# Default Test Key from api_spec.md
API_KEY_SECRET = os.getenv("API_KEY", "sc-key-secret-2026")

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
