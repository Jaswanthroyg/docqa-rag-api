from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from app.core.redis_client import get_redis
from app.db import engine

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@router.get("/health/ready")
def ready():
    """Readiness: dependencies are reachable."""
    checks = {}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception:  # noqa: BLE001
        checks["postgres"] = "down"
    try:
        get_redis().ping()
        checks["redis"] = "ok"
    except Exception:  # noqa: BLE001
        checks["redis"] = "down"
    if "down" in checks.values():
        raise HTTPException(503, detail=checks)
    return checks
