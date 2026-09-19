import datetime
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from backend.config import settings
from backend.database import check_database_connection
from backend.interfaces import registry
from backend.logger import logger

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def get_health():
    """
    Evaluates system operational health across:
    - Backend API service
    - Local SQLite database
    - AI model registry states
    """
    db_connected = check_database_connection()
    ai_models_status = registry.get_summary_status()
    models_detail = registry.get_detailed_status()

    # Determine overall system health
    is_healthy = db_connected
    system_status = "healthy" if is_healthy else "degraded"

    payload = {
        "status": system_status,
        "backend": "connected",
        "database": "connected" if db_connected else "disconnected",
        "ai_models": ai_models_status,
        "models": models_detail,
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "offline_mode": settings.OFFLINE_MODE,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if not is_healthy:
        logger.warning(f"Health check degraded: DB connected={db_connected}")
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)

    return payload
