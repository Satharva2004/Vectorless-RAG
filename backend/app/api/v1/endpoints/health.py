import time
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/")
async def get_health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "service": settings.PROJECT_NAME
    }
