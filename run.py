import uvicorn
from backend.config import settings

if __name__ == "__main__":
    print(f"Starting {settings.APP_NAME} on http://{settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
