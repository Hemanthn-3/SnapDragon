from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import init_database
from backend.errors import register_exception_handlers
from backend.health import router as health_router
from backend.routes_documents import router as documents_router
from backend.routes_knowledge import router as knowledge_router
from backend.routes_llm import llm_router
from backend.routes_agent import agent_router
from backend.routes_tools import router as tools_router
from backend.routes_verification import router as verification_router
from backend.routes_speech import router as speech_router
from backend.routes_vision import router as vision_router
from backend.routes_benchmarks import benchmarks_router
from backend.routes_network import network_router
from backend.routes_demo import router as demo_router
from backend.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initializes storage and logs startup."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION} [{settings.ENVIRONMENT}]")
    logger.info(f"Target Platform: {settings.TARGET_PLATFORM} | Offline Mode: {settings.OFFLINE_MODE}")

    # Initialize local SQLite database
    init_database()

    yield

    logger.info(f"Shutting down {settings.APP_NAME}...")


def create_app() -> FastAPI:
    """Factory creating and configuring the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_SUBTITLE,
        version=settings.VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration for offline local requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register error handlers
    register_exception_handlers(app)

    # Register API routers
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(knowledge_router)
    app.include_router(llm_router)
    app.include_router(agent_router)
    app.include_router(tools_router)
    app.include_router(verification_router)
    app.include_router(speech_router)
    app.include_router(vision_router)
    app.include_router(benchmarks_router)
    app.include_router(network_router)
    app.include_router(demo_router)

    # Mount static frontend directory
    frontend_dir = settings.BASE_DIR / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    else:
        logger.warning(f"Frontend directory not found at {frontend_dir}")

    return app


app = create_app()
