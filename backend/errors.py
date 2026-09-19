from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from backend.logger import logger


class NexusException(Exception):
    """Base exception for all NEXUS application errors."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ModelNotLoadedError(NexusException):
    """Raised when an operation is requested on an uninitialized AI model."""

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' is not loaded. Execution prohibited in Phase 1.",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        self.model_name = model_name


class DatabaseConnectionError(NexusException):
    """Raised when the local SQLite database fails connectivity checks."""

    def __init__(self, detail: str):
        super().__init__(
            message=f"Database connection error: {detail}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class ConfigurationError(NexusException):
    """Raised when invalid or conflicting configurations are detected."""

    def __init__(self, detail: str):
        super().__init__(
            message=f"Configuration error: {detail}",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


def register_exception_handlers(app: FastAPI) -> None:
    """Registers global exception handlers for consistent API error responses."""

    @app.exception_handler(NexusException)
    async def nexus_exception_handler(request: Request, exc: NexusException):
        logger.error(f"NexusException caught: {exc.message} (Path: {request.url.path})")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "message": exc.message,
                "status_code": exc.status_code,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred in the NEXUS backend.",
                "status_code": 500,
            },
        )
