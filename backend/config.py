from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class NexusSettings(BaseSettings):
    """Core configuration settings for the NEXUS offline-first application."""

    # Application Metadata
    APP_NAME: str = "NEXUS"
    APP_SUBTITLE: str = "Offline Multimodal Work Agent"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # Offline & Security
    OFFLINE_MODE: bool = True
    ALLOW_TELEMETRY: bool = False

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'nexus.db'}"
    LOG_FILE: Path = DATA_DIR / "nexus.log"

    # Hardware & Model Configuration Defaults
    TARGET_PLATFORM: str = "Snapdragon X Series"
    INFERENCE_BACKEND: str = "QNNExecutionProvider"  # Target: Hexagon NPU
    FALLBACK_BACKEND: str = "CPUExecutionProvider"

    model_config = SettingsConfigDict(
        env_prefix="NEXUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global settings instance
settings = NexusSettings()
