import logging
import sys
from pathlib import Path
from backend.config import settings


def setup_logger(name: str = "nexus") -> logging.Logger:
    """Configures structured logging to both console and a local log file."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on reloads
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (ensure directory exists)
    try:
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        console_handler.handle(
            logging.LogRecord(
                name=name,
                level=logging.WARNING,
                pathname=__file__,
                lineno=35,
                msg=f"Failed to initialize file logger: {e}",
                args=(),
                exc_info=None,
            )
        )

    return logger


logger = setup_logger("nexus")
get_logger = setup_logger
