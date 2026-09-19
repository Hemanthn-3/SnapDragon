from typing import Generator
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from backend.config import settings
from backend.logger import logger

# Ensure local data directory exists before establishing SQLite file
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

# SQLite engine with check_same_thread=False for FastAPI concurrency
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=settings.DEBUG,
)


# Configure SQLite pragmas on connection: WAL journal mode and foreign keys
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
    finally:
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency for providing request-scoped database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> bool:
    """Performs a lightweight ping against the SQLite database."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            return result == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def init_database() -> None:
    """Initializes tables in the SQLite database."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database initialized successfully at {settings.DATABASE_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}", exc_info=True)
        raise
