from sqlalchemy import text
from backend.database import check_database_connection, engine, SessionLocal
from backend.models_db import SystemMetadata


def test_database_connection_check():
    """Verifies that the SQLite database responds to lightweight connectivity checks."""
    assert check_database_connection() is True


def test_database_pragma_wal():
    """Verifies that SQLite is operating with WAL mode enabled."""
    with engine.connect() as conn:
        journal_mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        assert journal_mode.lower() == "wal"


def test_system_metadata_table():
    """Verifies that tables can be queried and records inserted and retrieved."""
    db = SessionLocal()
    try:
        # Insert test key
        meta = SystemMetadata(key="test_init_key", value="active")
        db.add(meta)
        db.commit()

        # Retrieve and verify
        retrieved = db.query(SystemMetadata).filter_by(key="test_init_key").first()
        assert retrieved is not None
        assert retrieved.value == "active"

        # Cleanup
        db.delete(retrieved)
        db.commit()
    finally:
        db.close()
