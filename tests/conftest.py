import os
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.database import init_database


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Sets up test environment variables and database."""
    os.environ["NEXUS_ENVIRONMENT"] = "testing"
    os.environ["NEXUS_DEBUG"] = "true"
    # Ensure database is initialized for tests
    init_database()
    yield


@pytest.fixture
def client():
    """Provides a TestClient instance for API integration tests."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
