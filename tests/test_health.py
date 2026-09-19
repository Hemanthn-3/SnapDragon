from fastapi import status


def test_backend_startup_and_health_endpoint(client):
    """Verifies that the backend starts and GET /health returns the required status structure."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["status"] == "healthy"
    assert data["backend"] == "connected"
    assert data["database"] == "connected"
    assert data["ai_models"] == "not_loaded"
    assert "models" in data
    assert data["models"]["speech"] == "not_loaded"
    assert data["models"]["vision"] == "not_loaded"
    assert data["models"]["ocr"] == "not_loaded"
    assert data["models"]["embedding"] == "not_loaded"
    assert data["models"]["language"] == "not_loaded"
    assert data["app_name"] == "NEXUS"
    assert data["offline_mode"] is True
    assert "timestamp" in data


def test_static_index_serving(client):
    """Verifies that the frontend index.html is properly mounted and served at root /."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert "text/html" in response.headers.get("content-type", "")
    assert "NEXUS" in response.text
    assert "Offline Multimodal Work Agent" in response.text
    assert "System Status" in response.text
