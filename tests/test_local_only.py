"""
NEXUS Phase 11 Tests: Strict Local-Only Mode & Offline Workflow Integration
Validates:
- Socket layer air-gap enforcement (loopback permitted, external blocked)
- Empirical network request counting (zero fabricated metrics)
- Network adapter status detection without cloud pings
- Complete 7-step offline workflow integration test:
  1. Start application
  2. Disable network
  3. Import documents
  4. Run retrieval
  5. Run local LLM
  6. Run agent planner & tools
  7. Generate and export report
"""

import io
import os
import socket
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import create_app
from backend.config import settings
from backend.database import SessionLocal
from backend.models_db import DocumentRecord
from backend.ingestion.service import ingestion_service
from backend.knowledge.service import knowledge_service
from backend.routes_llm import local_llama_model
from backend.agent.planner import NEXUSPlanner
from backend.tools.registry import tool_registry
from backend.tools.base import ToolContext
from backend.network.guard import (
    LocalNetworkGuard,
    StrictLocalOnlyViolationError,
    network_guard,
)


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


# =====================================================================
# 1. Socket Layer Air-Gap Guard Tests
# =====================================================================

def test_network_guard_blocks_external_and_allows_loopback():
    """
    Verifies that LocalNetworkGuard permits loopback connections
    while intercepting and blocking external host socket connections.
    """
    guard = LocalNetworkGuard()
    guard.enable()

    try:
        # Loopback check
        assert guard.is_loopback(("127.0.0.1", 8000)) is True
        assert guard.is_loopback(("localhost", 80)) is True
        assert guard.is_loopback(("::1", 443)) is True

        # External check
        assert guard.is_loopback(("8.8.8.8", 53)) is False
        assert guard.is_loopback(("api.openai.com", 443)) is False

        # Attempt external socket connection: MUST raise StrictLocalOnlyViolationError
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(StrictLocalOnlyViolationError) as exc_info:
            s.connect(("8.8.8.8", 80))
        assert "Strict Local-Only Violation" in str(exc_info.value)
        s.close()

        # Verify blocked count incremented
        assert guard.external_blocked_count >= 1
    finally:
        guard.disable()
        # Re-enable global guard
        network_guard.enable()


def test_detect_network_status_structure():
    """Verifies that detect_network_status returns verified fields and adapter metrics."""
    status = network_guard.detect_network_status()
    assert status["local_only_mode"] == "ACTIVE"
    assert "internet_status" in status
    assert status["cloud_ai"] == "DISABLED"
    assert status["ai_processing"] == "LOCAL"
    assert "loopback_served" in status["network_requests"]
    assert "external_blocked" in status["network_requests"]
    assert isinstance(status["interfaces"], list)
    assert "psutil.net_if_stats" in status["detection_method"]


# =====================================================================
# 2. End-to-End Offline Workflow Integration Test
# =====================================================================

def test_full_offline_core_workflow(client):
    """
    Validates that the complete NEXUS core workflow continues uninterrupted 100% offline:
    Step 1: Start application and verify air-gap guard is active.
    Step 2: Disable network (simulated offline mode).
    Step 3: Import local documents.
    Step 4: Run local knowledge retrieval.
    Step 5: Run local language model (LLM).
    Step 6: Run autonomous agent planner and execute tools.
    Step 7: Generate and export report to disk.
    """
    # -------------------------------------------------------------
    # Step 1: Start application & verify air-gap guard
    # -------------------------------------------------------------
    assert network_guard.is_active is True
    initial_blocked = network_guard.external_blocked_count

    # -------------------------------------------------------------
    # Step 2: Disable network (simulated offline)
    # -------------------------------------------------------------
    network_guard.set_simulated_offline(True)
    net_status = network_guard.detect_network_status()
    assert net_status["internet_status"] == "OFFLINE"
    assert net_status["cloud_ai"] == "DISABLED"
    assert net_status["ai_processing"] == "LOCAL"

    # -------------------------------------------------------------
    # Step 3: Import documents locally
    # -------------------------------------------------------------
    doc_content = (
        "OFFLINE INSPECTION REPORT 2026\n"
        "Structural Safety Value: 42.5 MPa.\n"
        "Reference Operational Threshold: 35.0 MPa.\n"
        "Finding: Measured structural load exceeds reference threshold by 21.4%.\n"
        "Recommendation: Replace coupling gasket before operating turbine."
    )
    doc_bytes = doc_content.encode("utf-8")

    db = SessionLocal()
    doc_record = None
    exported_file = None

    try:
        # Ingest document offline
        doc_record = ingestion_service.ingest_document(
            filename="offline_inspection.txt",
            content=doc_bytes,
            db=db,
        )
        assert doc_record.id is not None
        assert doc_record.filename == "offline_inspection.txt"

        # Index into knowledge service
        index_res = knowledge_service.index_document(doc_record.id, db=db)
        assert index_res["chunks_indexed"] > 0

        # -------------------------------------------------------------
        # Step 4: Run local retrieval (vector search)
        # -------------------------------------------------------------
        search_results = knowledge_service.search(
            query="structural safety value threshold",
            top_k=3,
            db=db,
        )
        assert len(search_results) > 0
        top_hit = search_results[0]
        assert "42.5 MPa" in top_hit["chunk_text"]

        # -------------------------------------------------------------
        # Step 5: Run local LLM (offline grounded QA)
        # -------------------------------------------------------------
        if not local_llama_model.is_loaded:
            local_llama_model.load()

        prompt = f"Given context: '{top_hit['chunk_text']}', what is the safety value?"
        answer = local_llama_model.generate(prompt=prompt, max_tokens=16)
        assert len(answer) > 0

        # -------------------------------------------------------------
        # Step 6: Run agent planner & tools
        # -------------------------------------------------------------
        planner = NEXUSPlanner()
        plan_res = planner.create_plan(goal="Analyze inspection documents and create an action report.")

        assert plan_res.status == "success"
        assert len(plan_res.plan.tasks) >= 3

        # Execute read_document tool offline
        read_tool = tool_registry.get_tool("read_document")
        ctx = ToolContext(task_id="task_1", is_approved=True)
        read_output = read_tool.execute({"document_id": doc_record.id}, context=ctx)
        assert read_output.success is True
        assert "42.5 MPa" in read_output.output["content"]

        # -------------------------------------------------------------
        # Step 7: Generate and export report to disk
        # -------------------------------------------------------------
        create_tool = tool_registry.get_tool("create_report")
        report_output = create_tool.execute(
            {
                "title": "Offline Turbine Safety Audit",
                "summary": "Structural safety value exceeds reference threshold.",
                "sections": [
                    {"title": "Findings", "content": "Measured load is 42.5 MPa vs 35.0 MPa threshold."},
                    {"title": "Action", "content": "Replace coupling gasket immediately."},
                ],
            },
            context=ctx,
        )
        assert report_output.success is True
        report_content = report_output.output["content"]
        assert "Confidential & Local-Only" in report_content

        # Export report safely to disk with approval
        export_tool = tool_registry.get_tool("export_report")
        export_filename = f"offline_audit_{uuid.uuid4().hex[:6]}.md"
        export_output = export_tool.execute(
            {
                "filename": export_filename,
                "content": report_content,
                "approved": True,
            },
            context=ctx,
        )
        assert export_output.success is True
        exported_file = Path(export_output.output["exported_path"])
        assert exported_file.exists()
        assert exported_file.read_text(encoding="utf-8") == report_content

        # -------------------------------------------------------------
        # Step 8: Verify zero outbound network leakage
        # -------------------------------------------------------------
        final_status = network_guard.detect_network_status()
        assert final_status["internet_status"] == "OFFLINE"
        assert final_status["cloud_ai"] == "DISABLED"
        assert final_status["ai_processing"] == "LOCAL"
        # Ensure no unexpected external outbound connections occurred
        assert network_guard.external_blocked_count == initial_blocked

    finally:
        # Cleanup test document and report
        if doc_record:
            try:
                ingestion_service.delete_document(doc_record.id, db=db)
            except Exception:
                pass
        db.close()
        if exported_file and exported_file.exists():
            exported_file.unlink()


# =====================================================================
# 3. REST API Endpoint Tests
# =====================================================================

def test_api_network_status_endpoint(client):
    """Verifies GET /network/status returns expected air-gap telemetry."""
    response = client.get("/network/status")
    assert response.status_code == 200
    data = response.json()
    assert data["local_only_mode"] == "ACTIVE"
    assert data["cloud_ai"] == "DISABLED"
    assert data["ai_processing"] == "LOCAL"
    assert "network_requests" in data
    assert "loopback_served" in data["network_requests"]


def test_api_simulate_offline_toggle(client):
    """Verifies POST /network/simulate-offline toggles simulated network state."""
    response = client.post("/network/simulate-offline", json={"offline": True})
    assert response.status_code == 200
    data = response.json()
    assert data["simulated_offline"] is True
    assert data["network_status"]["internet_status"] == "OFFLINE"
