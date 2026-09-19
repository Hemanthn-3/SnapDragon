"""
NEXUS Phase 15: Competition Demo Automated Test Suite
Tests:
- Synthetic demonstration dataset existence and integrity in demo_data/
- Deterministic ingestion & vector indexing via DemoOrchestrator
- Complete 10-stage multimodal workflow execution
- Air-gap hardware telemetry reporting (NETWORK: OFFLINE, AI: LOCAL, NPU)
- Report file generation and disk persistence
- API route verification: POST /demo/load, POST /demo/run, GET /demo/status, GET /demo/report
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import settings
from backend.database import SessionLocal
from backend.demo.service import demo_orchestrator, DEMO_GOAL, DEMO_DATA_DIR

client = TestClient(app)


def test_demo_dataset_files_exist():
    """Verify that all 5 synthetic assets exist in demo_data/ and are non-empty."""
    expected_files = [
        "reference_manual.txt",
        "inspection_report.pdf",
        "scanned_maintenance_log.png",
        "bearing_assembly_inspection.png",
        "sensor_telemetry.json",
    ]
    assert DEMO_DATA_DIR.exists()
    for filename in expected_files:
        fpath = DEMO_DATA_DIR / filename
        assert fpath.exists(), f"Missing synthetic demo asset: {filename}"
        assert fpath.stat().st_size > 0, f"Empty asset: {filename}"


def test_load_demo_dataset_service():
    """Verify deterministic dataset loading into SQLite and knowledge store."""
    with SessionLocal() as db:
        res = demo_orchestrator.load_demo_dataset(db=db)
    
    assert res["status"] == "SUCCESS"
    assert res["demo_mode"] == "ACTIVE"
    assert res["total_documents"] == 5
    doc_names = [d["filename"] for d in res["loaded_documents"]]
    assert "reference_manual.txt" in doc_names
    assert "inspection_report.pdf" in doc_names
    assert "scanned_maintenance_log.png" in doc_names
    assert "bearing_assembly_inspection.png" in doc_names
    assert "sensor_telemetry.json" in doc_names


def test_run_competition_demo_workflow():
    """Verify end-to-end execution of all 10 stages in the competition demo."""
    with SessionLocal() as db:
        res = demo_orchestrator.run_competition_demo(db=db)

    assert res["status"] == "SUCCESS"
    assert res["goal"] == DEMO_GOAL
    assert res["total_stages"] == 10
    stages = res["stages"]
    assert len(stages) == 10

    # 1. User Goal
    assert stages[0]["name"] == "User Goal"
    assert stages[0]["details"]["goal"] == DEMO_GOAL

    # 2. Agent Plan
    assert stages[1]["name"] == "Agent Plan"
    assert stages[1]["details"]["total_tasks"] >= 5
    assert stages[1]["details"]["dag_valid"] is True

    # 3. Document Discovery
    assert stages[2]["name"] == "Document Discovery"
    assert stages[2]["details"]["package_count"] == 5

    # 4. OCR
    assert stages[3]["name"] == "OCR"
    assert "95 Nm" in stages[3]["details"]["extracted_key_text"]
    assert stages[3]["details"]["detected_value"] == "95 Nm"

    # 5. Retrieval
    assert stages[4]["name"] == "Retrieval"
    assert stages[4]["details"]["queries_executed"] >= 3

    # 6. Vision
    assert stages[5]["name"] == "Vision"
    assert "bearing_assembly_inspection.png" in stages[5]["details"]["target_image"]
    assert "observed" in stages[5]["details"]
    assert "inferred" in stages[5]["details"]

    # 7. Local Reasoning
    assert stages[6]["name"] == "Local Reasoning"
    assert stages[6]["details"]["total_issues_identified"] >= 4
    discrepancies = stages[6]["details"]["discrepancies"]
    params = [d["parameter"] for d in discrepancies]
    assert any("Hydraulic" in p for p in params)
    assert any("Bolt Torque" in p for p in params)
    assert any("Vibration" in p for p in params)

    # 8. Evidence
    assert stages[7]["name"] == "Evidence"
    assert stages[7]["details"]["evidence_chain_count"] >= 4

    # 9. Verification
    assert stages[8]["name"] == "Verification"
    assert stages[8]["details"]["claims_verified"] >= 4

    # 10. Report Generation
    assert stages[9]["name"] == "Report Generation"
    exported_path = Path(stages[9]["details"]["exported_path"])
    assert exported_path.exists()
    report_text = exported_path.read_text(encoding="utf-8")
    assert "Turbine Generator TR-900" in report_text
    assert "DISC-01" in report_text
    assert "245.0 Bar" in report_text
    assert "95.0 Nm" in report_text

    # Hardware Status
    hw = res["hardware_status"]
    assert hw["network"] == "OFFLINE"
    assert hw["ai"] == "LOCAL"
    assert hw["cloud_ai"] == "DISABLED"
    assert "NPU" in hw["npu"] or "CPU" in hw["npu"]


def test_api_demo_endpoints():
    """Verify demo HTTP endpoints via FastAPI TestClient."""
    # GET /demo/status
    s_res = client.get("/demo/status")
    assert s_res.status_code == 200
    s_data = s_res.json()
    assert "demo_mode" in s_data
    assert "hardware_status" in s_data
    assert s_data["hardware_status"]["network"] == "OFFLINE"

    # POST /demo/load
    l_res = client.post("/demo/load")
    assert l_res.status_code == 200
    l_data = l_res.json()
    assert l_data["status"] == "SUCCESS"
    assert l_data["total_documents"] == 5

    # POST /demo/run
    r_res = client.post("/demo/run")
    assert r_res.status_code == 200
    r_data = r_res.json()
    assert r_data["status"] == "SUCCESS"
    assert len(r_data["stages"]) == 10

    # GET /demo/report
    rep_res = client.get("/demo/report")
    assert rep_res.status_code == 200
    rep_data = rep_res.json()
    assert "title" in rep_data
    assert "content" in rep_data
    assert "Turbine Generator TR-900" in rep_data["title"]
