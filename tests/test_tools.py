"""
NEXUS Phase 6: Controlled Tool Execution Layer Tests
Tests all security constraints and operational requirements:
- Valid tool calls
- Invalid tool calls (unregistered tools, shell, python, deletion)
- Malicious parameters (path traversal, null bytes, command injection characters)
- Unauthorized path access (sandbox escape)
- Tool timeout enforcement
- Tool failure handling
- User approval gating for export
- Plan execution pipeline: Plan -> Task -> Tool -> Result -> Next task
"""

import time
import pytest
from pathlib import Path
from pydantic import BaseModel, Field

from backend.config import settings
from backend.database import SessionLocal
from backend.models_db import DocumentRecord, DocumentChunk
from backend.tools.base import BaseTool, ToolContext, ToolResult
from backend.tools.registry import ToolRegistry, tool_registry
from backend.tools.implementations import (
    ListDocumentsTool,
    ReadDocumentTool,
    SearchKnowledgeTool,
    AnalyzeImageTool,
    RunOCRTool,
    CreateReportTool,
    ExportReportTool,
    REPORTS_DIR,
    DOCUMENTS_DIR,
)
from backend.agent.schemas import (
    StructuredPlan,
    PlannedTask,
    TaskType,
    TaskStatus,
    AgentExecuteRequest,
)
from backend.agent.executor import PlanExecutor


# --- Fixtures for Tool Testing ---

@pytest.fixture
def sample_doc_in_db():
    """Creates a mock document record in SQLite for tool testing."""
    db = SessionLocal()
    doc_id = "test-doc-uuid-12345"
    doc_dir = DOCUMENTS_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    file_path = doc_dir / "sample_doc.txt"
    file_path.write_text("This is test document content for tool execution testing.", encoding="utf-8")

    # Clean existing
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
    db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).delete()
    db.commit()

    doc = DocumentRecord(
        id=doc_id,
        filename="sample_doc.txt",
        file_type="txt",
        file_size_bytes=len(file_path.read_bytes()),
        sha256_hash="dummyhash12345",
        mime_type="text/plain",
        local_path=str(file_path),
        page_count=1,
        ocr_status="NOT_REQUIRED",
    )
    db.add(doc)

    chunk = DocumentChunk(
        id="chunk-test-001",
        document_id=doc_id,
        chunk_index=0,
        filename="sample_doc.txt",
        page_number=1,
        source_location="sample_doc.txt: Page 1",
        text="This is test document content for tool execution testing.",
        char_count=57,
        word_count=8,
    )
    db.add(chunk)
    db.commit()
    db.refresh(doc)
    db.close()

    yield doc_id

    # Cleanup
    db = SessionLocal()
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
    db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).delete()
    db.commit()
    db.close()


@pytest.fixture
def sample_image_in_db():
    """Creates a mock PNG image record in SQLite for image tool testing."""
    from PIL import Image
    db = SessionLocal()
    doc_id = "test-img-uuid-67890"
    doc_dir = DOCUMENTS_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    img_path = doc_dir / "test_diagram.png"

    # Create real 100x100 RGB image
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    img.save(img_path)

    # Clean existing
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
    db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).delete()
    db.commit()

    doc = DocumentRecord(
        id=doc_id,
        filename="test_diagram.png",
        file_type="png",
        file_size_bytes=img_path.stat().st_size,
        sha256_hash="imagehash67890",
        mime_type="image/png",
        local_path=str(img_path),
        page_count=1,
        ocr_status="NOT_ATTEMPTED",
    )
    db.add(doc)
    db.commit()
    db.close()

    yield doc_id

    # Cleanup
    db = SessionLocal()
    db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).delete()
    db.commit()
    db.close()


# =====================================================================
# 1. Test Valid Tool Calls
# =====================================================================

def test_valid_list_documents(sample_doc_in_db):
    tool = ListDocumentsTool()
    result = tool.execute({"limit": 10})
    assert result.success is True
    assert result.output is not None
    assert result.output["total_count"] >= 1
    doc_ids = [d["id"] for d in result.output["documents"]]
    assert sample_doc_in_db in doc_ids


def test_valid_read_document(sample_doc_in_db):
    tool = ReadDocumentTool()
    result = tool.execute({"document_id": sample_doc_in_db})
    assert result.success is True
    assert result.output["document_id"] == sample_doc_in_db
    assert result.output["chunk_count"] >= 1
    assert "This is test document content" in result.output["content"]


def test_valid_search_knowledge():
    tool = SearchKnowledgeTool()
    result = tool.execute({"query": "testing query", "top_k": 3})
    assert result.success is True
    assert result.output["query"] == "testing query"
    assert "citations" in result.output


def test_valid_analyze_image(sample_image_in_db):
    tool = AnalyzeImageTool()
    result = tool.execute({"document_id": sample_image_in_db, "prompt": "Inspect layout"})
    assert result.success is True
    assert result.output["status"] == "VERIFIED"
    assert result.output["image_info"]["width"] == 100
    assert result.output["image_info"]["height"] == 100
    assert "test_diagram.png" in result.output["analysis"]


def test_valid_run_ocr(sample_image_in_db):
    tool = RunOCRTool()
    result = tool.execute({"document_id": sample_image_in_db})
    assert result.success is True
    assert result.output["ocr_status"] == "OCR STATUS: NOT YET IMPLEMENTED"
    assert "NOT YET IMPLEMENTED" in result.output["extracted_text"]


def test_valid_create_report():
    tool = CreateReportTool()
    params = {
        "title": "Quarterly Technical Audit",
        "summary": "Everything is functioning correctly.",
        "sections": [
            {"title": "Section 1", "content": "Findings details."},
            {"title": "Section 2", "content": "Recommendations details."},
        ],
    }
    result = tool.execute(params)
    assert result.success is True
    assert "Quarterly Technical Audit" in result.output["content"]
    assert "Findings details." in result.output["content"]
    assert result.output["char_count"] > 0


def test_valid_export_report():
    tool = ExportReportTool()
    content = "# Test Audit\n\nContent verified."
    filename = "test_valid_export.md"
    result = tool.execute(
        {"filename": filename, "content": content, "approved": True},
        context=ToolContext(is_approved=True),
    )
    assert result.success is True
    assert result.output["filename"] == filename
    assert result.output["status"] == "EXPORTED"

    # Verify file actually written to data/reports/
    exported_file = REPORTS_DIR / filename
    assert exported_file.exists()
    assert exported_file.read_text(encoding="utf-8") == content

    # Cleanup
    exported_file.unlink(missing_ok=True)


# =====================================================================
# 2. Test Invalid Tool Call & Security Whitelist
# =====================================================================

def test_invalid_tool_rejection():
    """Verifies that arbitrary tools, shell commands, or python execution are strictly rejected."""
    forbidden_tools = [
        "execute_shell",
        "run_bash",
        "eval_python",
        "delete_file",
        "download_url",
        "system_command",
        "format_disk",
    ]
    for tool_name in forbidden_tools:
        # Check registry directly
        assert tool_registry.has_tool(tool_name) is False
        with pytest.raises(PermissionError) as exc_info:
            tool_registry.get_tool(tool_name)
        assert "Unauthorized tool execution" in str(exc_info.value)

        # Execute attempt should fail safely
        res = tool_registry.execute_tool(tool_name, {})
        assert res.success is False
        assert "Unauthorized" in res.error


def test_invalid_parameters_schema_rejection():
    """Verifies that missing required fields or incorrect types are rejected by strict schemas."""
    tool = ReadDocumentTool()
    # Missing required 'document_id'
    res = tool.execute({})
    assert res.success is False
    assert "Invalid parameters" in res.error

    # Incorrect type for top_k (must be int <= 20)
    search_tool = SearchKnowledgeTool()
    res2 = search_tool.execute({"query": "hello", "top_k": 999})
    assert res2.success is False
    assert "Invalid parameters" in res2.error


# =====================================================================
# 3. Test Malicious Parameters & Path Traversal
# =====================================================================

def test_path_traversal_in_read_document():
    tool = ReadDocumentTool()
    traversal_attacks = [
        "../../Windows/System32",
        "../data/nexus.db",
        "..\\..\\Windows\\System32\\cmd.exe",
        "doc/../../etc/passwd",
    ]
    for attack in traversal_attacks:
        res = tool.execute({"document_id": attack})
        assert res.success is False
        assert "Path traversal sequence detected" in res.error or "Invalid parameters" in res.error


def test_null_byte_rejection():
    tool = ExportReportTool()
    res = tool.execute(
        {"filename": "test\x00_bypass.md", "content": "test", "approved": True},
        context=ToolContext(is_approved=True),
    )
    assert res.success is False
    assert "Null byte detected" in res.error or "Invalid parameters" in res.error


def test_command_injection_characters_rejection():
    tool = ExportReportTool()
    injection_names = [
        "report.md; rm -rf /",
        "test | whoami.md",
        "report`id`.md",
        "file$test.md",
        "test&echo.md",
    ]
    for name in injection_names:
        res = tool.execute(
            {"filename": name, "content": "test", "approved": True},
            context=ToolContext(is_approved=True),
        )
        assert res.success is False
        assert "Invalid character" in res.error or "Invalid parameters" in res.error


def test_disallowed_extension_rejection():
    tool = ExportReportTool()
    dangerous_extensions = [
        "exploit.exe",
        "script.py",
        "payload.bat",
        "key.pem",
        "document.docx",
    ]
    for filename in dangerous_extensions:
        res = tool.execute(
            {"filename": filename, "content": "test", "approved": True},
            context=ToolContext(is_approved=True),
        )
        assert res.success is False
        assert "Disallowed file extension" in res.error or "Invalid parameters" in res.error


# =====================================================================
# 4. Test Unauthorized Path Access
# =====================================================================

def test_unauthorized_path_access():
    tool = ExportReportTool()
    unauthorized_destinations = [
        "C:\\Windows\\System32\\drivers\\etc\\hosts.md",
        "/etc/shadow.md",
        "\\\\remote-share\\exploit.md",
    ]
    for path in unauthorized_destinations:
        res = tool.execute(
            {"filename": path, "content": "malicious content", "approved": True},
            context=ToolContext(is_approved=True),
        )
        assert res.success is False
        assert (
            "Path traversal" in res.error
            or "Invalid character" in res.error
            or "Invalid parameters" in res.error
        )


# =====================================================================
# 5. Test Tool Timeout Enforcement
# =====================================================================

class SlowDummyInput(BaseModel):
    sleep_duration: float = Field(default=0.5)


class SlowDummyOutput(BaseModel):
    message: str


class SlowDummyTool(BaseTool):
    name = "slow_dummy"
    description = "Simulates a slow tool to test timeout enforcement"
    input_schema = SlowDummyInput
    output_schema = SlowDummyOutput
    requires_approval = False
    timeout_seconds = 0.1  # Fast timeout

    def _run(self, params: SlowDummyInput, context: ToolContext) -> SlowDummyOutput:
        time.sleep(params.sleep_duration)
        return SlowDummyOutput(message="Finished")


def test_tool_timeout_enforcement():
    slow_tool = SlowDummyTool()
    res = slow_tool.execute({"sleep_duration": 0.5})
    assert res.success is False
    assert "timed out" in res.error
    assert "0.1" in res.error


# =====================================================================
# 6. Test Tool Failure Handling
# =====================================================================

def test_tool_failure_on_missing_document():
    tool = ReadDocumentTool()
    res = tool.execute({"document_id": "non-existent-doc-id-99999"})
    assert res.success is False
    assert "not found" in res.error


def test_image_tool_failure_on_non_image(sample_doc_in_db):
    tool = AnalyzeImageTool()
    # sample_doc_in_db is a text file (.txt)
    res = tool.execute({"document_id": sample_doc_in_db})
    assert res.success is False
    assert "not an image" in res.error


# =====================================================================
# 7. Test Approval Support for File Creation/Export
# =====================================================================

def test_export_report_without_approval_blocked():
    tool = ExportReportTool()
    # Both context.is_approved=False and approved=False
    res = tool.execute(
        {"filename": "unapproved.md", "content": "# Data", "approved": False},
        context=ToolContext(is_approved=False),
    )
    assert res.success is False
    assert "requires explicit user approval" in res.error

    # Ensure no file was created
    assert not (REPORTS_DIR / "unapproved.md").exists()


def test_export_report_with_approval_succeeds():
    tool = ExportReportTool()
    filename = "approved_audit.md"
    content = "# Approved Content\n\nVerified by user."

    res = tool.execute(
        {"filename": filename, "content": content, "approved": True},
        context=ToolContext(is_approved=True),
    )
    assert res.success is True
    assert res.output["status"] == "EXPORTED"

    file_path = REPORTS_DIR / filename
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == content

    # Cleanup
    file_path.unlink(missing_ok=True)


# =====================================================================
# 8. Test Plan Execution Pipeline (Plan -> Task -> Tool -> Result -> Next)
# =====================================================================

def test_plan_execution_sequential_pipeline(sample_doc_in_db):
    """
    Verifies full execution pipeline:
    Plan -> Task 1 (list_documents) -> Task 2 (read_document) -> Task 3 (create_report) -> Result
    """
    executor = PlanExecutor()

    task1 = PlannedTask(
        id="task_1",
        description="Find relevant documents",
        type=TaskType.DOCUMENT_RETRIEVAL.value,
        input={"tool": "list_documents", "limit": 5},
        status=TaskStatus.PENDING,
    )
    task2 = PlannedTask(
        id="task_2",
        description="Extract text content",
        type=TaskType.CONTENT_EXTRACTION.value,
        dependencies=["task_1"],
        input={"tool": "read_document", "document_id": sample_doc_in_db},
        status=TaskStatus.PENDING,
    )
    task3 = PlannedTask(
        id="task_3",
        description="Generate summary report",
        type=TaskType.REPORT_GENERATION.value,
        dependencies=["task_2"],
        input={
            "tool": "create_report",
            "title": "Document Inspection Report",
            "summary": "Inspection concluded without errors.",
        },
        status=TaskStatus.PENDING,
    )

    plan = StructuredPlan(
        goal="Analyze documents and create report",
        tasks=[task1, task2, task3],
        estimated_steps=3,
        created_at="2026-09-19T00:00:00Z",
    )

    req = AgentExecuteRequest(plan=plan)
    res = executor.execute_plan(req)

    assert res.status == "completed"
    assert res.executed_tasks == 3
    assert res.failed_tasks == 0
    assert res.plan.tasks[0].status == TaskStatus.COMPLETED
    assert res.plan.tasks[1].status == TaskStatus.COMPLETED
    assert res.plan.tasks[2].status == TaskStatus.COMPLETED

    # Check outputs were stored
    assert res.plan.tasks[0].output["total_count"] >= 1
    assert "This is test document content" in res.plan.tasks[1].output["content"]
    assert "Document Inspection Report" in res.plan.tasks[2].output["content"]


def test_plan_execution_pauses_for_approval():
    """
    Verifies that the plan pauses when encountering a task requiring approval,
    and resumes to completion once approval is granted.
    """
    executor = PlanExecutor()

    task1 = PlannedTask(
        id="task_1",
        description="Draft report in memory",
        type=TaskType.REPORT_GENERATION.value,
        input={
            "tool": "create_report",
            "title": "Audit Summary",
            "summary": "Passed all integrity checks.",
        },
        status=TaskStatus.PENDING,
    )
    task2 = PlannedTask(
        id="task_2",
        description="Export report to disk",
        type=TaskType.REPORT_GENERATION.value,
        dependencies=["task_1"],
        input={
            "tool": "export_report",
            "filename": "pipeline_audit.md",
        },
        status=TaskStatus.PENDING,
    )

    plan = StructuredPlan(
        goal="Draft and export audit report",
        tasks=[task1, task2],
        estimated_steps=2,
        created_at="2026-09-19T00:00:00Z",
    )

    # 1. Execute without approval -> should pause at task_2
    req1 = AgentExecuteRequest(plan=plan, approved_task_ids=[])
    res1 = executor.execute_plan(req1)

    assert res1.status == "paused_for_approval"
    assert res1.pending_approval_task_id == "task_2"
    assert res1.plan.tasks[0].status == TaskStatus.COMPLETED
    assert res1.plan.tasks[1].status == TaskStatus.REQUIRES_APPROVAL

    # 2. Re-submit with approval for task_2 -> should complete
    req2 = AgentExecuteRequest(plan=res1.plan, approved_task_ids=["task_2"])
    res2 = executor.execute_plan(req2)

    assert res2.status == "completed"
    assert res2.plan.tasks[1].status == TaskStatus.COMPLETED
    assert res2.plan.tasks[1].output["filename"] == "pipeline_audit.md"

    # Cleanup
    (REPORTS_DIR / "pipeline_audit.md").unlink(missing_ok=True)


# =====================================================================
# 9. Test Tools REST API Integration
# =====================================================================

def test_api_list_tools(client):
    response = client.get("/tools")
    assert response.status_code == 200
    data = response.json()
    assert data["total_tools"] == 7
    tool_names = [t["name"] for t in data["tools"]]
    expected = [
        "analyze_image",
        "create_report",
        "export_report",
        "list_documents",
        "read_document",
        "run_ocr",
        "search_knowledge",
    ]
    assert sorted(tool_names) == sorted(expected)


def test_api_execute_tool_success(client):
    response = client.post(
        "/tools/execute",
        json={
            "tool": "create_report",
            "parameters": {
                "title": "API Test Report",
                "summary": "Executed through REST endpoint.",
                "sections": [],
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "API Test Report" in data["output"]["content"]


def test_api_execute_unauthorized_tool_rejected(client):
    response = client.post(
        "/tools/execute",
        json={
            "tool": "system_terminal_shell",
            "parameters": {"command": "dir"},
        },
    )
    assert response.status_code == 403
    assert "not permitted" in response.json()["detail"]


def test_api_execute_unapproved_export_rejected(client):
    response = client.post(
        "/tools/execute",
        json={
            "tool": "export_report",
            "parameters": {
                "filename": "api_export.md",
                "content": "Secret content",
                "approved": False,
            },
            "approved": False,
        },
    )
    assert response.status_code == 403
    assert "approval" in response.json()["detail"].lower()
