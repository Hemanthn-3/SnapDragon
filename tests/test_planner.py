"""
NEXUS Phase 5: Planner Tests
Tests:
- Structured task schema representation (8 required fields)
- Valid plan generation and DAG topological validity
- Malformed LLM output rejection (non-JSON, truncated, corrupt)
- Invalid schema structure rejection (missing tasks, bad types)
- Cycle detection (Kahn's DAG algorithm)
- Missing/self-dependency reference rejection
- Propose-only status enforcement (tasks remain PENDING, no execution)
- API endpoint integration: POST /agent/plan
"""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.agent.schemas import (
    PlannedTask,
    StructuredPlan,
    TaskStatus,
    TaskType,
    AgentPlanRequest,
)
from backend.agent.planner import NEXUSPlanner, nexus_planner
from backend.interfaces.base import ModelStatus

client = TestClient(app)


def test_structured_task_schema_fields():
    """Verify all 8 mandatory fields on PlannedTask."""
    task = PlannedTask(
        id="task_1",
        description="Find relevant inspection documents",
        type=TaskType.DOCUMENT_RETRIEVAL.value,
        dependencies=[],
        status=TaskStatus.PENDING,
        input={"query": "inspection report"},
        output=None,
        error=None,
    )
    assert task.id == "task_1"
    assert task.description == "Find relevant inspection documents"
    assert task.type == "DOCUMENT_RETRIEVAL"
    assert task.dependencies == []
    assert task.status == TaskStatus.PENDING
    assert task.input == {"query": "inspection report"}
    assert task.output is None
    assert task.error is None

    # Test JSON serialization
    serialized = task.model_dump()
    for field in ["id", "description", "type", "dependencies", "status", "input", "output", "error"]:
        assert field in serialized


def test_plan_creation_and_topological_validity():
    """Verify that a valid LLM response is parsed into a valid DAG plan."""
    mock_llm_json = {
        "goal": "Analyze these inspection documents and create an action report.",
        "tasks": [
            {
                "id": "task_1",
                "description": "Find relevant inspection documents in local workspace",
                "type": "DOCUMENT_RETRIEVAL",
                "dependencies": [],
                "status": "PENDING",
                "input": {"query": "inspection report"},
                "output": None,
                "error": None
            },
            {
                "id": "task_2",
                "description": "Extract text and findings from identified documents",
                "type": "CONTENT_EXTRACTION",
                "dependencies": ["task_1"],
                "status": "PENDING",
                "input": {"source": "task_1"},
                "output": None,
                "error": None
            },
            {
                "id": "task_3",
                "description": "Generate action report summarizing findings",
                "type": "REPORT_GENERATION",
                "dependencies": ["task_2"],
                "status": "PENDING",
                "input": {"format": "markdown"},
                "output": None,
                "error": None
            }
        ]
    }

    planner = NEXUSPlanner()
    with patch.object(planner.model, "health_check", return_value=ModelStatus.READY):
        with patch.object(planner.model, "generate", return_value=json.dumps(mock_llm_json)):
            response = planner.create_plan(
                goal="Analyze these inspection documents and create an action report."
            )
            assert response.status == "success"
            assert response.plan is not None
            assert response.plan.estimated_steps == 3
            assert len(response.plan.tasks) == 3
            assert response.validation.valid is True
            assert response.validation.has_cycles is False
            assert len(response.validation.errors) == 0

            # Verify DAG ordering
            task_ids = [t.id for t in response.plan.tasks]
            assert task_ids == ["task_1", "task_2", "task_3"]


def test_malformed_llm_output_rejection():
    """Verify that non-JSON, truncated, or conversational filler is safely rejected."""
    planner = NEXUSPlanner()

    # Case 1: Plain conversational text
    with patch.object(planner.model, "health_check", return_value=ModelStatus.READY):
        with patch.object(planner.model, "generate", return_value="Sure, here is your plan: First, I will look at the files."):
            response = planner.create_plan("Analyze docs")
            assert response.status == "rejected"
            assert response.plan is None
            assert response.validation.valid is False
            assert "malformed" in response.message.lower()

    # Case 2: Truncated JSON
    with patch.object(planner.model, "health_check", return_value=ModelStatus.READY):
        with patch.object(planner.model, "generate", return_value='{"goal": "Analyze", "tasks": [{"id": "task_1"'):
            response = planner.create_plan("Analyze docs")
            assert response.status == "rejected"
            assert response.plan is None
            assert response.validation.valid is False


def test_invalid_schema_structure_rejection():
    """Verify rejection when LLM outputs JSON missing required fields or empty tasks."""
    planner = NEXUSPlanner()

    # Empty tasks list
    with patch.object(planner.model, "health_check", return_value=ModelStatus.READY):
        with patch.object(planner.model, "generate", return_value='{"goal": "Test", "tasks": []}'):
            response = planner.create_plan("Test goal")
            assert response.status == "rejected"
            assert response.plan is None
            assert response.validation.valid is False
            assert any("empty" in e.lower() for e in response.validation.errors)

    # Missing task description
    bad_json = {
        "goal": "Test",
        "tasks": [{"id": "task_1", "type": "GENERAL_REASONING"}]  # Missing description
    }
    with patch.object(planner.model, "health_check", return_value=ModelStatus.READY):
        with patch.object(planner.model, "generate", return_value=json.dumps(bad_json)):
            response = planner.create_plan("Test goal")
            assert response.status == "rejected"
            assert response.plan is None
            assert response.validation.valid is False


def test_cycle_detection_in_task_dependencies():
    """Verify that circular dependency cycles are detected and rejected via Kahn's algorithm."""
    planner = NEXUSPlanner()
    cyclic_json = {
        "goal": "Test cycle",
        "tasks": [
            {
                "id": "task_1",
                "description": "Step 1",
                "type": "DOCUMENT_RETRIEVAL",
                "dependencies": ["task_2"],  # Cycles with task_2
                "status": "PENDING"
            },
            {
                "id": "task_2",
                "description": "Step 2",
                "type": "CONTENT_EXTRACTION",
                "dependencies": ["task_1"],  # Cycles with task_1
                "status": "PENDING"
            }
        ]
    }
    with patch.object(planner.model, "health_check", return_value=ModelStatus.READY):
        with patch.object(planner.model, "generate", return_value=json.dumps(cyclic_json)):
            response = planner.create_plan("Test cycle")
            assert response.status == "rejected"
            assert response.plan is None
            assert response.validation.has_cycles is True
            assert any("circular" in e.lower() or "cycle" in e.lower() for e in response.validation.errors)


def test_missing_and_self_dependency_rejection():
    """Verify rejection when tasks reference non-existent tasks or self-depend."""
    planner = NEXUSPlanner()

    # Self-dependency
    self_dep_json = {
        "goal": "Test self",
        "tasks": [
            {"id": "task_1", "description": "Step 1", "dependencies": ["task_1"]}
        ]
    }
    with patch.object(planner.model, "health_check", return_value=ModelStatus.READY):
        with patch.object(planner.model, "generate", return_value=json.dumps(self_dep_json)):
            response = planner.create_plan("Test self")
            assert response.status == "rejected"
            assert any("itself" in e.lower() for e in response.validation.errors)

    # Missing reference
    missing_ref_json = {
        "goal": "Test missing",
        "tasks": [
            {"id": "task_1", "description": "Step 1", "dependencies": ["non_existent_task"]}
        ]
    }
    with patch.object(planner.model, "health_check", return_value=ModelStatus.READY):
        with patch.object(planner.model, "generate", return_value=json.dumps(missing_ref_json)):
            response = planner.create_plan("Test missing")
            assert response.status == "rejected"
            assert any("non-existent" in e.lower() for e in response.validation.errors)


def test_tasks_remain_pending_no_tool_execution():
    """Verify that tasks strictly remain in PENDING status and plan status is PROPOSED."""
    mock_json = {
        "goal": "Execute shell command",
        "tasks": [
            {
                "id": "task_1",
                "description": "Propose inspection",
                "type": "DOCUMENT_RETRIEVAL",
                "status": "IN_PROGRESS",  # LLM tries to say IN_PROGRESS
                "dependencies": []
            }
        ]
    }
    planner = NEXUSPlanner()
    with patch.object(planner.model, "health_check", return_value=ModelStatus.READY):
        with patch.object(planner.model, "generate", return_value=json.dumps(mock_json)):
            response = planner.create_plan("Test pending enforcement")
            assert response.status == "success"
            # Must be forced to PENDING
            assert response.plan.tasks[0].status == TaskStatus.PENDING
            assert response.plan.status == "PROPOSED"
            assert response.plan.tasks[0].output is None


def test_agent_plan_endpoint_success_and_failures():
    """Verify POST /agent/plan integration with valid requests, empty goals, and rejected plans."""
    # 1. Empty goal (422)
    resp = client.post("/agent/plan", json={"goal": "   "})
    assert resp.status_code == 422

    # 2. Valid goal
    valid_mock = {
        "goal": "Analyze inspection documents",
        "tasks": [
            {"id": "task_1", "description": "Fetch documents", "type": "DOCUMENT_RETRIEVAL", "dependencies": []}
        ]
    }
    with patch.object(nexus_planner.model, "health_check", return_value=ModelStatus.READY):
        with patch.object(nexus_planner.model, "generate", return_value=json.dumps(valid_mock)):
            resp = client.post("/agent/plan", json={"goal": "Analyze inspection documents"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert data["plan"]["goal"] == "Analyze inspection documents"
            assert len(data["plan"]["tasks"]) == 1
            assert data["validation"]["valid"] is True

    # 3. Model outputs invalid plan (422 rejection)
    with patch.object(nexus_planner.model, "health_check", return_value=ModelStatus.READY):
        with patch.object(nexus_planner.model, "generate", return_value="I cannot create a JSON plan for this."):
            resp = client.post("/agent/plan", json={"goal": "Do something invalid"})
            assert resp.status_code == 422
            assert "validation" in resp.json()["detail"]
