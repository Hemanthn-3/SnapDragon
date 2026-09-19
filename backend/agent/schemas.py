"""
NEXUS Phase 5: Agent Planning Schemas
Strict structured JSON representations for goals, tasks, dependencies, and validation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    DOCUMENT_RETRIEVAL = "DOCUMENT_RETRIEVAL"
    CONTENT_EXTRACTION = "CONTENT_EXTRACTION"
    KNOWLEDGE_SEARCH = "KNOWLEDGE_SEARCH"
    DATA_COMPARISON = "DATA_COMPARISON"
    EVIDENCE_VERIFICATION = "EVIDENCE_VERIFICATION"
    REPORT_GENERATION = "REPORT_GENERATION"
    GENERAL_REASONING = "GENERAL_REASONING"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


class PlannedTask(BaseModel):
    """
    Structured representation of an individual planned action.
    Strictly initialized in PENDING state; the LLM only proposes the task.
    """
    id: str = Field(..., description="Unique task identifier, e.g. 'task_1'")
    description: str = Field(..., description="Action description")
    type: str = Field(default=TaskType.GENERAL_REASONING.value, description="Task category")
    dependencies: List[str] = Field(default_factory=list, description="IDs of prerequisite tasks")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current execution state")
    input: Dict[str, Any] = Field(default_factory=dict, description="Input parameters for task")
    output: Optional[Dict[str, Any]] = Field(default=None, description="Task execution output (null in proposal)")
    error: Optional[str] = Field(default=None, description="Error message if failed (null in proposal)")

    @field_validator("input", mode="before")
    @classmethod
    def normalize_input(cls, v):
        if v is None:
            return {}
        return v


class StructuredPlan(BaseModel):
    """
    Structured plan comprising a sequence of planned tasks.
    """
    goal: str = Field(..., description="Original user goal")
    tasks: List[PlannedTask] = Field(..., description="Ordered list of proposed tasks")
    estimated_steps: int = Field(..., description="Total task count")
    created_at: str = Field(..., description="ISO timestamp of plan creation")
    status: str = Field(default="PROPOSED", description="Plan status (always 'PROPOSED' in Phase 5)")


class PlanValidationResult(BaseModel):
    """
    Validation report detailing graph integrity, cycle detection, and schema adherence.
    """
    valid: bool = Field(..., description="Whether plan is structurally valid")
    task_count: int = Field(default=0, description="Number of validated tasks")
    has_cycles: bool = Field(default=False, description="Whether circular dependencies were detected")
    errors: List[str] = Field(default_factory=list, description="Validation failure descriptions")


class AgentPlanRequest(BaseModel):
    """
    Request payload for proposing an agent plan.
    """
    goal: str = Field(..., min_length=1, description="Natural language goal to plan")
    context: List[str] = Field(default_factory=list, description="Optional grounding context or documents")


class AgentPlanResponse(BaseModel):
    """
    Response payload containing the proposed plan and validation diagnostics.
    """
    status: str = Field(..., description="'success' or 'rejected'")
    plan: Optional[StructuredPlan] = Field(default=None, description="The validated proposed plan")
    validation: PlanValidationResult = Field(..., description="Validation outcome")
    message: Optional[str] = Field(default=None, description="Diagnostic or error explanation")
    raw_model_response: Optional[str] = Field(default=None, description="Raw LLM output snippet for transparency")


class AgentExecuteRequest(BaseModel):
    """
    Request payload to safely execute a validated structured plan.
    """
    plan: StructuredPlan = Field(..., description="The structured plan to execute")
    approved_task_ids: List[str] = Field(default_factory=list, description="IDs of tasks explicitly approved by user")
    auto_approve_exports: bool = Field(default=False, description="Global approval flag for file creation/export")


class AgentExecuteResponse(BaseModel):
    """
    Response payload following plan execution step by step.
    """
    status: str = Field(..., description="'completed', 'paused_for_approval', or 'failed'")
    plan: StructuredPlan = Field(..., description="The plan with updated task outputs and states")
    executed_tasks: int = Field(default=0, description="Count of successfully executed tasks")
    failed_tasks: int = Field(default=0, description="Count of failed tasks")
    pending_approval_task_id: Optional[str] = Field(default=None, description="Task ID currently blocking on approval")
    message: str = Field(..., description="Execution status summary")
