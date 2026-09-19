"""
NEXUS Phase 5: Agent Planning Route
Endpoint: POST /agent/plan
Transforms user goals into structured, validated plans without tool execution.
"""

from fastapi import APIRouter, HTTPException, status
from backend.agent.schemas import (
    AgentPlanRequest,
    AgentPlanResponse,
    AgentExecuteRequest,
    AgentExecuteResponse,
)
from backend.agent.planner import nexus_planner
from backend.agent.executor import plan_executor
from backend.logger import get_logger

logger = get_logger("nexus.routes_agent")

agent_router = APIRouter(prefix="/agent", tags=["Agent Planner & Executor"])


@agent_router.post(
    "/plan",
    response_model=AgentPlanResponse,
    summary="Generate Structured Agent Plan",
    description="Decomposes a user goal into an acyclic task graph (DAG). Propose-only: does not execute tools.",
)
def create_agent_plan(request: AgentPlanRequest):
    goal_clean = request.goal.strip()
    if not goal_clean:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Goal cannot be empty or whitespace only.",
        )

    response = nexus_planner.create_plan(goal_clean, request.context)

    if response.status == "rejected":
        # Check reason
        if any("unavailable" in err.lower() for err in response.validation.errors):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.message or "Local language model unavailable.",
            )
        # Validation failure or malformed JSON
        logger.warning(f"Plan proposal rejected for goal '{goal_clean}': {response.validation.errors}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": response.message,
                "validation": response.validation.model_dump(),
                "raw_model_response": response.raw_model_response,
            },
        )

    return response


@agent_router.post(
    "/execute",
    response_model=AgentExecuteResponse,
    summary="Execute Controlled Structured Plan",
    description="Executes a validated structured plan sequentially through registered tools. Pauses if approval required.",
)
def execute_agent_plan(request: AgentExecuteRequest):
    logger.info(f"Received plan execution request: goal='{request.plan.goal}' ({len(request.plan.tasks)} tasks)")
    response = plan_executor.execute_plan(request)
    return response
