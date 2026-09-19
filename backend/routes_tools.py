"""
NEXUS Controlled Tool Execution Layer: Tools API Routes
Provides endpoints for inspecting permitted tools and executing registered tools safely.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.logger import logger
from backend.tools.base import ToolContext, ToolResult
from backend.tools.registry import tool_registry

router = APIRouter(prefix="/tools", tags=["Tools"])


class ExecuteToolRequest(BaseModel):
    tool: str = Field(..., description="Name of registered tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool input parameters")
    approved: bool = Field(default=False, description="User approval flag for sensitive operations")


@router.get("", status_code=status.HTTP_200_OK)
def list_available_tools():
    """
    Returns metadata, descriptions, schemas, and approval rules
    for all 7 strictly permitted NEXUS tools.
    """
    tools = tool_registry.list_tools()
    return {
        "total_tools": len(tools),
        "tools": tools,
    }


@router.post("/execute", status_code=status.HTTP_200_OK)
def execute_tool(request: ExecuteToolRequest):
    """
    Executes a registered tool within strict security sandboxes.
    Rejects any unregistered tools, shell access, or unapproved exports.
    """
    if not tool_registry.has_tool(request.tool):
        logger.error(f"[SECURITY_ALERT] Blocked execution of unauthorized tool: '{request.tool}'")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Tool '{request.tool}' is not permitted. Only registered tools can be executed.",
        )

    context = ToolContext(is_approved=request.approved)
    result = tool_registry.execute_tool(
        name=request.tool,
        params=request.parameters,
        context=context,
    )

    if not result.success:
        if "approval" in (result.error or "").lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=result.error,
            )
        # Return result with 200 containing error details or 422 for bad input
        if "invalid parameters" in (result.error or "").lower():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result.error,
            )

    return result
