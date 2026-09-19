"""
NEXUS Controlled Tool Execution Layer: Base Tool Infrastructure
Enforces strict schema validation, permission checks, timeouts, logging, and error handling.
"""

import time
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field

from backend.logger import logger


class ToolContext(BaseModel):
    """
    Context passed during tool execution, carrying security tokens and approval state.
    """
    is_approved: bool = Field(default=False, description="Whether explicit approval is granted for write/export")
    session_id: Optional[str] = Field(default=None, description="Optional session tracking ID")
    user_id: Optional[str] = Field(default="local_user", description="Authenticated user identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")


class ToolResult(BaseModel):
    """
    Strict result container for every executed tool.
    """
    tool_name: str = Field(..., description="Name of executed tool")
    success: bool = Field(..., description="Whether tool execution succeeded")
    output: Optional[Dict[str, Any]] = Field(default=None, description="Structured tool output")
    error: Optional[str] = Field(default=None, description="Detailed error message if failed")
    duration_ms: float = Field(default=0.0, description="Execution time in milliseconds")


class BaseTool(ABC):
    """
    Abstract base class for all NEXUS registered tools.
    Every tool strictly defines schemas, timeout, approval rules, and isolation.
    """
    name: str
    description: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    requires_approval: bool = False
    timeout_seconds: float = 30.0

    def validate_permissions(self, params: BaseModel, context: ToolContext) -> None:
        """
        Validates whether context or parameters satisfy the tool's security requirements.
        Raises PermissionError if approval is missing for high-impact operations.
        """
        if self.requires_approval:
            # Check context-level approval or parameter-level approval
            param_approved = getattr(params, "approved", False)
            if not (context.is_approved or param_approved):
                raise PermissionError(
                    f"Tool '{self.name}' requires explicit user approval before execution."
                )

    @abstractmethod
    def _run(self, params: BaseModel, context: ToolContext) -> BaseModel:
        """
        Concrete tool logic. Must return an instance of self.output_schema.
        """
        pass

    def execute(self, params: Dict[str, Any], context: Optional[ToolContext] = None) -> ToolResult:
        """
        Safe executor wrapping input validation, permission check, timeout enforcement,
        structured output validation, error isolation, and audit logging.
        """
        start_time = time.perf_counter()
        ctx = context or ToolContext()

        logger.info(f"[TOOL_EXEC] Starting '{self.name}' | Approval={ctx.is_approved}")

        try:
            # 1. Strict Input Schema Validation
            try:
                validated_input = self.input_schema.model_validate(params)
            except Exception as e:
                logger.warning(f"[TOOL_EXEC] Schema validation failed for '{self.name}': {e}")
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Invalid parameters for tool '{self.name}': {str(e)}",
                    duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
                )

            # 2. Permission and Approval Validation
            try:
                self.validate_permissions(validated_input, ctx)
            except PermissionError as pe:
                logger.warning(f"[TOOL_EXEC] Permission denied for '{self.name}': {pe}")
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=str(pe),
                    duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
                )

            # 3. Timeout Enforcement using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run, validated_input, ctx)
                try:
                    raw_result = future.result(timeout=self.timeout_seconds)
                except concurrent.futures.TimeoutError:
                    logger.error(f"[TOOL_EXEC] Tool '{self.name}' timed out after {self.timeout_seconds}s")
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        error=f"Tool '{self.name}' timed out after {self.timeout_seconds} seconds",
                        duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
                    )

            # 4. Strict Output Schema Validation
            if not isinstance(raw_result, self.output_schema):
                validated_output = self.output_schema.model_validate(raw_result)
            else:
                validated_output = raw_result

            duration = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(f"[TOOL_EXEC] Finished '{self.name}' successfully in {duration}ms")

            return ToolResult(
                tool_name=self.name,
                success=True,
                output=validated_output.model_dump(),
                duration_ms=duration,
            )

        except Exception as e:
            duration = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"[TOOL_EXEC] Unhandled exception in '{self.name}': {e}", exc_info=True)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Execution error in tool '{self.name}': {str(e)}",
                duration_ms=duration,
            )
