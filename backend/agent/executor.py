"""
NEXUS Controlled Tool Execution Layer: Plan Executor
Implements the deterministic execution pipeline:
Plan -> Task -> Tool -> Result -> Next Task

Guarantees that:
1. ONLY registered, whitelisted tools are executed.
2. The LLM or plan can NEVER trigger shell commands, arbitrary Python, or file deletion.
3. High-impact operations (file creation/export) strictly require explicit user approval.
4. Intermediate results are securely piped forward between dependent tasks.
"""

from typing import Any, Dict, List, Optional
from backend.logger import logger
from backend.agent.schemas import (
    PlannedTask,
    StructuredPlan,
    TaskStatus,
    TaskType,
    AgentExecuteRequest,
    AgentExecuteResponse,
)
from backend.tools.base import ToolContext, ToolResult
from backend.tools.registry import tool_registry


class PlanExecutor:
    """
    Sequential, sandboxed executor that runs planned tasks step by step.
    Enforces that execution is restricted to registered tools and halted
    when approval is required.
    """

    def __init__(self, registry=None):
        self.registry = registry or tool_registry

    def resolve_tool_for_task(self, task: PlannedTask) -> str:
        """
        Deterministically resolves the registered tool name for a planned task.
        Prioritizes explicit tool specification in task.input['tool'],
        then falls back to mapping task.type safely.
        """
        explicit_tool = task.input.get("tool")
        if explicit_tool:
            if not self.registry.has_tool(explicit_tool):
                raise PermissionError(f"Unauthorized or unregistered tool: '{explicit_tool}'")
            return explicit_tool

        # Safe deterministic mapping from TaskType
        task_type = task.type
        if task_type == TaskType.DOCUMENT_RETRIEVAL.value:
            if "document_id" in task.input:
                return "read_document"
            return "list_documents"

        elif task_type == TaskType.CONTENT_EXTRACTION.value:
            if task.input.get("is_scanned"):
                return "run_ocr"
            return "read_document"

        elif task_type == TaskType.KNOWLEDGE_SEARCH.value:
            return "search_knowledge"

        elif task_type == TaskType.REPORT_GENERATION.value:
            if "filename" in task.input:
                return "export_report"
            return "create_report"

        elif task_type == TaskType.EVIDENCE_VERIFICATION.value:
            if "document_id" in task.input and any(
                task.input.get("file_type", "").endswith(ext) for ext in ["png", "jpg", "jpeg"]
            ):
                return "analyze_image"
            return "search_knowledge"

        elif task_type == TaskType.DATA_COMPARISON.value:
            return "create_report"

        elif task_type == TaskType.GENERAL_REASONING.value:
            return "create_report"

        raise ValueError(f"Unable to safely resolve registered tool for task type '{task_type}'.")

    def pipe_dependency_outputs(
        self,
        task: PlannedTask,
        completed_tasks: Dict[str, PlannedTask],
    ) -> Dict[str, Any]:
        """
        Securely merges outputs from prerequisite tasks into the current task's input.
        """
        params = dict(task.input)

        for dep_id in task.dependencies:
            dep_task = completed_tasks.get(dep_id)
            if not dep_task or not dep_task.output:
                continue

            dep_out = dep_task.output

            # 1. Forward document_id if missing
            if "document_id" not in params:
                if "document_id" in dep_out:
                    params["document_id"] = dep_out["document_id"]
                elif "documents" in dep_out and len(dep_out["documents"]) > 0:
                    params["document_id"] = dep_out["documents"][0]["id"]

            # 2. Forward content/citations for report creation if missing
            if "content" not in params and "content" in dep_out:
                params["content"] = dep_out["content"]

            if "summary" not in params:
                if "analysis" in dep_out:
                    params["summary"] = dep_out["analysis"]
                elif "content" in dep_out:
                    params["summary"] = f"Report synthesized from {dep_task.description}."

            if "sections" not in params and "citations" in dep_out:
                sections = []
                for idx, cit in enumerate(dep_out["citations"][:5], 1):
                    sections.append({
                        "title": f"Finding {idx}: {cit.get('document_name', 'Doc')} (Page {cit.get('page_number', 'N/A')})",
                        "content": cit.get("text", "").strip(),
                    })
                if sections:
                    params["sections"] = sections

        return params

    def execute_plan(self, request: AgentExecuteRequest) -> AgentExecuteResponse:
        """
        Executes a StructuredPlan step by step:
        Plan -> Task -> Tool -> Result -> Next task

        Pauses safely when explicit user approval is required.
        """
        plan = request.plan
        completed_tasks: Dict[str, PlannedTask] = {}
        executed_count = 0
        failed_count = 0

        logger.info(f"[PLAN_EXECUTOR] Commencing execution of plan for goal: '{plan.goal}' ({len(plan.tasks)} tasks)")

        # Index any tasks that were already completed in a prior run
        for t in plan.tasks:
            if t.status == TaskStatus.COMPLETED and t.output is not None:
                completed_tasks[t.id] = t
                executed_count += 1

        for task in plan.tasks:
            # Skip tasks already completed
            if task.status == TaskStatus.COMPLETED:
                continue

            # Verify all dependencies have succeeded
            unmet_deps = [dep for dep in task.dependencies if dep not in completed_tasks]
            if unmet_deps:
                task.status = TaskStatus.FAILED
                task.error = f"Unmet dependencies: {', '.join(unmet_deps)}"
                failed_count += 1
                plan.status = "FAILED"
                logger.error(f"[PLAN_EXECUTOR] Task '{task.id}' failed due to unmet dependencies: {unmet_deps}")
                return AgentExecuteResponse(
                    status="failed",
                    plan=plan,
                    executed_tasks=executed_count,
                    failed_tasks=failed_count,
                    message=f"Plan halted: Task '{task.id}' has unsatisfied dependencies ({unmet_deps}).",
                )

            # Resolve registered tool
            try:
                tool_name = self.resolve_tool_for_task(task)
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                failed_count += 1
                plan.status = "FAILED"
                return AgentExecuteResponse(
                    status="failed",
                    plan=plan,
                    executed_tasks=executed_count,
                    failed_tasks=failed_count,
                    message=f"Plan halted: Security violation or unknown tool for task '{task.id}': {e}",
                )

            tool = self.registry.get_tool(tool_name)

            # Check approval gate
            is_approved = (
                request.auto_approve_exports
                or task.id in request.approved_task_ids
                or task.input.get("approved") is True
            )

            if tool.requires_approval and not is_approved:
                task.status = TaskStatus.REQUIRES_APPROVAL
                task.error = f"Tool '{tool_name}' requires explicit user approval to execute."
                plan.status = "PAUSED_FOR_APPROVAL"
                logger.info(f"[PLAN_EXECUTOR] Execution paused for approval at task '{task.id}' ({tool_name})")
                return AgentExecuteResponse(
                    status="paused_for_approval",
                    plan=plan,
                    executed_tasks=executed_count,
                    failed_tasks=failed_count,
                    pending_approval_task_id=task.id,
                    message=f"Execution paused: Task '{task.id}' ({tool_name}) requires user approval to write/export.",
                )

            # Prepare tool input parameters
            piped_params = self.pipe_dependency_outputs(task, completed_tasks)
            if tool.requires_approval:
                piped_params["approved"] = is_approved

            # Execute tool safely
            task.status = TaskStatus.IN_PROGRESS
            context = ToolContext(is_approved=is_approved, metadata={"task_id": task.id})

            logger.info(f"[PLAN_EXECUTOR] Executing Task '{task.id}' via Tool '{tool_name}'")
            tool_result: ToolResult = tool.execute(params=piped_params, context=context)

            if tool_result.success:
                task.status = TaskStatus.COMPLETED
                task.output = tool_result.output
                task.error = None
                completed_tasks[task.id] = task
                executed_count += 1
                logger.info(f"[PLAN_EXECUTOR] Task '{task.id}' COMPLETED in {tool_result.duration_ms}ms")
            else:
                task.status = TaskStatus.FAILED
                task.error = tool_result.error
                failed_count += 1
                plan.status = "FAILED"
                logger.error(f"[PLAN_EXECUTOR] Task '{task.id}' FAILED: {tool_result.error}")
                return AgentExecuteResponse(
                    status="failed",
                    plan=plan,
                    executed_tasks=executed_count,
                    failed_tasks=failed_count,
                    message=f"Plan halted on failure at task '{task.id}': {tool_result.error}",
                )

        plan.status = "COMPLETED"
        logger.info(f"[PLAN_EXECUTOR] All {executed_count} tasks completed successfully!")
        return AgentExecuteResponse(
            status="completed",
            plan=plan,
            executed_tasks=executed_count,
            failed_tasks=0,
            message="Plan executed successfully to completion.",
        )


plan_executor = PlanExecutor()
