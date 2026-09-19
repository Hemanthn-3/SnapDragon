"""
NEXUS Controlled Tool Execution Layer: Tool Registry
Enforces strict whitelisting of the 7 allowed tools.
Explicitly prohibits arbitrary shell, Python execution, file deletion, and unregistered tools.
"""

from typing import Any, Dict, List, Optional
from backend.logger import logger
from backend.tools.base import BaseTool, ToolContext, ToolResult
from backend.tools.implementations import (
    ListDocumentsTool,
    ReadDocumentTool,
    SearchKnowledgeTool,
    AnalyzeImageTool,
    RunOCRTool,
    CreateReportTool,
    ExportReportTool,
)


class ToolRegistry:
    """
    Central registry for strictly permitted tools.
    Enforces that the LLM or executor can NEVER call arbitrary tools or shell commands.
    """

    ALLOWED_TOOLS = {
        "list_documents",
        "read_document",
        "search_knowledge",
        "analyze_image",
        "run_ocr",
        "create_report",
        "export_report",
    }

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Instantiates and registers the 7 allowed tools."""
        tools: List[BaseTool] = [
            ListDocumentsTool(),
            ReadDocumentTool(),
            SearchKnowledgeTool(),
            AnalyzeImageTool(),
            RunOCRTool(),
            CreateReportTool(),
            ExportReportTool(),
        ]
        for tool in tools:
            if tool.name not in self.ALLOWED_TOOLS:
                raise ValueError(f"Security violation: Tool '{tool.name}' is not in allowed list.")
            self._tools[tool.name] = tool
            logger.debug(f"[TOOL_REGISTRY] Registered tool: '{tool.name}' (Approval={tool.requires_approval})")

    def get_tool(self, name: str) -> BaseTool:
        """
        Retrieves a registered tool by name.
        Raises PermissionError if tool is unknown or unauthorized.
        """
        if name not in self.ALLOWED_TOOLS or name not in self._tools:
            logger.error(f"[SECURITY_ALERT] Attempted access to unauthorized tool '{name}'")
            raise PermissionError(
                f"Unauthorized tool execution attempted: '{name}'. "
                f"Permitted tools: {sorted(list(self.ALLOWED_TOOLS))}"
            )
        return self._tools[name]

    def has_tool(self, name: str) -> bool:
        """Returns True if the tool is registered and allowed."""
        return name in self.ALLOWED_TOOLS and name in self._tools

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns metadata and schemas for all registered tools."""
        tool_specs = []
        for name in sorted(self._tools.keys()):
            tool = self._tools[name]
            tool_specs.append({
                "name": tool.name,
                "description": tool.description,
                "requires_approval": tool.requires_approval,
                "timeout_seconds": tool.timeout_seconds,
                "input_schema": tool.input_schema.model_json_schema(),
                "output_schema": tool.output_schema.model_json_schema(),
            })
        return tool_specs

    def execute_tool(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """
        Executes a registered tool with full security guards.
        """
        try:
            tool = self.get_tool(name)
            return tool.execute(params=params, context=context)
        except PermissionError as pe:
            return ToolResult(
                tool_name=name,
                success=False,
                error=str(pe),
                duration_ms=0.0,
            )
        except Exception as e:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Registry error for '{name}': {str(e)}",
                duration_ms=0.0,
            )


# Global registry singleton
tool_registry = ToolRegistry()
