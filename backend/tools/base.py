from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.database.models import ToolError, ToolResult


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    sensitive: bool = False
    required: list[str] = Field(default_factory=list)


def ok(tool: str, observation: str, data: Any = None, artifacts: list | None = None) -> ToolResult:
    return ToolResult(success=True, tool=tool, data=data, observation=observation, artifacts=artifacts or [])


def fail(tool: str, observation: str, error_type: str, recoverable: bool, data: Any = None) -> ToolResult:
    return ToolResult(
        success=False,
        tool=tool,
        data=data,
        observation=observation,
        error=ToolError(type=error_type, recoverable=recoverable, message=observation),
    )
