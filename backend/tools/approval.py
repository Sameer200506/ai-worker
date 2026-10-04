from __future__ import annotations

from typing import Any

from backend.database.models import ToolResult
from backend.tools.base import ok


def request(summary: str, payload: dict[str, Any]) -> ToolResult:
    return ok(
        "approval.request",
        summary,
        data={"summary": summary, "payload": payload},
        artifacts=[{"type": "approval_request", "value": payload}],
    )
