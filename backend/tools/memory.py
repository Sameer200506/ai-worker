from __future__ import annotations

from backend.database.models import ToolResult
from backend.database.store import retrieve_memory, store_memory
from backend.tools.base import ok


def store_item(key: str, value: object, task_id: str | None = None) -> ToolResult:
    store_memory(key, value, task_id)
    return ok("memory.store", f"Stored memory for {key}.", data={"key": key, "value": value})


def retrieve(key: str) -> ToolResult:
    values = retrieve_memory(key)
    return ok("memory.retrieve", f"Retrieved {len(values)} memor(ies) for {key}.", data={"key": key, "values": values})
