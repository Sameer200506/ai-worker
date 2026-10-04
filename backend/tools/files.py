from __future__ import annotations

from apps.sandbox.store import store
from backend.database.models import ToolResult
from backend.tools.base import fail, ok


def search(query: str) -> ToolResult:
    rows = store.search_files(query)
    return ok("files.search", f"Found {len(rows)} file(s).", data={"files": rows})


def read(path: str) -> ToolResult:
    row = store.read_file(path)
    if not row:
        return fail("files.read", f"File {path} was not found.", "NOT_FOUND", False)
    return ok("files.read", f"Read file {path}.", data=row)
