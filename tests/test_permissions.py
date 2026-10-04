from backend.permissions.policies import validate_browser_url, validate_tool_name
from backend.tools.registry import execute_tool
from backend.database.models import TaskState


def test_unknown_tool_rejected():
    state = TaskState(task_id="t", goal="x")
    result = execute_tool("shell.exec", {"cmd": "rm -rf /"}, state)
    assert not result.success
    assert result.error and result.error.type == "PERMISSION_DENIED"


def test_browser_url_restricted():
    assert validate_browser_url("https://evil.example")
    assert validate_tool_name("company.search_invoice") is None
