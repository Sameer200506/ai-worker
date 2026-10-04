"""Deterministic permission policy.

The model proposes actions; this module decides what is *allowed*. Nothing here knows
anything about business workflows — it only knows which tool names exist, which
arguments are structurally acceptable, which resources may be touched, and which
actions need human approval.
"""
from __future__ import annotations

from urllib.parse import urlparse

from backend import config

# --- Tool allowlist --------------------------------------------------------
# Any tool name outside this set is rejected before execution. There is no shell,
# no Python eval, no raw SQL, no arbitrary filesystem and no free-form fetch.
ALLOWED_TOOLS = frozenset(
    {
        # company sandbox
        "company.login",
        "company.search_invoice",
        "company.get_invoice",
        "company.create_finance_record",
        "company.get_finance_record",
        "company.search_customer",
        "company.get_customer",
        "company.update_customer",
        # sandbox files (virtual table, path-allowlisted)
        "files.search",
        "files.read",
        # browser (sandbox-restricted)
        "browser.open",
        "browser.click",
        "browser.type",
        "browser.read",
        "browser.screenshot",
        "browser.go_back",
        # agent control
        "approval.request",
        "memory.store",
        "memory.retrieve",
        "task.complete",
        "task.fail",
        "task.clarify",
    }
)

# Hard denylist, checked before the allowlist so the intent is explicit in logs/tests.
FORBIDDEN_TOOL_PATTERNS = (
    "shell",
    "bash",
    "exec",
    "eval",
    "python",
    "sql",
    "os.",
    "subprocess",
    "fs.",
    "http.",
    "fetch",
    "download",
    "upload",
    "sudo",
)

# --- Approval policy -------------------------------------------------------
# Tools that mutate company state and therefore need an explicit human decision.
SENSITIVE_TOOLS = frozenset(
    {
        "company.create_finance_record",
        "company.update_customer",
    }
)

# Approval policies, chosen by configuration and never by the model.
APPROVAL_POLICY_SENSITIVE = "sensitive"
APPROVAL_POLICY_DEMO = "demo"


def validate_tool_name(name: str) -> str | None:
    """Return a rejection reason for an unknown or forbidden tool, else ``None``."""
    if not isinstance(name, str) or not name.strip():
        return "Tool name is empty."
    lowered = name.lower()
    for pattern in FORBIDDEN_TOOL_PATTERNS:
        if pattern in lowered:
            return f"Tool '{name}' is not allowed: arbitrary execution primitives are forbidden."
    if name not in ALLOWED_TOOLS:
        return f"Tool '{name}' is not allowed or does not exist."
    return None


def _sandbox_port() -> int:
    return urlparse(config.SANDBOX_URL).port or 3001


def resolve_sandbox_url(url: str) -> str:
    """Resolve a root-relative sandbox path (e.g. ``/invoices``) against the sandbox."""
    if isinstance(url, str) and url.startswith("/"):
        return f"{config.SANDBOX_URL}{url}"
    return url


def validate_browser_url(url: str) -> str | None:
    """Browser navigation is confined to the local company sandbox."""
    if not isinstance(url, str) or not url.strip():
        return "Browser URL is empty."
    parsed = urlparse(resolve_sandbox_url(url.strip()))
    host = (parsed.hostname or "").lower()
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if parsed.scheme not in config.ALLOWED_BROWSER_SCHEMES:
        return f"Invalid URL scheme '{parsed.scheme}'."
    if host not in config.ALLOWED_BROWSER_HOSTS:
        return "Browser navigation is restricted to the local sandbox."
    if port != _sandbox_port():
        return "Browser navigation is restricted to the sandbox port."
    return None


def validate_file_path(path: str) -> str | None:
    """`files.*` only reads the sandbox's virtual file table, under allowed prefixes.

    The agent never touches the real filesystem: no absolute paths, no traversal,
    no device paths, no backslashes.
    """
    if not isinstance(path, str) or not path.strip():
        return "File path is empty."
    if len(path) > config.ALLOWED_FILE_MAX_LENGTH:
        return "File path is too long."
    if "\\" in path or path.startswith("/") or ":" in path:
        return "File path must be a relative sandbox path (for example notes/process-guide.txt)."
    if ".." in path.split("/"):
        return "File path traversal is not allowed."
    if not any(path.startswith(prefix) for prefix in config.ALLOWED_FILE_PREFIXES):
        allowed = ", ".join(config.ALLOWED_FILE_PREFIXES)
        return f"File access is restricted to: {allowed}"
    return None


def requires_approval(tool_name: str, demo_mode: str = "normal", approval_status: str | None = None) -> bool:
    """Whether a tool call must stop and wait for a human decision.

    ``APPROVAL_POLICY=sensitive`` (default) gates every sensitive tool in every mode.
    ``APPROVAL_POLICY=demo`` gates them only in the ``approval_required`` demo mode.
    """
    if approval_status == "approved":
        return False
    if tool_name not in SENSITIVE_TOOLS:
        return False
    if config.APPROVAL_POLICY == APPROVAL_POLICY_DEMO:
        return demo_mode == "approval_required"
    return True
