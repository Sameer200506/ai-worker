"""Tool registry.

Responsibilities (one per concern, all deterministic):
1. registration
2. lookup
3. schema generation for native LLM tool calling
4. validation of incoming arguments against the declared JSON schema
5. safe, timeout-bounded execution

Only tools registered here are exposed to the model. Unknown names and malformed
arguments fail safely with a typed error instead of reaching a handler.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from backend import config
from backend.database.models import TaskState, ToolResult
from backend.permissions.policies import (
    validate_browser_url,
    validate_file_path,
    validate_tool_name,
)
from backend.tools import approval, company, files, memory
from backend.tools.base import fail, ok
from backend.tools.browser import dispatch as browser_dispatch

#: JSON Schema keywords understood by the local validator.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], TaskState], ToolResult]
    required: list[str] = field(default_factory=list)
    sensitive: bool = False
    timeout_seconds: float | None = None

    def schema(self) -> dict[str, Any]:
        parameters = dict(self.parameters)
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters["required"] = list(self.required)
        parameters.setdefault("additionalProperties", False)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    # Kept for readability at call sites that only care about the payload.
    to_schema = schema


class ValidationError(ValueError):
    """Raised internally when arguments do not satisfy the declared schema."""

    def __init__(self, message: str, kind: str = "INVALID_DATA") -> None:
        super().__init__(message)
        self.kind = kind


def validate_arguments(name: str, schema: dict[str, Any], arguments: dict[str, Any] | None) -> None:
    """Validate arguments against the declared JSON schema.

    Raises ``ValidationError`` with ``PERMISSION_DENIED`` when the failure is a policy
    violation (disallowed URL, disallowed path) and ``INVALID_DATA`` otherwise.
    """
    args = arguments or {}
    if not isinstance(args, dict):
        raise ValidationError(f"Arguments for {name} must be a JSON object.", "INVALID_DATA")

    if name == "browser.open":
        url = args.get("url")
        if url:
            url_error = validate_browser_url(url)
            if url_error:
                raise ValidationError(url_error, "PERMISSION_DENIED")
    if name == "files.read":
        path = args.get("path")
        if path:
            path_error = validate_file_path(path)
            if path_error:
                raise ValidationError(path_error, "PERMISSION_DENIED")

    properties: dict[str, Any] = schema.get("properties") or {}
    required: list[str] = schema.get("required") or []

    missing = [key for key in required if args.get(key) in (None, "")]
    if missing:
        raise ValidationError(f"Missing required argument(s) for {name}: {', '.join(sorted(missing))}.")

    unknown = [key for key in args if key not in properties]
    if unknown:
        allowed = ", ".join(sorted(properties)) or "none"
        raise ValidationError(
            f"Unknown argument(s) for {name}: {', '.join(sorted(unknown))}. Accepted arguments: {allowed}."
        )

    for key, value in args.items():
        spec = properties.get(key) or {}
        if value is None:
            continue
        expected = spec.get("type")
        if isinstance(expected, list):
            expected_types = expected
        else:
            expected_types = [expected] if expected else []
        if not expected_types:
            continue
        python_types: tuple[type, ...] = tuple(
            _JSON_TYPES[item] for item in expected_types if item in _JSON_TYPES
        )
        if not python_types:
            continue
        if expected_types == ["integer"] and isinstance(value, bool):
            raise ValidationError(f"Argument '{key}' of {name} must be an integer.")
        if not isinstance(value, python_types):
            names = " or ".join(expected_types)
            raise ValidationError(f"Argument '{key}' of {name} must be of type {names}, got {type(value).__name__}.")

        enum = spec.get("enum")
        if enum and value not in enum:
            raise ValidationError(f"Argument '{key}' of {name} must be one of: {', '.join(map(str, enum))}.")

    # Coerce numeric strings that models sometimes emit for integer fields.
    for key, spec in properties.items():
        if spec.get("type") == "integer" and isinstance(args.get(key), str) and args[key].strip().lstrip("-").isdigit():
            args[key] = int(args[key])
        if spec.get("type") in {"object", "array"} and isinstance(args.get(key), str) and key == "value":
            import json as _json

            try:
                args[key] = _json.loads(args[key])
            except (TypeError, ValueError):
                pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # -- 1. registration ----------------------------------------------------
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    # -- 2. lookup ----------------------------------------------------------
    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # -- 3. schema generation ----------------------------------------------
    def schemas(self) -> list[dict[str, Any]]:
        """Native structured tool definitions sent to the model on every request."""
        return [tool.schema() for tool in self._tools.values()]

    def flat_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "sensitive": tool.sensitive,
                "parameters": tool.parameters.get("properties", {}),
                "required": list(tool.required),
            }
            for tool in self._tools.values()
        ]

    # -- 4. validation ------------------------------------------------------
    def validate(self, name: str, arguments: dict[str, Any] | None = None) -> str | None:
        """Return a human-readable rejection reason, or ``None`` when the call is valid."""
        forbidden = validate_tool_name(name)
        if forbidden:
            return forbidden
        tool = self.get(name)
        if not tool:
            return f"Tool '{name}' is not registered."
        try:
            validate_arguments(name, tool.schema()["function"]["parameters"], arguments)
        except ValidationError as exc:
            return str(exc)
        return None

    def validate_or_raise(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        forbidden = validate_tool_name(name)
        if forbidden:
            raise ValidationError(forbidden, "PERMISSION_DENIED")
        tool = self.get(name)
        if not tool:
            raise ValidationError(f"Tool '{name}' is not registered.", "UNKNOWN_TOOL")
        validate_arguments(name, tool.schema()["function"]["parameters"], arguments)
        return arguments or {}

    # -- 5. safe execution --------------------------------------------------
    def execute(self, name: str, arguments: dict[str, Any] | None, state: TaskState) -> ToolResult:
        """Synchronous execution (used by the verifier and by tests)."""
        try:
            payload = self.validate_or_raise(name, arguments)
        except ValidationError as exc:
            return fail(name, str(exc), exc.kind, False)

        tool = self.get(name)
        if tool is None:  # pragma: no cover - guarded by validate_or_raise
            return fail(name, f"Tool '{name}' is not registered.", "UNKNOWN_TOOL", False)

        started = time.perf_counter()
        try:
            result = tool.handler(payload, state)
        except KeyError as exc:
            return fail(name, f"Missing argument: {exc}", "INVALID_DATA", False)
        except Exception as exc:  # noqa: BLE001 - the model must never see a traceback
            return fail(name, f"Tool failed: {type(exc).__name__}: {exc}", "UNKNOWN_ERROR", False)
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result

    async def execute_async(self, name: str, arguments: dict[str, Any] | None, state: TaskState) -> ToolResult:
        """Bounded execution: a hung tool cannot stall the agent loop forever."""
        timeout = (self.get(name).timeout_seconds if self.get(name) else None) or config.TOOL_TIMEOUT_SECONDS
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.execute, name, arguments, state),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return fail(
                name,
                f"Tool '{name}' exceeded the {timeout:g}s execution limit and was aborted.",
                "TIMEOUT",
                True,
            )
        except Exception as exc:  # noqa: BLE001
            return fail(name, f"Tool failed: {type(exc).__name__}: {exc}", "UNKNOWN_ERROR", False)
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result


def _build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(
        Tool(
            name="company.login",
            description=(
                "Authenticate against the company sandbox. The runtime injects the sandbox "
                "credentials, so call this with no arguments. Use it when a session is missing "
                "or a tool reports that authentication expired."
            ),
            parameters={"type": "object", "properties": {}},
            required=[],
            handler=lambda args, state: company.login(
                args.get("username") or config.SANDBOX_USERNAME,
                args.get("password") or config.SANDBOX_PASSWORD,
            ),
        )
    )

    reg.register(
        Tool(
            name="company.search_invoice",
            description=(
                "Search company invoices. Returns full invoice rows; pick the relevant one "
                "yourself using issued_at, due_date and status."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Company name to filter by."},
                    "status": {"type": "string", "description": "Optional status filter, e.g. Pending or Paid."},
                    "sort": {
                        "type": "string",
                        "enum": ["latest", "oldest", "amount_desc"],
                        "description": "Optional ordering hint.",
                    },
                },
            },
            required=["company"],
            handler=lambda args, state: company.search_invoice(args.get("company"), args.get("status"), args.get("sort")),
        )
    )

    reg.register(
        Tool(
            name="company.get_invoice",
            description="Read one invoice by its ID. Use it to obtain the authoritative amount, company and due date.",
            parameters={
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string", "description": "Invoice identifier, e.g. INV-1042."},
                },
            },
            required=["invoice_id"],
            handler=lambda args, state: company.get_invoice(args["invoice_id"]),
        )
    )

    reg.register(
        Tool(
            name="company.create_finance_record",
            description=(
                "Record an invoice in the finance system. Use this once you have read the invoice "
                "and know its exact amount and due date. Copy the values from the invoice you read; "
                "never invent them."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string", "description": "Invoice ID to record, e.g. INV-1042."},
                    "company": {"type": "string", "description": "Company name exactly as shown on the invoice."},
                    "amount": {"type": "integer", "description": "Invoice amount in INR, copied from the invoice."},
                    "due_date": {"type": "string", "description": "ISO due date copied from the invoice, e.g. 2026-10-15."},
                },
            },
            required=["invoice_id", "company", "amount", "due_date"],
            sensitive=True,
            handler=lambda args, state: company.create_finance_record(
                args["invoice_id"],
                args["company"],
                int(args["amount"]),
                args["due_date"],
                state.finance_token,
            ),
        )
    )

    reg.register(
        Tool(
            name="company.get_finance_record",
            description="Read a finance record back from the finance system by record_id or invoice_id.",
            parameters={
                "type": "object",
                "properties": {
                    "record_id": {"type": "string", "description": "Finance record identifier, e.g. FIN-8801."},
                    "invoice_id": {"type": "string", "description": "Associated invoice identifier."},
                },
            },
            required=[],
            handler=lambda args, state: company.get_finance_record(args.get("record_id"), args.get("invoice_id")),
        )
    )

    reg.register(
        Tool(
            name="company.search_customer",
            description="Search customers by name, customer ID or email.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Customer name, company name, or search keyword."},
                },
            },
            required=["query"],
            handler=lambda args, state: company.search_customer(args.get("query") or ""),
        )
    )

    reg.register(
        Tool(
            name="company.get_customer",
            description="Read one customer record by customer ID.",
            parameters={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer identifier, e.g. CUST-102."},
                },
            },
            required=["customer_id"],
            handler=lambda args, state: company.get_customer(args["customer_id"]),
        )
    )

    reg.register(
        Tool(
            name="company.update_customer",
            description=(
                "Update supported fields on a customer record. Use this when the user asks for a "
                "customer's status or contact details to change."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer identifier."},
                    "status": {"type": "string", "description": "New customer status, e.g. Follow Up or Active."},
                    "email": {"type": "string", "description": "New contact email address."},
                    "contact": {"type": "string", "description": "New primary contact name."},
                },
            },
            required=["customer_id"],
            sensitive=True,
            handler=lambda args, state: company.update_customer(
                args["customer_id"],
                args.get("status"),
                **{key: args[key] for key in ("email", "contact") if key in args},
            ),
        )
    )

    reg.register(
        Tool(
            name="files.search",
            description="Search the sandbox document store for paths matching a term.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search term."}},
            },
            required=["query"],
            handler=lambda args, state: files.search(args.get("query") or ""),
        )
    )

    reg.register(
        Tool(
            name="files.read",
            description=(
                "Read a document from the sandbox document store. Paths must be relative and "
                f"start with one of: {', '.join(config.ALLOWED_FILE_PREFIXES)}."
            ),
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative sandbox path, e.g. notes/process-guide.txt"}},
            },
            required=["path"],
            handler=lambda args, state: files.read(args["path"]),
        )
    )

    reg.register(
        Tool(
            name="memory.store",
            description=(
                "Remember a fact for later tasks (company identifiers, account details, "
                "context discovered while working)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Entity key, e.g. 'Acme Corporation'."},
                    "value": {"type": "object", "description": "Structured value to remember."},
                },
            },
            required=["key", "value"],
            handler=lambda args, state: memory.store_item(args["key"], args.get("value"), state.task_id),
        )
    )

    reg.register(
        Tool(
            name="memory.retrieve",
            description="Recall previously stored facts for a key.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string", "description": "Entity key to look up."}},
            },
            required=["key"],
            handler=lambda args, state: memory.retrieve(args["key"]),
        )
    )

    reg.register(
        Tool(
            name="browser.open",
            description=f"Open a page in the local sandbox at {config.SANDBOX_URL}. External URLs are blocked.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Sandbox URL, absolute or root-relative."}},
            },
            required=["url"],
            handler=lambda args, state: browser_dispatch("browser.open", args, state.task_id),
        )
    )

    reg.register(
        Tool(
            name="browser.click",
            description="Click an element in the current sandbox page.",
            parameters={
                "type": "object",
                "properties": {"selector": {"type": "string", "description": "CSS selector to click."}},
            },
            required=["selector"],
            handler=lambda args, state: browser_dispatch("browser.click", args, state.task_id),
        )
    )

    reg.register(
        Tool(
            name="browser.type",
            description="Type text into an element in the current sandbox page.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string", "description": "CSS selector to type into."},
                    "text": {"type": "string", "description": "Text to enter."},
                },
            },
            required=["selector", "text"],
            handler=lambda args, state: browser_dispatch("browser.type", args, state.task_id),
        )
    )

    reg.register(
        Tool(
            name="browser.read",
            description="Read the visible text of the current sandbox page.",
            parameters={"type": "object", "properties": {}},
            required=[],
            handler=lambda args, state: browser_dispatch("browser.read", args, state.task_id),
        )
    )

    reg.register(
        Tool(
            name="browser.screenshot",
            description="Capture a PNG screenshot of the current sandbox page as evidence.",
            parameters={"type": "object", "properties": {}},
            required=[],
            handler=lambda args, state: browser_dispatch("browser.screenshot", args, state.task_id),
        )
    )

    reg.register(
        Tool(
            name="browser.go_back",
            description="Navigate back to the previous sandbox page.",
            parameters={"type": "object", "properties": {}},
            required=[],
            handler=lambda args, state: browser_dispatch("browser.go_back", args, state.task_id),
        )
    )

    reg.register(
        Tool(
            name="approval.request",
            description="Explicitly ask the operator to approve a high-impact action before it is executed.",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Why approval is needed."},
                    "payload": {"type": "object", "description": "Details of the proposed action."},
                },
            },
            required=["summary"],
            handler=lambda args, state: approval.request(args.get("summary") or "Approval required.", args.get("payload") or {}),
        )
    )

    reg.register(
        Tool(
            name="task.complete",
            description=(
                "Declare that the requested outcome is ready for review. This does NOT complete "
                "the task: the runtime independently re-reads the persisted state and only "
                "verification can mark the task completed."
            ),
            parameters={
                "type": "object",
                "properties": {"summary": {"type": "string", "description": "Operational summary of what was achieved."}},
            },
            required=[],
            handler=lambda args, state: ok(
                "task.complete",
                args.get("summary") or "Outcome ready for independent verification.",
                data=args,
            ),
        )
    )

    reg.register(
        Tool(
            name="task.fail",
            description="Stop safely with a reason when the request cannot be satisfied.",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "Why the task cannot proceed."}},
            },
            required=["reason"],
            handler=lambda args, state: fail("task.fail", args.get("reason") or "Stopped safely.", "UNKNOWN_ERROR", False),
        )
    )

    reg.register(
        Tool(
            name="task.clarify",
            description=(
                "Ask the operator a clarifying question when the request is ambiguous. Use this "
                "instead of guessing when several materially different actions are possible."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "A single, specific question for the operator."}
                },
            },
            required=["question"],
            handler=lambda args, state: ok("task.clarify", args.get("question") or "Need clarification.", data=args),
        )
    )

    return reg


registry = _build_default_registry()

#: Public schema snapshot (used by /api/tools and by tests).
TOOL_SCHEMAS: list[dict[str, Any]] = registry.schemas()
TOOL_SPECS: list[dict[str, Any]] = registry.flat_specs()
REQUIRED_ARGUMENTS: dict[str, list[str]] = {
    tool.name: list(tool.required) for tool in registry._tools.values()
}


def execute_tool(name: str, arguments: dict[str, Any], state: TaskState) -> ToolResult:
    return registry.execute(name, arguments, state)


async def execute_tool_async(name: str, arguments: dict[str, Any], state: TaskState) -> ToolResult:
    return await registry.execute_async(name, arguments, state)


__all__ = [
    "Tool",
    "ToolRegistry",
    "ValidationError",
    "registry",
    "validate_arguments",
    "execute_tool",
    "execute_tool_async",
    "TOOL_SCHEMAS",
    "TOOL_SPECS",
    "REQUIRED_ARGUMENTS",
]
