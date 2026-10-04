"""Deterministic safety layer.

The model decides *what* to do. This module decides whether the model is *allowed* to do
it, and it is the last thing between a model proposal and a real side effect.

It never selects business actions. It only answers four questions:
1. Is this tool registered and permitted?
2. Are the arguments structurally valid and policy-compliant?
3. Does this call need human approval?
4. Is a retry still allowed?

Order matters: a proposal is validated *before* any tool handler runs, and approval is
checked *before* execution as well.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend import config
from backend.agent.llm import LLMDecision
from backend.database.models import TaskState
from backend.permissions.policies import requires_approval, validate_tool_name
from backend.tools.registry import registry


@dataclass
class SafetyVerdict:
    allowed: bool
    error: str | None = None
    error_type: str = "INVALID_DATA"
    recoverable: bool = True
    requires_approval: bool = False


def evaluate(decision: LLMDecision, state: TaskState) -> SafetyVerdict:
    """Validate a proposed tool call against every deterministic control."""
    if decision.validation_error:
        return SafetyVerdict(
            allowed=False,
            error=f"The model's tool arguments could not be parsed: {decision.validation_error}",
            error_type="INVALID_DATA",
            recoverable=True,
        )

    if not decision.tool:
        return SafetyVerdict(
            allowed=False,
            error="The model did not call a tool. Respond with a registered tool call.",
            error_type="INVALID_DATA",
            recoverable=True,
        )

    forbidden = validate_tool_name(decision.tool)
    if forbidden:
        # Not recoverable: a forbidden tool is a policy decision, not a transient fault.
        return SafetyVerdict(allowed=False, error=forbidden, error_type="PERMISSION_DENIED", recoverable=False)

    if decision.tool not in registry:
        return SafetyVerdict(
            allowed=False,
            error=f"Tool '{decision.tool}' is not registered.",
            error_type="UNKNOWN_TOOL",
            recoverable=True,
        )

    arguments: dict[str, Any] = decision.arguments or {}
    problem = registry.validate(decision.tool, arguments)
    if problem:
        kind = "PERMISSION_DENIED" if "restricted" in problem or "not allowed" in problem else "INVALID_DATA"
        return SafetyVerdict(
            allowed=False,
            error=problem,
            error_type=kind,
            # A bad URL/path is a policy boundary; a malformed argument is retryable.
            recoverable=kind != "PERMISSION_DENIED",
        )

    needs_approval = requires_approval(decision.tool, state.demo_mode, state.approval_status)
    return SafetyVerdict(allowed=True, requires_approval=needs_approval)


def retries_exhausted(state: TaskState) -> bool:
    return state.retry_count >= config.MAX_RETRIES


def blocked_summary(error: str) -> str:
    """Short operational line shown in the UI for a rejected proposal."""
    return f"Blocked by the safety layer: {error}"
