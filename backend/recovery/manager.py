"""Deterministic failure classification and recovery policy.

The *runtime* decides whether a failure is retryable and how many attempts remain. The *model*
decides what to do next. This module never picks a business action; it produces an operational
classification plus a bounded mitigation hint that is surfaced to the model.

Failure taxonomy
----------------
AUTH_EXPIRED      authentication/session expired      recoverable -> re-authenticate
NETWORK_ERROR     transient transport failure         recoverable -> retry
TIMEOUT           tool exceeded its execution budget  recoverable -> retry
INVALID_DATA      input could not be processed        not recoverable
PERMISSION_DENIED blocked by policy                   not recoverable
UNKNOWN_TOOL      tool does not exist                 not recoverable (safety)
NOT_FOUND         the requested record does not exist not recoverable
UNKNOWN_ERROR     anything else                       recoverable once
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.config import MAX_RETRIES
from backend.database.models import TaskState, ToolResult

RECOVERABLE = {"AUTH_EXPIRED", "AUTH_FAILED", "NETWORK_ERROR", "TIMEOUT", "UNKNOWN_ERROR"}
FATAL = {"INVALID_DATA", "PERMISSION_DENIED", "UNKNOWN_TOOL", "NOT_FOUND", "SANDBOX_ERROR"}

#: Operational hint shown to the model. It is guidance, not an executed action:
#: the runtime never calls a tool on the model's behalf.
MITIGATION_HINTS = {
    "AUTH_EXPIRED": "The company session expired. Call company.login with no arguments, then repeat the failed action.",
    "AUTH_FAILED": "Authentication was rejected. Call company.login with no arguments, then repeat the failed action.",
    "NETWORK_ERROR": "A transient transport error occurred. Repeat the failed action once.",
    "TIMEOUT": "The tool exceeded its execution budget. Repeat the action once, or use a narrower read.",
    "UNKNOWN_ERROR": "An unexpected error occurred. Re-read the record and try a narrower action.",
    "INVALID_DATA": "The values were rejected as invalid. Re-read the source record and use exactly the values it returned.",
    "NOT_FOUND": "The requested record does not exist. Do not invent it; either search again with different filters or stop with task.fail.",
    "PERMISSION_DENIED": "The action is blocked by policy. Do not attempt a workaround; stop with task.fail.",
    "UNKNOWN_TOOL": "That tool does not exist. Choose from the tools provided in this request.",
}


def classify(result: ToolResult) -> str:
    if result.success:
        return "OK"
    return (result.error.type if result.error else "UNKNOWN_ERROR")


@dataclass
class RecoveryPolicy:
    """Deterministic verdict about a failure."""

    error_type: str
    recoverable: bool
    retry_allowed: bool
    retries_remaining: int
    hint: str
    terminal_reason: str | None = None

    @property
    def should_continue(self) -> bool:
        return self.retry_allowed


def evaluate(state: TaskState, result: ToolResult) -> RecoveryPolicy:
    """Classify a failed tool result and decide whether the loop may continue."""
    error_type = classify(result)
    declared = bool(result.error and result.error.recoverable)
    recoverable = declared and error_type in RECOVERABLE
    retries_remaining = max(0, MAX_RETRIES - state.retry_count)
    hint = MITIGATION_HINTS.get(error_type, MITIGATION_HINTS["UNKNOWN_ERROR"])

    if error_type == "OK":
        return RecoveryPolicy(error_type, False, True, retries_remaining, hint)
    if not recoverable:
        return RecoveryPolicy(
            error_type=error_type,
            recoverable=False,
            retry_allowed=False,
            retries_remaining=retries_remaining,
            hint=hint,
            terminal_reason=result.observation,
        )
    if retries_remaining <= 0:
        return RecoveryPolicy(
            error_type=error_type,
            recoverable=True,
            retry_allowed=False,
            retries_remaining=0,
            hint=hint,
            terminal_reason=(
                f"Retry limit reached after {state.retry_count} recovery attempt(s) "
                f"while handling {error_type}. The task was stopped safely."
            ),
        )
    return RecoveryPolicy(
        error_type=error_type,
        recoverable=True,
        retry_allowed=True,
        retries_remaining=retries_remaining,
        hint=hint,
    )


def recovery_notice(policy: RecoveryPolicy, tool: str) -> dict[str, Any]:
    """The operational recovery message injected into the model's context."""
    return {
        "role": "user",
        "content": (
            f"SYSTEM RECOVERY NOTICE (deterministic runtime, not a tool result)\n"
            f"{tool} failed with {policy.error_type}.\n"
            f"{policy.hint}\n"
            f"Recovery attempts used: {MAX_RETRIES - policy.retries_remaining}/{MAX_RETRIES}."
        ),
    }
