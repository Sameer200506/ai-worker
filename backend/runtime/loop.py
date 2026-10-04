"""Autonomous TaskPilot execution loop.

    goal -> LLM -> structured tool call -> safety layer -> tool -> observation
        -> back to the LLM with the real tool result -> ... -> verification

The model owns every business decision: which tool, which arguments, whether to adapt, when
the outcome looks ready. This runtime owns everything a model must not control: the tool
allowlist, argument validation, URL/path boundaries, approval gating, retry ceilings,
cancellation, context size, and the independent verification that decides completion.

There is no `if intent == ...` anywhere in this file.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from typing import Any

from backend import config
from backend.agent import llm, safety
from backend.agent.llm import LLMDecision, LLMError
from backend.agent.prompts import SYSTEM_PROMPT
from backend.database.models import LLMCall, TaskState, ToolError, ToolResult
from backend.database.store import audit, load_task, save_task, store_memory
from backend.permissions.policies import requires_approval
from backend.recovery.manager import RecoveryPolicy, evaluate as evaluate_recovery, recovery_notice
from backend.runtime.events import emit
from backend.tools.registry import registry
from backend.verification.engine import verify

TERMINAL = {"completed", "failed", "cancelled", "awaiting_clarification"}

# Session tokens must never reach task evidence, summaries or the UI.
SECRET_ARTIFACT_TYPES = {"sandbox_token", "finance_token"}

#: Control tools. They are interpreted by the runtime, not dispatched as side effects.
CONTROL_TOOLS = {"task.complete", "task.fail", "task.clarify"}


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
def _progress(state: TaskState) -> int:
    if state.status == "completed":
        return 100
    if state.verification_status == "passed":
        return 98
    return min(95, max(8, 8 + len(state.completed_actions) * 18))


# ---------------------------------------------------------------------------
# Message history
# ---------------------------------------------------------------------------
def initial_messages(goal: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]


def _state_digest(state: TaskState) -> str:
    lines = [f"Goal: {state.goal}", f"Status so far: {state.status}."]
    if state.completed_actions:
        done = ", ".join(f"{item['action']}" for item in state.completed_actions[-6:])
        lines.append(f"Actions completed: {done}.")
    if state.failed_actions:
        failures = ", ".join(
            f"{item['action']} -> {(item.get('error') or {}).get('type', 'UNKNOWN_ERROR')}"
            for item in state.failed_actions[-4:]
        )
        lines.append(f"Actions that failed: {failures}.")
    if state.artifacts.get("invoice_id"):
        lines.append(f"Invoice in context: {state.artifacts['invoice_id']}.")
    if state.artifacts.get("customer_id"):
        lines.append(f"Customer in context: {state.artifacts['customer_id']}.")
    if state.artifacts.get("finance_record"):
        record = state.artifacts["finance_record"]
        lines.append(f"Finance record created: {record.get('record_id')}.")
    if state.retry_count:
        lines.append(f"Recovery attempts used: {state.retry_count}/{config.MAX_RETRIES}.")
    if state.verification_status != "pending":
        lines.append(f"Last verification: {state.verification_status}.")
    return "STATE SUMMARY (compressed history follows)\n" + "\n".join(lines)


def build_context(messages: list[dict[str, Any]], max_messages: int, digest: str | None = None) -> list[dict[str, Any]]:
    """Return a bounded, *structurally valid* context window.

    Providers reject a ``tool`` message whose originating ``tool_calls`` message was cut
    away, and vice versa. Blocks are therefore added whole: an assistant tool call and the
    tool results answering it always travel together.
    """
    if len(messages) <= max_messages:
        return messages

    head = [messages[0]] if messages and messages[0].get("role") == "system" else []
    if len(messages) > 1 and messages[1].get("role") == "user":
        head.append(messages[1])

    body = messages[len(head) :]
    blocks: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []
    answered_ids: set[str] = set()

    for message in body:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                answered_ids.add(str(call.get("id", "")))
            pending.append(message)
            continue
        if role == "tool":
            if pending:
                pending.append(message)
            continue
        if pending:
            blocks.append(pending)
            pending = []
        blocks.append([message])
    if pending:
        blocks.append(pending)

    budget = max(2, max_messages - len(head))
    tail: list[dict[str, Any]] = []
    for block in reversed(blocks):
        if len(tail) + len(block) > budget:
            break
        tail = block + tail

    context = list(head)
    if digest:
        context.append({"role": "user", "content": digest})
    context.extend(tail)
    return context


def _append_assistant_call(state: TaskState, decision: LLMDecision, call_id: str) -> None:
    state.messages.append(
        {
            "role": "assistant",
            "content": decision.text or decision.summary or "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": decision.tool or "task.unknown",
                        "arguments": json.dumps(decision.arguments or {}, default=str),
                    },
                }
            ],
        }
    )


def _append_tool_result(state: TaskState, call_id: str, result: ToolResult) -> None:
    payload: dict[str, Any] = {
        "success": result.success,
        "observation": result.observation,
        "data": result.data,
        "latency_ms": result.latency_ms,
    }
    if result.error:
        payload["error"] = result.error.model_dump()
    state.messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": result.tool,
            "content": json.dumps(payload, default=str),
        }
    )


def _call_id(state: TaskState, decision: LLMDecision) -> str:
    return decision.tool_call_id or f"call_{state.step_count}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Observation capture
# ---------------------------------------------------------------------------
def _apply_artifacts(state: TaskState, result: ToolResult, arguments: dict[str, Any]) -> None:
    if not result.success:
        return
    data = result.data or {}

    if result.tool == "company.login":
        for item in result.artifacts:
            if item.get("type") == "sandbox_token":
                state.sandbox_token = item.get("value")
            if item.get("type") == "finance_token":
                state.finance_token = item.get("value")
        state.last_error = None
    elif result.tool == "company.search_invoice":
        state.artifacts["invoice_candidates"] = data.get("invoices") or []
    elif result.tool == "company.get_invoice":
        state.artifacts["invoice"] = data
        state.artifacts["invoice_id"] = data.get("invoice_id")
    elif result.tool == "company.search_customer":
        state.artifacts["customer_candidates"] = data.get("customers") or []
    elif result.tool == "company.get_customer":
        state.artifacts["customer"] = data
        state.artifacts["customer_id"] = data.get("customer_id")
    elif result.tool == "company.update_customer":
        state.artifacts["customer_updated"] = data
        state.artifacts["customer"] = data
        state.artifacts["customer_id"] = data.get("customer_id")
        # Expected values come from the *submitted intent*, never from the tool's echo,
        # so verification compares intent against independently re-read state.
        for field_name in ("status", "email", "contact"):
            if field_name in arguments:
                state.artifacts[f"requested_{field_name}"] = arguments[field_name]
    elif result.tool == "company.create_finance_record":
        state.artifacts["finance_record"] = data
        state.pending_action = None
        state.last_error = None
    elif result.tool == "company.get_finance_record":
        state.artifacts["finance_read"] = data

    for item in result.artifacts:
        if item.get("type") in SECRET_ARTIFACT_TYPES:
            continue
        if item.get("type") and item.get("value") is not None:
            state.evidence.append(
                {
                    "kind": item["type"],
                    "value": item["value"],
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )


def observe(
    state: TaskState,
    decision: LLMDecision | Any,
    result: ToolResult,
    step_id: str,
    latency_ms: int | None = None,
) -> None:
    action_name = getattr(decision, "tool", None) or getattr(decision, "action", str(decision))
    arguments = getattr(decision, "arguments", {}) or {}
    state.observations.append(
        {
            "step": step_id,
            "tool": result.tool,
            "success": result.success,
            "observation": result.observation,
            "latency_ms": latency_ms if latency_ms is not None else result.latency_ms,
            "data": result.data,
        }
    )
    if result.success:
        state.completed_actions.append(
            {"action": action_name, "arguments": arguments, "observation": result.observation}
        )
        _apply_artifacts(state, result, arguments)
        state.last_error = None
    else:
        error = result.error.model_dump() if result.error else {"type": "UNKNOWN_ERROR", "recoverable": False}
        state.failed_actions.append({"action": action_name, "arguments": arguments, "error": error})
        state.last_error = error
    audit(
        task_id=state.task_id,
        agent_step=step_id,
        tool=result.tool,
        arguments=_redact(arguments),
        result=_redact(result.model_dump()),
        success=result.success,
        error=result.error.model_dump() if result.error else None,
        retry_count=state.retry_count,
        latency_ms=result.latency_ms,
        provider=state.llm_provider,
        model=state.llm_model,
    )


def _redact(payload: Any) -> Any:
    """Strip credentials and session tokens before anything is persisted or shown."""
    if isinstance(payload, dict):
        # Session tokens travel inside artifacts as {"type": ..., "value": ...}
        if payload.get("type") in SECRET_ARTIFACT_TYPES:
            return {"type": payload["type"], "value": "[redacted]"}
        cleaned = {}
        for key, value in payload.items():
            if key in {"password", "username", "token", "finance_token", "sandbox_token"}:
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = _redact(value)
        return cleaned
    if isinstance(payload, list):
        return [_redact(item) for item in payload]
    return payload


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def _format_amount(amount: Any) -> str:
    try:
        return f"₹{int(amount):,}"
    except (TypeError, ValueError):
        return str(amount)


def _format_date(value: Any) -> str:
    try:
        return date.fromisoformat(str(value)).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return str(value)


def _approval_payload(state: TaskState, decision: LLMDecision) -> dict[str, Any]:
    invoice = state.artifacts.get("invoice") or {}
    arguments = decision.arguments or {}
    impact: dict[str, Any] = {}
    if decision.tool == "company.create_finance_record":
        impact = {
            "Action": "Create finance record",
            "Invoice": arguments.get("invoice_id") or invoice.get("invoice_id"),
            "Company": arguments.get("company") or invoice.get("company"),
            "Amount": _format_amount(arguments.get("amount") if arguments.get("amount") is not None else invoice.get("amount")),
            "Due date": _format_date(arguments.get("due_date") or invoice.get("due_date")),
        }
    elif decision.tool == "company.update_customer":
        customer_id = arguments.get("customer_id")
        name = (state.artifacts.get("customer") or {}).get("name")
        if not name and customer_id:
            for candidate in state.artifacts.get("customer_candidates") or []:
                if candidate.get("customer_id") == customer_id:
                    name = candidate.get("name")
                    break
        impact = {
            "Action": "Update customer",
            "Customer ID": customer_id,
            "Name": name,
            "Requested change": {key: value for key, value in arguments.items() if key != "customer_id"},
        }
    return {
        "action": decision.tool,
        "arguments": arguments,
        "summary": decision.summary or f"{decision.tool} needs your approval before it can run.",
        "impact": impact,
    }


def _recovery_note(state: TaskState) -> str:
    if state.recovery_count == 0 and not state.failed_actions:
        return "No recovery was required."
    errors: list[str] = []
    for item in state.failed_actions:
        error_type = (item.get("error") or {}).get("type")
        if error_type and error_type not in errors:
            errors.append(error_type)
    if errors:
        return (
            f"Encountered {' and '.join(errors)}. The runtime classified the failure as recoverable, "
            f"the agent chose a mitigation and used {state.recovery_count} of {config.MAX_RETRIES} "
            "permitted recovery attempts; subsequent actions succeeded."
        )
    return f"A recoverable failure was handled automatically using {state.recovery_count} recovery attempt(s)."


#: Human-readable labels for verification fields.
CHECK_LABELS = {
    "invoice_id": "Invoice ID",
    "company": "Company",
    "amount": "Amount",
    "due_date": "Due date",
    "record_id": "Finance record",
    "customer_id": "Customer ID",
    "status": "Status",
    "email": "Email",
    "contact": "Contact",
    "tools_executed": "Tools executed",
    "failed_actions": "Failed actions",
}


def _verification_block(state: TaskState) -> str:
    if not state.verification or not state.verification.checks:
        return "No verification record."
    return "\n".join(
        f"{'✓' if check.passed else '✗'} {CHECK_LABELS.get(check.field, check.field)}"
        for check in state.verification.checks
    )


def _evidence_block(state: TaskState) -> str:
    lines: list[str] = []
    for item in state.evidence:
        kind, value = item.get("kind"), item.get("value")
        text: str | None = None
        if kind == "finance_record" and isinstance(value, dict):
            text = f"Finance record {value.get('record_id')} for invoice {value.get('invoice_id')}"
        elif kind == "invoice" and isinstance(value, dict):
            text = f"Invoice {value.get('invoice_id')} from {value.get('company')}"
        elif kind == "customer" and isinstance(value, dict):
            text = f"Customer {value.get('customer_id')} ({value.get('name')})"
        elif kind == "screenshot" and value:
            text = f"Screenshot {value}"
        if text and text not in lines:
            lines.append(text)
    return "\n".join(lines) if lines else "No artifacts were captured."


def _final_summary(state: TaskState) -> str:
    verification = _verification_block(state)
    recovery = _recovery_note(state)
    evidence = _evidence_block(state)
    header = "Task completed successfully. Independent verification passed.\n"

    record = state.artifacts.get("finance_record") or {}
    invoice = state.artifacts.get("invoice") or {}
    if record:
        return (
            f"{header}\nInvoice:\n{record.get('invoice_id') or invoice.get('invoice_id') or 'N/A'}\n\n"
            f"Company:\n{record.get('company') or invoice.get('company') or 'N/A'}\n\n"
            f"Amount:\n{_format_amount(record.get('amount'))}\n\n"
            f"Due date:\n{_format_date(record.get('due_date'))}\n\n"
            f"Finance record:\n{record.get('record_id') or 'N/A'}\n\n"
            f"Verification:\n{verification}\n\n"
            f"Recovery:\n{recovery}\n\n"
            f"Evidence:\n{evidence}\n"
        )

    customer = state.artifacts.get("customer_updated") or state.artifacts.get("customer") or {}
    if customer:
        changes = ", ".join(
            f"{key.replace('requested_', '')}={state.artifacts[key]}"
            for key in ("requested_status", "requested_email", "requested_contact")
            if state.artifacts.get(key)
        )
        return (
            f"{header}\nCustomer:\n{customer.get('name') or 'N/A'}\n\n"
            f"Customer ID:\n{customer.get('customer_id') or 'N/A'}\n\n"
            f"Requested change:\n{changes or 'n/a'}\n\n"
            f"Current status:\n{customer.get('status') or 'N/A'}\n\n"
            f"Verification:\n{verification}\n\n"
            f"Recovery:\n{recovery}\n\n"
            f"Evidence:\n{evidence}\n"
        )

    return f"{header}\nVerification:\n{verification}\n\nRecovery:\n{recovery}\n\nEvidence:\n{evidence}\n"


# ---------------------------------------------------------------------------
# Terminal handlers
# ---------------------------------------------------------------------------
def _fail(state: TaskState, reason: str, error_type: str = "TASK_FAILED") -> None:
    state.status = "failed"
    state.summary = reason
    state.last_error = {"type": error_type, "recoverable": False, "message": reason}
    emit(state.task_id, "task.failed", message=reason, error_type=error_type)


def _stop_requested(task_id: str) -> bool:
    """Whether an operator asked to stop, judged from the persisted state.

    Read fresh from the store on every step so the running loop can never overwrite a
    cancellation issued through the API.
    """
    persisted = load_task(task_id)
    if persisted is None:
        return False
    return bool(persisted.cancel_requested) or persisted.status == "cancelled"


def _apply_stop(state: TaskState) -> None:
    state.cancel_requested = True
    state.status = "cancelled"
    state.pending_action = None
    if not state.summary:
        state.summary = "Cancelled by the operator."
    emit(state.task_id, "task.cancelled", message=state.summary)


def _persist(state: TaskState) -> None:
    """Save task state, without ever clobbering an operator cancellation."""
    if state.status not in {"cancelled", "failed"} and _stop_requested(state.task_id):
        _apply_stop(state)
    save_task(state)


async def _run_verification(state: TaskState) -> bool:
    """Independent verification. Only a passing result can complete the task."""
    emit(state.task_id, "verification.started", message="Independently verifying the persisted outcome.")
    result = verify(state)
    state.verification = result
    state.verification_status = "passed" if result.verified else "failed"
    emit(
        state.task_id,
        "verification.completed",
        message=result.observation,
        verified=result.verified,
        checks=[check.model_dump() for check in result.checks],
    )
    if result.verified:
        state.status = "completed"
        state.progress = 100
        state.summary = _final_summary(state)
        emit(
            state.task_id,
            "task.completed",
            message="Task completed successfully.",
            summary=state.summary,
            evidence=state.evidence,
        )
        return True
    state.status = "failed"
    state.summary = result.observation
    emit(state.task_id, "task.failed", message=result.observation, error_type="VERIFICATION_FAILED")
    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def run_task(task_id: str, state: TaskState | None = None) -> TaskState:
    if state is None:
        loaded = load_task(task_id)
        if not loaded:
            raise RuntimeError(f"task {task_id} not found")
        state = loaded

    now_iso = datetime.now(timezone.utc).isoformat()
    if not state.created_at:
        state.created_at = now_iso
    state.updated_at = now_iso
    state.plan = None  # plans are a model product, never a runtime script
    state.degraded = False

    if not state.messages:
        state.messages = initial_messages(state.goal)
        emit(
            task_id,
            "task.started",
            message="Task started.",
            goal=state.goal,
            provider=config.LLM_PROVIDER or None,
            model=config.LLM_MODEL,
        )

    # Resume: an approved action that has not run yet.
    if state.pending_action and state.approval_status == "approved":
        await _resume_approved_action(state)

    # Resume: the operator answered a clarification question.
    if state.status == "running" and state.clarification_answer:
        state.messages.append({"role": "user", "content": f"OPERATOR ANSWER: {state.clarification_answer}"})
        state.clarification_answer = None
        state.clarification_question = None
        state.status = "running"
        emit(task_id, "task.resumed", message="Operator answered the clarification question. Continuing.")

    empty_turns = 0

    while state.status not in TERMINAL:
        if state.step_count >= config.MAX_STEPS:
            _fail(
                state,
                f"Stopped safely after reaching the maximum step limit of {config.MAX_STEPS}.",
                "MAX_STEPS_EXCEEDED",
            )
            break

        # Honour cancellation issued through the API while this loop was running.
        if _stop_requested(task_id):
            _apply_stop(state)
            _persist(state)
            break

        state.status = "running"
        state.step_count += 1
        state.progress = _progress(state)
        state.updated_at = datetime.now(timezone.utc).isoformat()
        step_id = f"step_{state.step_count:02d}"

        # --- 1. the model decides -----------------------------------------
        client = llm.build_client(state)
        state.llm_provider = client.name
        state.llm_model = getattr(client, "model", config.LLM_MODEL)
        context = build_context(state.messages, config.MAX_CONTEXT_MESSAGES, _state_digest(state))
        try:
            decision = await client.create_response(messages=context, tools=registry.schemas())
        except LLMError as exc:
            state.status = "failed"
            state.summary = f"{exc.error_type}: {exc}"
            state.last_error = {"type": exc.error_type, "recoverable": False, "message": str(exc)}
            emit(task_id, "task.failed", message=state.summary, error_type=exc.error_type)
            _persist(state)
            break
        except Exception as exc:  # noqa: BLE001 - never leak a traceback as a "result"
            state.status = "failed"
            state.summary = f"LLM error: {type(exc).__name__}: {exc}"
            state.last_error = {"type": "LLM_ERROR", "recoverable": False, "message": str(exc)}
            emit(task_id, "task.failed", message=state.summary, error_type="LLM_ERROR")
            _persist(state)
            break

        record = LLMCall(
            step=state.step_count,
            provider=decision.provider or client.name,
            model=decision.model or state.llm_model,
            latency_ms=decision.latency_ms,
            tool=decision.tool,
            finish_reason=decision.finish_reason,
            prompt_tokens=decision.prompt_tokens,
            completion_tokens=decision.completion_tokens,
            validation="pending",
            at=datetime.now(timezone.utc).isoformat(),
        )
        state.llm_calls.append(record)
        audit(
            task_id=task_id,
            agent_step=step_id,
            tool=decision.tool,
            arguments=_redact(decision.arguments),
            result={"summary": decision.summary, "text": decision.text},
            success=None,
            error=None,
            retry_count=state.retry_count,
            latency_ms=decision.latency_ms,
            provider=record.provider,
            model=record.model,
            kind="llm_decision",
        )

        state.current_action = decision.tool
        emit(
            task_id,
            "ai.decision",
            message=decision.summary or f"Proposed {decision.tool or 'no tool call'}.",
            tool=decision.tool,
            arguments=_redact(decision.arguments),
            step=step_id,
            provider=record.provider,
            model=record.model,
            latency_ms=decision.latency_ms,
        )

        # An operator may have cancelled while the model was thinking: do not act on it.
        if _stop_requested(task_id):
            _apply_stop(state)
            _persist(state)
            break

        # --- 2. no tool call: ask the model to commit to one --------------
        if not decision.tool:
            record.validation = "no_tool_call"
            state.messages.append({"role": "assistant", "content": decision.text or decision.summary or ""})
            state.messages.append(
                {
                    "role": "user",
                    "content": (
                        "No tool call was returned. Reply with exactly one call to a tool listed in this "
                        "request, or task.clarify if the request is ambiguous, or task.fail if it cannot be done."
                    ),
                }
            )
            empty_turns += 1
            if empty_turns >= 3:
                _fail(state, "The model stopped returning tool calls. The task was stopped safely.", "NO_TOOL_CALL")
                break
            _persist(state)
            continue
        empty_turns = 0

        call_id = _call_id(state, decision)

        # --- 3. control tools ---------------------------------------------
        if decision.tool == "task.clarify":
            record.validation = "clarify"
            question = (decision.arguments or {}).get("question") or decision.summary or "I need clarification."
            state.clarification_question = question
            state.status = "awaiting_clarification"
            state.summary = question
            _append_assistant_call(state, decision, call_id)
            _append_tool_result(
                state, call_id, ToolResult(success=True, tool="task.clarify", observation=question, data=decision.arguments)
            )
            emit(task_id, "task.waiting_for_user", message=question, clarification=True)
            _persist(state)
            break

        if decision.tool == "task.fail":
            record.validation = "fail"
            reason = (decision.arguments or {}).get("reason") or decision.summary or "The task cannot proceed."
            _append_assistant_call(state, decision, call_id)
            _append_tool_result(
                state,
                call_id,
                ToolResult(
                    success=False,
                    tool="task.fail",
                    observation=reason,
                    error=ToolError(type="TASK_FAILED", recoverable=False, message=reason),
                ),
            )
            _fail(state, reason, "TASK_FAILED")
            _persist(state)
            break

        if decision.tool == "task.complete":
            record.validation = "verification_requested"
            summary_hint = (decision.arguments or {}).get("summary") or decision.summary or ""
            _append_assistant_call(state, decision, call_id)
            _append_tool_result(
                state,
                call_id,
                ToolResult(
                    success=True,
                    tool="task.complete",
                    observation=f"Outcome submitted for independent verification. {summary_hint}".strip(),
                ),
            )
            await _run_verification(state)
            _persist(state)
            break

        # --- 4. deterministic safety layer ---------------------------------
        verdict = safety.evaluate(decision, state)
        record.validation = "blocked" if not verdict.allowed else "approved_or_allowed"
        if not verdict.allowed:
            call_id = _call_id(state, decision)
            _append_assistant_call(state, decision, call_id)
            result = ToolResult(
                success=False,
                tool=decision.tool or "unknown",
                observation=safety.blocked_summary(verdict.error or "Rejected."),
                error=ToolError(
                    type=verdict.error_type,
                    recoverable=verdict.recoverable,
                    message=verdict.error,
                ),
            )
            _append_tool_result(state, call_id, result)
            observe(state, decision, result, step_id)
            emit(
                task_id,
                "tool.failed",
                message=result.observation,
                tool=result.tool,
                error=result.error.model_dump(),
                step=step_id,
                blocked=True,
            )
            if not verdict.recoverable:
                _fail(state, f"Blocked by the safety layer: {verdict.error}", verdict.error_type)
                break
            # Recoverable: the model sees the rejection and can correct itself.
            if safety.retries_exhausted(state):
                _fail(
                    state,
                    f"Blocked repeatedly by the safety layer and the retry limit of {config.MAX_RETRIES} was reached.",
                    verdict.error_type,
                )
                break
            state.retry_count += 1
            state.recovery_count += 1
            _persist(state)
            continue

        # --- 5. human approval gate ----------------------------------------
        if requires_approval(decision.tool, state.demo_mode, state.approval_status):
            state.status = "awaiting_approval"
            state.approval_status = "pending"
            state.pending_action = {
                "action": decision.tool,
                "arguments": decision.arguments,
                "summary": decision.summary,
                "step": step_id,
            }
            state.approval_payload = _approval_payload(state, decision)
            _append_assistant_call(state, decision, call_id)
            _append_tool_result(
                state,
                call_id,
                ToolResult(
                    success=False,
                    tool=decision.tool,
                    observation="Execution paused: this action requires human approval.",
                    data=state.approval_payload,
                    error=ToolError(type="APPROVAL_REQUIRED", recoverable=True, message="Awaiting human approval."),
                ),
            )
            emit(
                task_id,
                "approval.required",
                message=state.approval_payload["summary"],
                payload=state.approval_payload,
                step=step_id,
            )
            _persist(state)
            break

        # --- 6. bounded tool execution -------------------------------------
        emit(
            task_id,
            "tool.started",
            message=decision.summary or f"Calling {decision.tool}.",
            tool=decision.tool,
            arguments=_redact(decision.arguments),
            step=step_id,
        )
        result = await registry.execute_async(decision.tool, decision.arguments, state)
        _append_assistant_call(state, decision, call_id)
        _append_tool_result(state, call_id, result)
        observe(state, decision, result, step_id)

        emit(
            task_id,
            "observation.created",
            message=result.observation,
            tool=result.tool,
            success=result.success,
            data=_redact(result.data),
            step=step_id,
            latency_ms=result.latency_ms,
        )

        if result.success:
            emit(
                task_id,
                "tool.completed",
                message=result.observation,
                tool=result.tool,
                data=_redact(result.data),
                step=step_id,
                latency_ms=result.latency_ms,
            )
            state.retry_count = state.retry_count if result.tool != "company.login" else 0
            if result.tool in {"company.get_invoice", "company.get_customer"}:
                key = (result.data or {}).get("company") or (result.data or {}).get("name")
                if key:
                    store_memory(key, result.data, state.task_id)
                    state.memory.append({"key": key, "value": result.data})
            if result.tool == "company.login" and any(
                (item.get("error") or {}).get("type") in {"AUTH_EXPIRED", "AUTH_FAILED"}
                for item in state.failed_actions
            ):
                emit(task_id, "recovery.completed", message="Authentication restored; the agent can retry.")
            _persist(state)
            continue

        emit(
            task_id,
            "tool.failed",
            message=result.observation,
            tool=result.tool,
            error=result.error.model_dump() if result.error else None,
            step=step_id,
            latency_ms=result.latency_ms,
        )

        # --- 7. deterministic recovery policy ------------------------------
        policy: RecoveryPolicy = evaluate_recovery(state, result)
        if policy.error_type != "OK":
            emit(
                task_id,
                "recovery.started",
                message=f"{result.tool} failed with {policy.error_type}. {policy.hint}",
                strategy="retry" if policy.retry_allowed else "stop",
                error_type=policy.error_type,
                recoverable=policy.recoverable,
                retries_remaining=policy.retries_remaining,
            )
        if not policy.retry_allowed:
            if policy.terminal_reason and policy.error_type == "OK":  # pragma: no cover
                policy.terminal_reason = "Stopped safely."
            _fail(
                state,
                policy.terminal_reason or f"{result.tool} failed: {policy.error_type}",
                policy.error_type,
            )
            _persist(state)
            break

        state.retry_count += 1
        state.recovery_count += 1
        state.messages.append(recovery_notice(policy, result.tool))
        emit(
            task_id,
            "recovery.attempted",
            message=f"Recovery attempt {state.recovery_count} used by the agent.",
            retries_used=state.recovery_count,
        )
        _persist(state)

    _persist(state)
    return state


async def _resume_approved_action(state: TaskState) -> None:
    """Execute the action the operator just approved, then hand control back to the model."""
    pending = state.pending_action or {}
    tool = pending.get("action")
    arguments = pending.get("arguments") or {}
    step_id = pending.get("step") or f"step_{state.step_count:02d}_approved"

    state.pending_action = None
    state.status = "running"

    emit(state.task_id, "tool.started", message=f"Executing approved action: {tool}.", tool=tool, arguments=_redact(arguments), step=step_id)
    result = await registry.execute_async(tool, arguments, state)
    decision = LLMDecision(tool=tool, arguments=arguments, summary=f"Approved action {tool} executed.")
    call_id = _call_id(state, decision)
    _append_assistant_call(state, decision, call_id)
    _append_tool_result(state, call_id, result)
    observe(state, decision, result, step_id)

    emit(
        state.task_id,
        "observation.created",
        message=result.observation,
        tool=result.tool,
        success=result.success,
        step=step_id,
    )
    if result.success:
        emit(state.task_id, "tool.completed", message=result.observation, tool=result.tool, data=_redact(result.data), step=step_id)
        _persist(state)
        return

    emit(
        state.task_id,
        "tool.failed",
        message=result.observation,
        tool=result.tool,
        error=result.error.model_dump() if result.error else None,
        step=step_id,
    )

    # The approved action failed: apply exactly the same deterministic recovery policy as
    # any other tool failure, so the operator's approval does not bypass the failure path.
    policy = evaluate_recovery(state, result)
    emit(
        state.task_id,
        "recovery.started",
        message=f"{result.tool} failed with {policy.error_type}. {policy.hint}",
        strategy="retry" if policy.retry_allowed else "stop",
        error_type=policy.error_type,
        recoverable=policy.recoverable,
        retries_remaining=policy.retries_remaining,
    )
    if not policy.retry_allowed:
        _fail(
            state,
            policy.terminal_reason or f"{result.tool} failed: {policy.error_type}",
            policy.error_type,
        )
        _persist(state)
        return

    state.retry_count += 1
    state.recovery_count += 1
    state.messages.append(recovery_notice(policy, result.tool))
    emit(
        state.task_id,
        "recovery.attempted",
        message=f"Recovery attempt {state.recovery_count} used by the agent.",
        retries_used=state.recovery_count,
    )
    _persist(state)
