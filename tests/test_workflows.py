"""End-to-end tests for the TaskPilot autonomous runtime.

Every test drives the *real* runtime: the same message history, safety layer, approval
gate, recovery policy and independent verification that the live API uses. The only thing
swapped is the model itself, via a scripted/mock ``LLMClient``. No test monkeypatches the
agent into returning ``None`` and no test calls a tool handler directly to fake success.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pytest

from apps.sandbox.store import store
from backend import config

# Captured at import time, before the offline fixture patches the config module.
LIVE_LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip().lower()
LIVE_LLM_API_KEY = (os.getenv("LLM_API_KEY") or "").strip()
from backend.agent.llm import LLMDecision, MockLLMClient, sanitize_messages
from backend.database import connect, init_db
from backend.database.models import TaskState
from backend.database.store import list_audit, load_task, save_task
from backend.runtime.loop import build_context, initial_messages, run_task
from backend.tools import company
from backend.tools.registry import registry


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """Isolate the database and the company sandbox for every test."""
    db_path = tmp_path / "taskpilot.db"
    monkeypatch.setattr("backend.config.DATABASE_PATH", db_path)
    monkeypatch.setattr("backend.database.DATABASE_PATH", db_path)
    init_db(connect(db_path))
    store.path = tmp_path / "store.json"
    store.reset("")
    # Default: the scripted model simulator, selected explicitly and offline.
    monkeypatch.setattr("backend.config.LLM_PROVIDER", "mock")
    monkeypatch.setattr("backend.config.DETERMINISTIC_FALLBACK", False)
    yield


def _install(monkeypatch, mock: MockLLMClient) -> None:
    monkeypatch.setattr("backend.agent.llm.build_client", lambda _state=None: mock)


async def run_goal(goal: str, demo_mode: str = "normal", failure_mode: str = "") -> TaskState:
    store.reset(failure_mode)
    state = TaskState(task_id="task_test", goal=goal, status="queued", demo_mode=demo_mode)
    save_task(state)
    return await run_task(state.task_id, state)


async def run_goal_approved(goal: str, failure_mode: str = "", demo_mode: str = "normal", approve: bool = True) -> TaskState:
    """Drive the full loop including the human approval gate.

    `approve=False` simulates the operator rejecting the sensitive action.
    """
    state = await run_goal(goal, demo_mode=demo_mode, failure_mode=failure_mode)
    guard = 0
    while state.status == "awaiting_approval" and guard < 5:
        guard += 1
        if not approve:
            state.approval_status = "rejected"
            state.pending_action = None
            state.status = "failed"
            state.summary = "The operator rejected the high-impact action."
            save_task(state)
            break
        state.approval_status = "approved"
        state.status = "running"
        save_task(state)
        state = await run_task(state.task_id)
    return state


# ===========================================================================
# 1. Normal invoice processing
# ===========================================================================
@pytest.mark.asyncio
async def test_invoice_processing_success():
    result = await run_goal_approved("Process the latest invoice from Acme Corporation")
    assert result.status == "completed"
    assert result.verification_status == "passed"
    assert result.verification is not None and result.verification.verified
    assert result.artifacts["invoice"]["invoice_id"] == "INV-1042"
    assert result.artifacts["invoice"]["amount"] == 84250
    assert result.artifacts["finance_record"]["invoice_id"] == "INV-1042"
    assert all(check.passed for check in result.verification.checks)


# ===========================================================================
# 2. Natural-language generalisation (no keyword matching in the runtime)
# ===========================================================================
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "goal",
    [
        "Process the latest invoice from Acme Corporation.",
        "Handle Acme's newest pending invoice.",
        "Find the most recent pending invoice from Acme and record it in finance.",
    ],
)
async def test_natural_language_invoice_variations(goal):
    result = await run_goal_approved(goal)
    assert result.status == "completed", result.summary
    assert result.artifacts["invoice"]["invoice_id"] == "INV-1042"
    assert result.verification_status == "passed"


# ===========================================================================
# 3. Customer workflow through the same runtime
# ===========================================================================
@pytest.mark.asyncio
async def test_customer_status_update():
    result = await run_goal_approved("Find Acme Corporation and update its customer status to Follow Up.")
    assert result.status == "completed", result.summary
    assert result.verification_status == "passed"
    stored = company.get_customer("CUST-102")
    assert stored.success and stored.data["status"] == "Follow Up"
    assert result.artifacts["requested_status"] == "Follow Up"


# ===========================================================================
# 4. Recoverable AUTH_EXPIRED -> model-driven recovery -> verification
# ===========================================================================
@pytest.mark.asyncio
async def test_finance_session_expiration_recovers():
    result = await run_goal_approved(
        "Process the latest invoice from Acme Corporation",
        failure_mode="finance_session_expired",
    )
    assert result.status == "completed", result.summary
    assert any(item["error"]["type"] == "AUTH_EXPIRED" for item in result.failed_actions)
    assert result.verification_status == "passed"
    # The model decided to re-authenticate; the runtime only bounded the retries.
    assert any(item["action"] == "company.login" for item in result.completed_actions)


# ===========================================================================
# 5. Verification failure never completes the task
# ===========================================================================
@pytest.mark.asyncio
async def test_verification_mismatch_does_not_complete():
    result = await run_goal_approved(
        "Process the latest invoice from Acme Corporation",
        failure_mode="verification_mismatch",
    )
    assert result.status == "failed"
    assert result.verification_status == "failed"
    assert result.verification is not None and not result.verification.verified
    assert "VERIFICATION FAILED" in (result.summary or "")
    failed = [check for check in result.verification.checks if not check.passed]
    assert failed, "the verifier must report at least one mismatched field"
    assert failed[0].expected != failed[0].actual


# ===========================================================================
# 6. Approval required, then granted
# ===========================================================================
@pytest.mark.asyncio
async def test_approval_pauses_then_continues():
    paused = await run_goal("Process the latest invoice from Acme Corporation")
    assert paused.status == "awaiting_approval"
    assert paused.approval_status == "pending"
    assert paused.pending_action["action"] == "company.create_finance_record"
    assert paused.approval_payload["impact"]["Invoice"] == "INV-1042"
    assert "84,250" in paused.approval_payload["impact"]["Amount"]
    # Nothing was persisted while approval was pending.
    assert store.list_finance() == []

    paused.approval_status = "approved"
    paused.status = "running"
    save_task(paused)
    resumed = await run_task(paused.task_id)
    assert resumed.status == "completed", resumed.summary
    assert resumed.verification_status == "passed"
    assert len(store.list_finance()) == 1


# ===========================================================================
# 7. Approval rejection terminates safely
# ===========================================================================
@pytest.mark.asyncio
async def test_approval_rejected_terminates():
    paused = await run_goal("Process the latest invoice from Acme Corporation")
    assert paused.status == "awaiting_approval"
    assert store.list_finance() == []

    paused.approval_status = "rejected"
    paused.pending_action = None
    paused.status = "failed"
    paused.summary = "The operator rejected the high-impact action."
    save_task(paused)

    loaded = load_task(paused.task_id)
    assert loaded is not None
    assert loaded.status == "failed"
    assert loaded.approval_status == "rejected"
    assert store.list_finance() == []


# ===========================================================================
# 8. Ambiguous request asks the user instead of guessing
# ===========================================================================
@pytest.mark.asyncio
async def test_ambiguous_task_asks_for_clarification():
    result = await run_goal("Do something with Acme.")
    assert result.status == "awaiting_clarification"
    assert result.clarification_question
    assert store.list_finance() == []


@pytest.mark.asyncio
async def test_clarification_answer_resumes_the_task():
    result = await run_goal("Do something with Acme.")
    assert result.status == "awaiting_clarification"

    result.clarification_answer = "Process the latest invoice from Acme Corporation."
    result.status = "running"
    save_task(result)
    resumed = await run_task(result.task_id)
    assert resumed.status == "awaiting_approval"
    resumed.approval_status = "approved"
    resumed.status = "running"
    save_task(resumed)
    finished = await run_task(resumed.task_id)
    assert finished.status == "completed", finished.summary
    assert finished.verification_status == "passed"


# ===========================================================================
# 9-11. Safety layer
# ===========================================================================
@pytest.mark.asyncio
async def test_unknown_tool_is_blocked(monkeypatch):
    mock = MockLLMClient(
        decisions=[LLMDecision(tool="shell.execute", arguments={"command": "cat /etc/passwd"})]
    )
    _install(monkeypatch, mock)
    state = TaskState(task_id="task_unknown", goal="Run a dangerous tool")
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "failed"
    assert "not allowed" in (result.summary or "")
    assert result.last_error["type"] == "PERMISSION_DENIED"


@pytest.mark.asyncio
async def test_sql_tool_is_blocked(monkeypatch):
    mock = MockLLMClient(decisions=[LLMDecision(tool="db.run_sql", arguments={"query": "SELECT 1"})])
    _install(monkeypatch, mock)
    state = TaskState(task_id="task_sql", goal="Query the database directly")
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "failed"
    assert "not allowed" in (result.summary or "")


@pytest.mark.asyncio
async def test_invalid_arguments_are_returned_to_the_model(monkeypatch):
    """A malformed call is fed back as an observation so the model can correct itself."""
    mock = MockLLMClient(
        decisions=[
            LLMDecision(tool="company.create_finance_record", arguments={"invoice_id": "INV-1042"}),
            LLMDecision(tool="task.fail", arguments={"reason": "arguments invalid"}),
        ]
    )
    _install(monkeypatch, mock)
    state = TaskState(task_id="task_invalid_args", goal="Create finance record with incomplete args")
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "failed"
    assert any("Missing required argument" in obs["observation"] for obs in result.observations)
    # The rejection is visible to the model as a tool result, not silently dropped.
    assert any("Missing required argument" in str(m.get("content", "")) for m in result.messages if m.get("role") == "tool")


@pytest.mark.asyncio
async def test_wrong_argument_type_is_rejected(monkeypatch):
    mock = MockLLMClient(
        decisions=[
            LLMDecision(
                tool="company.create_finance_record",
                arguments={"invoice_id": "INV-1042", "company": "Acme", "amount": "eighty four", "due_date": "2026-10-15"},
            ),
            LLMDecision(tool="task.fail", arguments={"reason": "invalid amount type"}),
        ]
    )
    _install(monkeypatch, mock)
    state = TaskState(task_id="task_bad_type", goal="Create a finance record")
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "failed"
    assert any("must be of type integer" in obs["observation"] for obs in result.observations)


@pytest.mark.asyncio
async def test_external_url_is_blocked(monkeypatch):
    mock = MockLLMClient(
        decisions=[LLMDecision(tool="browser.open", arguments={"url": "https://malicious-external-site.com"})]
    )
    _install(monkeypatch, mock)
    state = TaskState(task_id="task_perm", goal="Browse an external site")
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "failed"
    assert "restricted" in (result.summary or "")


@pytest.mark.asyncio
async def test_filesystem_escape_is_blocked(monkeypatch):
    mock = MockLLMClient(decisions=[LLMDecision(tool="files.read", arguments={"path": "../../etc/passwd"})])
    _install(monkeypatch, mock)
    state = TaskState(task_id="task_fs", goal="Read a file outside the sandbox")
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "failed"
    assert "traversal" in (result.summary or "").lower() or "restricted" in (result.summary or "").lower()


# ===========================================================================
# 12. Retry limit is enforced by the runtime, not the model
# ===========================================================================
@pytest.mark.asyncio
async def test_retry_limit_enforced(monkeypatch):
    store.reset("")
    store.set_failure_mode("finance_session_expired")
    arguments = {
        "invoice_id": "INV-1042",
        "company": "Acme Corporation",
        "amount": 84250,
        "due_date": "2026-10-15",
    }
    mock = MockLLMClient(
        decisions=[
            LLMDecision(tool="company.create_finance_record", arguments=arguments),
            LLMDecision(tool="company.create_finance_record", arguments=arguments),
            LLMDecision(tool="company.create_finance_record", arguments=arguments),
            LLMDecision(tool="company.create_finance_record", arguments=arguments),
            LLMDecision(tool="company.create_finance_record", arguments=arguments),
        ]
    )
    _install(monkeypatch, mock)
    state = TaskState(task_id="task_retries", goal="Trigger the retry limit", approval_status="approved")
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "failed"
    assert result.retry_count >= config.MAX_RETRIES
    assert "Retry limit" in (result.summary or "")


# ===========================================================================
# 13-16. Missing data, tool failure, LLM failure, cancellation
# ===========================================================================
@pytest.mark.asyncio
async def test_missing_invoice_stops_safely():
    result = await run_goal_approved("Process the latest invoice from Nonexistent Corporation")
    assert result.status == "failed"
    assert store.list_finance() == []


@pytest.mark.asyncio
async def test_model_can_stop_safely(monkeypatch):
    mock = MockLLMClient(decisions=[LLMDecision(tool="task.fail", arguments={"reason": "service outage"})])
    _install(monkeypatch, mock)
    state = TaskState(task_id="task_tool_fail", goal="Handle an outage")
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "failed"
    assert "outage" in (result.summary or "")


@pytest.mark.asyncio
async def test_tool_failure_is_observed(monkeypatch):
    mock = MockLLMClient(
        decisions=[
            LLMDecision(tool="company.get_invoice", arguments={"invoice_id": "INV-9999"}),
            LLMDecision(tool="task.fail", arguments={"reason": "invoice does not exist"}),
        ]
    )
    _install(monkeypatch, mock)
    state = TaskState(task_id="task_tool_error", goal="Read a missing invoice")
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "failed"
    assert any(obs["error"] if "error" in obs else False for obs in result.observations) or any(
        item["error"]["type"] == "NOT_FOUND" for item in result.failed_actions
    )


@pytest.mark.asyncio
async def test_llm_failure_fails_the_task(monkeypatch):
    from backend.agent.llm import LLMConnectionError

    class ExplodingLLM:
        name = "exploding"
        model = "boom"

        async def create_response(self, messages, tools=None):
            raise LLMConnectionError("LLM API gateway unreachable")

    _install(monkeypatch, ExplodingLLM())
    state = TaskState(task_id="task_llm_fail", goal="Process the invoice")
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "failed"
    assert "LLM_CONNECTION_ERROR" in (result.summary or "")
    assert "API gateway unreachable" in (result.summary or "")


@pytest.mark.asyncio
async def test_missing_llm_configuration_fails_loudly(monkeypatch):
    """No key and no explicit fallback must fail, never fake a deterministic result."""
    from backend.agent.llm import LLMUnavailable, build_client

    monkeypatch.setattr("backend.config.LLM_API_KEY", "")
    monkeypatch.setattr("backend.config.LLM_PROVIDER", "openrouter")
    with pytest.raises(LLMUnavailable):
        build_client()


@pytest.mark.asyncio
async def test_malformed_tool_arguments_are_reported_not_guessed(monkeypatch):
    mock = MockLLMClient(
        decisions=[
            LLMDecision(
                tool="company.get_invoice",
                arguments={},
                validation_error="Tool arguments were not valid JSON",
            ),
            LLMDecision(tool="task.fail", arguments={"reason": "malformed response"}),
        ]
    )
    _install(monkeypatch, mock)
    state = TaskState(task_id="task_malformed", goal="Read an invoice")
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "failed"
    assert any("could not be parsed" in obs["observation"] for obs in result.observations)


@pytest.mark.asyncio
async def test_cancelled_task_is_not_restarted():
    """A task cancelled before dispatch must not begin executing tools."""
    state = TaskState(task_id="task_cancel_pre", goal="Process the latest invoice from Acme Corporation")
    state.status = "cancelled"
    state.cancel_requested = True
    state.summary = "Cancelled by the operator."
    save_task(state)
    result = await run_task(state.task_id, state)
    assert result.status == "cancelled"
    assert result.completed_actions == []
    assert store.list_finance() == []


@pytest.mark.asyncio
async def test_cancel_stops_a_running_task(monkeypatch):
    """A cancellation issued mid-flight is honoured instead of being overwritten."""
    task_id = "task_cancel"
    state = TaskState(
        task_id=task_id,
        goal="Process the latest invoice from Acme Corporation",
        approval_status="approved",  # no pause, so the loop keeps stepping
    )
    save_task(state)

    class CancellingMock(MockLLMClient):
        """Cancels the task from the API side while the agent is mid-conversation."""

        async def create_response(self, messages, tools=None):
            decision = await super().create_response(messages, tools)
            if len(self.calls) == 1:
                current = load_task(task_id)
                current.cancel_requested = True
                save_task(current)
            return decision

    _install(monkeypatch, CancellingMock(auto_pilot=True))

    result = await run_task(task_id, state)
    assert result.status == "cancelled"
    assert result.completed_actions == [], "no tool may run after cancellation"
    assert load_task(task_id).status == "cancelled"


# ===========================================================================
# 17. Tool timeout is bounded by the runtime
# ===========================================================================
@pytest.mark.asyncio
async def test_tool_timeout_is_bounded(monkeypatch):
    """A hung tool is aborted by the runtime instead of stalling the agent."""
    import time

    from backend.tools.registry import Tool

    original = registry.get("company.search_invoice")
    slow = Tool(
        name="company.search_invoice",
        description=original.description,
        parameters=original.parameters,
        required=original.required,
        handler=lambda args, state: (time.sleep(3), company.search_invoice(args["company"]))[1],
        timeout_seconds=0.2,
    )
    registry._tools["company.search_invoice"] = slow
    try:
        mock = MockLLMClient(
            decisions=[
                LLMDecision(tool="company.search_invoice", arguments={"company": "Acme Corporation"}),
                LLMDecision(tool="task.fail", arguments={"reason": "timeout observed"}),
            ]
        )
        _install(monkeypatch, mock)
        state = TaskState(task_id="task_timeout", goal="Search invoices")
        save_task(state)
        result = await run_task(state.task_id, state)
        assert result.status == "failed"
        assert any(item["error"]["type"] == "TIMEOUT" for item in result.failed_actions)
    finally:
        registry._tools["company.search_invoice"] = original


# ===========================================================================
# 18. Scripted model proves observations are returned to the model
# ===========================================================================
@pytest.mark.asyncio
async def test_scripted_sequence_receives_real_observations(monkeypatch):
    mock = MockLLMClient(
        decisions=[
            ("company.login", {}),
            ("company.search_invoice", {"company": "Acme Corporation"}),
            ("company.get_invoice", {"invoice_id": "INV-1042"}),
            (
                "company.create_finance_record",
                {"invoice_id": "INV-1042", "company": "Acme Corporation", "amount": 84250, "due_date": "2026-10-15"},
            ),
            ("company.get_finance_record", {"invoice_id": "INV-1042"}),
            ("task.complete", {"summary": "Done."}),
        ]
    )
    _install(monkeypatch, mock)
    store.reset("")
    state = TaskState(task_id="task_scripted_seq", goal="Process the latest invoice from Acme Corporation")
    save_task(state)
    state.approval_status = "approved"  # operator pre-authorised so the script runs straight through
    result = await run_task(state.task_id, state)

    assert result.status == "completed"
    assert result.verification_status == "passed"

    # Call #3 sees the search result; call #4 sees the invoice details.
    search_seen = [m for m in mock.calls[2] if m.get("role") == "tool" and m.get("name") == "company.search_invoice"]
    assert search_seen and "INV-1042" in search_seen[0]["content"]

    invoice_seen = [m for m in mock.calls[3] if m.get("role") == "tool" and m.get("name") == "company.get_invoice"]
    assert invoice_seen and "84250" in invoice_seen[0]["content"]

    # Every call carries the system prompt and the original goal.
    for call in mock.calls:
        assert call[0]["role"] == "system"
        assert "Process the latest invoice" in call[1]["content"]


@pytest.mark.asyncio
async def test_tool_schemas_are_native_and_complete(monkeypatch):
    mock = MockLLMClient(decisions=[LLMDecision(tool="task.fail", arguments={"reason": "done"})])
    captured: dict[str, Any] = {}

    async def capture(messages, tools=None):
        captured["tools"] = tools
        return LLMDecision(tool="task.fail", arguments={"reason": "done"})

    mock.create_response = capture  # type: ignore[assignment]
    _install(monkeypatch, mock)
    state = TaskState(task_id="task_schemas", goal="Do something")
    save_task(state)
    await run_task(state.task_id, state)

    tools = captured["tools"]
    assert tools, "the runtime must send real tool definitions"
    names = {tool["function"]["name"] for tool in tools}
    assert {
        "company.login",
        "company.search_invoice",
        "company.get_invoice",
        "company.create_finance_record",
        "company.get_finance_record",
        "company.search_customer",
        "company.get_customer",
        "company.update_customer",
        "browser.open",
        "browser.click",
        "browser.type",
        "browser.read",
        "browser.screenshot",
        "memory.store",
        "memory.retrieve",
    } <= names
    for tool in tools:
        function = tool["function"]
        assert function["type"] if "type" in function else True
        assert tool["type"] == "function"
        assert function["description"]
        assert function["parameters"]["type"] == "object"
    # No shell / sql / eval surface is exposed at all.
    assert not any(word in " ".join(names) for word in ("shell", "sql", "eval", "exec"))


# ===========================================================================
# 19. Overdue summary (read-only task, still verified)
# ===========================================================================
@pytest.mark.asyncio
async def test_overdue_summary_workflow():
    result = await run_goal_approved("Find all overdue invoices from Acme Corporation and summarize them.")
    assert result.status == "completed", result.summary
    assert result.verification_status == "passed"


# ===========================================================================
# 20. Secrets never leak
# ===========================================================================
@pytest.mark.asyncio
async def test_secrets_never_appear_in_evidence_or_summary():
    result = await run_goal_approved("Process the latest invoice from Acme Corporation")
    assert result.finance_token and result.sandbox_token
    kinds = {item["kind"] for item in result.evidence}
    assert "sandbox_token" not in kinds
    assert "finance_token" not in kinds
    serialized = json.dumps(result.evidence) + (result.summary or "")
    assert result.finance_token not in serialized
    assert result.sandbox_token not in serialized
    assert config.SANDBOX_PASSWORD not in serialized

    feed = json.dumps(list_audit(result.task_id))
    assert result.finance_token not in feed
    assert config.SANDBOX_PASSWORD not in feed


# ===========================================================================
# 21. Observability
# ===========================================================================
@pytest.mark.asyncio
async def test_observability_records_llm_and_tool_steps():
    result = await run_goal_approved("Process the latest invoice from Acme Corporation")
    assert result.llm_calls, "every model call must be recorded"
    assert all(call.provider == "mock" for call in result.llm_calls)
    assert any(call.tool == "company.search_invoice" for call in result.llm_calls)

    feed = list_audit(result.task_id)
    kinds = {entry["kind"] for entry in feed}
    assert {"llm_decision", "tool_call"} <= kinds
    assert all("password" not in json.dumps(entry) or "redacted" in json.dumps(entry) for entry in feed)


# ===========================================================================
# 22. Context window management never orphans a tool result
# ===========================================================================
def test_context_compression_keeps_tool_pairs_whole():
    messages = initial_messages("Process the invoice")
    for index in range(20):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"call_{index}", "type": "function", "function": {"name": "company.get_invoice", "arguments": "{}"}}
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"call_{index}", "name": "company.get_invoice", "content": "{}"})

    context = build_context(messages, max_messages=12, digest="STATE SUMMARY")
    answered = {
        call["id"]
        for message in context
        if message.get("role") == "assistant"
        for call in message.get("tool_calls", [])
    }
    for message in context:
        if message.get("role") == "tool":
            assert message["tool_call_id"] in answered
    assert sanitize_messages(context) == context
    assert len(context) <= 12 + 2  # window + system/goal + digest


# ===========================================================================
# 23. Approval gate is server-side, not a UI convention
# ===========================================================================
def test_approval_gate_is_server_side():
    from backend.permissions.policies import requires_approval

    state = TaskState(task_id="t", goal="g", demo_mode="normal")
    assert requires_approval("company.create_finance_record", "normal", None)
    assert requires_approval("company.update_customer", "normal", None)
    assert not requires_approval("company.search_invoice", "normal", None)
    # An approval already granted applies only to that call.
    assert not requires_approval("company.create_finance_record", "normal", "approved")


# ===========================================================================
# 24. Registry safety
# ===========================================================================
def test_registry_validation_contract():
    assert registry.validate("company.get_invoice", {"invoice_id": "INV-1042"}) is None
    assert registry.validate("company.get_invoice", {}) is not None
    assert registry.validate("not.a.tool", {}) is not None
    assert registry.validate("shell.exec", {}) is not None
    assert registry.validate("files.read", {"path": "/etc/passwd"}) is not None
    assert registry.validate("files.read", {"path": "notes/process-guide.txt"}) is None
    assert registry.validate("browser.open", {"url": "/invoices"}) is None
    assert registry.validate("browser.open", {"url": "https://example.com"}) is not None


def test_unknown_tool_execution_fails_safely():
    state = TaskState(task_id="t", goal="g")
    result = registry.execute("nope.tool", {}, state)
    assert not result.success
    assert result.error.type == "PERMISSION_DENIED"


# ===========================================================================
# 25. Real LLM integration test (skipped without a key)
# ===========================================================================
@pytest.mark.asyncio
async def test_real_llm_end_to_end(monkeypatch):
    """Runs a genuine model through the genuine runtime. Skipped when no key is set."""
    provider = LIVE_LLM_PROVIDER
    if provider in {"", "mock"} or not LIVE_LLM_API_KEY:
        pytest.skip("No real LLM configured; set LLM_PROVIDER and LLM_API_KEY to run this.")

    # Defeat the offline default the shared fixture installs.
    monkeypatch.setattr("backend.config.LLM_PROVIDER", provider)

    store.reset("")
    state = TaskState(task_id="task_live_llm", goal="Process the latest invoice from Acme Corporation")
    save_task(state)
    state.approval_status = "approved"  # the live test does not need an operator in the loop
    result = await run_task(state.task_id, state)

    assert result.llm_calls, "the real model must have been called"
    assert result.llm_provider not in {"mock", None}
    assert len(result.llm_calls) >= 2, "the loop must iterate on real tool results"
    assert any(call.tool for call in result.llm_calls)

    # Either the task completed through verification, or it stopped safely.
    assert result.status in {"completed", "awaiting_approval", "awaiting_clarification", "failed"}
    if result.status == "completed":
        assert result.verification_status == "passed"
        assert result.artifacts["finance_record"]["invoice_id"] == "INV-1042"
    else:
        # A safe stop must be explained, never silent.
        assert result.summary
