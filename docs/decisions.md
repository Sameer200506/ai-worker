# Technical decisions

This document records the choices that shape TaskPilot, and the trade-offs behind them.

## 1. Explicit runtime loop, not a single LLM call

`backend/runtime/loop.py` drives a `while state.status not in TERMINAL` loop. Every
iteration reads the persisted `TaskState`, asks the planner for exactly one next
action, executes it, records the observation, and decides whether to verify, recover,
or complete.

Why: the brief requires a reusable worker, not a chatbot. A visible loop makes each
stage (understand, plan, execute, observe, adapt, verify, complete) inspectable and
testable, and keeps control flow out of hidden model output.

Trade-off: more orchestration code than a single prompt, but far easier to reason
about, extend, and test deterministically.

## 2. Observation-driven planner with an optional LLM

`backend/agent/planner.py` contains `decide()`, a deterministic planner that chooses
the next tool from the current artifacts and the last error. `llm_decide()` is an
optional OpenAI-compatible planner used only when `LLM_API_KEY` is set.

Why: tests, demos, and CI must be reproducible without a vendor key or network access.
The deterministic planner still performs genuine dynamic selection — it does not match
on phrases to call a hard-coded workflow. It inspects artifacts (`invoice_candidates`,
`customer_candidates`, `finance_record`, `last_error`) and derives the next action, so
the same code path serves every workflow.

Trade-off: the deterministic planner encodes domain-aware rules for the sandbox
entities. This is intentional: business rules are acceptable for validation and safety,
as long as the execution loop is genuinely dynamic. The LLM planner is the extension
point for open-ended reasoning.

## 3. One tool registry with structured results

All tools are registered in `backend/tools/registry.py` and return a uniform
`ToolResult` (`success`, `tool`, `data`, `observation`, `artifacts`, `error`). Unknown
tools, missing arguments, and malformed calls are converted into structured failures
rather than exceptions.

Why: the runtime can treat every action identically, and the observer never has to
special-case a tool that failed by raising.

## 4. Verification is separate from execution

`backend/verification/engine.py` never trusts the payload returned by a create/update
tool. It re-reads the finance or customer record and compares expected vs actual field
by field. A mismatch marks the task failed even though the write "succeeded".

Why: tool success is not task success. This is the single most important reliability
property of the worker.

## 5. Deterministic, observable failure injection

The sandbox store in `apps/sandbox/store.py` injects real failures (`finance_session_expired`,
`network_error`, `verification_mismatch`) that mutate actual state — a finance session is
truly invalidated on the first create attempt, so the recovery is not faked. Demo modes
map onto these failure modes so a technical interview is repeatable.

## 6. Recovery is classified, bounded, and typed

`backend/recovery/manager.py` classifies each failure and maps it to a strategy:

| Error type | Strategy |
| --- | --- |
| `AUTH_EXPIRED` / `AUTH_FAILED` | re-authenticate |
| `NETWORK_ERROR` | retry the previous action |
| `INVALID_DATA` | stop safely (re-plan needs new input) |
| `NOT_FOUND` | stop safely |
| `PERMISSION_DENIED` | ask the user to intervene |
| unknown | stop safely |

Retries are bounded by `MAX_RETRIES` (default 2). It never retries indefinitely.

## 7. Human approval pauses persisted state

Sensitive tools (`company.create_finance_record`, `company.update_customer`) can require
approval. When they do, the runtime persists `status = "awaiting_approval"` and stops;
the API resumes the same task on `POST /api/tasks/:id/approve`. Rejection terminates the
task safely with an explanation. The pause survives a process restart because it is stored
in SQLite, not held in memory.

## 8. SQLite for state, events, memory, and audit

A single SQLite file (`data/taskpilot.db`) holds task payloads, lifecycle events, learned
facts, and the audit log. There is deliberately no vector database: the prototype's memory
is small, structured, and keyed.

Why: one file, no infrastructure, trivial to reset and inspect. Distributed workers and a
durable queue are explicitly out of scope (see README limitations).

## 9. SSE for live updates

The operator UI consumes `text/event-stream` from `GET /api/tasks/:id/events`. SSE is
simpler than WebSockets for one-way progress and reconnects natively in the browser.

## 10. Security boundaries in code, not just in prompts

- Browser navigation is allowlisted to the local sandbox host/port
  (`backend/permissions/policies.py`); off-sandbox URLs are rejected before Playwright runs.
- Tool names are checked against an allowlist; there is no shell tool.
- Session and finance tokens are never written to `TaskState.evidence` or summaries
  (see `SECRET_ARTIFACT_TYPES` in `backend/runtime/loop.py`).
- Only fake sandbox credentials are used.

The LLM never receives unrestricted system access; its proposed actions pass through the
same validation as any other planner.
