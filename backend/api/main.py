"""TaskPilot Agent API.

Endpoints
---------
POST /api/tasks                  start an autonomous task
GET  /api/tasks/:id              full task state
GET  /api/tasks/:id/events       SSE live event stream
GET  /api/tasks/:id/audit        observability feed (LLM + tool steps)
POST /api/tasks/:id/approve      approve the pending sensitive action and resume
POST /api/tasks/:id/reject       reject it and terminate safely
POST /api/tasks/:id/clarify      answer a clarification question and resume
POST /api/tasks/:id/cancel       cancel a running task
GET  /api/tools                  the tool schemas exposed to the model
GET  /api/demo, POST /api/demo   switch sandbox demo behaviour
GET  /health                     service + LLM configuration status
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from apps.sandbox.store import store as company_store
from backend import config
from backend.agent.llm import describe_configuration
from backend.database import connect, init_db
from backend.database.models import TaskState
from backend.database.store import list_audit, list_tasks, load_task, save_task
from backend.runtime.events import emit, history, subscribe, unsubscribe
from backend.runtime.loop import run_task
from backend.tools.registry import registry

#: Demo modes only change *sandbox* behaviour. They never bypass the LLM.
DEMO_MODES = ("normal", "recoverable_failure", "verification_failure", "approval_required")
DEMO_TO_FAILURE = {
    "normal": "",
    "recoverable_failure": "finance_session_expired",
    "verification_failure": "verification_mismatch",
    "approval_required": "",
}

_demo_mode = config.DEMO_MODE if config.DEMO_MODE in DEMO_MODES else "normal"
_running: dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db(connect())
    yield
    for task in list(_running.values()):
        task.cancel()


app = FastAPI(title="TaskPilot API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.WEB_ORIGIN, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class TaskCreate(BaseModel):
    goal: str = Field(min_length=3, max_length=2000)
    demo_mode: str | None = None


class DemoBody(BaseModel):
    mode: str = Field(pattern="^(normal|recoverable_failure|verification_failure|approval_required)$")


class ClarifyBody(BaseModel):
    answer: str = Field(min_length=1, max_length=2000)


def _spawn(task_id: str, coro) -> None:
    task = asyncio.create_task(coro)
    _running[task_id] = task

    def _cleanup(finished: asyncio.Task) -> None:
        _running.pop(task_id, None)
        if not finished.cancelled():
            finished.exception()  # surface the error without crashing the loop

    task.add_done_callback(_cleanup)


def _require_task(task_id: str) -> TaskState:
    state = load_task(task_id)
    if not state:
        raise HTTPException(404, "Task not found")
    return state


# ---------------------------------------------------------------------------
# Health and configuration
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    llm = describe_configuration()
    return {
        "ok": True,
        "app": "taskpilot",
        "version": "2.0.0",
        "demo_mode": _demo_mode,
        "llm": {
            "provider": llm["provider"],
            "model": llm["model"],
            "configured": llm["configured"],
            "api_key_present": llm["api_key_present"],
            "error": llm["error"],
        },
        "autonomous": llm["configured"],
        "tools_registered": len(registry),
        "approval_policy": config.APPROVAL_POLICY,
        "max_steps": config.MAX_STEPS,
        "max_retries": config.MAX_RETRIES,
    }


@app.get("/api/tools")
def get_tools() -> dict[str, Any]:
    return {"tools": registry.flat_specs(), "schemas": registry.schemas()}


# ---------------------------------------------------------------------------
# Demo modes
# ---------------------------------------------------------------------------
@app.get("/api/demo")
def get_demo() -> dict[str, str]:
    return {"mode": _demo_mode}


@app.post("/api/demo")
def set_demo(body: DemoBody) -> dict[str, str]:
    global _demo_mode
    _demo_mode = body.mode
    company_store.set_failure_mode(DEMO_TO_FAILURE.get(body.mode, ""))
    return {"mode": _demo_mode}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
@app.post("/api/tasks")
async def create_task(body: TaskCreate) -> dict[str, Any]:
    mode = body.demo_mode if body.demo_mode in DEMO_MODES else _demo_mode
    company_store.reset(DEMO_TO_FAILURE.get(mode, ""))
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    state = TaskState(task_id=task_id, goal=body.goal.strip(), status="queued", demo_mode=mode)
    save_task(state)
    _spawn(task_id, run_task(task_id, state))
    return {"task_id": task_id, "status": "queued", "demo_mode": mode}


@app.get("/api/tasks")
def get_tasks() -> dict[str, Any]:
    return {"tasks": [item.model_dump() for item in list_tasks()]}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    return _require_task(task_id).model_dump()


@app.get("/api/tasks/{task_id}/audit")
def get_task_audit(task_id: str) -> dict[str, Any]:
    _require_task(task_id)
    return {"task_id": task_id, "audit": list_audit(task_id)}


@app.get("/api/tasks/{task_id}/events/history")
def task_events_history(task_id: str) -> dict[str, Any]:
    """The same events as the SSE stream, as plain JSON (handy for CLI demos)."""
    _require_task(task_id)
    return {"task_id": task_id, "events": history(task_id)}


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str):
    _require_task(task_id)
    queue = subscribe(task_id)

    async def gen():
        try:
            for event in history(task_id):
                yield {"event": event.get("type") or "message", "data": json.dumps(event, default=str)}
            while True:
                event = await queue.get()
                yield {"event": event.get("type") or "message", "data": json.dumps(event, default=str)}
        finally:
            unsubscribe(task_id, queue)

    return EventSourceResponse(gen())


# ---------------------------------------------------------------------------
# Human decisions
# ---------------------------------------------------------------------------
@app.post("/api/tasks/{task_id}/approve")
async def approve_task(task_id: str) -> dict[str, Any]:
    state = _require_task(task_id)
    if state.status != "awaiting_approval" or not state.pending_action:
        raise HTTPException(409, "No action is awaiting approval.")
    state.approval_status = "approved"
    state.status = "running"
    save_task(state)
    emit(task_id, "approval.approved", message="Approved by the operator. Resuming execution.")
    _spawn(task_id, run_task(task_id))
    return {"task_id": task_id, "approval_status": "approved", "resumed": True}


@app.post("/api/tasks/{task_id}/reject")
async def reject_task(task_id: str) -> dict[str, Any]:
    state = _require_task(task_id)
    if state.status != "awaiting_approval":
        raise HTTPException(409, "No action is awaiting approval.")
    state.approval_status = "rejected"
    state.pending_action = None
    state.status = "failed"
    state.summary = (
        "The operator rejected the high-impact action, so the tool was never executed "
        "and the task was terminated safely."
    )
    save_task(state)
    emit(task_id, "approval.rejected", message="Rejected by the operator. The tool will not run.")
    emit(task_id, "task.failed", message=state.summary, error_type="APPROVAL_REJECTED")
    return {"task_id": task_id, "status": "failed", "approval_status": "rejected"}


@app.post("/api/tasks/{task_id}/clarify")
async def clarify_task(task_id: str, body: ClarifyBody) -> dict[str, Any]:
    state = _require_task(task_id)
    if state.status != "awaiting_clarification":
        raise HTTPException(409, "The task is not waiting for clarification.")
    state.clarification_answer = body.answer.strip()
    state.status = "running"
    save_task(state)
    emit(task_id, "task.clarified", message=f"Operator answered: {state.clarification_answer}")
    _spawn(task_id, run_task(task_id))
    return {"task_id": task_id, "status": "running", "resumed": True}


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict[str, Any]:
    state = _require_task(task_id)
    if state.status in {"completed", "failed", "cancelled"}:
        return {"task_id": task_id, "status": state.status, "cancelled": False}
    state.status = "cancelled"
    state.summary = "Cancelled by the operator."
    state.pending_action = None
    state.cancel_requested = True
    save_task(state)
    running = _running.pop(task_id, None)
    if running and not running.done():
        running.cancel()
    emit(task_id, "task.cancelled", message=state.summary)
    emit(task_id, "task.failed", message=state.summary, error_type="CANCELLED")
    return {"task_id": task_id, "status": "cancelled", "cancelled": True}
