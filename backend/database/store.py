from __future__ import annotations

from typing import Any

from backend.database import db_session, dump_json, utcnow
from backend.database.models import TaskState


def save_task(state: TaskState) -> None:
    state.updated_at = utcnow()
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO tasks(task_id, payload, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
            """,
            (state.task_id, state.model_dump_json(), state.updated_at),
        )


def load_task(task_id: str) -> TaskState | None:
    with db_session() as conn:
        row = conn.execute("SELECT payload FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    if not row:
        return None
    return TaskState.model_validate_json(row["payload"])


def list_tasks(limit: int = 50) -> list[TaskState]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT payload FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [TaskState.model_validate_json(row["payload"]) for row in rows]


def append_event(task_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    created = utcnow()
    event = {"task_id": task_id, "type": event_type, "created_at": created, **payload}
    with db_session() as conn:
        conn.execute(
            "INSERT INTO events(task_id, event_type, payload, created_at) VALUES(?,?,?,?)",
            (task_id, event_type, dump_json(event), created),
        )
    return event


def list_events(task_id: str) -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT payload FROM events WHERE task_id = ? ORDER BY id ASC", (task_id,)
        ).fetchall()
    import json

    return [json.loads(row["payload"]) for row in rows]


def store_memory(key: str, value: Any, task_id: str | None = None) -> None:
    with db_session() as conn:
        conn.execute(
            "INSERT INTO memory(key, value, task_id, created_at) VALUES(?,?,?,?)",
            (key, dump_json(value), task_id, utcnow()),
        )


def retrieve_memory(key: str) -> list[Any]:
    import json

    with db_session() as conn:
        rows = conn.execute(
            "SELECT value FROM memory WHERE key = ? ORDER BY id DESC LIMIT 20", (key,)
        ).fetchall()
    return [json.loads(row["value"]) for row in rows]


def audit(
    task_id: str | None,
    agent_step: str,
    tool: str | None,
    arguments: Any,
    result: Any,
    success: bool | None,
    error: Any,
    retry_count: int,
    latency_ms: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    kind: str = "tool_call",
) -> None:
    """Record one observable step. Callers must redact credentials before calling."""
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO audit_log(
                task_id, agent_step, kind, tool, arguments, result, success, error,
                retry_count, latency_ms, provider, model, created_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                task_id,
                agent_step,
                kind,
                tool,
                dump_json(arguments),
                dump_json(result),
                None if success is None else int(success),
                dump_json(error),
                retry_count,
                latency_ms,
                provider,
                model,
                utcnow(),
            ),
        )


def list_audit(task_id: str, limit: int = 500) -> list[dict[str, Any]]:
    """Observability feed for a task: one row per LLM decision and per tool call."""
    import json

    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT agent_step, kind, tool, arguments, result, success, error,
                   retry_count, latency_ms, provider, model, created_at
            FROM audit_log WHERE task_id = ? ORDER BY id ASC LIMIT ?
            """,
            (task_id, limit),
        ).fetchall()

    feed: list[dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        for field in ("arguments", "result", "error"):
            raw = entry.get(field)
            if not raw:
                entry[field] = None
                continue
            try:
                entry[field] = json.loads(raw)
            except (TypeError, ValueError):
                pass
        feed.append(entry)
    return feed
