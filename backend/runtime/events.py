from __future__ import annotations

import asyncio
from typing import Any

from backend.database.store import append_event, list_events

_subscribers: dict[str, list[asyncio.Queue]] = {}


def subscribe(task_id: str) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.setdefault(task_id, []).append(queue)
    return queue


def unsubscribe(task_id: str, queue: asyncio.Queue) -> None:
    subs = _subscribers.get(task_id) or []
    if queue in subs:
        subs.remove(queue)


def emit(task_id: str, event_type: str, **payload: Any) -> dict[str, Any]:
    event = append_event(task_id, event_type, payload)
    for queue in list(_subscribers.get(task_id) or []):
        try:
            queue.put_nowait(event)
        except Exception:
            pass
    return event


def history(task_id: str) -> list[dict[str, Any]]:
    return list_events(task_id)
