"""Drive a full TaskPilot task through the live HTTP API and print the real event stream.

Usage:
    python scripts/demo_run.py "Process the latest invoice from Acme Corporation." normal
    python scripts/demo_run.py "Process the latest invoice from Acme Corporation." recoverable_failure
    python scripts/demo_run.py "Process the latest invoice from Acme Corporation." verification_failure

Every line printed comes from an event the running backend actually emitted.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

API = "http://127.0.0.1:8000"
TERMINAL = {"completed", "failed", "cancelled", "awaiting_clarification"}


def post(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else b""
    request = urllib.request.Request(
        f"{API}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=60) as response:
        return json.loads(response.read())


def render(event: dict, counter: list[int]) -> None:
    kind = event.get("type", "")
    if kind == "ai.decision":
        counter[0] += 1
        print(f"\nAI DECISION #{counter[0]}: {event.get('message', '')}")
        if event.get("arguments"):
            print(f"  tool     : {event.get('tool')}")
            print(f"  arguments: {json.dumps(event['arguments'], ensure_ascii=False)}")
        if event.get("latency_ms") is not None:
            print(f"  model    : {event.get('provider')}/{event.get('model')} ({event['latency_ms']}ms)")
    elif kind == "tool.started":
        print(f"TOOL      : {event.get('tool')}")
    elif kind == "tool.completed":
        print(f"RESULT    : {event.get('message', '')[:120]}")
    elif kind == "tool.failed":
        error = event.get("error") or {}
        print(f"FAILED    : {event.get('tool')} -> {error.get('type')}: {(error.get('message') or '')[:90]}")
    elif kind == "observation.created":
        print(f"OBSERVED  : {event.get('message', '')[:120]}")
    elif kind.startswith("recovery."):
        print(f"RECOVERY  : {event.get('message', '')[:140]}")
    elif kind == "approval.required":
        payload = event.get("payload") or {}
        print(f"APPROVAL  : {payload.get('summary')}")
        for key, value in (payload.get("impact") or {}).items():
            print(f"            {key}: {value}")
    elif kind in {"approval.approved", "approval.rejected"}:
        print(f"APPROVAL  : {kind.split('.')[1].upper()}")
    elif kind == "verification.started":
        print("VERIFYING : independently re-reading persisted state ...")
    elif kind == "verification.completed":
        print(f"VERIFIED  : {event.get('message', '')[:150]}")
        for check in event.get("checks", []):
            mark = "PASS" if check["passed"] else "FAIL"
            print(f"            [{mark}] {check['field']}: expected {check['expected']} / actual {check['actual']}")
    elif kind == "task.waiting_for_user":
        print(f"QUESTION  : {event.get('message')}")
    elif kind == "task.cancelled":
        print(f"CANCELLED : {event.get('message')}")


def main(goal: str, mode: str, auto_approve: bool = True, timeout: int = 900) -> int:
    health = get("/health")
    post("/api/demo", {"mode": mode})
    created = post("/api/tasks", {"goal": goal, "demo_mode": mode})
    task_id = created["task_id"]

    print(f"goal      : {goal}")
    print(f"demo mode : {mode}")
    print(f"task id   : {task_id}")
    print(f"llm       : {health['llm']['provider']} / {health['llm']['model']} (autonomous={health['autonomous']})")
    print(f"policy    : approval={health['approval_policy']} max_retries={health['max_retries']}")
    print("-" * 78)

    cursor = 0
    decisions = [0]
    approved_once = False
    deadline = time.time() + timeout

    while time.time() < deadline:
        state = get(f"/api/tasks/{task_id}")
        events = get(f"/api/tasks/{task_id}/events/history")["events"]
        for event in events[cursor:]:
            render(event, decisions)
        cursor = len(events)

        status = state["status"]
        if status == "awaiting_approval" and not approved_once:
            approved_once = True
            if auto_approve:
                post(f"/api/tasks/{task_id}/approve")
                print("APPROVAL  : operator approved, resuming.\n")
            else:
                post(f"/api/tasks/{task_id}/reject")
                print("REJECTING : operator rejected the action.\n")
        elif status in TERMINAL:
            for event in get(f"/api/tasks/{task_id}/events/history")["events"][cursor:]:
                render(event, decisions)
            print("-" * 78)
            print(f"status    : {status}")
            print(f"verifier  : {state.get('verification_status')}")
            print(f"recovery  : {state.get('recovery_count', 0)} attempt(s)")
            print(f"evidence  : {len(state.get('evidence') or [])} artifact(s)")
            print(f"llm calls : {len(state.get('llm_calls') or [])}")
            print()
            print(state.get("summary") or "")
            return 0 if status == "completed" else 1
        time.sleep(1.5)

    print("timed out waiting for the task to finish")
    return 2


if __name__ == "__main__":
    goal = sys.argv[1] if len(sys.argv) > 1 else "Process the latest invoice from Acme Corporation."
    mode = sys.argv[2] if len(sys.argv) > 2 else "normal"
    reject = "--reject" in sys.argv
    sys.exit(main(goal, mode, auto_approve=not reject))
