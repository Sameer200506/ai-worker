"""Manual end-to-end trace against the live LLM. Not part of the test-suite.

Usage:  python scripts/live_trace.py "Process the latest invoice from Acme Corporation"
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.database import connect, init_db  # noqa: E402
from backend.database.models import TaskState  # noqa: E402
from backend.database.store import save_task  # noqa: E402
from backend.runtime.loop import run_task  # noqa: E402

from apps.sandbox.store import store  # noqa: E402


async def main(goal: str, auto_approve: bool = True) -> None:
    tmp = Path(tempfile.mkdtemp())
    config.DATABASE_PATH = tmp / "trace.db"
    import backend.database as db

    db.DATABASE_PATH = tmp / "trace.db"
    init_db(connect(tmp / "trace.db"))
    store.path = tmp / "store.json"
    store.reset("")

    state = TaskState(task_id="trace", goal=goal, status="queued")
    save_task(state)

    print(f"LLM      : {config.LLM_PROVIDER} / {config.LLM_MODEL}")
    print(f"APPROVAL : {config.APPROVAL_POLICY}")
    print(f"GOAL     : {goal}\n" + "-" * 78)

    while True:
        state = await run_task("trace", state)
        if state.status == "awaiting_approval" and auto_approve:
            print(f"  [approval] {state.approval_payload['summary']}")
            for key, value in state.approval_payload["impact"].items():
                print(f"             {key}: {value}")
            state.approval_status = "approved"
            state.status = "running"
            save_task(state)
            continue
        break

    for call in state.llm_calls:
        print(f"LLM #{call.step:<2} {call.provider}/{call.model} {call.latency_ms:>5}ms -> {call.tool}")
    for obs in state.observations:
        flag = "ok " if obs["success"] else "ERR"
        print(f"TOOL     {flag} {obs['tool']:<32} {obs['observation'][:70]}")
    print("-" * 78)
    print(f"STATUS       : {state.status}")
    print(f"VERIFICATION : {state.verification_status}")
    print(f"RETRIES      : {state.retry_count}")
    print("SUMMARY:\n" + (state.summary or ""))


if __name__ == "__main__":
    goal = sys.argv[1] if len(sys.argv) > 1 else "Process the latest invoice from Acme Corporation."
    asyncio.run(main(goal))
