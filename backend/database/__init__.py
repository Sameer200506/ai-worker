from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from backend.config import DATABASE_PATH

_lock = threading.Lock()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or DATABASE_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            task_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            agent_step TEXT,
            kind TEXT NOT NULL DEFAULT 'tool_call',
            tool TEXT,
            arguments TEXT,
            result TEXT,
            success INTEGER,
            error TEXT,
            retry_count INTEGER,
            latency_ms INTEGER,
            provider TEXT,
            model TEXT,
            created_at TEXT NOT NULL
        );
        """
    )
    _migrate(conn)
    conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations so an existing database keeps working."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(audit_log)")}
    for name, ddl in (
        ("kind", "kind TEXT NOT NULL DEFAULT 'tool_call'"),
        ("latency_ms", "latency_ms INTEGER"),
        ("provider", "provider TEXT"),
        ("model", "model TEXT"),
    ):
        if name not in columns:
            conn.execute(f"ALTER TABLE audit_log ADD COLUMN {ddl}")


@contextmanager
def db_session(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    with _lock:
        conn = connect(path)
        try:
            init_db(conn)
            yield conn
            conn.commit()
        finally:
            conn.close()


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
