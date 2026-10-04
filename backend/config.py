"""Central configuration.

Every runtime-critical knob is read from the environment (or `.env`). No secrets are
hard-coded here, and no value is derived from the LLM.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        return default


# --- Company sandbox -------------------------------------------------------
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://127.0.0.1:3001").rstrip("/")
SANDBOX_USERNAME = os.getenv("SANDBOX_USERNAME", "demo")
SANDBOX_PASSWORD = os.getenv("SANDBOX_PASSWORD", "demo123")

# --- Storage ---------------------------------------------------------------
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(ROOT / "data" / "taskpilot.db")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- LLM -------------------------------------------------------------------
# Supported providers: openrouter | openai | anthropic | mock
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or "openai/gpt-4o-mini"

# Default endpoint per provider so LLM_BASE_URL is optional.
PROVIDER_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "openai-compatible": "http://127.0.0.1:11434/v1",
    "ollama": "http://127.0.0.1:11434/v1",
}
LLM_BASE_URL = (
    os.getenv("LLM_BASE_URL", "").strip() or PROVIDER_BASE_URLS.get(LLM_PROVIDER, PROVIDER_BASE_URLS["openai"])
).rstrip("/")
LLM_TIMEOUT_SECONDS = _float("LLM_TIMEOUT_SECONDS", 90.0)
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 1024)

# Optional attribution headers (OpenRouter app metadata). Never secrets.
LLM_APP_URL = os.getenv("LLM_APP_URL", "http://127.0.0.1:3000").strip()
LLM_APP_TITLE = os.getenv("LLM_APP_TITLE", "TaskPilot").strip()

# Explicitly opted-in only. Never activated implicitly. Even when true, a missing
# LLM fails the task fast instead of producing a fabricated "autonomous" result.
DETERMINISTIC_FALLBACK = _bool("DETERMINISTIC_FALLBACK", False)
DEGRADED_DETERMINISTIC_MODE = DETERMINISTIC_FALLBACK

# --- Sandbox behaviour (demo modes) ---------------------------------------
FAILURE_MODE = os.getenv("FAILURE_MODE", "").strip()
DEMO_MODE = os.getenv("DEMO_MODE", "normal").strip() or "normal"

# --- Deterministic safety ceilings (never model controlled) ----------------
# sensitive = every sensitive tool needs human approval (default)
# demo      = approval only when the task runs in the approval_required demo mode
APPROVAL_POLICY = os.getenv("APPROVAL_POLICY", "sensitive").strip().lower() or "sensitive"
MAX_RETRIES = _int("MAX_RETRIES", 2)
MAX_STEPS = _int("MAX_STEPS", 24)
TOOL_TIMEOUT_SECONDS = _float("TOOL_TIMEOUT_SECONDS", 20.0)
MAX_CONTEXT_MESSAGES = _int("MAX_CONTEXT_MESSAGES", 40)

# --- Browser tools (real Playwright, sandbox-restricted) ------------------
USE_BROWSER = _bool("USE_BROWSER", False)
# "" uses Playwright's bundled Chromium; "chrome"/"msedge" reuse an installed browser.
BROWSER_CHANNEL = os.getenv("BROWSER_CHANNEL", "").strip()
ALLOWED_BROWSER_HOSTS = {"127.0.0.1", "localhost"}
ALLOWED_BROWSER_SCHEMES = {"http", "https"}

# --- Filesystem tools ------------------------------------------------------
# The agent has no real filesystem access. `files.*` reads only inside the
# company sandbox's virtual file table, and only these prefixes are allowed.
ALLOWED_FILE_PREFIXES = ("notes/",)
ALLOWED_FILE_MAX_LENGTH = 200

# --- Web / services --------------------------------------------------------
WEB_ORIGIN = os.getenv("WEB_ORIGIN", "http://127.0.0.1:3000").rstrip("/")
ARTIFACT_DIR = ROOT / "data" / "artifacts"
