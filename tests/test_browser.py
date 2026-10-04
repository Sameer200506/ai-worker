"""Browser-automation tests for the agent's Playwright tool layer.

These tests start a real sandbox server, enable the browser tools, and drive the
simulated company app exactly the way the agent would: open a page, log in, read
the rendered text, follow a link to the invoice, and capture evidence. They also
assert the security boundary that keeps navigation inside the sandbox.
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from backend.database.models import TaskState
from backend.tools.registry import execute_tool

pytestmark = pytest.mark.browser

ROOT = Path(__file__).resolve().parents[1]
SANDBOX_PORT = 3399
SANDBOX_BASE = f"http://127.0.0.1:{SANDBOX_PORT}"


def _health_ok(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=2) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def sandbox_server():
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "apps.sandbox.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(SANDBOX_PORT),
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    healthy = False
    while time.time() < deadline:
        if _health_ok(SANDBOX_BASE):
            healthy = True
            break
        if proc.poll() is not None:
            raise RuntimeError("sandbox server exited during startup")
        time.sleep(0.3)
    if not healthy:
        proc.terminate()
        raise RuntimeError("sandbox server did not become healthy")

    # Deterministic seed data for the browser session.
    urllib.request.urlopen(
        urllib.request.Request(f"{SANDBOX_BASE}/api/admin/reset", method="POST"), timeout=10
    ).read()

    yield SANDBOX_BASE
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def browser_tools(monkeypatch, sandbox_server):
    from backend import config
    from backend.tools import browser

    monkeypatch.setattr(browser, "USE_BROWSER", True)
    monkeypatch.setattr(browser, "SANDBOX_URL", sandbox_server)
    monkeypatch.setattr(config, "SANDBOX_URL", sandbox_server)
    try:
        browser._ensure_page()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Playwright chromium is not installed: {exc}")
    yield sandbox_server
    browser.close()


def _run(state: TaskState, name: str, **arguments):
    return execute_tool(name, arguments, state)


def test_agent_logs_in_and_reads_latest_acme_invoice(browser_tools):
    base = browser_tools
    state = TaskState(task_id="task_browser", goal="Find the latest Acme invoice in the browser")

    assert _run(state, "browser.open", url=f"{base}/login").success
    assert _run(state, "browser.type", selector="input[name=username]", text="demo").success
    assert _run(state, "browser.type", selector="input[name=password]", text="demo123").success
    assert _run(state, "browser.click", selector="#login-submit").success

    dashboard = _run(state, "browser.read")
    assert dashboard.success
    assert "Dashboard" in dashboard.data["text"]

    assert _run(state, "browser.open", url=f"{base}/invoices?company=Acme+Corporation").success
    listing = _run(state, "browser.read")
    assert listing.success
    assert "INV-1042" in listing.data["text"]

    assert _run(state, "browser.click", selector="a[href='/invoices/INV-1042']").success
    detail = _run(state, "browser.read")
    assert detail.success
    assert "84,250" in detail.data["text"]
    assert "Acme Corporation" in detail.data["text"]

    assert _run(state, "browser.go_back").success

    shot = _run(state, "browser.screenshot")
    assert shot.success
    artifact = Path(shot.data["path"])
    assert artifact.exists()
    assert artifact.stat().st_size > 0


def test_browser_navigation_is_restricted_to_the_sandbox(browser_tools):
    state = TaskState(task_id="task_browser_block", goal="Try to leave the sandbox")
    result = _run(state, "browser.open", url="https://example.com")
    assert not result.success
    assert result.error is not None
    assert result.error.type == "PERMISSION_DENIED"
