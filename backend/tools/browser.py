from __future__ import annotations

from typing import Any

from backend.config import ARTIFACT_DIR, BROWSER_CHANNEL, SANDBOX_URL, USE_BROWSER
from backend.database.models import ToolResult
from backend.permissions.policies import resolve_sandbox_url, validate_browser_url
from backend.tools.base import fail, ok

# SANDBOX_URL is re-exported so the sandbox origin stays overridable at runtime.
__all__ = [
    "SANDBOX_URL",
    "dispatch",
    "open_url",
    "click",
    "type_text",
    "read",
    "screenshot",
    "go_back",
    "close",
]

_page = None
_browser = None
_playwright = None


def _ensure_page():
    global _page, _browser, _playwright
    if not USE_BROWSER:
        raise RuntimeError("browser disabled")
    from playwright.sync_api import sync_playwright

    if _page is None:
        _playwright = sync_playwright().start()
        _browser = _launch(_playwright)
        _page = _browser.new_page()
    return _page


def _launch(playwright):
    """Launch a browser, preferring the configured channel then bundled Chromium.

    Falls back to an installed Chrome/Edge so the tool works on machines where the
    Playwright browser download is unavailable.
    """
    candidates: list[str | None] = []
    if BROWSER_CHANNEL:
        candidates.append(BROWSER_CHANNEL)
    candidates.append(None)  # bundled Chromium, if present
    if "chrome" not in candidates:
        candidates.append("chrome")
    last_error: Exception | None = None
    for channel in candidates:
        kwargs: dict[str, object] = {"headless": True}
        if channel:
            kwargs["channel"] = channel
        try:
            return playwright.chromium.launch(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise last_error or RuntimeError("no browser available")


def open_url(url: str | None = None) -> ToolResult:
    target = resolve_sandbox_url(url or "/login")
    err = validate_browser_url(target)
    if err:
        return fail("browser.open", err, "PERMISSION_DENIED", False)
    if not USE_BROWSER:
        return ok("browser.open", f"Browser open skipped in API mode. Target would be {target}.", data={"url": target})
    try:
        page = _ensure_page()
        page.goto(target)
        return ok("browser.open", f"Opened {target}.", data={"url": page.url, "title": page.title()})
    except Exception as exc:  # noqa: BLE001
        return fail("browser.open", str(exc), "NETWORK_ERROR", True)


def click(selector: str) -> ToolResult:
    if not USE_BROWSER:
        return ok("browser.click", f"Would click {selector}.", data={"selector": selector})
    try:
        page = _ensure_page()
        page.click(selector)
        return ok("browser.click", f"Clicked {selector}.", data={"url": page.url})
    except Exception as exc:  # noqa: BLE001
        return fail("browser.click", str(exc), "UNEXPECTED_STATE", True)


def type_text(selector: str, text: str) -> ToolResult:
    if not USE_BROWSER:
        return ok("browser.type", f"Would type into {selector}.", data={"selector": selector})
    try:
        page = _ensure_page()
        page.fill(selector, text)
        return ok("browser.type", f"Typed into {selector}.", data={"selector": selector})
    except Exception as exc:  # noqa: BLE001
        return fail("browser.type", str(exc), "UNEXPECTED_STATE", True)


def read() -> ToolResult:
    if not USE_BROWSER:
        return ok("browser.read", "Browser read skipped in API mode.", data={"text": ""})
    try:
        page = _ensure_page()
        text = page.inner_text("body")
        return ok("browser.read", "Captured page text.", data={"url": page.url, "text": text[:4000]})
    except Exception as exc:  # noqa: BLE001
        return fail("browser.read", str(exc), "UNEXPECTED_STATE", True)


def screenshot(task_id: str) -> ToolResult:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{task_id}-screenshot.png"
    if not USE_BROWSER:
        path.write_bytes(b"")
        return ok("browser.screenshot", "Screenshot placeholder stored.", data={"path": str(path)})
    try:
        page = _ensure_page()
        page.screenshot(path=str(path))
        return ok("browser.screenshot", "Captured screenshot.", data={"path": str(path)}, artifacts=[{"type": "screenshot", "value": str(path)}])
    except Exception as exc:  # noqa: BLE001
        return fail("browser.screenshot", str(exc), "UNKNOWN_ERROR", False)


def go_back() -> ToolResult:
    if not USE_BROWSER:
        return ok("browser.go_back", "Would go back.", data={})
    try:
        page = _ensure_page()
        page.go_back()
        return ok("browser.go_back", f"Navigated back to {page.url}.", data={"url": page.url})
    except Exception as exc:  # noqa: BLE001
        return fail("browser.go_back", str(exc), "UNEXPECTED_STATE", True)


def close() -> None:
    """Tear down the persistent browser session. Safe to call repeatedly."""
    global _page, _browser, _playwright
    for resource in (_page, _browser):
        if resource is not None:
            try:
                resource.close()
            except Exception:  # noqa: BLE001
                pass
    if _playwright is not None:
        try:
            _playwright.stop()
        except Exception:  # noqa: BLE001
            pass
    _page = None
    _browser = None
    _playwright = None


def dispatch(name: str, arguments: dict[str, Any], task_id: str) -> ToolResult:
    if name == "browser.open":
        return open_url(arguments.get("url"))
    if name == "browser.click":
        return click(arguments["selector"])
    if name == "browser.type":
        return type_text(arguments["selector"], arguments["text"])
    if name == "browser.read":
        return read()
    if name == "browser.screenshot":
        return screenshot(task_id)
    if name == "browser.go_back":
        return go_back()
    return fail(name, "Unknown browser tool.", "UNKNOWN_ERROR", False)
