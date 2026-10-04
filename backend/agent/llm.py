"""LLM provider abstraction.

This module is the *only* place that talks to a model vendor. It exposes a single
interface, ``LLMClient.create_response(messages, tools)``, and returns a structured
``LLMDecision``. The runtime depends on that interface only, never on a vendor SDK.

Design rules enforced here:
- Native structured tool/function calling. Tool definitions are sent as real
  ``tools`` payloads, never as prose inside the prompt.
- Real tool results are appended to the message history by the runtime and sent
  back verbatim on the next request.
- No silent fallback. If the provider is unconfigured or the call fails, a typed
  error is raised so the task fails loudly instead of pretending to be autonomous.
- Credentials come from the environment only.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from backend import config

#: Bounded transport retries with linear backoff. Transport faults only: an HTTP error
#: response is never retried, because a 400 will fail again and a 401 needs an operator.
LLM_MAX_ATTEMPTS = 3
LLM_RETRY_BACKOFF_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class LLMError(RuntimeError):
    """Base class for every LLM failure."""

    error_type = "LLM_ERROR"


class LLMUnavailable(LLMError):
    """No usable provider configuration (missing key, unknown provider)."""

    error_type = "LLM_UNAVAILABLE"


class LLMAuthError(LLMError):
    """Provider rejected the credentials."""

    error_type = "LLM_AUTH_ERROR"


class LLMConnectionError(LLMError):
    """Provider unreachable / timeout."""

    error_type = "LLM_CONNECTION_ERROR"


class LLMResponseError(LLMError):
    """Provider replied with something that is not a usable tool decision."""

    error_type = "LLM_MALFORMED_RESPONSE"


# ---------------------------------------------------------------------------
# Structured decision
# ---------------------------------------------------------------------------
class LLMDecision(BaseModel):
    """One structured decision produced by the model."""

    summary: str = ""
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    tool_call_id: str | None = None
    model: str | None = None
    provider: str | None = None
    latency_ms: int | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    validation_error: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    """The interface the runtime depends on."""

    name: str
    model: str

    async def create_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMDecision: ...


# ---------------------------------------------------------------------------
# Message hygiene
# ---------------------------------------------------------------------------
def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise the message history for an OpenAI-style chat-completions API.

    Two guarantees matter here:
    1. Tool messages are only kept when they answer an assistant ``tool_calls``
       message that is still present. Providers reject orphaned tool results,
       which is what happens when context compression cuts a pair in half.
    2. An assistant message with tool calls always carries a string ``content``.
    """
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in {"system", "user"}:
            cleaned.append({"role": role, "content": str(message.get("content", ""))})
        elif role == "assistant":
            assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                assistant["tool_calls"] = tool_calls
            cleaned.append(assistant)
        elif role == "tool":
            cleaned.append(
                {
                    "role": "tool",
                    "tool_call_id": str(message.get("tool_call_id", "")),
                    "name": message.get("name", ""),
                    "content": str(message.get("content", "")),
                }
            )

    answered: set[str] = set()
    for message in cleaned:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                answered.add(str(call.get("id", "")))

    paired: list[dict[str, Any]] = []
    for message in cleaned:
        if message.get("role") == "tool" and str(message.get("tool_call_id", "")) not in answered:
            continue
        paired.append(message)

    # A leading orphan assistant tool call (its tool result was compressed away)
    # must also be dropped, otherwise the provider sees an unanswered call.
    while paired and paired[0].get("role") == "assistant" and paired[0].get("tool_calls"):
        first_call = str(paired[0]["tool_calls"][0].get("id", ""))
        if any(m.get("role") == "tool" and str(m.get("tool_call_id")) == first_call for m in paired):
            break
        paired.pop(0)

    return paired


def _decode_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Parse tool-call arguments, reporting malformed payloads honestly."""
    if raw in (None, ""):
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return {}, f"Tool arguments were not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return {}, "Tool arguments must be a JSON object."
    return parsed, None


# ---------------------------------------------------------------------------
# OpenAI-compatible providers (OpenRouter, OpenAI, Ollama, ...)
# ---------------------------------------------------------------------------
class OpenAICompatibleClient:
    """Native structured tool calling against any OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, base_url: str, model: str, provider: str = "openai") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = provider or "openai"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.name == "openrouter":
            headers["HTTP-Referer"] = config.LLM_APP_URL
            headers["X-Title"] = config.LLM_APP_TITLE
        return headers

    async def create_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMDecision:
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": config.LLM_MAX_TOKENS,
            "messages": sanitize_messages(messages),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        started = time.perf_counter()
        response = None
        last_error: Exception | None = None
        for attempt in range(LLM_MAX_ATTEMPTS):
            try:
                async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )
                break
            except httpx.TimeoutException as exc:
                last_error = LLMConnectionError(f"LLM request timed out after {config.LLM_TIMEOUT_SECONDS}s: {exc}")
            except httpx.HTTPError as exc:
                last_error = LLMConnectionError(f"Could not reach the LLM provider: {exc}")
            if attempt < LLM_MAX_ATTEMPTS - 1:
                await asyncio.sleep(LLM_RETRY_BACKOFF_SECONDS * (attempt + 1))

        if response is None:
            raise last_error or LLMConnectionError("The LLM provider could not be reached.")

        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code in {401, 403}:
            raise LLMAuthError(
                f"LLM provider '{self.name}' rejected the API key (HTTP {response.status_code}). "
                "Check LLM_API_KEY."
            )
        if response.status_code == 429:
            raise LLMConnectionError(
                "LLM provider rate limit reached (HTTP 429). Wait and retry, or use a model with capacity."
            )
        if response.status_code >= 500:
            raise LLMConnectionError(f"LLM provider is unavailable (HTTP {response.status_code}).")
        if response.status_code >= 400:
            detail = response.text[:400]
            raise LLMResponseError(f"LLM provider error (HTTP {response.status_code}): {detail}")

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError(f"LLM provider returned a non-JSON body: {exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise LLMResponseError("LLM provider returned no choices.")

        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        content = message.get("content") or ""
        finish_reason = choices[0].get("finish_reason")

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return LLMDecision(
                text=content,
                summary=content[:400],
                model=self.model,
                provider=self.name,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )

        call = tool_calls[0]
        function = call.get("function") or {}
        arguments, parse_error = _decode_arguments(function.get("arguments"))
        if parse_error:
            # Surface the malformed payload to the runtime instead of guessing.
            return LLMDecision(
                tool=function.get("name"),
                arguments={},
                text=content,
                summary=content or f"Proposed {function.get('name')} with unreadable arguments.",
                tool_call_id=call.get("id"),
                model=self.model,
                provider=self.name,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                validation_error=parse_error,
            )

        return LLMDecision(
            tool=function.get("name"),
            arguments=arguments,
            text=content,
            summary=content or f"Proposed {function.get('name')}.",
            tool_call_id=call.get("id"),
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
class AnthropicClient:
    """Native tool calling against Anthropic's Messages API."""

    endpoint = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.name = "anthropic"
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")

    @staticmethod
    def _split(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        conversation: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                system_parts.append(str(message.get("content", "")))
            elif role == "user":
                conversation.append({"role": "user", "content": str(message.get("content", ""))})
            elif role == "assistant":
                content: list[dict[str, Any]] = []
                if message.get("content"):
                    content.append({"type": "text", "text": str(message["content"])})
                for call in message.get("tool_calls") or []:
                    function = call.get("function", {})
                    arguments, _ = _decode_arguments(function.get("arguments"))
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id", "call_0"),
                            "name": function.get("name"),
                            "input": arguments,
                        }
                    )
                conversation.append({"role": "assistant", "content": content or [{"type": "text", "text": "."}]})
            elif role == "tool":
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": str(message.get("tool_call_id", "")),
                                "content": str(message.get("content", "")),
                            }
                        ],
                    }
                )
        return "\n\n".join(system_parts), conversation

    async def create_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMDecision:
        import httpx

        system_prompt, conversation = self._split(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": 0,
            "system": system_prompt,
            "messages": conversation,
        }
        if tools:
            payload["tools"] = [
                {
                    "name": (tool.get("function") or {}).get("name"),
                    "description": (tool.get("function") or {}).get("description", ""),
                    "input_schema": (tool.get("function") or {}).get("parameters", {"type": "object", "properties": {}}),
                }
                for tool in tools
            ]

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise LLMConnectionError(f"LLM request timed out after {config.LLM_TIMEOUT_SECONDS}s: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(f"Could not reach the LLM provider: {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code in {401, 403}:
            raise LLMAuthError(f"Anthropic rejected the API key (HTTP {response.status_code}). Check LLM_API_KEY.")
        if response.status_code >= 400:
            raise LLMResponseError(f"Anthropic error (HTTP {response.status_code}): {response.text[:400]}")

        data = response.json()
        usage = data.get("usage") or {}
        for block in data.get("content") or []:
            if block.get("type") == "tool_use":
                return LLMDecision(
                    tool=block.get("name"),
                    arguments=block.get("input") or {},
                    tool_call_id=block.get("id"),
                    model=self.model,
                    provider=self.name,
                    latency_ms=latency_ms,
                    finish_reason=data.get("stop_reason"),
                    prompt_tokens=usage.get("input_tokens"),
                    completion_tokens=usage.get("output_tokens"),
                    summary=f"Proposed {block.get('name')}.",
                )
        text = " ".join(block.get("text", "") for block in data.get("content") or [] if block.get("type") == "text")
        return LLMDecision(
            text=text,
            summary=text[:400],
            model=self.model,
            provider=self.name,
            latency_ms=latency_ms,
            finish_reason=data.get("stop_reason"),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
SUPPORTED_PROVIDERS = {"openrouter", "openai", "openai-compatible", "ollama", "anthropic", "mock"}


def describe_configuration() -> dict[str, Any]:
    """Non-secret description of the LLM configuration, for /health and the UI."""
    client_type: str | None = None
    error: str | None = None
    try:
        build_client()
        client_type = "configured"
    except LLMUnavailable as exc:
        error = str(exc)
    return {
        "provider": config.LLM_PROVIDER or None,
        "model": config.LLM_MODEL,
        "base_url": config.LLM_BASE_URL,
        "api_key_present": bool(config.LLM_API_KEY),
        "configured": client_type == "configured",
        "error": error,
        "deterministic_fallback": config.DETERMINISTIC_FALLBACK,
    }


def build_client(task_state: Any | None = None) -> LLMClient:
    """Return the configured LLM client.

    Raises ``LLMUnavailable`` when no provider is usable. There is no implicit
    deterministic fallback: the runtime must fail loudly rather than simulate.
    """
    from backend.agent.mock_model import MockLLMClient

    provider = (config.LLM_PROVIDER or "").lower()

    if not provider:
        raise LLMUnavailable(
            "LLM unavailable: LLM_PROVIDER is not configured. "
            "Set LLM_PROVIDER, LLM_API_KEY and LLM_MODEL (see .env.example)."
        )

    if provider not in SUPPORTED_PROVIDERS:
        raise LLMUnavailable(
            f"LLM unavailable: unsupported LLM_PROVIDER '{provider}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_PROVIDERS))}."
        )

    if provider == "mock":
        # Scripted model simulator for offline tests. Never a production path.
        return MockLLMClient(auto_pilot=True)

    if not config.LLM_API_KEY:
        raise LLMUnavailable(
            "LLM unavailable: LLM_API_KEY is not configured for provider "
            f"'{provider}'. Autonomous execution requires a real LLM."
        )

    if provider == "anthropic":
        return AnthropicClient(api_key=config.LLM_API_KEY, model=config.LLM_MODEL, base_url=config.LLM_BASE_URL)

    return OpenAICompatibleClient(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        model=config.LLM_MODEL,
        provider=provider,
    )


# The mock model is a test double and lives in its own module; it is re-exported here
# so `from backend.agent.llm import MockLLMClient` keeps working.
from backend.agent.mock_model import FakeLLMClient, MockLLMClient, ScriptedLLMClient  # noqa: E402

__all__ = [
    "LLMClient",
    "LLMDecision",
    "LLMError",
    "LLMUnavailable",
    "LLMAuthError",
    "LLMConnectionError",
    "LLMResponseError",
    "OpenAICompatibleClient",
    "AnthropicClient",
    "MockLLMClient",
    "FakeLLMClient",
    "ScriptedLLMClient",
    "build_client",
    "describe_configuration",
    "sanitize_messages",
    "SUPPORTED_PROVIDERS",
]
