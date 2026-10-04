from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

TaskStatus = Literal[
    "queued",
    "running",
    "awaiting_approval",
    "awaiting_clarification",
    "completed",
    "failed",
    "cancelled",
]


class PlanStep(BaseModel):
    """A step in the agent's self-reported plan.

    Steps are produced by the model as operational summaries, never as an executable
    workflow. Execution is always driven by live tool decisions.
    """

    id: str
    description: str
    status: Literal["pending", "running", "completed", "failed", "skipped"] = "pending"
    tool: str | None = None
    requires: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)


class ToolError(BaseModel):
    type: str
    recoverable: bool = False
    message: str | None = None


class ToolResult(BaseModel):
    success: bool
    tool: str
    data: Any = None
    observation: str
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    error: ToolError | None = None
    latency_ms: int | None = None


class CheckResult(BaseModel):
    field: str
    expected: Any
    actual: Any
    passed: bool


class VerificationResult(BaseModel):
    verified: bool
    checks: list[CheckResult] = Field(default_factory=list)
    observation: str = ""


class Decision(BaseModel):
    """Legacy structured decision shape kept for API compatibility."""

    summary: str
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    done: bool = False
    fail: bool = False
    reason: str | None = None


class LLMCall(BaseModel):
    """Observability record for one model call. Never stores prompts or secrets."""

    step: int
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    tool: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    validation: str | None = None
    at: str | None = None


class TaskState(BaseModel):
    task_id: str
    goal: str
    status: TaskStatus = "queued"
    current_step: str | None = None
    current_action: str | None = None
    progress: int = 0
    plan: Plan | None = None
    completed_actions: list[dict[str, Any]] = Field(default_factory=list)
    failed_actions: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    memory: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    approval_status: Literal["pending", "approved", "rejected"] | None = None
    approval_payload: dict[str, Any] | None = None
    verification_status: Literal["pending", "passed", "failed"] = "pending"
    verification: VerificationResult | None = None
    summary: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    # Real message history: system prompt, user goal, assistant tool calls, tool results.
    messages: list[dict[str, Any]] = Field(default_factory=list)
    # Observability
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_calls: list[LLMCall] = Field(default_factory=list)
    degraded: bool = False
    demo_mode: str = "normal"
    retry_count: int = 0
    recovery_count: int = 0
    step_count: int = 0
    last_error: dict[str, Any] | None = None
    pending_action: dict[str, Any] | None = None
    clarification_question: str | None = None
    clarification_answer: str | None = None
    cancel_requested: bool = False
    sandbox_token: str | None = None
    finance_token: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
