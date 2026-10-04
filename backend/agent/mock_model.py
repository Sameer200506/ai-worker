"""Mock model used for offline tests and local development without an API key.

IMPORTANT
---------
This file is a **test double for the LLM**, not part of the agent runtime. It stands in
for the model so the real loop (safety layer, tool dispatch, message history, recovery,
approval, verification) can be exercised deterministically and offline.

It contains goal-reading reasoning only because a model reads the goal. None of this
code is imported by the runtime decision path: ``backend/runtime/loop.py`` only ever
calls ``LLMClient.create_response``. ``LLM_PROVIDER=mock`` selects it explicitly and
never implicitly.

``backend/agent/llm.py`` re-exports ``MockLLMClient`` for backwards compatibility.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from backend.agent.llm import LLMDecision


Policy = Callable[[list[dict[str, Any]]], "tuple[str, dict[str, Any]] | LLMDecision | None"]


class MockLLMClient:
    """Scripted model client.

    Two modes:
    - ``decisions``: an ordered script of tool calls (used to prove the runtime).
    - ``auto_pilot=True``: a capability-driven simulator that reacts to the tool
      observations in the message history.
    """

    name: str = "mock"

    def __init__(
        self,
        decisions: list[tuple[str, dict[str, Any]] | LLMDecision] | None = None,
        policy: Policy | None = None,
        auto_pilot: bool = False,
        model: str = "mock-model",
        goal: str | None = None,
    ) -> None:
        self.decisions = list(decisions or [])
        self.policy = policy
        self.auto_pilot = auto_pilot
        self.model = model
        self.goal_override = goal
        self.calls: list[list[dict[str, Any]]] = []

    # -- helpers ------------------------------------------------------------
    def _normalize(self, value: tuple[str, dict[str, Any]] | LLMDecision | None) -> LLMDecision:
        if value is None:
            return LLMDecision(tool="task.complete", summary="No further action.", arguments={"summary": "Done."}, model=self.model)
        if isinstance(value, LLMDecision):
            if not value.model:
                value.model = self.model
            value.provider = value.provider or self.name
            return value
        tool, arguments = value
        return LLMDecision(
            tool=tool,
            arguments=arguments or {},
            summary=f"Proposed {tool}.",
            model=self.model,
            provider=self.name,
        )

    @staticmethod
    def _observations(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Every tool result the runtime has returned so far, oldest first."""
        results: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "tool":
                continue
            try:
                content = json.loads(message.get("content", "{}"))
            except (TypeError, ValueError):
                content = {}
            results.append({"name": message.get("name"), "content": content if isinstance(content, dict) else {}})
        return results

    @staticmethod
    def _successful(results: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
        return [item for item in results if item["name"] == name and item["content"].get("success")]

    @staticmethod
    def _last_call_arguments(messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                function = (message["tool_calls"][0].get("function") or {})
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except (TypeError, ValueError):
                    arguments = {}
                return function.get("name"), arguments if isinstance(arguments, dict) else {}
        return None

    @staticmethod
    def _effective_goal(messages: list[dict[str, Any]]) -> str:
        """The most recent operator instruction.

        Deterministic runtime notices (recovery policy) are user-role messages too, so
        they are skipped: they are not operator intent.
        """
        goal = ""
        for message in messages:
            if message.get("role") != "user":
                continue
            content = str(message.get("content", ""))
            if content.startswith("SYSTEM RECOVERY NOTICE") or content.startswith("No tool call was returned"):
                continue
            goal = content
        return goal

    # -- simulated reasoning ------------------------------------------------
    def _simulate(self, messages: list[dict[str, Any]]) -> LLMDecision:
        goal = self.goal_override or self._effective_goal(messages)
        lowered = goal.lower()
        results = self._observations(messages)
        last = results[-1] if results else None
        last_error = (last["content"].get("error") or {}) if last else {}

        # --- failure handling: read the error, pick a mitigation -------------
        if last_error.get("type") in {"AUTH_EXPIRED", "AUTH_FAILED"}:
            return LLMDecision(
                tool="company.login",
                arguments={},
                summary="The company session expired. Re-authenticating before retrying.",
                model=self.model,
                provider=self.name,
            )
        if last_error.get("type") == "NETWORK_ERROR":
            previous = self._last_call_arguments(messages)
            if previous:
                tool, arguments = previous
                return LLMDecision(
                    tool=tool,
                    arguments=arguments,
                    summary="Transient network failure. Retrying the same action.",
                    model=self.model,
                    provider=self.name,
                )

        # --- ambiguity ------------------------------------------------------
        if self._is_ambiguous(lowered):
            return LLMDecision(
                tool="task.clarify",
                arguments={
                    "question": "I need more information about what you want me to do with Acme. "
                    "Should I find and process an invoice, or update the customer record?"
                },
                summary="The request does not specify an outcome, so I am asking for clarification.",
                model=self.model,
                provider=self.name,
            )

        company_name = self._target_company(lowered, goal)

        if self._is_customer_goal(lowered):
            return self._simulate_customer(company_name, lowered, results)
        return self._simulate_invoice(company_name, results)

    @staticmethod
    def _is_ambiguous(lowered_goal: str) -> bool:
        if "do something" in lowered_goal:
            return True
        intent_words = {
            "process",
            "update",
            "record",
            "summar",
            "find",
            "handle",
            "create",
            "set",
            "change",
            "report",
        }
        return not any(word in lowered_goal for word in intent_words)

    @staticmethod
    def _is_customer_goal(lowered_goal: str) -> bool:
        return "customer" in lowered_goal or "status" in lowered_goal

    @staticmethod
    def _target_company(lowered_goal: str, original_goal: str) -> str:
        """Extract the company the operator named (a model would read it, not pattern-match)."""
        for needle, name in (("beta", "Beta Industries"), ("nimbus", "Nimbus Labs")):
            if needle in lowered_goal:
                return name
        match = re.search(r"\bfrom\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)", original_goal)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
        match = re.search(r"\bfor\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*)", original_goal)
        if match:
            return match.group(1).strip()
        return "Acme Corporation"

    def _simulate_customer(self, company_name: str, lowered_goal: str, results: list[dict[str, Any]]) -> LLMDecision:
        requested_status = "Follow Up"
        match = re.search(r"status to ([A-Za-z ]+)", lowered_goal, re.IGNORECASE)
        if match:
            requested_status = re.sub(r"\s+", " ", match.group(1)).strip().strip(".").title()

        searches = self._successful(results, "company.search_customer")
        if not searches:
            return LLMDecision(
                tool="company.search_customer",
                arguments={"query": company_name},
                summary=f"Looking up the customer record for {company_name}.",
                model=self.model,
                provider=self.name,
            )

        customers = (searches[-1]["content"].get("data") or {}).get("customers") or []
        customer_id = customers[0]["customer_id"] if customers else "CUST-102"

        if not self._successful(results, "company.get_customer"):
            return LLMDecision(
                tool="company.get_customer",
                arguments={"customer_id": customer_id},
                summary=f"Reading {customer_id} to confirm the current status.",
                model=self.model,
                provider=self.name,
            )

        updates = self._successful(results, "company.update_customer")
        if not updates:
            return LLMDecision(
                tool="company.update_customer",
                arguments={"customer_id": customer_id, "status": requested_status},
                summary=f"Setting {customer_id} status to {requested_status}.",
                model=self.model,
                provider=self.name,
            )

        if len(self._successful(results, "company.get_customer")) < 2:
            return LLMDecision(
                tool="company.get_customer",
                arguments={"customer_id": customer_id},
                summary="Re-reading the customer record so the update can be verified independently.",
                model=self.model,
                provider=self.name,
            )

        return LLMDecision(
            tool="task.complete",
            arguments={"summary": f"{company_name} ({customer_id}) status set to {requested_status}."},
            summary="The customer record is updated and re-read. Requesting verification.",
            model=self.model,
            provider=self.name,
        )

    def _simulate_invoice(self, company_name: str, results: list[dict[str, Any]]) -> LLMDecision:
        searches = self._successful(results, "company.search_invoice")
        if not searches:
            return LLMDecision(
                tool="company.search_invoice",
                arguments={"company": company_name, "status": "Pending", "sort": "latest"},
                summary=f"Searching for the newest pending invoice from {company_name}.",
                model=self.model,
                provider=self.name,
            )

        invoices = (searches[-1]["content"].get("data") or {}).get("invoices") or []
        pending = [row for row in invoices if row.get("status") == "Pending"] or invoices
        invoice_id = max(
            pending,
            key=lambda row: (row.get("issued_at") or "", row.get("invoice_id") or ""),
        )["invoice_id"]

        retrieved = self._successful(results, "company.get_invoice")
        if not retrieved:
            return LLMDecision(
                tool="company.get_invoice",
                arguments={"invoice_id": invoice_id},
                summary=f"Retrieving {invoice_id} to read the amount and due date.",
                model=self.model,
                provider=self.name,
            )

        invoice = retrieved[-1]["content"].get("data") or {}

        if not self._successful(results, "company.create_finance_record"):
            return LLMDecision(
                tool="company.create_finance_record",
                arguments={
                    "invoice_id": invoice.get("invoice_id", invoice_id),
                    "company": invoice.get("company", company_name),
                    "amount": int(invoice.get("amount") or 0),
                    "due_date": invoice.get("due_date", ""),
                },
                summary=f"Recording {invoice_id} in finance using the values read from the invoice.",
                model=self.model,
                provider=self.name,
            )

        if not self._successful(results, "company.get_finance_record"):
            return LLMDecision(
                tool="company.get_finance_record",
                arguments={"invoice_id": invoice.get("invoice_id", invoice_id)},
                summary="Reading the finance record back from the finance system.",
                model=self.model,
                provider=self.name,
            )

        return LLMDecision(
            tool="task.complete",
            arguments={"summary": f"{invoice_id} recorded in finance."},
            summary="The finance record has been read back. Requesting independent verification.",
            model=self.model,
            provider=self.name,
        )

    # -- LLMClient interface ------------------------------------------------
    async def create_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMDecision:
        self.calls.append(json.loads(json.dumps(messages, default=str)))

        if self.policy is not None:
            scripted = self.policy(messages)
            if scripted is not None:
                return self._normalize(scripted)

        if self.decisions:
            return self._normalize(self.decisions.pop(0))

        if self.auto_pilot:
            return self._simulate(messages)

        return LLMDecision(
            tool="task.complete",
            summary="Script exhausted.",
            arguments={"summary": "Task complete."},
            model=self.model,
            provider=self.name,
        )


# Backwards-compatible aliases used by the existing test-suite.
FakeLLMClient = MockLLMClient
ScriptedLLMClient = MockLLMClient
