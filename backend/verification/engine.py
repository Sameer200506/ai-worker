"""Independent verification.

Verification is deliberately separated from the agent:

- It never trusts the payload a mutating tool returned.
- It never asks the model whether it succeeded.
- It re-reads the *source of truth* from the company sandbox and compares it against the
  *persisted* state.

For a finance record the expected values are re-read from the invoice itself, so a model
that invents an amount, or a store that corrupts one, is both caught. Only this module can
set ``TaskState.status = "completed"``.
"""
from __future__ import annotations

from backend.database.models import CheckResult, TaskState, VerificationResult
from backend.tools import company


def verify(state: TaskState) -> VerificationResult:
    """Verify whatever outcome the task actually produced."""
    if state.artifacts.get("finance_record"):
        return _verify_finance(state)
    if state.artifacts.get("customer_updated") or (
        state.artifacts.get("requested_status") and state.artifacts.get("customer_id")
    ):
        return _verify_customer(state)
    return _verify_read_only(state)


# ---------------------------------------------------------------------------
# Finance record
# ---------------------------------------------------------------------------
def _verify_finance(state: TaskState) -> VerificationResult:
    record = state.artifacts.get("finance_record") or {}
    record_id = record.get("record_id")
    invoice_id = record.get("invoice_id") or (state.artifacts.get("invoice") or {}).get("invoice_id")

    persisted = company.get_finance_record(record_id=record_id, invoice_id=invoice_id)
    if not persisted.success or not persisted.data:
        return VerificationResult(
            verified=False,
            checks=[],
            observation=(
                f"VERIFICATION FAILED: finance record {record_id or invoice_id or 'unknown'} could not be "
                "read back from the finance system. The task is not completed."
            ),
        )
    actual = persisted.data

    # Expected values come from the invoice, re-read independently from the source system.
    source = company.get_invoice(actual.get("invoice_id") or invoice_id or "")
    if not source.success or not source.data:
        return VerificationResult(
            verified=False,
            checks=[],
            observation=(
                f"VERIFICATION FAILED: the invoice behind record {actual.get('record_id')} could not be "
                "re-read, so the stored values cannot be confirmed."
            ),
        )
    expected = source.data

    checks = [
        CheckResult(
            field="invoice_id",
            expected=expected.get("invoice_id"),
            actual=actual.get("invoice_id"),
            passed=actual.get("invoice_id") == expected.get("invoice_id"),
        ),
        CheckResult(
            field="company",
            expected=expected.get("company"),
            actual=actual.get("company"),
            passed=actual.get("company") == expected.get("company"),
        ),
        CheckResult(
            field="amount",
            expected=expected.get("amount"),
            actual=actual.get("amount"),
            passed=_same_amount(expected.get("amount"), actual.get("amount")),
        ),
        CheckResult(
            field="due_date",
            expected=expected.get("due_date"),
            actual=actual.get("due_date"),
            passed=actual.get("due_date") == expected.get("due_date"),
        ),
        CheckResult(
            field="record_id",
            expected=record_id,
            actual=actual.get("record_id"),
            passed=bool(record_id) and actual.get("record_id") == record_id,
        ),
    ]

    verified = all(check.passed for check in checks)
    if verified:
        observation = (
            f"VERIFICATION PASSED: finance record {actual.get('record_id')} matches invoice "
            f"{expected.get('invoice_id')} on every checked field."
        )
    else:
        mismatched = [check for check in checks if not check.passed]
        details = ", ".join(
            f"{check.field} expected {check.expected!r} but stored {check.actual!r}" for check in mismatched
        )
        observation = f"VERIFICATION FAILED: {details}. The task is not completed."

    return VerificationResult(verified=verified, checks=checks, observation=observation)


def _same_amount(expected: object, actual: object) -> bool:
    try:
        return int(expected) == int(actual)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return expected == actual


# ---------------------------------------------------------------------------
# Customer record
# ---------------------------------------------------------------------------
def _verify_customer(state: TaskState) -> VerificationResult:
    customer_id = state.artifacts.get("customer_id") or _last_argument(state, "customer_id")
    if not customer_id:
        return VerificationResult(
            verified=False,
            checks=[],
            observation="VERIFICATION FAILED: no customer was updated, so there is nothing to verify.",
        )

    # Expected status = the intent the operator saw and approved, never the tool's echo.
    expected_status = state.artifacts.get("requested_status") or _last_argument(state, "status")
    expected_email = state.artifacts.get("requested_email") or _last_argument(state, "email")
    expected_contact = state.artifacts.get("requested_contact") or _last_argument(state, "contact")

    persisted = company.get_customer(customer_id)
    if not persisted.success or not persisted.data:
        return VerificationResult(
            verified=False,
            checks=[],
            observation=f"VERIFICATION FAILED: customer {customer_id} could not be read back from the sandbox.",
        )
    actual = persisted.data

    checks = [
        CheckResult(
            field="customer_id",
            expected=customer_id,
            actual=actual.get("customer_id"),
            passed=actual.get("customer_id") == customer_id,
        )
    ]
    for field_name, expected_value in (
        ("status", expected_status),
        ("email", expected_email),
        ("contact", expected_contact),
    ):
        if expected_value in (None, ""):
            continue
        checks.append(
            CheckResult(
                field=field_name,
                expected=expected_value,
                actual=actual.get(field_name),
                passed=actual.get(field_name) == expected_value,
            )
        )

    verified = all(check.passed for check in checks)
    if verified:
        observation = (
            f"VERIFICATION PASSED: customer {customer_id} persisted as "
            f"status={actual.get('status')}."
        )
    else:
        mismatched = [check for check in checks if not check.passed]
        details = ", ".join(
            f"{check.field} expected {check.expected!r} but stored {check.actual!r}" for check in mismatched
        )
        observation = f"VERIFICATION FAILED: {details}. The task is not completed."

    return VerificationResult(verified=verified, checks=checks, observation=observation)


def _last_argument(state: TaskState, name: str) -> object | None:
    """The most recent argument the agent submitted for a mutating tool."""
    for tool_name in ("company.update_customer", "company.create_finance_record"):
        for action in reversed(state.completed_actions):
            if action.get("action") == tool_name:
                value = (action.get("arguments") or {}).get(name)
                if value not in (None, ""):
                    return value
    return None


# ---------------------------------------------------------------------------
# Read-only outcomes
# ---------------------------------------------------------------------------
def _verify_read_only(state: TaskState) -> VerificationResult:
    if not state.completed_actions:
        return VerificationResult(
            verified=False,
            checks=[],
            observation="VERIFICATION FAILED: the task produced no tool activity, so no outcome can be verified.",
        )
    return VerificationResult(
        verified=True,
        checks=[
            CheckResult(
                field="tools_executed",
                expected=">=1",
                actual=len(state.completed_actions),
                passed=True,
            ),
            CheckResult(
                field="failed_actions",
                expected=0,
                actual=len(state.failed_actions),
                passed=len(state.failed_actions) == 0,
            ),
        ],
        observation=(
            f"VERIFICATION PASSED: this task was read-only. {len(state.completed_actions)} tool call(s) "
            "succeeded against the live sandbox and nothing needed to be persisted."
        ),
    )
