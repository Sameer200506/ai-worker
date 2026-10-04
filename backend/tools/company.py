from __future__ import annotations

from typing import Any

from apps.sandbox.store import store
from backend.database.models import ToolResult
from backend.tools.base import fail, ok


def login(username: str, password: str) -> ToolResult:
    result = store.login(username, password)
    if not result.get("success"):
        return fail("company.login", "Authentication failed.", "AUTH_FAILED", True)
    return ok(
        "company.login",
        "Authentication restored.",
        data={"username": result["username"]},
        artifacts=[
            {"type": "sandbox_token", "value": result["token"]},
            {"type": "finance_token", "value": result["finance_token"]},
        ],
    )


def search_invoice(company: str | None = None, status: str | None = None, sort: str | None = None) -> ToolResult:
    rows = store.search_invoices(company=company, status=status)
    if sort == "latest":
        rows = sorted(rows, key=lambda row: (row.get("issued_at") or "", row.get("invoice_id") or ""), reverse=True)
    elif sort == "oldest":
        rows = sorted(rows, key=lambda row: (row.get("issued_at") or "", row.get("invoice_id") or ""))
    elif sort == "amount_desc":
        rows = sorted(rows, key=lambda row: row.get("amount") or 0, reverse=True)
    if not rows:
        return fail(
            "company.search_invoice",
            f"No invoices found for company={company!r}.",
            "NOT_FOUND",
            False,
            data={"invoices": []},
        )
    return ok(
        "company.search_invoice",
        f"Found {len(rows)} invoice(s). The agent must choose the relevant one from issued_at/due_date.",
        data={"invoices": rows},
    )


def get_invoice(invoice_id: str) -> ToolResult:
    invoice = store.get_invoice(invoice_id)
    if not invoice:
        return fail("company.get_invoice", f"Invoice {invoice_id} was not found.", "NOT_FOUND", False)
    return ok(
        "company.get_invoice",
        f"Extracted invoice {invoice['invoice_id']} for {invoice['company']}.",
        data=invoice,
        artifacts=[{"type": "invoice", "value": invoice}],
    )


def search_customer(query: str) -> ToolResult:
    rows = store.search_customers(query)
    if not rows:
        return fail("company.search_customer", f"No customers matched {query!r}.", "NOT_FOUND", False, data={"customers": []})
    return ok("company.search_customer", f"Found {len(rows)} customer(s).", data={"customers": rows})


def get_customer(customer_id: str) -> ToolResult:
    customer = store.get_customer(customer_id)
    if not customer:
        return fail("company.get_customer", f"Customer {customer_id} was not found.", "NOT_FOUND", False)
    return ok(
        "company.get_customer",
        f"Loaded customer {customer['name']} ({customer['customer_id']}).",
        data=customer,
        artifacts=[{"type": "customer", "value": customer}],
    )


def update_customer(customer_id: str, status: str | None = None, **fields: Any) -> ToolResult:
    payload = {k: v for k, v in {"status": status, **fields}.items() if v is not None}
    if not payload:
        return fail("company.update_customer", "No valid customer fields were provided.", "INVALID_DATA", False)
    customer = store.update_customer(customer_id, payload)
    if not customer:
        return fail("company.update_customer", f"Customer {customer_id} was not found.", "NOT_FOUND", False)
    return ok(
        "company.update_customer",
        f"Updated customer {customer['name']} status to {customer.get('status')}.",
        data=customer,
        artifacts=[{"type": "customer", "value": customer}],
    )


def create_finance_record(
    invoice_id: str,
    company: str,
    amount: int,
    due_date: str,
    finance_token: str | None = None,
) -> ToolResult:
    if not invoice_id or not company or amount is None or not due_date:
        return fail("company.create_finance_record", "Invoice data is incomplete.", "INVALID_DATA", False)
    result = store.create_finance_record(invoice_id, company, int(amount), due_date, finance_token)
    if not result.get("success"):
        error = result.get("error") or {}
        return fail(
            "company.create_finance_record",
            error.get("message") or "Finance action failed.",
            error.get("type") or "UNKNOWN_ERROR",
            bool(error.get("recoverable")),
        )
    record = result["record"]
    return ok(
        "company.create_finance_record",
        f"Finance record {record['record_id']} created for {record['invoice_id']}.",
        data=record,
        artifacts=[{"type": "finance_record", "value": record}],
    )


def get_finance_record(record_id: str | None = None, invoice_id: str | None = None) -> ToolResult:
    record = None
    if record_id:
        record = store.get_finance_record(record_id)
    if not record and invoice_id:
        record = store.find_finance_by_invoice(invoice_id)
    if not record:
        return fail("company.get_finance_record", "Finance record was not found.", "NOT_FOUND", False)
    return ok(
        "company.get_finance_record",
        f"Read finance record {record['record_id']} from the finance system.",
        data=record,
        artifacts=[{"type": "finance_record", "value": record}],
    )
