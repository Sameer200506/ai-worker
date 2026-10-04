from __future__ import annotations

import copy
import json
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "store.json"
_lock = threading.Lock()

SEED_INVOICES = [
    {
        "invoice_id": "INV-1038",
        "company": "Acme Corporation",
        "amount": 15500,
        "due_date": "2026-08-01",
        "status": "Pending",
        "issued_at": "2026-07-02",
    },
    {
        "invoice_id": "INV-1039",
        "company": "Acme Corporation",
        "amount": 41200,
        "due_date": "2026-09-01",
        "status": "Paid",
        "issued_at": "2026-08-01",
    },
    {
        "invoice_id": "INV-1040",
        "company": "Beta Industries",
        "amount": 92100,
        "due_date": "2026-09-10",
        "status": "Pending",
        "issued_at": "2026-08-12",
    },
    {
        "invoice_id": "INV-1041",
        "company": "Acme Corporation",
        "amount": 62400,
        "due_date": "2026-10-08",
        "status": "Pending",
        "issued_at": "2026-09-08",
    },
    {
        "invoice_id": "INV-1042",
        "company": "Acme Corporation",
        "amount": 84250,
        "due_date": "2026-10-15",
        "status": "Pending",
        "issued_at": "2026-09-20",
    },
]

SEED_CUSTOMERS = [
    {
        "customer_id": "CUST-102",
        "name": "Acme Corporation",
        "email": "accounts@acme.example",
        "status": "Active",
        "contact": "Raj Mehta",
        "last_contact": "2026-09-20",
        "finance_category": "Accounts Payable",
    },
    {
        "customer_id": "CUST-201",
        "name": "Beta Industries",
        "email": "ap@beta.example",
        "status": "Active",
        "contact": "Priya Shah",
        "last_contact": "2026-08-14",
        "finance_category": "Accounts Payable",
    },
    {
        "customer_id": "CUST-310",
        "name": "Nimbus Labs",
        "email": "billing@nimbus.example",
        "status": "Lead",
        "contact": "Omar Ali",
        "last_contact": "2026-09-02",
        "finance_category": "Accounts Receivable",
    },
]

SEED_FILES = [
    {
        "path": "notes/acme-vendor.txt",
        "content": "Acme Corporation vendor onboarding complete. Finance portal category: Accounts Payable. Primary contact Raj Mehta.",
    },
    {
        "path": "notes/process-guide.txt",
        "content": "Invoice processing: locate the latest pending invoice, extract amount and due date, then create an Accounts Payable finance record.",
    },
]


def _empty_state() -> dict[str, Any]:
    return {
        "invoices": copy.deepcopy(SEED_INVOICES),
        "customers": copy.deepcopy(SEED_CUSTOMERS),
        "finance": [],
        "files": copy.deepcopy(SEED_FILES),
        "sessions": {},
        "finance_sessions": {},
        "failure_mode": "",
        "finance_fail_once": False,
        "network_fail_once": False,
    }


class CompanyStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DATA_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.reset()

    def reset(self, failure_mode: str = "") -> None:
        state = _empty_state()
        state["failure_mode"] = failure_mode
        # Arm the one-shot transient failure immediately so reset() and
        # set_failure_mode() behave identically for every failure mode.
        state["network_fail_once"] = failure_mode == "network_error"
        self._write(state)

    def _read(self) -> dict[str, Any]:
        with _lock:
            return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, state: dict[str, Any]) -> None:
        with _lock:
            self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def set_failure_mode(self, mode: str) -> None:
        state = self._read()
        state["failure_mode"] = mode or ""
        state["finance_fail_once"] = False
        state["network_fail_once"] = bool(mode == "network_error")
        self._write(state)

    def failure_mode(self) -> str:
        return str(self._read().get("failure_mode") or "")

    def login(self, username: str, password: str) -> dict[str, Any]:
        if username != "demo" or password != "demo123":
            return {"success": False, "error": "Invalid credentials"}
        token = uuid.uuid4().hex
        finance_token = uuid.uuid4().hex
        state = self._read()
        state["sessions"][token] = {
            "username": username,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        state["finance_sessions"][finance_token] = {
            "username": username,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "valid": True,
        }
        self._write(state)
        return {"success": True, "token": token, "finance_token": finance_token, "username": username}

    def session_valid(self, token: str | None) -> bool:
        if not token:
            return False
        return token in self._read().get("sessions", {})

    def finance_valid(self, finance_token: str | None) -> bool:
        if not finance_token:
            return False
        session = self._read().get("finance_sessions", {}).get(finance_token)
        return bool(session and session.get("valid"))

    def expire_finance(self, finance_token: str | None) -> None:
        if not finance_token:
            return
        state = self._read()
        session = state.get("finance_sessions", {}).get(finance_token)
        if session:
            session["valid"] = False
            self._write(state)

    def search_invoices(self, company: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        invoices = self._read()["invoices"]
        results = invoices
        if company:
            needle = company.lower()
            results = [row for row in results if needle in row["company"].lower()]
        if status:
            results = [row for row in results if row["status"].lower() == status.lower()]
        return copy.deepcopy(results)

    def get_invoice(self, invoice_id: str) -> dict[str, Any] | None:
        for row in self._read()["invoices"]:
            if row["invoice_id"].lower() == invoice_id.lower():
                return copy.deepcopy(row)
        return None

    def search_customers(self, query: str | None = None) -> list[dict[str, Any]]:
        customers = self._read()["customers"]
        if not query:
            return copy.deepcopy(customers)
        needle = query.lower()
        return copy.deepcopy(
            [
                row
                for row in customers
                if needle in row["name"].lower()
                or needle in row["customer_id"].lower()
                or needle in row["email"].lower()
            ]
        )

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        for row in self._read()["customers"]:
            if row["customer_id"].lower() == customer_id.lower():
                return copy.deepcopy(row)
        return None

    def update_customer(self, customer_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        state = self._read()
        for row in state["customers"]:
            if row["customer_id"].lower() == customer_id.lower():
                allowed = {"status", "email", "contact", "last_contact"}
                for key, value in fields.items():
                    if key in allowed:
                        row[key] = value
                self._write(state)
                return copy.deepcopy(row)
        return None

    def create_finance_record(
        self,
        invoice_id: str,
        company: str,
        amount: int,
        due_date: str,
        finance_token: str | None,
        force_mismatch: bool = False,
    ) -> dict[str, Any]:
        state = self._read()
        mode = state.get("failure_mode") or ""
        token = finance_token or ""
        if mode == "finance_session_expired" and not state.get("finance_fail_once"):
            state["finance_fail_once"] = True
            session = state.get("finance_sessions", {}).get(token)
            if session:
                session["valid"] = False
            self._write(state)
            return {
                "success": False,
                "error": {"type": "AUTH_EXPIRED", "recoverable": True, "message": "Finance session expired."},
            }
        session = state.get("finance_sessions", {}).get(token)
        if not session or not session.get("valid"):
            return {
                "success": False,
                "error": {"type": "AUTH_EXPIRED", "recoverable": True, "message": "Finance session expired."},
            }
        if mode == "network_error" and state.get("network_fail_once"):
            state["network_fail_once"] = False
            self._write(state)
            return {
                "success": False,
                "error": {"type": "NETWORK_ERROR", "recoverable": True, "message": "Temporary network error."},
            }
        stored_amount = (amount - 2000) if (force_mismatch or mode == "verification_mismatch") else amount
        record = {
            "record_id": f"FIN-{8800 + len(state['finance']) + 1}",
            "invoice_id": invoice_id,
            "company": company,
            "amount": stored_amount,
            "due_date": due_date,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "Posted",
        }
        state["finance"].append(record)
        self._write(state)
        return {"success": True, "record": copy.deepcopy(record)}

    def get_finance_record(self, record_id: str) -> dict[str, Any] | None:
        for row in self._read()["finance"]:
            if row["record_id"].lower() == record_id.lower():
                return copy.deepcopy(row)
        return None

    def find_finance_by_invoice(self, invoice_id: str) -> dict[str, Any] | None:
        matches = [row for row in self._read()["finance"] if row["invoice_id"].lower() == invoice_id.lower()]
        return copy.deepcopy(matches[-1]) if matches else None

    def list_finance(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._read()["finance"])

    def search_files(self, query: str) -> list[dict[str, Any]]:
        needle = query.lower()
        return [
            {"path": row["path"]}
            for row in self._read()["files"]
            if needle in row["path"].lower() or needle in row["content"].lower()
        ]

    def read_file(self, path: str) -> dict[str, Any] | None:
        for row in self._read()["files"]:
            if row["path"] == path:
                return copy.deepcopy(row)
        return None


def format_inr(amount: int) -> str:
    return f"₹{amount:,}"


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def is_overdue(invoice: dict[str, Any], today: date | None = None) -> bool:
    today = today or date(2026, 10, 4)
    return invoice["status"].lower() != "paid" and parse_date(invoice["due_date"]) < today


store = CompanyStore()
