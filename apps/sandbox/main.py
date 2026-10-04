from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import Cookie, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.sandbox.store import format_inr, is_overdue, store

app = FastAPI(title="CentrAlign Sandbox")
BASE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE / "templates"))
templates.env.globals["format_inr"] = format_inr
templates.env.globals["is_overdue"] = is_overdue
(BASE / "static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")


class LoginBody(BaseModel):
    username: str
    password: str


class FinanceBody(BaseModel):
    invoice_id: str
    company: str
    amount: int
    due_date: str


class CustomerPatch(BaseModel):
    status: str | None = None
    email: str | None = None
    contact: str | None = None


def _auth(token: str | None) -> bool:
    return store.session_valid(token)


def page(request: Request, name: str, token: str | None, **context: Any) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name=name,
        context={"logged_in": _auth(token), "username": "demo" if _auth(token) else None, **context},
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(sandbox_session):
        return RedirectResponse("/login", status_code=302)
    return RedirectResponse("/dashboard", status_code=302)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, sandbox_session: str | None = Cookie(default=None), error: str | None = None):
    if _auth(sandbox_session):
        return RedirectResponse("/dashboard", status_code=302)
    return page(request, "login.html", sandbox_session, error=error)


@app.post("/login")
def login_form(username: str = Form(...), password: str = Form(...)):
    result = store.login(username, password)
    if not result["success"]:
        return RedirectResponse("/login?error=1", status_code=303)
    response = RedirectResponse("/dashboard", status_code=303)
    response.set_cookie("sandbox_session", result["token"], httponly=False)
    response.set_cookie("finance_session", result["finance_token"], httponly=False)
    return response


@app.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("sandbox_session")
    response.delete_cookie("finance_session")
    return response


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(sandbox_session):
        return RedirectResponse("/login", status_code=302)
    invoices = store.search_invoices()
    customers = store.search_customers()
    finance = store.list_finance()
    return page(
        request,
        "dashboard.html",
        sandbox_session,
        invoice_count=len(invoices),
        customer_count=len(customers),
        finance_count=len(finance),
        pending=[row for row in invoices if row["status"] == "Pending"][:5],
    )


@app.get("/invoices", response_class=HTMLResponse)
def invoices_page(
    request: Request,
    company: str | None = None,
    sandbox_session: str | None = Cookie(default=None),
):
    if not _auth(sandbox_session):
        return RedirectResponse("/login", status_code=302)
    rows = store.search_invoices(company=company)
    return page(request, "invoices.html", sandbox_session, invoices=rows, company=company or "")


@app.get("/invoices/{invoice_id}", response_class=HTMLResponse)
def invoice_detail(invoice_id: str, request: Request, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(sandbox_session):
        return RedirectResponse("/login", status_code=302)
    invoice = store.get_invoice(invoice_id)
    if not invoice:
        return page(request, "missing.html", sandbox_session, entity="Invoice", ident=invoice_id)
    return page(request, "invoice_detail.html", sandbox_session, invoice=invoice)


@app.get("/customers", response_class=HTMLResponse)
def customers_page(request: Request, q: str | None = None, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(sandbox_session):
        return RedirectResponse("/login", status_code=302)
    rows = store.search_customers(q)
    return page(request, "customers.html", sandbox_session, customers=rows, q=q or "")


@app.get("/customers/{customer_id}", response_class=HTMLResponse)
def customer_detail(customer_id: str, request: Request, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(sandbox_session):
        return RedirectResponse("/login", status_code=302)
    customer = store.get_customer(customer_id)
    if not customer:
        return page(request, "missing.html", sandbox_session, entity="Customer", ident=customer_id)
    return page(request, "customer_detail.html", sandbox_session, customer=customer)


@app.post("/customers/{customer_id}")
def customer_update_form(
    customer_id: str,
    status: str = Form(...),
    sandbox_session: str | None = Cookie(default=None),
):
    if not _auth(sandbox_session):
        return RedirectResponse("/login", status_code=302)
    store.update_customer(customer_id, {"status": status})
    return RedirectResponse(f"/customers/{customer_id}", status_code=303)


@app.get("/finance", response_class=HTMLResponse)
def finance_page(request: Request, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(sandbox_session):
        return RedirectResponse("/login", status_code=302)
    return page(request, "finance.html", sandbox_session, records=store.list_finance())


@app.get("/finance/{record_id}", response_class=HTMLResponse)
def finance_detail(record_id: str, request: Request, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(sandbox_session):
        return RedirectResponse("/login", status_code=302)
    record = store.get_finance_record(record_id)
    if not record:
        return page(request, "missing.html", sandbox_session, entity="Finance record", ident=record_id)
    return page(request, "finance_detail.html", sandbox_session, record=record)


@app.post("/finance")
def finance_create_form(
    invoice_id: str = Form(...),
    company: str = Form(...),
    amount: int = Form(...),
    due_date: str = Form(...),
    sandbox_session: str | None = Cookie(default=None),
    finance_session: str | None = Cookie(default=None),
):
    if not _auth(sandbox_session):
        return RedirectResponse("/login", status_code=302)
    result = store.create_finance_record(invoice_id, company, amount, due_date, finance_session)
    if not result.get("success"):
        return RedirectResponse("/finance?error=session", status_code=303)
    record_id = result["record"]["record_id"]
    return RedirectResponse(f"/finance/{record_id}", status_code=303)


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(sandbox_session):
        return RedirectResponse("/login", status_code=302)
    return page(request, "settings.html", sandbox_session, failure_mode=store.failure_mode())


@app.post("/api/auth/login")
def api_login(body: LoginBody):
    result = store.login(body.username, body.password)
    if not result["success"]:
        return JSONResponse({"success": False, "error": {"type": "AUTH_FAILED", "recoverable": True}}, status_code=401)
    return result


@app.get("/api/invoices")
def api_invoices(company: str | None = None, status: str | None = None, token: str | None = None, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(token or sandbox_session):
        return JSONResponse({"success": False, "error": {"type": "AUTH_EXPIRED", "recoverable": True}}, status_code=401)
    return {"success": True, "invoices": store.search_invoices(company, status)}


@app.get("/api/invoices/{invoice_id}")
def api_invoice(invoice_id: str, token: str | None = None, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(token or sandbox_session):
        return JSONResponse({"success": False, "error": {"type": "AUTH_EXPIRED", "recoverable": True}}, status_code=401)
    invoice = store.get_invoice(invoice_id)
    if not invoice:
        return JSONResponse({"success": False, "error": {"type": "NOT_FOUND", "recoverable": False}}, status_code=404)
    return {"success": True, "invoice": invoice}


@app.get("/api/customers")
def api_customers(q: str | None = None, token: str | None = None, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(token or sandbox_session):
        return JSONResponse({"success": False, "error": {"type": "AUTH_EXPIRED", "recoverable": True}}, status_code=401)
    return {"success": True, "customers": store.search_customers(q)}


@app.get("/api/customers/{customer_id}")
def api_customer(customer_id: str, token: str | None = None, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(token or sandbox_session):
        return JSONResponse({"success": False, "error": {"type": "AUTH_EXPIRED", "recoverable": True}}, status_code=401)
    customer = store.get_customer(customer_id)
    if not customer:
        return JSONResponse({"success": False, "error": {"type": "AUTH_EXPIRED" if False else "NOT_FOUND", "recoverable": False}}, status_code=404)
    return {"success": True, "customer": customer}


@app.patch("/api/customers/{customer_id}")
def api_customer_patch(
    customer_id: str,
    body: CustomerPatch,
    token: str | None = None,
    sandbox_session: str | None = Cookie(default=None),
):
    if not _auth(token or sandbox_session):
        return JSONResponse({"success": False, "error": {"type": "AUTH_EXPIRED", "recoverable": True}}, status_code=401)
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    customer = store.update_customer(customer_id, fields)
    if not customer:
        return JSONResponse({"success": False, "error": {"type": "NOT_FOUND", "recoverable": False}}, status_code=404)
    return {"success": True, "customer": customer}


@app.get("/api/finance")
def api_finance(token: str | None = None, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(token or sandbox_session):
        return JSONResponse({"success": False, "error": {"type": "AUTH_EXPIRED", "recoverable": True}}, status_code=401)
    return {"success": True, "records": store.list_finance()}


@app.get("/api/finance/{record_id}")
def api_finance_one(record_id: str, token: str | None = None, sandbox_session: str | None = Cookie(default=None)):
    if not _auth(token or sandbox_session):
        return JSONResponse({"success": False, "error": {"type": "AUTH_EXPIRED", "recoverable": True}}, status_code=401)
    record = store.get_finance_record(record_id)
    if not record:
        by_invoice = store.find_finance_by_invoice(record_id)
        if by_invoice:
            return {"success": True, "record": by_invoice}
        return JSONResponse({"success": False, "error": {"type": "NOT_FOUND", "recoverable": False}}, status_code=404)
    return {"success": True, "record": record}


@app.post("/api/finance")
def api_finance_create(
    body: FinanceBody,
    token: str | None = None,
    finance_token: str | None = None,
    sandbox_session: str | None = Cookie(default=None),
    finance_session: str | None = Cookie(default=None),
):
    if not _auth(token or sandbox_session):
        return JSONResponse({"success": False, "error": {"type": "AUTH_EXPIRED", "recoverable": True}}, status_code=401)
    result = store.create_finance_record(
        body.invoice_id,
        body.company,
        body.amount,
        body.due_date,
        finance_token or finance_session,
    )
    if not result.get("success"):
        return JSONResponse(result, status_code=401)
    return {"success": True, "record": result["record"]}


@app.post("/api/admin/reset")
def api_reset(mode: str = ""):
    store.reset(mode)
    return {"success": True, "failure_mode": mode}


@app.post("/api/admin/failure-mode")
def api_failure_mode(mode: str = ""):
    store.set_failure_mode(mode)
    return {"success": True, "failure_mode": mode}


@app.get("/health")
def health():
    return {"ok": True, "app": "sandbox"}
