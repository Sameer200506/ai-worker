from backend.tools import company
from apps.sandbox.store import store


def setup_function():
    store.reset("")


def test_login_and_search():
    login = company.login("demo", "demo123")
    assert login.success
    invoices = company.search_invoice("Acme Corporation")
    assert invoices.success
    assert len(invoices.data["invoices"]) >= 4


def test_bad_login():
    result = company.login("demo", "wrong")
    assert not result.success
    assert result.error.type == "AUTH_FAILED"
