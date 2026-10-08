"""Tests for Shop Simulator following MEGAPLAN.md §6."""

import os
import pytest
from fastapi.testclient import TestClient

from industrial_lab.shop import fixtures
from industrial_lab.shop.app import app, control_app


@pytest.fixture(autouse=True)
def reset_fixtures():
    """Ensure clean fixture state before each test."""
    fixtures.reset_fixtures_state()
    # Reset LAB_CONTROL_TOKEN in env if set
    old_token = os.environ.get("LAB_CONTROL_TOKEN")
    yield
    fixtures.reset_fixtures_state()
    if old_token is not None:
        os.environ["LAB_CONTROL_TOKEN"] = old_token
    else:
        os.environ.pop("LAB_CONTROL_TOKEN", None)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def control_client():
    return TestClient(control_app)


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    # Verify simulation badge
    assert "CATÁLOGO DE SIMULACIÓN - PRECIOS Y STOCK SIMULADOS" in html
    # Verify 3 products are listed
    assert "P_X4" in html
    assert "P_THT" in html
    assert "P_UHEAT" in html
    # Verify links to product details
    assert "/products/P_X4" in html
    assert "/products/P_THT" in html
    assert "/products/P_UHEAT" in html
    # Verify price and stock indication
    assert "Stock:" in html or "disponibles" in html


def test_product_detail_page(client):
    response = client.get("/products/P_X4")
    assert response.status_code == 200
    html = response.text
    assert "CATÁLOGO DE SIMULACIÓN - PRECIOS Y STOCK SIMULADOS" in html
    assert "P_X4" in html
    assert any(term in html for term in ["Horner", "HE-X4", "P_X4"])
    # Verify specs extracted section
    assert "Especificaciones Técnicas Extraídas" in html
    # Verify document links
    assert "/documents/" in html

    # Non-existent product gives 404
    bad_resp = client.get("/products/NONEXISTENT")
    assert bad_resp.status_code == 404


def test_documents_pdf(client):
    response = client.get("/documents/D_X4_MANUAL_MAN1137.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    content = response.content
    assert content.startswith(b"%PDF-")

    # Document route without .pdf extension also works
    response_no_ext = client.get("/documents/D_X4_MANUAL_MAN1137")
    assert response_no_ext.status_code == 200
    assert response_no_ext.headers["content-type"] == "application/pdf"
    assert response_no_ext.content.startswith(b"%PDF-")


def test_api_catalog_no_curated_facts(client):
    response = client.get("/api/catalog")
    assert response.status_code == 200
    data = response.json()

    assert "catalog_id" in data
    assert "products" in data
    assert len(data["products"]) == 3

    for p in data["products"]:
        assert "product_id" in p
        assert "sku" in p
        assert "name" in p
        assert "url" in p
        assert "documents" in p
        # §6.1 / §3.1: NO curated technical facts in /api/catalog!
        assert "facts" not in p
        assert "specs" not in p
        assert "rules" not in p


def test_api_commerce(client):
    response = client.get("/api/commerce/P_X4")
    assert response.status_code == 200
    data = response.json()

    assert data["product_id"] == "P_X4"
    assert isinstance(data["price_minor"], int)
    assert data["price_minor"] > 0
    assert data["currency"] in {"EUR", "USD"}
    assert isinstance(data["stock_available"], int)
    assert data["is_simulated"] is True
    assert "revision" in data
    assert "observed_at" in data

    # 404 for unknown product
    bad_resp = client.get("/api/commerce/UNKNOWN")
    assert bad_resp.status_code == 404


def test_api_quotes_integer_arithmetic_and_stock(client):
    payload = {
        "lines": [
            {"product_id": "P_X4", "quantity": 2},
            {"product_id": "P_THT", "quantity": 3},
        ]
    }
    response = client.post("/api/quotes", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["is_simulated"] is True
    assert data["scenario_id"] == "S0001"
    assert "revision" in data
    assert len(data["lines"]) == 2

    l1 = data["lines"][0]
    assert l1["product_id"] == "P_X4"
    assert l1["quantity"] == 2
    assert isinstance(l1["unit_price_minor"], int)
    assert isinstance(l1["line_total_minor"], int)
    assert l1["line_total_minor"] == l1["unit_price_minor"] * 2
    assert l1["available"] is True

    l2 = data["lines"][1]
    assert l2["product_id"] == "P_THT"
    assert l2["quantity"] == 3
    assert l2["line_total_minor"] == l2["unit_price_minor"] * 3
    assert l2["available"] is True

    expected_subtotal = l1["line_total_minor"] + l2["line_total_minor"]
    assert data["subtotal_minor"] == expected_subtotal
    assert data["tax_minor"] == 0
    assert data["total_minor"] == expected_subtotal
    assert data["status"] == "valid"
    assert data["availability_status"] == "available"


def test_api_quotes_out_of_stock(client):
    # Request more than available stock (e.g. 9999 units)
    payload = {
        "lines": [
            {"product_id": "P_X4", "quantity": 9999},
        ]
    }
    response = client.post("/api/quotes", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["lines"][0]["available"] is False
    assert data["availability_status"] == "unavailable"
    assert data["status"] == "unavailable"


def test_api_quotes_invalid_inputs(client):
    # Quantity <= 0
    bad_qty = {"lines": [{"product_id": "P_X4", "quantity": 0}]}
    r1 = client.post("/api/quotes", json=bad_qty)
    assert r1.status_code == 422

    # Empty lines
    empty_lines = {"lines": []}
    r2 = client.post("/api/quotes", json=empty_lines)
    assert r2.status_code == 422

    # Unknown product
    bad_prod = {"lines": [{"product_id": "UNKNOWN_PRODUCT", "quantity": 1}]}
    r3 = client.post("/api/quotes", json=bad_prod)
    assert r3.status_code == 404


def test_scenario_control_lifecycle_and_auth(client):
    # 1. Check default scenario
    res = client.get("/control/scenario")
    assert res.status_code == 200
    assert res.json()["scenario_id"] == "S0001"
    rev1 = res.json()["revision"]

    # 2. Test auth protection when LAB_CONTROL_TOKEN is configured
    os.environ["LAB_CONTROL_TOKEN"] = "secret-token-123"

    # Request without token -> 403 Forbidden
    unauth = client.post("/control/scenario", json={"scenario_id": "S0002"})
    assert unauth.status_code == 403

    # Request with wrong token -> 403 Forbidden
    wrong_token = client.post(
        "/control/scenario",
        json={"scenario_id": "S0002"},
        headers={"X-Control-Token": "wrong-token"},
    )
    assert wrong_token.status_code == 403

    # Request with valid X-Control-Token -> 200 OK
    auth_resp = client.post(
        "/control/scenario",
        json={"scenario_id": "S0002"},
        headers={"X-Control-Token": "secret-token-123"},
    )
    assert auth_resp.status_code == 200
    assert auth_resp.json()["scenario_id"] == "S0002"
    assert auth_resp.json()["status"] == "activated"

    # Verify scenario was actually updated
    res2 = client.get("/control/scenario")
    assert res2.status_code == 200
    assert res2.json()["scenario_id"] == "S0002"
    rev2 = res2.json()["revision"]
    assert rev2 != rev1

    # Verify commerce endpoint now serves new scenario data
    # In S0002, P_THT stock is 0
    tht_comm = client.get("/api/commerce/P_THT").json()
    assert tht_comm["stock_available"] == 0
    assert tht_comm["revision"] == rev2

    # Switch back using Authorization: Bearer token header
    bearer_resp = client.post(
        "/control/scenario",
        json={"scenario_id": "S0001"},
        headers={"Authorization": "Bearer secret-token-123"},
    )
    assert bearer_resp.status_code == 200
    assert bearer_resp.json()["scenario_id"] == "S0001"


def test_scenario_control_invalid_scenario(client):
    os.environ.pop("LAB_CONTROL_TOKEN", None)
    res = client.post("/control/scenario", json={"scenario_id": "NON_EXISTENT_SCENARIO"})
    assert res.status_code == 400
    assert "not found in allowed scenarios" in res.json()["detail"]


def test_dedicated_control_server(control_client):
    # Dedicated control server on port 8082
    res = control_client.get("/healthz")
    assert res.status_code == 200

    scen = control_client.get("/control/scenario")
    assert scen.status_code == 200
    assert scen.json()["scenario_id"] == "S0001"
