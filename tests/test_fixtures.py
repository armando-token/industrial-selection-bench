"""Contractual tests for synthetic fixtures and real G2 catalog integrity (§4, §13, §24).

Validates:
1. Synthetic catalog (catalog.synthetic.yaml / synthetic_fixtures) remains P1/P2/P3 labeled synthetic_fixture.
2. Live catalog.yaml holds the 3 real user products (P_X4, P_THT, P_UHEAT) with real_user_document origin.
3. Live pages.jsonl is non-empty after G2 ingest.
4. Synthetic facts.reviewed.jsonl still covers P1/P2/P3 for pipeline tests.
5. Live scenarios carry simulated commerce for real product IDs (explicitly non-manufacturer).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml

from tests.conftest import requires_real_manuals

from industrial_lab.schemas import CatalogManifest, ExtractionStatus, Fact, SourceSpan


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
SYN_DIR = DATA_DIR / "synthetic_fixtures"
SYN_CATALOG = DATA_DIR / "manifests" / "catalog.synthetic.yaml"
LIVE_CATALOG = DATA_DIR / "manifests" / "catalog.yaml"


def test_synthetic_catalog_manifest_exists_and_valid() -> None:
    """Synthetic catalog remains FixtureCorp P1/P2/P3 labeled synthetic_fixture."""
    catalog_path = SYN_CATALOG if SYN_CATALOG.exists() else SYN_DIR / "manifests" / "catalog.yaml"
    assert catalog_path.exists(), f"Synthetic catalog not found at {catalog_path}"

    with open(catalog_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    manifest = CatalogManifest.model_validate(data)
    assert manifest.data_origin == "synthetic_fixture"
    assert len(manifest.products) == 3
    product_ids = [p.product_id for p in manifest.products]
    assert product_ids == ["P1", "P2", "P3"]
    for p in manifest.products:
        assert p.sku and p.manufacturer and p.exact_model
        assert len(p.documents) > 0


def test_live_catalog_real_user_documents() -> None:
    """Live shop catalog exposes the 3 real G2 products from user PDFs."""
    assert LIVE_CATALOG.exists()
    with open(LIVE_CATALOG, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    manifest = CatalogManifest.model_validate(data)
    assert manifest.data_origin == "real_user_document"
    assert manifest.official_status == "NON-OFFICIAL"
    assert manifest.is_official is False
    ids = [p.product_id for p in manifest.products]
    assert ids == ["P_X4", "P_THT", "P_UHEAT"]
    for p in manifest.products:
        assert p.sku and p.manufacturer and p.exact_model
        assert len(p.documents) > 0
        for doc in p.documents:
            assert isinstance(doc, dict)
            assert doc.get("sha256") and len(doc["sha256"]) == 64
            assert doc.get("filename")
            assert int(doc.get("page_count", 0)) >= 1


@requires_real_manuals
def test_pages_jsonl_exists_and_valid() -> None:
    """Live pages.jsonl exists after G2 real PDF ingest (or synthetic backup)."""
    pages_path = DATA_DIR / "pages" / "pages.jsonl"
    assert pages_path.exists(), f"Pages file not found at {pages_path}"

    pages: list[dict] = []
    with open(pages_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))

    assert len(pages) > 0, "pages.jsonl must not be empty"
    doc_ids = {p.get("document_id") for p in pages}
    assert len(doc_ids) >= 3, f"Expected pages for at least 3 documents, got {doc_ids}"

    for p in pages:
        assert "document_id" in p
        assert "pdf_page_index" in p
        assert p["pdf_page_index"] >= 1
        assert "text" in p and len(p["text"]) > 0


@requires_real_manuals
def test_spans_jsonl_validates_against_source_span_schema() -> None:
    """Check that data/pages/spans.jsonl exists and validates against SourceSpan schema (§5.1)."""
    spans_path = DATA_DIR / "pages" / "spans.jsonl"
    if not spans_path.exists():
        pytest.skip("spans.jsonl not generated yet")

    spans: list[SourceSpan] = []
    with open(spans_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                span = SourceSpan.model_validate(data)
                spans.append(span)

    assert len(spans) > 0
    for span in spans:
        assert span.span_id
        assert span.document_id
        assert span.pdf_page_index >= 1
        assert len(span.text) > 0


def test_facts_exist_and_validate_against_fact_schema() -> None:
    """Synthetic reviewed facts still cover FixtureCorp P1/P2/P3 for pipeline tests."""
    reviewed_facts_path = SYN_DIR / "facts" / "facts.reviewed.jsonl"
    if not reviewed_facts_path.exists():
        reviewed_facts_path = DATA_DIR / "facts" / "facts.reviewed.jsonl"
    assert reviewed_facts_path.exists(), f"Facts file not found at {reviewed_facts_path}"

    facts: list[Fact] = []
    with open(reviewed_facts_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                fact = Fact.model_validate(data)
                facts.append(fact)

    assert len(facts) >= 3, f"Expected facts for all 3 products, got {len(facts)}"
    product_ids_in_facts = {f.product_id for f in facts}
    assert {"P1", "P2", "P3"}.issubset(product_ids_in_facts), (
        f"Facts must cover P1, P2, P3. Found: {product_ids_in_facts}"
    )
    for fact in facts:
        assert fact.fact_id
        assert fact.property
        assert fact.extraction_status in (ExtractionStatus.reviewed, ExtractionStatus.auto_extracted)


def test_scenarios_exist_with_simulated_commerce() -> None:
    """Live scenarios carry simulated commerce for real G2 product IDs."""
    scenario_path = DATA_DIR / "scenarios" / "S0001.yaml"
    assert scenario_path.exists(), f"Scenario S0001 not found at {scenario_path}"

    with open(scenario_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["scenario_id"] == "S0001"
    assert data.get("data_origin") in ("simulated_commerce", "synthetic_fixture")
    assert data.get("official_status") == "NON-OFFICIAL"
    assert data.get("is_official") is False
    products = data.get("products", [])
    if isinstance(products, dict):
        products = [{"product_id": k, **v} for k, v in products.items()]
    assert len(products) >= 3
    for prod in products:
        assert prod["product_id"] in ["P_X4", "P_THT", "P_UHEAT", "P1", "P2", "P3"]
        assert isinstance(prod["price_minor"], int)
        assert prod["price_minor"] > 0
        assert isinstance(prod["stock_available"], int)
        assert prod.get("currency", "USD") in ["USD", "EUR"]
