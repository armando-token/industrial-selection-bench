"""Unit and integration tests for API service and Ingest/Knowledge modules.

Adheres strictly to MEGAPLAN.md §3.2, §4.3, §5.1, §5.2, §5.3, §19, §22.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from tests.conftest import requires_real_manuals

def _clear_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure blocked-provider assertions are not polluted by host .env exports."""
    for key in (
        "TYPESAFE_API_KEY",
        "JEV_API_KEY",
        "LLM_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

from fastapi.testclient import TestClient

from industrial_lab.api.app import app
from industrial_lab.schemas import (
    CheckStatus,
    ExtractionStatus,
    Fact,
    Port,
    QueryRequest,
    Relation,
    RelationType,
    SourceSpan,
    TechnicalVerdict,
)
from industrial_lab.ingest.documents import (
    compute_file_sha256,
    get_document_page,
    get_document_page_spans,
    ingest_documents,
    load_pages,
    load_spans,
)
from industrial_lab.ingest.facts import (
    extract_facts,
    load_facts,
    save_facts,
)
from industrial_lab.ingest.review import (
    export_review,
    import_review,
)
from industrial_lab.knowledge.store import (
    KnowledgeStore,
    build_knowledge_store,
)
from industrial_lab.knowledge.graph import (
    KnowledgeGraph,
    build_knowledge_graph,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ==============================================================================
# 1. Ingest Documents Tests
# ==============================================================================

def test_document_ingestion_and_spans(tmp_path: Path) -> None:
    """Tests document parsing, sha256 hashing, and span generation on synthetic fixtures (isolated)."""
    # Test SHA256 helper
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello Industrial Lab", encoding="utf-8")
    h = compute_file_sha256(test_file)
    assert len(h) == 64

    root = Path(__file__).resolve().parent.parent
    syn_catalog = root / "data" / "manifests" / "catalog.synthetic.yaml"
    syn_raw = root / "data" / "synthetic_fixtures" / "raw"
    if not syn_raw.exists() or not any(syn_raw.iterdir()):
        syn_raw = root / "data" / "raw"
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    # Ingest synthetic fixtures into tmp — do NOT overwrite live real G2 pages
    res = ingest_documents(catalog_path=syn_catalog, raw_dir=syn_raw, pages_dir=pages_dir)
    assert res["documents_count"] >= 3
    assert res["pages_count"] >= 3
    assert res["spans_count"] > 0

    pages = load_pages(pages_dir / "pages.jsonl")
    assert len(pages) >= 3
    spans = load_spans(pages_dir / "spans.jsonl")
    assert len(spans) > 0

    s0 = spans[0]
    assert isinstance(s0, SourceSpan)
    assert s0.span_id.startswith(s0.document_id)
    assert s0.pdf_page_index >= 1
    assert len(s0.text) > 0

    page_rec = get_document_page("D1_P1_MANUAL", 2, pages_path=pages_dir / "pages.jsonl")
    assert page_rec is not None
    assert page_rec["pdf_page_index"] == 2
    doc_spans = get_document_page_spans(
        "D1_P1_MANUAL", 2, spans_path=pages_dir / "spans.jsonl"
    )
    assert len(doc_spans) > 0


# ==============================================================================
# 2. Ingest Facts Tests
# ==============================================================================

def test_fact_extraction_and_persistence(tmp_path: Path) -> None:
    """Tests schema-guided fact extraction on synthetic fixture pages (not live real PDFs)."""
    root = Path(__file__).resolve().parent.parent
    syn_spans = root / "data" / "pages" / "spans.synthetic.jsonl"
    if not syn_spans.exists():
        syn_spans = root / "data" / "synthetic_fixtures" / "pages" / "spans.jsonl"
    facts = extract_facts(spans_path=syn_spans)
    assert len(facts) > 0

    for f in facts:
        assert isinstance(f, Fact)
        assert f.fact_id.startswith("F_")
        assert f.product_id in ("P1", "P2", "P3")
        assert len(f.evidence_ids) > 0
        assert f.extraction_status == ExtractionStatus.auto_extracted

    temp_facts_file = tmp_path / "test_facts.jsonl"
    save_facts(facts, temp_facts_file)
    loaded = load_facts(temp_facts_file)
    assert len(loaded) == len(facts)
    assert loaded[0].fact_id == facts[0].fact_id


# ==============================================================================
# 3. Review Workflow Tests
# ==============================================================================

def test_review_export_and_import(tmp_path: Path) -> None:
    """Tests review-export and review-import flow (§22)."""
    export_file = tmp_path / "pending_review.json"
    reviewed_file = tmp_path / "reviewed_facts.jsonl"

    exp_res = export_review(output_review_file=export_file)
    assert exp_res["status"] == "exported"
    assert export_file.is_file()

    with open(export_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "facts" in data
    assert len(data["facts"]) > 0

    # Mark first fact as rejected to test filtering
    data["facts"][0]["review_decision"] = "rejected"
    with open(export_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    imported_facts = import_review(review_file=export_file, output_reviewed_file=reviewed_file)
    assert len(imported_facts) == len(data["facts"]) - 1
    for f in imported_facts:
        assert f.extraction_status == ExtractionStatus.reviewed
        assert f.reviewed_by is not None


# ==============================================================================
# 4. Knowledge Store Tests
# ==============================================================================

def test_knowledge_store(tmp_path: Path) -> None:
    """Tests SQLite and JSON knowledge store CRUD operations."""
    db_file = tmp_path / "test_knowledge.db"
    store = KnowledgeStore(db_path=db_file)

    # 1. Product
    store.save_product("P1", sku="SKU-1", manufacturer="Corp", exact_model="M-1")
    p1 = store.get_product("P1")
    assert p1 is not None
    assert p1["sku"] == "SKU-1"

    # 2. Fact
    f1 = Fact(
        fact_id="F100",
        product_id="P1",
        property="voltage_nominal",
        value=24.0,
        unit="V",
        extraction_status=ExtractionStatus.reviewed,
        document_revision="revA",
    )
    store.save_fact(f1)
    got_f = store.get_fact("F100")
    assert got_f is not None
    assert got_f.value == 24.0

    facts_p = store.get_facts_by_product("P1")
    assert len(facts_p) >= 1

    # 3. Port
    p_port = Port(
        port_id="P1_PORT_1",
        product_id="P1",
        direction="in",
        physical_interface="terminal",
        signal_kind="power_dc",
    )
    store.save_port(p_port)
    got_port = store.get_port("P1_PORT_1")
    assert got_port is not None
    assert got_port.signal_kind == "power_dc"

    # 4. Relation
    rel = Relation(
        relation_id="REL_1",
        source_id="P1",
        target_id="P1_PORT_1",
        relation_type=RelationType.HAS_PORT,
    )
    store.save_relation(rel)
    rels = store.get_relations(source_id="P1")
    assert len(rels) == 1

    # Build knowledge store helper
    populated_store = build_knowledge_store(db_path=tmp_path / "pop_store.db")
    assert len(populated_store.list_products()) == 3
    assert len(populated_store.list_all_ports()) >= 5
    assert len(populated_store.list_all_relations()) >= 5


# ==============================================================================
# 5. Knowledge Graph Tests
# ==============================================================================

def test_knowledge_graph(tmp_path: Path) -> None:
    """Tests NetworkX graph modeling of components, ports, and relations."""
    store = build_knowledge_store(db_path=tmp_path / "kg_store.db")
    kg = build_knowledge_graph(store)

    assert len(kg.graph.nodes) >= 8
    assert len(kg.graph.edges) >= 5

    # Test port discovery
    p1_ports = kg.get_product_ports("P1")
    assert len(p1_ports) >= 2

    # Candidate connections between P3 current loop out and P1 analog in
    candidates = kg.find_candidate_connections("P3_PORT_ANALOG_LOOP", "P1_PORT_ANALOG_IN")
    assert len(candidates) == 1
    assert candidates[0]["candidate"] is True

    # JSON save and load roundtrip
    graph_json = tmp_path / "graph.json"
    kg.save_graph(graph_json)
    kg_reloaded = KnowledgeGraph.load_graph(graph_json)
    assert len(kg_reloaded.graph.nodes) == len(kg.graph.nodes)


# ==============================================================================
# 6. API Endpoints Tests
# ==============================================================================

def test_api_healthz_and_readyz(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests GET /healthz and GET /readyz endpoints (§19)."""
    _clear_provider_keys(monkeypatch)

    # Healthz
    r = client.get("/healthz")
    assert r.status_code == 200
    data = r.json()
    assert data["alive"] is True
    assert data["service"] == "industrial-selection-lab-api"

    # Readyz: JEV blocked (no key) -> ready=True with note (§19)
    r = client.get("/readyz")
    assert r.status_code == 200
    ready_data = r.json()
    assert ready_data["ready"] is True
    assert "structured_jev blocked (no API key)" in ready_data["note"]
    assert ready_data["components"]["catalog"]["ready"] is True
    assert ready_data["components"]["providers"]["structured_jev"]["blocked"] is True


def test_api_engines(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests GET /api/engines discovery (§19)."""
    _clear_provider_keys(monkeypatch)

    r = client.get("/api/engines")
    assert r.status_code == 200
    engines = r.json()["engines"]
    assert len(engines) >= 6

    engines_by_id = {e["id"]: e for e in engines}
    assert "structured_jev" in engines_by_id
    assert "rag_llm" in engines_by_id
    assert "scrape_llm" in engines_by_id
    assert "structured_rules" in engines_by_id
    assert "rag_llm_dryrun" in engines_by_id
    assert "scrape_llm_dryrun" in engines_by_id

    # Verify statuses and notes
    jev = engines_by_id["structured_jev"]
    assert jev["status"] == "blocked"
    assert jev["blocked"] is True
    assert jev["note"] == "structured_jev blocked (no API key)"

    rag = engines_by_id["rag_llm"]
    assert rag["status"] == "unavailable"
    assert "Missing LLM API key" in rag["note"]

    scrape = engines_by_id["scrape_llm"]
    assert scrape["status"] == "unavailable"
    assert "Missing LLM API key" in scrape["note"]

    rules = engines_by_id["structured_rules"]
    assert rules["status"] == "available"
    assert rules["blocked"] is False
    assert rules["note"] == "Ready (local evaluation)"

    rag_dry = engines_by_id["rag_llm_dryrun"]
    assert rag_dry["status"] == "available"
    assert rag_dry["blocked"] is False
    assert "Dry-run mode (synthetic fixtures, non-official)" in rag_dry["note"]

    scrape_dry = engines_by_id["scrape_llm_dryrun"]
    assert scrape_dry["status"] == "available"
    assert scrape_dry["blocked"] is False
    assert "Dry-run mode (synthetic fixtures, non-official)" in scrape_dry["note"]


def test_api_query_dryruns(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests POST /api/query for dry-run engines (rag_llm_dryrun and scrape_llm_dryrun)."""
    _clear_provider_keys(monkeypatch)
    # 1. RAG Dry-run
    req_rag = {
        "request_id": "req-dryrun-rag-1",
        "query_text": "Controlador FX-CPU-24V y fuente 24V",
        "engine": "rag_llm_dryrun",
        "mode": "M1",
        "scenario_id": "S0001",
        "catalog_version": "v1",
        "knowledge_version": "v1",
        "requested_product_ids": ["P1", "P2"],
        "include_quote": True,
    }
    r = client.post("/api/query", json=req_rag)
    assert r.status_code == 200
    resp_rag = r.json()
    assert resp_rag["engine"] == "rag_llm_dryrun"
    assert resp_rag["execution_status"] == "completed"
    assert resp_rag["technical_verdict"] == "COMPATIBLE"
    assert resp_rag["quote"] is not None

    # 2. Scrape Dry-run
    req_scrape = {
        "request_id": "req-dryrun-scrape-1",
        "query_text": "Controlador FX-CPU-24V",
        "engine": "scrape_llm_dryrun",
        "mode": "M1",
        "scenario_id": "S0001",
        "catalog_version": "v1",
        "knowledge_version": "v1",
        "requested_product_ids": ["P1"],
        "include_quote": True,
    }
    r2 = client.post("/api/query", json=req_scrape)
    assert r2.status_code == 200
    resp_scrape = r2.json()
    assert resp_scrape["engine"] == "scrape_llm_dryrun"
    assert resp_scrape["execution_status"] == "completed"
    assert resp_scrape["technical_verdict"] == "COMPATIBLE"
    assert resp_scrape["quote"] is not None


def test_api_query_system_a_blocked(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests POST /api/query for System A when JEV key is missing.
    
    Adheres to MEGAPLAN §0.1, §9.5, §19:
    Returns provider_error and does NOT mock JEV.
    """
    _clear_provider_keys(monkeypatch)

    req_payload = {
        "request_id": "req-unit-jev-1",
        "query_text": "Fuente 24V para PLC",
        "engine": "structured_jev",
        "mode": "M1",
        "scenario_id": "S0001",
        "catalog_version": "v1",
        "knowledge_version": "v1",
        "include_quote": False,
    }
    r = client.post("/api/query", json=req_payload)
    assert r.status_code == 200
    resp = r.json()
    assert resp["engine"] == "structured_jev"
    assert resp["execution_status"] == "provider_error"
    assert resp["technical_verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "Error 30" in resp["summary"]
    assert "structured_jev blocked (no API key)" in resp["summary"]


def test_api_query_structured_rules(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests POST /api/query for ablation structured_rules with quote generation."""
    _clear_provider_keys(monkeypatch)
    req_payload = {
        "request_id": "req-unit-rules-1",
        "query_text": "Controlador FX-CPU-24V",
        "engine": "structured_rules",
        "mode": "M1",
        "scenario_id": "S0001",
        "catalog_version": "v1",
        "knowledge_version": "v1",
        "requested_product_ids": ["P1"],
        "include_quote": True,
        "requirements": [
            {
                "requirement_id": "supply_voltage_nominal_v",
                "kind": "exact_property",
                "operator": "eq",
                "target": 24.0,
                "unit": "V",
                "hard": True,
            }
        ],
    }
    r = client.post("/api/query", json=req_payload)
    assert r.status_code == 200
    resp = r.json()
    assert resp["execution_status"] == "completed"
    assert resp["technical_verdict"] == "COMPATIBLE"
    assert len(resp["checks"]) == 1
    assert resp["checks"][0]["status"] == "PASS"
    assert resp["quote"] is not None
    assert len(resp["quote"]["lines"]) >= 1


@requires_real_manuals
def test_api_documents_endpoint(client: TestClient) -> None:
    """Tests GET /api/documents/{document_id}/pages/{page} endpoint (§19)."""
    # Prefer live real G2 pages; fall back to synthetic id if present
    import json
    from pathlib import Path as _P
    pages_file = _P(__file__).resolve().parent.parent / "data" / "pages" / "pages.jsonl"
    if not pages_file.exists() or pages_file.stat().st_size == 0:
        pages_file = _P(__file__).resolve().parent.parent / "data" / "synthetic_fixtures" / "pages" / "pages.jsonl"
    doc_id = None
    page_idx = 1
    sample_text = ""
    with open(pages_file, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("pdf_page_index", 0) >= 1 and rec.get("text", "").strip():
                doc_id = rec["document_id"]
                page_idx = int(rec["pdf_page_index"])
                sample_text = rec["text"][:40]
                if page_idx >= 1:
                    break
    assert doc_id, "pages.jsonl must contain at least one page"
    r = client.get(f"/api/documents/{doc_id}/pages/{page_idx}")
    assert r.status_code == 200
    data = r.json()
    assert data["document_id"] == doc_id
    assert data["page"] == page_idx
    assert len(data["text"]) > 0
    assert len(data["spans"]) >= 0

    # Non-existing document
    r404 = client.get("/api/documents/NON_EXISTENT_DOC/pages/1")
    assert r404.status_code == 404


def test_api_demo_ui(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests GET /demo endpoint rendering demo UI (§19.1)."""
    _clear_provider_keys(monkeypatch)
    r = client.get("/demo")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    html = r.text
    assert "Industrial Selection Lab" in html
    assert "PRECIO Y STOCK SIMULADOS" in html
    assert "NON-OFFICIAL" in html
    assert "P_X4" in html and "P_THT" in html and "P_UHEAT" in html
    assert "Horner" in html or "HE-X4" in html


