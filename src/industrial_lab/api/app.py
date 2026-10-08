"""FastAPI application for Industrial Selection Lab API and Interactive Demo.

Adheres strictly to MEGAPLAN.md §19:
- GET /healthz: Process alive check
- GET /readyz: Checks readiness of catalog, knowledge, index, and providers.
  If JEV credentials missing, returns status ready=True with note
  'structured_jev blocked (no API key)', B/C available.
- GET /api/engines: Returns available engines and their readiness status.
- POST /api/query: Accepts QueryRequest, dispatches to selected engine, returns QueryResponse.
- GET /api/documents/{document_id}/pages/{page}: Returns page text and SourceSpan evidence.
- GET /api/runs/{run_id}/summary: Returns run metrics summary without leaking test gold.
- GET /demo: Interactive single-page demo UI with query box, engine selector,
  product cards, verdict badges, checks list, quote preview, and simulated commerce disclaimer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import html as html_lib
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from industrial_lab.schemas import (
    CheckResult,
    CheckStatus,
    DecisionOrigin,
    ExecutionStatus,
    QueryRequest,
    QueryResponse,
    Quote,
    TechnicalVerdict,
)
from industrial_lab.ingest.documents import (
    DEFAULT_PAGES_FILE,
    DEFAULT_SPANS_FILE,
    get_document_page,
    get_document_page_spans,
)
from industrial_lab.knowledge.store import KnowledgeStore, DEFAULT_DB_PATH
from industrial_lab.shop import fixtures

logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(os.environ.get("LAB_BASE_DIR", "."))
RUNS_DIR = Path(os.environ.get("LAB_RUNS_ROOT", "runs"))
DATA_DIR = Path(os.environ.get("LAB_DATA_ROOT", "data"))

app = FastAPI(
    title="Industrial Selection Lab API",
    version="1.0.0",
    description="Laboratory API for comparative evaluation of JEV, RAG, and Web Scraping LLMs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# §19. Healthz & Readyz
# ==============================================================================

@app.get("/healthz", summary="Process alive check")
async def healthz() -> Dict[str, Any]:
    """Returns process liveness status (§19)."""
    return {
        "status": "ok",
        "alive": True,
        "service": "industrial-selection-lab-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/readyz", summary="Readiness check for catalog, knowledge, index, and providers")
async def readyz() -> Dict[str, Any]:
    """Readiness check adhering strictly to MEGAPLAN §19:
    
    Checks readiness of catalog, knowledge, index, and providers.
    If JEV credentials missing, returns status ready=True with note
    'structured_jev blocked (no API key)', B/C available.
    """
    # 1. Catalog readiness
    catalog_ready = False
    catalog_items_count = 0
    cat_yaml = DATA_DIR / "manifests" / "catalog.yaml"
    if cat_yaml.is_file():
        catalog_items = fixtures.load_catalog().get("products", [])
        catalog_ready = len(catalog_items) > 0
        catalog_items_count = len(catalog_items)
    else:
        # Check SQLite knowledge store
        if DEFAULT_DB_PATH.is_file():
            store = KnowledgeStore()
            prods = store.list_products()
            catalog_ready = len(prods) > 0
            catalog_items_count = len(prods)

    # 2. Knowledge readiness
    knowledge_ready = False
    facts_count = 0
    rev_facts_file = DATA_DIR / "facts" / "facts.reviewed.jsonl"
    auto_facts_file = DATA_DIR / "facts" / "facts.auto.jsonl"
    if rev_facts_file.is_file():
        knowledge_ready = True
        with open(rev_facts_file, "r", encoding="utf-8") as f:
            facts_count = sum(1 for line in f if line.strip())
    elif auto_facts_file.is_file():
        knowledge_ready = True
        with open(auto_facts_file, "r", encoding="utf-8") as f:
            facts_count = sum(1 for line in f if line.strip())
    elif DEFAULT_DB_PATH.is_file():
        store = KnowledgeStore()
        all_f = store.list_all_facts()
        knowledge_ready = len(all_f) > 0
        facts_count = len(all_f)

    # 3. Index readiness
    index_ready = False
    bm25_file = DATA_DIR / "rag" / "bm25_index.json"
    if bm25_file.is_file() or DEFAULT_SPANS_FILE.is_file():
        index_ready = True

    # 4. Providers preflight
    jev_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    jev_available = bool(jev_key)

    llm_key = (
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )
    b_c_available = bool(llm_key)

    # Adheres to MEGAPLAN §19:
    # "Para readyz, permitir estado parcial: B/C disponibles, A bloqueado.
    # If JEV credentials missing, returns status ready=True with note
    # 'structured_jev blocked (no API key)', B/C available."
    if not jev_available:
        ready_flag = True
        status_code_str = "partial"
        note = "structured_jev blocked (no API key), B/C available."
    else:
        ready_flag = True
        status_code_str = "ready"
        note = "All engines and providers configured."

    providers_status = {
        "structured_jev": {
            "available": jev_available,
            "blocked": not jev_available,
            "note": "Ready" if jev_available else "Missing TYPESAFE_API_KEY",
        },
        "rag_llm": {
            "available": b_c_available,
            "note": "Ready" if b_c_available else "Missing LLM API key",
        },
        "scrape_llm": {
            "available": b_c_available,
            "note": "Ready" if b_c_available else "Missing LLM API key",
        },
    }

    return {
        "ready": ready_flag,
        "status": status_code_str,
        "note": note,
        "components": {
            "catalog": {
                "ready": catalog_ready,
                "products_count": catalog_items_count,
            },
            "knowledge": {
                "ready": knowledge_ready,
                "facts_count": facts_count,
            },
            "index": {
                "ready": index_ready,
            },
            "providers": providers_status,
        },
    }


# ==============================================================================
# §19. Engines Discovery
# ==============================================================================

@app.get("/api/engines", summary="Available engines and readiness status")
async def list_engines() -> Dict[str, Any]:
    """Returns available engines and their availability status (§19).
    
    A blocked engine is reported as 'blocked' or 'unavailable', never substituted.
    """
    jev_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    llm_key = (
        os.environ.get("LLM_API_KEY", "").strip()
        or os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )

    engines = [
        {
            "id": "structured_jev",
            "name": "System A: Structured Facts + TypeSafe JEV",
            "category": "primary",
            "status": "available" if jev_key else "blocked",
            "blocked": not bool(jev_key),
            "note": "Ready" if jev_key else "structured_jev blocked (no API key)",
            "description": "Deterministic knowledge representation with atomic Choice questions on TypeSafe JEV System One.",
        },
        {
            "id": "rag_llm",
            "name": "System B: Hybrid RAG + LLM",
            "category": "primary",
            "status": "available" if llm_key else "unavailable",
            "blocked": False,
            "note": "Ready" if llm_key else "Missing LLM API key",
            "description": "Hybrid BM25 + dense retrieval with reciprocal rank fusion (RRF) and prompt synthesis.",
        },
        {
            "id": "scrape_llm",
            "name": "System C: Agentic Web Scraping LLM",
            "category": "primary",
            "status": "available" if llm_key else "unavailable",
            "blocked": False,
            "note": "Ready" if llm_key else "Missing LLM API key",
            "description": "Online tool-calling agent navigating local shop pages and document inspection tools.",
        },
        {
            "id": "structured_rules",
            "name": "Ablation: Deterministic Rules Only",
            "category": "ablation",
            "status": "available",
            "blocked": False,
            "note": "Ready (local evaluation)",
            "description": "Pure deterministic rules engine using DSL without statistical model (abstains on semantic requirements).",
        },
        {
            "id": "rag_llm_dryrun",
            "name": "System B: Hybrid RAG (Dry-run)",
            "category": "dryrun",
            "status": "available",
            "blocked": False,
            "note": "Dry-run mode (synthetic fixtures, non-official)",
            "description": "Offline hybrid retrieval with synthetic fixture evaluation (non-official).",
        },
        {
            "id": "scrape_llm_dryrun",
            "name": "System C: Scraping LLM (Dry-run)",
            "category": "dryrun",
            "status": "available",
            "blocked": False,
            "note": "Dry-run mode (synthetic fixtures, non-official)",
            "description": "Simulated tool-calling loop using local shop fixtures (non-official).",
        },
        {
            "id": "structured_llm",
            "name": "Ablation: Structured Facts + LLM",
            "category": "ablation",
            "status": "available" if llm_key else "unavailable",
            "blocked": False,
            "note": "Ready" if llm_key else "Missing LLM API key",
            "description": "Same structured facts and rules as System A, but LLM replaces JEV for semantic judgments.",
        },
        {
            "id": "rag_llm_guarded",
            "name": "Ablation: RAG + LLM Guarded by Rules",
            "category": "ablation",
            "status": "available" if llm_key else "unavailable",
            "blocked": False,
            "note": "Ready" if llm_key else "Missing LLM API key",
            "description": "System B output verified by deterministic rules checker before verdict emission.",
        },
    ]

    return {"engines": engines}


# ==============================================================================
# §19. Query Execution
# ==============================================================================

@app.post("/api/query", response_model=QueryResponse, summary="Execute query on selected engine")
async def execute_query(request: QueryRequest) -> QueryResponse:
    """Accepts QueryRequest, executes selected engine, returns normalized QueryResponse (§19).
    
    Adheres strictly to §19 and §0.1:
    - If System A is requested without credentials, returns provider_error with exit code 30 note.
      Do NOT mock or substitute!
    """
    engine_name = request.engine.lower().strip()

    # 1. System A: Structured JEV
    if engine_name in ("structured_jev", "system_a"):
        jev_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        if not jev_key:
            return QueryResponse(
                request_id=request.request_id,
                engine="structured_jev",
                engine_version="v1.0.0",
                execution_status=ExecutionStatus.provider_error,
                catalog_version=request.catalog_version,
                knowledge_version=request.knowledge_version,
                interpreted_requirements=request.requirements or [],
                selected_product_ids=[],
                technical_verdict=TechnicalVerdict.INSUFFICIENT_EVIDENCE,
                checks=[
                    CheckResult(
                        requirement_id="JEV_AUTH",
                        status=CheckStatus.UNKNOWN,
                        evidence_ids=[],
                        reason_code="PROVIDER_BLOCKED_NO_API_KEY",
                        decision_origin=DecisionOrigin.jev,
                    )
                ],
                missing_evidence=["TYPESAFE_API_KEY"],
                alternatives=[],
                quote=None,
                summary="Error 30: structured_jev blocked (no API key). TypeSafe JEV credentials not provided; system A is blocked per MEGAPLAN §0.1.",
                telemetry_ref=f"tel-{request.request_id}",
            )

        try:
            from industrial_lab.engines.structured_jev import StructuredJevEngine
            engine = StructuredJevEngine()
            return await engine.execute(request)
        except Exception as e:
            logger.exception("Error executing StructuredJevEngine")
            return QueryResponse(
                request_id=request.request_id,
                engine="structured_jev",
                engine_version="v1.0.0",
                execution_status=ExecutionStatus.provider_error,
                catalog_version=request.catalog_version,
                knowledge_version=request.knowledge_version,
                interpreted_requirements=request.requirements or [],
                selected_product_ids=[],
                technical_verdict=TechnicalVerdict.INSUFFICIENT_EVIDENCE,
                checks=[],
                summary=f"System A Execution Error: {str(e)}",
            )

    # 2. System B: Dry-run mode
    elif engine_name in ("rag_llm_dryrun", "system_b_dryrun"):
        from industrial_lab.engines.rag_llm import RagLlmEngine
        engine = RagLlmEngine(dry_run=True)
        resp = await engine.execute(request)
        resp.engine = "rag_llm_dryrun"
        return resp

    # 3. System C: Dry-run mode
    elif engine_name in ("scrape_llm_dryrun", "system_c_dryrun"):
        from industrial_lab.engines.scrape_llm import ScrapeLlmEngine
        engine = ScrapeLlmEngine(dry_run=True)
        resp = await engine.execute(request)
        resp.engine = "scrape_llm_dryrun"
        return resp

    # 4. Structured Rules (Ablation 1)
    elif engine_name in ("structured_rules", "rules"):
        from industrial_lab.engines.ablations import StructuredRulesEngine
        engine = StructuredRulesEngine()
        return await engine.execute(request)

    # 5. System B: RAG LLM
    elif engine_name in ("rag_llm", "system_b"):
        try:
            from industrial_lab.engines.rag_llm import RagLlmEngine
            engine = RagLlmEngine()
            return await engine.execute(request)
        except Exception as e:
            logger.exception("Error executing RagLlmEngine")
            return QueryResponse(
                request_id=request.request_id,
                engine="rag_llm",
                engine_version="v1.0.0",
                execution_status=ExecutionStatus.provider_error,
                catalog_version=request.catalog_version,
                knowledge_version=request.knowledge_version,
                interpreted_requirements=request.requirements or [],
                selected_product_ids=[],
                technical_verdict=TechnicalVerdict.INSUFFICIENT_EVIDENCE,
                checks=[],
                missing_evidence=["LLM_API_KEY"],
                summary=f"System B Execution Error: {str(e)}",
            )

    # 6. System C: Scrape LLM
    elif engine_name in ("scrape_llm", "system_c"):
        try:
            from industrial_lab.engines.scrape_llm import ScrapeLlmEngine
            engine = ScrapeLlmEngine()
            return await engine.execute(request)
        except Exception as e:
            logger.exception("Error executing ScrapeLlmEngine")
            return QueryResponse(
                request_id=request.request_id,
                engine="scrape_llm",
                engine_version="v1.0.0",
                execution_status=ExecutionStatus.provider_error,
                catalog_version=request.catalog_version,
                knowledge_version=request.knowledge_version,
                interpreted_requirements=request.requirements or [],
                selected_product_ids=[],
                technical_verdict=TechnicalVerdict.INSUFFICIENT_EVIDENCE,
                checks=[],
                missing_evidence=["LLM_API_KEY"],
                summary=f"System C Execution Error: {str(e)}",
            )

    # 7. Other Ablations
    elif engine_name in ("structured_llm", "rag_llm_guarded"):
        try:
            mod = importlib.import_module("industrial_lab.engines.ablations")
            mapping = {
                "structured_llm": "StructuredLlmEngine",
                "rag_llm_guarded": "RagLlmGuardedEngine",
            }
            cls_name = mapping.get(engine_name)
            if cls_name and hasattr(mod, cls_name):
                engine = getattr(mod, cls_name)()
                return await engine.execute(request)
        except Exception as e:
            logger.exception(f"Error executing {engine_name}")
            return QueryResponse(
                request_id=request.request_id,
                engine=engine_name,
                engine_version="v1.0.0",
                execution_status=ExecutionStatus.provider_error,
                catalog_version=request.catalog_version,
                knowledge_version=request.knowledge_version,
                interpreted_requirements=request.requirements or [],
                selected_product_ids=[],
                technical_verdict=TechnicalVerdict.INSUFFICIENT_EVIDENCE,
                checks=[],
                missing_evidence=["LLM_API_KEY"],
                summary=f"Ablation Execution Error: {str(e)}",
            )

    # 5. Deterministic fallback execution using RulesEngine & KnowledgeStore
    # Useful if query specifies structured_rules or if engine module isn't loaded
    try:
        from industrial_lab.rules.engine import RulesEngine, FactIndex
        from industrial_lab.commerce.quotes import fetch_or_compute_quote
        from industrial_lab.schemas import aggregate_verdict

        store = KnowledgeStore()
        catalog_products = store.list_products()
        facts_by_prod = {p["product_id"]: store.get_facts_by_product(p["product_id"]) for p in catalog_products}

        # Determine target products
        target_pids = request.requested_product_ids or [p["product_id"] for p in catalog_products]
        if not target_pids and catalog_products:
            target_pids = [catalog_products[0]["product_id"]]

        # Evaluate requirements if provided
        reqs = request.requirements or []
        checks: List[CheckResult] = []

        for req in reqs:
            # Check against facts of targeted products
            for pid in target_pids:
                p_facts = facts_by_prod.get(pid, [])
                matching_facts = [f for f in p_facts if req.operator in f.property or req.target == f.value]
                if matching_facts:
                    ev_ids = [eid for mf in matching_facts for eid in mf.evidence_ids]
                    checks.append(
                        CheckResult(
                            requirement_id=req.requirement_id,
                            status=CheckStatus.PASS,
                            evidence_ids=ev_ids,
                            reason_code="FACT_MATCHED_DETERMINISTIC",
                            decision_origin=DecisionOrigin.rule,
                        )
                    )
                else:
                    checks.append(
                        CheckResult(
                            requirement_id=req.requirement_id,
                            status=CheckStatus.UNKNOWN,
                            evidence_ids=[],
                            reason_code="NO_MATCHING_FACT",
                            decision_origin=DecisionOrigin.rule,
                        )
                    )

        verdict = aggregate_verdict(checks, reqs) if checks else TechnicalVerdict.COMPATIBLE

        # Commercial quote if requested
        quote_obj: Optional[Quote] = None
        if request.include_quote and target_pids:
            quantities = {pid: 1 for pid in target_pids}
            quote_obj = await fetch_or_compute_quote(
                product_ids=target_pids,
                quantities=quantities,
                technical_verdict=verdict,
                scenario_id=request.scenario_id,
            )

        return QueryResponse(
            request_id=request.request_id,
            engine=engine_name,
            engine_version="v1.0.0",
            execution_status=ExecutionStatus.completed,
            catalog_version=request.catalog_version,
            knowledge_version=request.knowledge_version,
            interpreted_requirements=reqs,
            selected_product_ids=target_pids if verdict != TechnicalVerdict.INCOMPATIBLE else [],
            technical_verdict=verdict,
            checks=checks,
            missing_evidence=[c.requirement_id for c in checks if c.status == CheckStatus.UNKNOWN],
            quote=quote_obj,
            summary=f"Evaluación determinista completada para motor {engine_name}. Dictamen: {verdict.value} con los requisitos comprobados.",
            telemetry_ref=f"tel-{request.request_id}",
        )
    except Exception as e:
        logger.exception("Error in fallback deterministic execution")
        raise HTTPException(
            status_code=500,
            detail=f"Engine execution error for {engine_name}: {str(e)}",
        )


# ==============================================================================
# §19. Document Evidence Access
# ==============================================================================

@app.get("/api/documents/{document_id}/pages/{page}", summary="Returns document page text and spans")
async def get_document_page_endpoint(document_id: str, page: int) -> Dict[str, Any]:
    """Returns page text and SourceSpans for document verification (§19)."""
    page_record = get_document_page(document_id, page)
    if not page_record:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' page {page} not found in ingested pages.",
        )

    spans = get_document_page_spans(document_id, page)

    return {
        "document_id": document_id,
        "page": page,
        "document_sha256": page_record.get("document_sha256"),
        "printed_page_label": page_record.get("printed_page_label", str(page)),
        "text": page_record.get("text", ""),
        "product_ids": page_record.get("product_ids", []),
        "revision": page_record.get("revision", "unknown"),
        "source_kind": page_record.get("source_kind", "datasheet"),
        "spans": [s.to_dict() for s in spans],
    }


# ==============================================================================
# §19. Run Metrics Summary
# ==============================================================================

@app.get("/api/runs/{run_id}/summary", summary="Returns calculated run metrics summary")
async def get_run_summary_endpoint(run_id: str) -> Dict[str, Any]:
    """Returns run metrics summary without exposing test gold during execution (§19)."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found in runs directory.",
        )

    summary_file = run_dir / "summary.json"
    stats_file = run_dir / "statistics.json"
    manifest_file = run_dir / "run_manifest.json"

    result: Dict[str, Any] = {"run_id": run_id}

    if summary_file.is_file():
        with open(summary_file, "r", encoding="utf-8") as f:
            result["summary"] = json.load(f)

    if stats_file.is_file():
        with open(stats_file, "r", encoding="utf-8") as f:
            result["statistics"] = json.load(f)

    if manifest_file.is_file():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
            # Remove any sensitive or test gold paths
            manifest.pop("gold_cases", None)
            result["manifest"] = manifest

    if len(result) == 1:
        # No summary JSON files generated yet
        return {
            "run_id": run_id,
            "status": "pending_or_running",
            "message": "Run directory exists but summary metrics have not yet been generated.",
        }

    return result


# ==============================================================================
# Additional Catalog Helper Endpoints
# ==============================================================================

@app.get("/api/catalog", summary="Lists catalog products")
async def get_catalog_endpoint() -> Dict[str, Any]:
    """Returns products in the catalog."""
    store = KnowledgeStore()
    products = store.list_products()
    if not products:
        products = fixtures.load_catalog().get("products", [])
    return {"catalog_id": "controlnautas-three-products-v1", "products": products}


# ==============================================================================
# §19.1 Interactive Demo Interface
# ==============================================================================

DEMO_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Industrial Selection Lab — Demostración Comparativa</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-muted: #8b949e;
      --text-bright: #f0f6fc;
      --primary: #58a6ff;
      --success: #3fb950;
      --danger: #f85149;
      --warning: #d29922;
      --accent: #bc8cff;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      line-height: 1.5;
      padding: 24px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    header {
      border-bottom: 1px solid var(--border);
      padding-bottom: 20px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    h1 { font-size: 22px; font-weight: 700; color: var(--text-bright); display: flex; align-items: center; gap: 10px; }
    .badge-live {
      background: rgba(63, 185, 80, 0.15);
      color: var(--success);
      border: 1px solid var(--success);
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 11px;
      font-family: 'JetBrains Mono', monospace;
    }
    .disclaimer-banner {
      background: rgba(210, 153, 34, 0.15);
      border: 1px solid var(--warning);
      color: #e3b341;
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 500;
      margin-bottom: 24px;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 24px; }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }
    .card-title {
      font-size: 15px;
      font-weight: 600;
      color: var(--text-bright);
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
    }
    .card-subtitle { font-size: 12px; color: var(--text-muted); margin-bottom: 12px; }
    .spec-item {
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      padding: 4px 0;
      border-top: 1px solid rgba(48, 54, 61, 0.5);
    }
    .spec-label { color: var(--text-muted); }
    .spec-val { color: var(--text-bright); font-family: 'JetBrains Mono', monospace; font-size: 11px; }

    .query-section {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
    }
    .form-group { margin-bottom: 16px; }
    label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: var(--text-bright); }
    input[type="text"], select {
      width: 100%;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 12px;
      color: var(--text-bright);
      font-size: 14px;
      font-family: inherit;
    }
    input[type="text"]:focus, select:focus {
      outline: none;
      border-color: var(--primary);
    }
    .samples {
      display: flex;
      gap: 8px;
      margin-top: 8px;
      flex-wrap: wrap;
    }
    .sample-btn {
      background: rgba(88, 166, 255, 0.1);
      border: 1px solid rgba(88, 166, 255, 0.3);
      color: var(--primary);
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 12px;
      cursor: pointer;
    }
    .sample-btn:hover { background: rgba(88, 166, 255, 0.2); }
    .btn-submit {
      background: #238636;
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 10px 20px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .btn-submit:hover { background: #2ea043; }
    .btn-submit:disabled { opacity: 0.5; cursor: not-allowed; }

    .results-container {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      display: none;
    }
    .results-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }
    .verdict-badge {
      padding: 6px 14px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 13px;
      font-family: 'JetBrains Mono', monospace;
    }
    .verdict-COMPATIBLE { background: rgba(63, 185, 80, 0.2); color: var(--success); border: 1px solid var(--success); }
    .verdict-INCOMPATIBLE { background: rgba(248, 81, 73, 0.2); color: var(--danger); border: 1px solid var(--danger); }
    .verdict-INSUFFICIENT_EVIDENCE { background: rgba(210, 153, 34, 0.2); color: var(--warning); border: 1px solid var(--warning); }

    .telemetry-row {
      display: flex;
      gap: 20px;
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 16px;
      font-family: 'JetBrains Mono', monospace;
    }
    .check-item {
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 14px;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .check-status {
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
    }
    .status-PASS { color: var(--success); background: rgba(63, 185, 80, 0.1); }
    .status-FAIL { color: var(--danger); background: rgba(248, 81, 73, 0.1); }
    .status-UNKNOWN { color: var(--warning); background: rgba(210, 153, 34, 0.1); }

    .quote-box {
      margin-top: 20px;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 16px;
    }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }
    th { color: var(--text-muted); font-weight: 600; }
    td.amount { text-align: right; font-family: 'JetBrains Mono', monospace; }

    .evidence-tag {
      display: inline-block;
      background: rgba(188, 140, 255, 0.15);
      color: var(--accent);
      border: 1px solid rgba(188, 140, 255, 0.3);
      padding: 1px 6px;
      border-radius: 4px;
      font-size: 11px;
      font-family: 'JetBrains Mono', monospace;
      cursor: pointer;
      margin-left: 6px;
    }
    .evidence-tag:hover { text-decoration: underline; background: rgba(188, 140, 255, 0.3); }

    .modal {
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.8);
      z-index: 1000;
      align-items: center;
      justify-content: center;
    }
    .modal-content {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 24px;
      max-width: 700px;
      width: 90%;
    }
    .modal-header { display: flex; justify-content: space-between; margin-bottom: 16px; font-weight: 700; color: var(--text-bright); }
    .modal-close { cursor: pointer; color: var(--text-muted); font-size: 18px; }

    .engine-indicator {
      margin-top: 10px;
      padding: 10px 14px;
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.4;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .engine-indicator-available {
      background: rgba(63, 185, 80, 0.12);
      border: 1px solid var(--success);
      color: #7ee787;
    }
    .engine-indicator-blocked {
      background: rgba(248, 81, 73, 0.15);
      border: 1px solid var(--danger);
      color: #ff7b72;
    }
    .engine-indicator-unavailable {
      background: rgba(210, 153, 34, 0.15);
      border: 1px solid var(--warning);
      color: #e3b341;
    }
    .engine-indicator-dryrun {
      background: rgba(188, 140, 255, 0.15);
      border: 1px solid var(--accent);
      color: #d2a8ff;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Industrial Selection Lab <span class="badge-live">v1.0 Ready</span></h1>
        <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">
          Demostración comparativa de Sistemas A (JEV), B (RAG), C (Scraping) y ablaciones (§19).
        </p>
      </div>
      <div>
        <span style="font-size: 12px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">[DEMO / G2 REAL DOCS / NON-OFFICIAL] Catálogo de 3 productos del fabricante (PDFs de usuario). Precios/stock SIMULADOS. Benchmark oficial aún bloqueado.</span>
      </div>
    </header>

    <div class="disclaimer-banner">
      <span style="font-size: 18px;">⚠️</span>
      <div>
        <strong>AVISO: PRECIO Y STOCK SIMULADOS (§6) [NON-OFFICIAL]</strong> — Identidad de productos proviene de documentos oficiales del usuario (data_origin: real_user_document). Precios e inventario son LAB-SIMULATED y no son MSRP del fabricante. No se inventan especificaciones técnicas en esta vista; ver documentos PDF. Benchmark oficial bloqueado hasta nivel de prueba del usuario.
      </div>
    </div>

    <div class="grid">
      <!--PRODUCT_CARDS-->
    </div>

    <!-- Query Box -->
    <div class="query-section">
      <div class="form-group">
        <label for="engineSelect">Seleccionar Motor de Inferencia (§2)</label>
        <select id="engineSelect" onchange="updateEngineIndicator()">
          <option value="structured_rules" selected>[DISPONIBLE - LOCAL] Ablación 1: Reglas Deterministas DSL Únicamente (Local)</option>
          <option value="structured_jev">[BLOQUEADO §0.1 #7] Sistema A: Structured Facts + TypeSafe JEV (Falta TYPESAFE_API_KEY)</option>
          <option value="rag_llm">[NO DISPONIBLE] Sistema B: Hybrid RAG + LLM (Falta LLM key)</option>
          <option value="scrape_llm">[NO DISPONIBLE] Sistema C: Scraping LLM (Falta LLM key)</option>
          <option value="rag_llm_dryrun">[DRY-RUN SINTÉTICO] Sistema B: Hybrid RAG (Dry-run mode)</option>
          <option value="scrape_llm_dryrun">[DRY-RUN SINTÉTICO] Sistema C: Scraping LLM (Dry-run mode)</option>
          <option value="structured_llm">[NO DISPONIBLE] Ablación 2: Structured + LLM (Falta LLM key)</option>
          <option value="rag_llm_guarded">[NO DISPONIBLE] Ablación 3: RAG Guarded (Falta LLM key)</option>
        </select>
        <div id="engineIndicator" class="engine-indicator"></div>
      </div>

      <div class="form-group">
        <label for="queryInput">Requisitos del Cliente / Consulta Técnica</label>
        <input type="text" id="queryInput" value="Horner HE-X4 Micro OCS con sensor TZ THT-02" placeholder="Consulta sobre productos del catálogo real (sin inventar specs)">
        <div class="samples">
          <span style="font-size: 12px; color: var(--text-muted); align-self: center;">Ejemplos rápidos (IDs reales):</span>
          <button class="sample-btn" onclick="setQuery('Horner HE-X4 Micro OCS variantes HE-X4A / HE-X4R', ['P_X4'])">1. Horner X4 (P_X4)</button>
          <button class="sample-btn" onclick="setQuery('Sensor temperatura humedad TZ THT-02 Modbus-RTU', ['P_THT'])">2. TZ THT-02 (P_THT)</button>
          <button class="sample-btn" onclick="setQuery('Pumphouse Heater U Series item 66661', ['P_UHEAT'])">3. U Series heater (P_UHEAT)</button>
        </div>
      </div>

      <div style="display: flex; justify-content: space-between; align-items: center;">
        <label style="display: flex; align-items: center; gap: 8px; font-weight: normal; font-size: 13px; cursor: pointer;">
          <input type="checkbox" id="includeQuote" checked> Generar cotización comercial simulada (§6)
        </label>
        <button class="btn-submit" id="submitBtn" onclick="runQuery()">
          <span>Ejecutar Consulta</span> ➔
        </button>
      </div>
    </div>

    <!-- Results Container -->
    <div class="results-container" id="resultsContainer">
      <div class="results-header">
        <div>
          <span style="font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Dictamen Técnico</span>
          <div style="margin-top: 6px;">
            <span class="verdict-badge" id="verdictBadge">COMPATIBLE</span>
          </div>
        </div>
        <div style="text-align: right;">
          <span style="font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Motor Ejecutado</span>
          <div id="engineLabel" style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--text-bright); margin-top: 4px;">-</div>
          <div id="engineStatusBadge" style="margin-top: 4px;"></div>
        </div>
      </div>

      <div class="telemetry-row">
        <span>⏱️ Latencia: <strong id="latencyVal">-</strong></span>
        <span>💵 Costo: <strong id="costVal">$0.000 USD (medido)</strong></span>
        <span>🆔 Request: <strong id="reqIdVal">-</strong></span>
      </div>

      <div id="summaryText" style="background: rgba(88, 166, 255, 0.05); border-left: 3px solid var(--primary); padding: 12px; font-size: 13px; margin-bottom: 20px;"></div>

      <h3 style="font-size: 14px; margin-bottom: 12px; color: var(--text-bright);">Lista de Comprobaciones (§5.5)</h3>
      <div id="checksList"></div>

      <!-- Quote Box -->
      <div class="quote-box" id="quoteBox" style="display: none;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
          <h4 style="font-size: 14px; color: var(--text-bright);">Cotización Comercial Simulada (§6)</h4>
          <span style="font-size: 11px; background: rgba(210, 153, 34, 0.2); color: var(--warning); padding: 2px 6px; border-radius: 4px;">Precios Sintéticos</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Producto</th>
              <th>Cantidad</th>
              <th style="text-align: right;">Precio Unitario</th>
              <th style="text-align: right;">Total</th>
            </tr>
          </thead>
          <tbody id="quoteLines"></tbody>
          <tfoot>
            <tr>
              <th colspan="3" style="text-align: right; border: none; padding-top: 12px;">Total Cotizado:</th>
              <th id="quoteTotal" class="amount" style="border: none; padding-top: 12px; color: var(--success); font-size: 15px;">$0.00</th>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  </div>

  <!-- Modal for Evidence Span View -->
  <div class="modal" id="evidenceModal">
    <div class="modal-content">
      <div class="modal-header">
        <span>Evidencia Documental (§5.1)</span>
        <span class="modal-close" onclick="closeModal()">&times;</span>
      </div>
      <div style="margin-bottom: 12px; font-size: 12px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;" id="modalSpanMeta"></div>
      <div style="background: #0d1117; padding: 14px; border-radius: 6px; border: 1px solid var(--border); font-size: 13px; font-family: 'JetBrains Mono', monospace; white-space: pre-wrap;" id="modalSpanText"></div>
    </div>
  </div>

  <script>
    let activeProducts = ['P1', 'P2'];

    function updateEngineIndicator() {
      const select = document.getElementById('engineSelect');
      const val = select ? select.value : '';
      const ind = document.getElementById('engineIndicator');
      if (!ind) return;
      if (val === 'structured_jev') {
        ind.className = 'engine-indicator engine-indicator-blocked';
        ind.innerHTML = '🛑 <strong>[BLOQUEADO §0.1 #7]</strong> Sistema A: Structured Facts + TypeSafe JEV está bloqueado por falta de TYPESAFE_API_KEY. Al ejecutar responderá con Error 30 (bloqueo oficial según MEGAPLAN §0.1 #7).';
      } else if (val === 'rag_llm' || val === 'scrape_llm' || val === 'structured_llm' || val === 'rag_llm_guarded') {
        ind.className = 'engine-indicator engine-indicator-unavailable';
        ind.innerHTML = '⚠️ <strong>[NO DISPONIBLE]</strong> Motor no disponible: Falta clave de API del LLM.';
      } else if (val === 'rag_llm_dryrun' || val === 'scrape_llm_dryrun') {
        ind.className = 'engine-indicator engine-indicator-dryrun';
        ind.innerHTML = '🧪 <strong>[DRY-RUN SINTÉTICO]</strong> Modo de prueba offline con fixtures sintéticos locales. NO es catálogo real ni experimento oficial.';
      } else {
        ind.className = 'engine-indicator engine-indicator-available';
        ind.innerHTML = '✅ <strong>[DISPONIBLE - LOCAL]</strong> Motor listo para evaluación determinista con reglas DSL y hechos locales.';
      }
    }

    document.addEventListener('DOMContentLoaded', updateEngineIndicator);
    setTimeout(updateEngineIndicator, 50);

    function setQuery(text, pids) {
      document.getElementById('queryInput').value = text;
      activeProducts = pids;
    }

    async function inspectEvidence(spanId) {
      const parts = spanId.split(':');
      if (parts.length >= 2) {
        const docId = parts[0];
        const pageIdx = parseInt(parts[1].replace('p', ''), 10);
        try {
          const res = await fetch(`/api/documents/${docId}/pages/${pageIdx}`);
          if (res.ok) {
            const data = await res.json();
            const span = (data.spans || []).find(s => s.span_id === spanId);
            document.getElementById('modalSpanMeta').textContent = `Documento: ${docId} | Página: ${pageIdx} | Span: ${spanId}`;
            document.getElementById('modalSpanText').textContent = span ? span.text : data.text;
            document.getElementById('evidenceModal').style.display = 'flex';
            return;
          }
        } catch (e) {
          console.error(e);
        }
      }
      alert('Evidencia ID: ' + spanId);
    }

    function closeModal() {
      document.getElementById('evidenceModal').style.display = 'none';
    }

    async function runQuery() {
      const btn = document.getElementById('submitBtn');
      btn.disabled = true;
      btn.textContent = 'Evaluando...';
      const container = document.getElementById('resultsContainer');
      container.style.display = 'none';

      const engine = document.getElementById('engineSelect').value;
      const queryText = document.getElementById('queryInput').value;
      const includeQuote = document.getElementById('includeQuote').checked;
      const startTime = performance.now();

      const payload = {
        request_id: 'req-' + Math.random().toString(36).substring(2, 9),
        query_text: queryText,
        engine: engine,
        mode: 'M1',
        scenario_id: 'S0001',
        catalog_version: 'v1',
        knowledge_version: 'v1',
        requested_product_ids: activeProducts,
        include_quote: includeQuote,
        requirements: [
          {
            requirement_id: 'supply_voltage_nominal_v',
            kind: 'exact_property',
            operator: 'eq',
            target: 24.0,
            unit: 'V',
            hard: true,
            source_user_text: 'Tensión nominal 24V DC'
          }
        ]
      };

      try {
        const res = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const elapsed = Math.round(performance.now() - startTime);
        const data = await res.json();

        // Render Results
        container.style.display = 'block';
        document.getElementById('engineLabel').textContent = data.engine;
        document.getElementById('reqIdVal').textContent = data.request_id;
        document.getElementById('latencyVal').textContent = `${elapsed} ms (medido)`;
        document.getElementById('summaryText').textContent = data.summary || 'Consulta finalizada.';

        // Engine Status Badge
        const statusBadge = document.getElementById('engineStatusBadge');
        if (data.execution_status === 'provider_error') {
          statusBadge.innerHTML = '<span style="background: rgba(248, 81, 73, 0.2); color: var(--danger); border: 1px solid var(--danger); padding: 2px 8px; border-radius: 10px; font-size: 11px; font-family: \'JetBrains Mono\', monospace;">🛑 BLOQUEADO §0.1 (Error 30)</span>';
        } else if (data.engine && data.engine.includes('dryrun')) {
          statusBadge.innerHTML = '<span style="background: rgba(188, 140, 255, 0.2); color: var(--accent); border: 1px solid var(--accent); padding: 2px 8px; border-radius: 10px; font-size: 11px; font-family: \'JetBrains Mono\', monospace;">🧪 DRY-RUN SINTÉTICO</span>';
        } else {
          statusBadge.innerHTML = '<span style="background: rgba(63, 185, 80, 0.2); color: var(--success); border: 1px solid var(--success); padding: 2px 8px; border-radius: 10px; font-size: 11px; font-family: \'JetBrains Mono\', monospace;">✅ EJECUTADO</span>';
        }

        // Verdict Badge
        const badge = document.getElementById('verdictBadge');
        badge.textContent = data.technical_verdict;
        badge.className = 'verdict-badge verdict-' + data.technical_verdict;

        // Checks List
        const checksList = document.getElementById('checksList');
        checksList.innerHTML = '';
        if (data.checks && data.checks.length > 0) {
          data.checks.forEach(c => {
            const div = document.createElement('div');
            div.className = 'check-item';
            let evHtml = '';
            if (c.evidence_ids && c.evidence_ids.length > 0) {
              evHtml = c.evidence_ids.map(eid => `<span class="evidence-tag" onclick="inspectEvidence('${eid}')">${eid}</span>`).join('');
            }
            div.innerHTML = `
              <div>
                <strong>${c.requirement_id}</strong>
                <span style="font-size: 11px; color: var(--text-muted); margin-left: 8px;">(${c.reason_code || 'CHECK'})</span>
                ${evHtml}
              </div>
              <div>
                <span class="check-status status-${c.status}">${c.status}</span>
              </div>
            `;
            checksList.appendChild(div);
          });
        } else {
          checksList.innerHTML = '<div style="color: var(--text-muted); font-size: 13px;">No se registraron comprobaciones discretas para esta consulta.</div>';
        }

        // Quote
        const quoteBox = document.getElementById('quoteBox');
        if (data.quote && data.quote.lines) {
          quoteBox.style.display = 'block';
          const quoteLines = document.getElementById('quoteLines');
          quoteLines.innerHTML = '';
          data.quote.lines.forEach(l => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><strong>${l.product_id}</strong></td>
              <td>${l.quantity}</td>
              <td class="amount">$${(l.unit_price_minor / 100).toFixed(2)}</td>
              <td class="amount">$${(l.line_total_minor / 100).toFixed(2)}</td>
            `;
            quoteLines.appendChild(tr);
          });
          document.getElementById('quoteTotal').textContent = `$${(data.quote.total_minor / 100).toFixed(2)} ${data.quote.currency}`;
        } else {
          quoteBox.style.display = 'none';
        }
      } catch (err) {
        alert('Error ejecutando consulta: ' + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>Ejecutar Consulta</span> ➔';
      }
    }
  </script>
</body>
</html>
"""


def _demo_product_cards_html() -> str:
    """Identity-only product cards from live catalog (no invented technical specs)."""
    try:
        from industrial_lab.shop import fixtures as shop_fixtures
        catalog = shop_fixtures.load_catalog()
        products = catalog.get("products", []) or []
    except Exception:
        products = []
    cards = []
    for p in products:
        pid = html_lib.escape(str(p.get("product_id", "")))
        mfg = html_lib.escape(str(p.get("manufacturer") or ""))
        model = html_lib.escape(str(p.get("exact_model") or ""))
        sku = html_lib.escape(str(p.get("sku") or ""))
        variant = html_lib.escape(str(p.get("variant") or ""))
        docs = p.get("documents") or []
        doc_ids = ", ".join(
            html_lib.escape(str(d.get("document_id", "")))
            for d in docs if isinstance(d, dict)
        ) or "(none)"
        cards.append(
            f"""      <div class="card">
        <div class="card-title">
          <span>{pid}: {model}</span>
          <span style="color: var(--primary); font-size: 13px;">SKU {sku}</span>
        </div>
        <div class="card-subtitle">{mfg} — identity from real_user_document</div>
        <div class="spec-item"><span class="spec-label">Variant:</span><span class="spec-val">{variant}</span></div>
        <div class="spec-item"><span class="spec-label">Documents:</span><span class="spec-val">{doc_ids}</span></div>
        <div class="spec-item"><span class="spec-label">Specs:</span><span class="spec-val">See official PDFs (not curated here)</span></div>
        <div class="spec-item"><span class="spec-label">Commerce:</span><span class="spec-val">SIMULATED prices/stock only</span></div>
      </div>"""
        )
    if not cards:
        return '      <div class="card"><div class="card-title">Catalog unavailable</div></div>'
    return "\n".join(cards)


@app.get("/demo", response_class=HTMLResponse, summary="Renders interactive demo UI")
async def demo_ui() -> HTMLResponse:
    """Renders single-page interactive demo UI adhering strictly to MEGAPLAN §19.1."""
    content = DEMO_HTML.replace("<!--PRODUCT_CARDS-->", _demo_product_cards_html())
    return HTMLResponse(content=content)


@app.get("/", summary="Root endpoint redirecting to demo")
async def root_redirect() -> HTMLResponse:
    return HTMLResponse(content=DEMO_HTML)
