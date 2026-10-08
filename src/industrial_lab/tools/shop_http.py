"""HTTP client tools for System C querying the shop simulator and document sources.

Adheres strictly to MEGAPLAN.md §6.1, §11.1, and §11.2:
- Explicit deterministic tools:
  - list_products()
  - fetch_product_page(product_id)
  - open_document(document_id)
  - read_document_pages(document_id, pages)
  - find_document_text(document_id, query)
  - get_commerce(product_ids)
  - create_quote(lines)
- Request-scoped document cache (per MEGAPLAN §11.2: reuse within request, cold across requests).
- Tool definitions for OpenAI-compatible function calling in LLMAdapter.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
from typing import Any, Mapping, Optional, Sequence
import zlib

import httpx

from industrial_lab.commerce.quotes import fetch_or_compute_quote
from industrial_lab.shop import fixtures

logger = logging.getLogger(__name__)

DEFAULT_SHOP_BASE_URL = os.environ.get("SHOP_BASE_URL", "http://127.0.0.1:8081")


def strip_html_tags(html_content: str) -> str:
    """Strips HTML tags and unescapes entities to produce clean text."""
    # Remove script and style elements
    clean = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
    # Replace breaks and paragraph tags with newlines
    clean = re.sub(r"<(br|p|div|tr|li)[^>]*>", "\n", clean, flags=re.IGNORECASE)
    # Strip remaining HTML tags
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = html.unescape(clean)
    # Collapse multiple whitespaces/newlines
    lines = [line.strip() for line in clean.splitlines()]
    return "\n".join(line for line in lines if line)


class ShopHttpClient:
    """HTTP client and document explorer tool-runner for System C (§11.1)."""

    _shared_doc_cache: dict[str, dict[int, str]] = {}

    @classmethod
    def clear_shared_cache(cls) -> None:
        """Clears the cross-request shared document cache (REPAIR3_PLAN §9)."""
        cls._shared_doc_cache.clear()

    def __init__(
        self,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout: float = 10.0,
        request_cache: Optional[dict[str, Any]] = None,
        cross_request_cache: bool = False,
    ) -> None:
        self.base_url = (base_url or os.environ.get("SHOP_BASE_URL") or DEFAULT_SHOP_BASE_URL).rstrip("/")
        self._http_client = http_client
        self.timeout = timeout
        self.cross_request_cache = cross_request_cache
        # Request-scoped cache (§11.2): cleared per query request in cold profile,
        # or shared across requests if cross_request_cache is enabled (REPAIR3_PLAN §9).
        if request_cache is not None:
            self.doc_cache: dict[str, dict[int, str]] = request_cache
        elif cross_request_cache:
            self.doc_cache = self._shared_doc_cache
        else:
            self.doc_cache = {}
        self.doc_metadata: dict[str, dict[str, Any]] = {}

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Returns OpenAPI tool definitions for LLM tool calling (§11.1)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_products",
                    "description": "Lista los identificadores, nombres, modelos y URLs de todos los equipos disponibles en el catálogo del simulador.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_product_page",
                    "description": "Obtiene la página de producto con texto limpio, resumen de especificaciones y enlaces a fichas técnicas y manuales en PDF.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "string",
                                "description": "ID del producto (ej. 'P_X4', 'P_THT', 'P_UHEAT').",
                            },
                        },
                        "required": ["product_id"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "open_document",
                    "description": "Descarga y procesa un manual o ficha técnica PDF en el request actual. Devuelve el número total de páginas, secciones y advertencias.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "ID del documento (ej. 'D_X4_MANUAL_MAN1137', 'D_THT_MANUAL', 'D_UHEAT_DATASHEET').",
                            },
                        },
                        "required": ["document_id"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_document_pages",
                    "description": "Lee el contenido textual detallado de páginas seleccionadas de un documento previamente abierto, junto con IDs de evidencia.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "ID del documento ya abierto.",
                            },
                            "pages": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Lista de números de página (1-indexed) a leer.",
                            },
                        },
                        "required": ["document_id", "pages"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "find_document_text",
                    "description": "Realiza una búsqueda léxica rápida dentro del documento abierto para ubicar menciones clave y números de página relevantes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                                "description": "ID del documento ya abierto.",
                            },
                            "query": {
                                "type": "string",
                                "description": "Término técnico o frase a buscar (ej. 'voltage', 'RS-485', '4-20mA', 'IP20').",
                            },
                        },
                        "required": ["document_id", "query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_commerce",
                    "description": "Consulta el precio unitario en centavos/céntimos, la moneda y el stock disponible para los productos dados.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Lista de IDs de productos a consultar comercialmente.",
                            },
                        },
                        "required": ["product_ids"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_quote",
                    "description": "Crea una cotización formal atómica con cálculo en unidades enteras menores para las cantidades indicadas.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lines": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "product_id": {"type": "string"},
                                        "quantity": {"type": "integer", "minimum": 1},
                                    },
                                    "required": ["product_id", "quantity"],
                                },
                                "description": "Líneas de producto y cantidad requerida.",
                            },
                        },
                        "required": ["lines"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Dispatches and executes a tool call by name."""
        logger.debug("Executing tool %s with args: %s", tool_name, arguments)
        if tool_name == "list_products":
            return await self.list_products()
        elif tool_name == "fetch_product_page":
            return await self.fetch_product_page(arguments.get("product_id", ""))
        elif tool_name == "open_document":
            return await self.open_document(arguments.get("document_id", ""))
        elif tool_name == "read_document_pages":
            pages = arguments.get("pages", [])
            if isinstance(pages, int):
                pages = [pages]
            return await self.read_document_pages(arguments.get("document_id", ""), pages)
        elif tool_name == "find_document_text":
            return await self.find_document_text(arguments.get("document_id", ""), arguments.get("query", ""))
        elif tool_name == "get_commerce":
            pids = arguments.get("product_ids", [])
            if isinstance(pids, str):
                pids = [pids]
            return await self.get_commerce(pids)
        elif tool_name == "create_quote":
            return await self.create_quote(arguments.get("lines", []))
        else:
            return {"error": f"Herramienta desconocida '{tool_name}'"}

    async def list_products(self) -> dict[str, Any]:
        """GET /api/catalog or fixtures fallback (§11.1)."""
        endpoint = f"{self.base_url}/api/catalog"
        client = self._http_client
        owns = client is None
        if owns:
            client = httpx.AsyncClient(timeout=self.timeout)

        try:
            try:
                resp = await client.get(endpoint)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as exc:
                logger.debug("GET /api/catalog HTTP error (%s); using fixtures fallback.", exc)
            return fixtures.get_catalog_public()
        finally:
            if owns and client is not None:
                await client.aclose()

    async def fetch_product_page(self, product_id: str) -> dict[str, Any]:
        """GET /products/{product_id} with clean text and document links (§11.1)."""
        pid = product_id.strip()
        endpoint = f"{self.base_url}/products/{pid}"
        client = self._http_client
        owns = client is None
        if owns:
            client = httpx.AsyncClient(timeout=self.timeout)

        clean_text = ""
        try:
            try:
                resp = await client.get(endpoint)
                if resp.status_code == 200:
                    clean_text = strip_html_tags(resp.text)
            except Exception as exc:
                logger.debug("GET /products/%s HTTP error (%s); using fixtures fallback.", pid, exc)
        finally:
            if owns and client is not None:
                await client.aclose()

        prod_data = fixtures.resolve_product(pid)
        if not prod_data:
            return {"error": f"Producto '{pid}' no encontrado en catálogo"}

        doc_links = []
        for d in prod_data.get("documents", []):
            doc_links.append(
                {
                    "document_id": d["document_id"],
                    "title": d.get("title", d["document_id"]),
                    "source_kind": d.get("source_kind", "datasheet"),
                    "url": f"/documents/{d['document_id']}.pdf",
                }
            )

        if not clean_text:
            # Build clean text description from product specs
            lines = [
                f"Producto: {prod_data.get('name', pid)}",
                f"ID: {pid}",
                f"Fabricante: {prod_data.get('manufacturer')}",
                f"Modelo: {prod_data.get('exact_model')}",
                f"Variante: {prod_data.get('variant')}",
                f"Descripción: {prod_data.get('description', '')}",
                "Especificaciones:",
            ]
            for k, v in prod_data.get("specs", {}).items():
                lines.append(f"- {k}: {v}")
            clean_text = "\n".join(lines)

        return {
            "product_id": pid,
            "page_text": clean_text,
            "documents": doc_links,
        }

    def _resolve_canonical_doc(self, document_id: str) -> tuple[str, list[str]]:
        """Resolves document_id or filename to canonical document_id and known aliases."""
        clean = document_id.strip()
        base_clean = clean[:-4] if clean.endswith(".pdf") else clean

        aliases = {clean, base_clean, f"{base_clean}.pdf"}
        canonical_id = base_clean

        catalog = fixtures.load_catalog()
        for p in catalog.get("products", []):
            for d in p.get("documents", []):
                doc_id = d.get("document_id", "")
                fn = d.get("filename", "")
                fn_no_ext = fn[:-4] if fn.endswith(".pdf") else fn

                doc_identifiers = {doc_id, f"{doc_id}.pdf", fn, fn_no_ext}
                if clean in doc_identifiers or base_clean in doc_identifiers or clean.lower() in {x.lower() for x in doc_identifiers}:
                    canonical_id = doc_id
                    aliases.update(doc_identifiers)
                    return canonical_id, list(aliases)

        syn = fixtures.load_synthetic_catalog()
        for p in syn.get("products", []):
            for d in p.get("documents", []):
                doc_id = d.get("document_id", "")
                fn = d.get("filename", "")
                fn_no_ext = fn[:-4] if fn.endswith(".pdf") else fn
                doc_identifiers = {doc_id, f"{doc_id}.pdf", fn, fn_no_ext}
                if clean in doc_identifiers or base_clean in doc_identifiers or clean.lower() in {x.lower() for x in doc_identifiers}:
                    canonical_id = doc_id
                    aliases.update(doc_identifiers)
                    return canonical_id, list(aliases)

        return canonical_id, list(aliases)

    def _get_cached_document(self, document_id: str) -> Optional[dict[int, str]]:
        canonical_id, aliases = self._resolve_canonical_doc(document_id)
        if canonical_id in self.doc_cache and self.doc_cache[canonical_id]:
            return self.doc_cache[canonical_id]
        for a in aliases:
            if a in self.doc_cache and self.doc_cache[a]:
                self.doc_cache[canonical_id] = self.doc_cache[a]
                return self.doc_cache[a]
        return None

    async def open_document(self, document_id: str) -> dict[str, Any]:
        """Downloads/retrieves PDF and parses pages into request-scoped cache (§11.1, §11.2)."""
        canonical_id, aliases = self._resolve_canonical_doc(document_id)

        # Check if already cached in current request
        cached = self._get_cached_document(canonical_id)
        if cached:
            return {
                "document_id": canonical_id,
                "status": "already_opened",
                "page_count": len(cached),
                "pages_available": sorted(cached.keys()),
                "warnings": [],
            }

        # First, check if pages are already available in data/pages/pages.jsonl
        pages_dict = self._load_pages_from_jsonl(canonical_id)

        # If not found in pages.jsonl, retrieve raw bytes from shop API or fixtures
        if not pages_dict:
            raw_pdf = await self._fetch_pdf_bytes(canonical_id)
            pages_dict = self._parse_pdf_bytes(canonical_id, raw_pdf)

        self.doc_cache[canonical_id] = pages_dict
        for a in aliases:
            self.doc_cache[a] = pages_dict

        page_numbers = sorted(pages_dict.keys())
        logger.info("Opened document '%s': extracted %d pages.", canonical_id, len(page_numbers))

        return {
            "document_id": canonical_id,
            "status": "opened",
            "page_count": len(page_numbers),
            "pages_available": page_numbers,
            "warnings": [] if page_numbers else ["No se pudo extraer texto legible del documento."],
        }

    def _load_pages_from_jsonl(self, doc_id: str) -> dict[int, str]:
        """Loads pages from data/pages/pages.jsonl if present."""
        pages: dict[int, str] = {}
        data_root = fixtures.get_data_root()
        page_files = [
            data_root / "pages" / "pages.jsonl",
            data_root / "pages" / "pages.synthetic.jsonl",
            data_root / "synthetic_fixtures" / "pages" / "pages.jsonl",
        ]

        for pages_file in page_files:
            if not pages_file.exists():
                continue
            try:
                with open(pages_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        item_doc_id = item.get("document_id", "")
                        if item_doc_id == doc_id or item_doc_id.replace("_", "-").lower() == doc_id.replace("_", "-").lower():
                            page_idx = int(item.get("pdf_page_index", 1))
                            pages.setdefault(page_idx, item.get("text", ""))
            except Exception as exc:
                logger.debug("Failed reading %s: %s", pages_file, exc)
            if pages:
                break

        return pages

    async def _fetch_pdf_bytes(self, doc_id: str) -> bytes:
        """Fetches raw PDF bytes from HTTP endpoint or fixtures."""
        endpoint = f"{self.base_url}/documents/{doc_id}.pdf"
        client = self._http_client
        owns = client is None
        if owns:
            client = httpx.AsyncClient(timeout=self.timeout)

        try:
            try:
                resp = await client.get(endpoint)
                if resp.status_code == 200 and resp.content:
                    return resp.content
            except Exception as exc:
                logger.debug("GET /documents/%s.pdf HTTP error: %s", doc_id, exc)
        finally:
            if owns and client is not None:
                await client.aclose()

        return fixtures.get_document_bytes(doc_id)

    def _parse_pdf_bytes(self, doc_id: str, raw_pdf: bytes) -> dict[int, str]:
        """Extracts text from PDF bytes using pypdf, falling back to FlateDecode and literal text streams."""
        pages: dict[int, str] = {}
        if not raw_pdf:
            return pages

        # 1. Preferred robust extractor: pypdf (MEGAPLAN §4.3, REPAIR_PLAN §8)
        try:
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw_pdf))
            for idx, page in enumerate(reader.pages, start=1):
                txt = page.extract_text() or ""
                pages[idx] = txt.strip()
            if pages and any(bool(t) for t in pages.values()):
                return pages
        except Exception as exc:
            logger.debug("pypdf parsing failed for %s: %s; falling back to stream parsing", doc_id, exc)

        # 2. Check for uncompressed text streams (like in synthetic PDF)
        tj_matches = re.findall(rb"\((.*?)\)\s*Tj", raw_pdf)
        if tj_matches:
            extracted_lines = [m.decode("latin1", errors="ignore").strip() for m in tj_matches]
            text = "\n".join(l for l in extracted_lines if l)
            pages[1] = text
            return pages

        # 3. Check for FlateDecode compressed streams
        stream_matches = re.findall(rb"stream[\r\n]+(.*?)[\r\n]+endstream", raw_pdf, flags=re.DOTALL)
        page_counter = 1

        for stream in stream_matches:
            try:
                decompressed = zlib.decompress(stream)
            except Exception:
                decompressed = stream

            inner_tj = re.findall(rb"\((.*?)\)\s*Tj", decompressed)
            inner_tj_array = re.findall(rb"\[(.*?)\]\s*TJ", decompressed)

            lines = []
            for m in inner_tj:
                t = m.decode("latin1", errors="ignore").strip()
                if t:
                    lines.append(t)
            for m in inner_tj_array:
                parts = re.findall(rb"\((.*?)\)", m)
                combined = "".join(p.decode("latin1", errors="ignore") for p in parts).strip()
                if combined:
                    lines.append(combined)

            if lines:
                pages[page_counter] = "\n".join(lines)
                page_counter += 1

        if not pages:
            # Fallback placeholder if no stream could be parsed
            pages[1] = f"[Documento {doc_id} cargado sin texto parseable]"

        return pages

    async def read_document_pages(self, document_id: str, pages: list[int], max_chars_per_page: int = 4000) -> dict[str, Any]:
        """Reads specific pages of an opened document and returns audited content (§11.1, REPAIR3_PLAN §9)."""
        canonical_id, aliases = self._resolve_canonical_doc(document_id)

        cached_pages = self._get_cached_document(canonical_id)
        if cached_pages is None:
            # Auto-open if not already opened
            await self.open_document(canonical_id)
            cached_pages = self._get_cached_document(canonical_id) or {}

        results: list[dict[str, Any]] = []

        for page_num in pages:
            text = cached_pages.get(page_num)
            if text is not None:
                span_id = f"{canonical_id}:p{page_num:02d}:s01"
                is_truncated = len(text) > max_chars_per_page
                snippet_text = text[:max_chars_per_page] if is_truncated else text
                page_data: dict[str, Any] = {
                    "document_id": canonical_id,
                    "page": page_num,
                    "span_id": span_id,
                    "text": snippet_text,
                    "truncated": is_truncated,
                }
                if is_truncated:
                    page_data["total_chars"] = len(text)
                results.append(page_data)
            else:
                results.append(
                    {
                        "document_id": canonical_id,
                        "page": page_num,
                        "error": f"Página {page_num} fuera de rango o no disponible (documento tiene {len(cached_pages)} páginas)",
                        "truncated": False,
                    }
                )

        return {
            "document_id": canonical_id,
            "read_pages_count": len(results),
            "pages": results,
            "truncated": any(p.get("truncated", False) for p in results),
        }

    async def find_document_text(self, document_id: str, query: str) -> dict[str, Any]:
        """Lexical search across pages of an opened document (§11.1, REPAIR3_PLAN §9)."""
        canonical_id, aliases = self._resolve_canonical_doc(document_id)

        cached_pages = self._get_cached_document(canonical_id)
        if cached_pages is None:
            await self.open_document(canonical_id)
            cached_pages = self._get_cached_document(canonical_id) or {}

        cleaned_query = query.strip().lower()
        matches: list[dict[str, Any]] = []

        if not cleaned_query:
            return {"document_id": canonical_id, "query": query, "matches": [], "match_count": 0, "truncated": False}

        for page_num, page_text in cached_pages.items():
            lower_text = page_text.lower()
            if cleaned_query in lower_text:
                # Extract snippet around match
                idx = lower_text.find(cleaned_query)
                start = max(0, idx - 100)
                end = min(len(page_text), idx + len(cleaned_query) + 100)
                snippet = page_text[start:end].replace("\n", " ").strip()
                matches.append(
                    {
                        "page": page_num,
                        "span_id": f"{canonical_id}:p{page_num:02d}:s01",
                        "snippet": f"...{snippet}...",
                    }
                )

        is_truncated = len(matches) > 10
        return {
            "document_id": canonical_id,
            "query": query,
            "match_count": len(matches),
            "matches": matches[:10],  # Return up to 10 best matches
            "truncated": is_truncated,
        }

    async def get_commerce(self, product_ids: list[str]) -> list[dict[str, Any]]:
        """GET /api/commerce/{product_id} for multiple products (§11.1)."""
        results: list[dict[str, Any]] = []
        client = self._http_client
        owns = client is None
        if owns:
            client = httpx.AsyncClient(timeout=self.timeout)

        try:
            for pid in product_ids:
                pid = pid.strip()
                endpoint = f"{self.base_url}/api/commerce/{pid}"
                data: Optional[dict[str, Any]] = None
                try:
                    resp = await client.get(endpoint)
                    if resp.status_code == 200:
                        data = resp.json()
                except Exception as exc:
                    logger.debug("GET /api/commerce/%s HTTP error: %s", pid, exc)

                if data is None:
                    # Fixtures fallback
                    comm = fixtures.get_commerce(pid)
                    if comm:
                        data = {
                            "product_id": pid,
                            "price_minor": comm.get("price_minor", 0),
                            "currency": comm.get("currency", "EUR"),
                            "stock_available": comm.get("stock_available", 0),
                            "revision": comm.get("revision", "rev-unknown"),
                            "is_simulated": True,
                        }
                    else:
                        data = {"product_id": pid, "error": "No disponible en tienda"}
                results.append(data)
        finally:
            if owns and client is not None:
                await client.aclose()

        return results

    async def create_quote(self, lines: list[dict[str, Any]]) -> dict[str, Any]:
        """POST /api/quotes or atomic quote creation (§11.1)."""
        quote = await fetch_or_compute_quote(
            product_ids=[line["product_id"] for line in lines if "product_id" in line],
            quantities={line["product_id"]: int(line.get("quantity", 1)) for line in lines if "product_id" in line},
            base_url=self.base_url,
        )
        return quote.to_dict()
