"""Document ingestion pipeline for Industrial Selection Lab.

Adheres strictly to MEGAPLAN.md §4.2, §4.3, §5.1, §7.1:
- Validates document identity and computes immutable SHA-256 byte hashes.
- Ingests documents (PDFs and text/markdown fixtures) into pages.
- Segments pages into stable, verifiable SourceSpan evidence items.
- Persists data/pages/pages.jsonl and data/pages/spans.jsonl.
- Exposes accessors for API and downstream knowledge engines.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from industrial_lab.schemas import DocumentMetadata, SourceKind, SourceSpan

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(os.environ.get("LAB_DATA_ROOT", "data"))
DEFAULT_CATALOG_PATH = DEFAULT_DATA_DIR / "manifests" / "catalog.yaml"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_PAGES_DIR = DEFAULT_DATA_DIR / "pages"
DEFAULT_PAGES_FILE = DEFAULT_PAGES_DIR / "pages.jsonl"
DEFAULT_SPANS_FILE = DEFAULT_PAGES_DIR / "spans.jsonl"


def compute_file_sha256(filepath: Path | str) -> str:
    """Computes the SHA-256 hex digest of file bytes for exact document identity (§4.3).
    
    Returns an empty string "" if the file does not exist or cannot be read.
    """
    path = Path(filepath)
    if not path.is_file():
        logger.warning(f"File not found for SHA-256 computation: {filepath}")
        return ""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        logger.warning(f"Error computing SHA-256 for {filepath}: {e}")
        return ""


def _extract_pages_from_pdf_poppler(file_path: Path) -> List[Tuple[int, str, str]]:
    """Local poppler pdftotext fallback (no remote calls)."""
    import subprocess
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(file_path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"pdftotext unavailable for {file_path}: {exc}")
        return []
    if proc.returncode != 0:
        logger.warning(f"pdftotext failed on {file_path}: {proc.stderr[:200]}")
        return []
    raw = proc.stdout or ""
    # pdftotext separates pages with form-feed
    parts = raw.split("\x0c")
    pages_out: List[Tuple[int, str, str]] = []
    for idx, part in enumerate(parts, start=1):
        text = part.strip()
        if text or idx == 1:
            pages_out.append((idx, str(idx), text))
    # Drop trailing empty page often produced by final form-feed
    while pages_out and not pages_out[-1][2] and len(pages_out) > 1:
        pages_out.pop()
    return pages_out


def _extract_pages_from_pdf(file_path: Path | str) -> List[Tuple[int, str, str]]:
    """Extracts pages from a PDF via pypdf, with local poppler fallback.
    
    Returns list of (pdf_page_index, printed_page_label, page_text).
    Page indices are 1-based (§4.3). Local only — no remote LLM calls.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.warning(f"PDF file not found: {file_path}")
        return []

    pages_out: List[Tuple[int, str, str]] = []
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        if reader.pages:
            for idx, page in enumerate(reader.pages, start=1):
                try:
                    text = (page.extract_text() or "").strip()
                except Exception as pe:
                    logger.warning(f"Failed to extract text from page {idx} in {file_path}: {pe}")
                    text = ""
                pages_out.append((idx, str(idx), text))
    except Exception as e:
        logger.warning(f"pypdf extraction failed on {file_path}: {e}")
        pages_out = []

    has_text = any(p[2].strip() for p in pages_out) if pages_out else False
    if not pages_out or not has_text:
        poppler_pages = _extract_pages_from_pdf_poppler(path)
        if poppler_pages and any(p[2].strip() for p in poppler_pages):
            return poppler_pages
    return pages_out


def _extract_pages_from_text(file_path: Path | str) -> List[Tuple[int, str, str]]:
    """Extracts pages from a text file, handling '=== PAGE X ===' delimiters or form-feeds."""
    path = Path(file_path)
    if not path.is_file():
        logger.warning(f"Text file not found: {file_path}")
        return []

    try:
        raw_content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"Failed reading text file {file_path}: {e}")
        return []

    if not raw_content.strip():
        return [(1, "1", "")]

    # Check for '=== PAGE (\d+) ===' pattern
    page_marker_re = re.compile(r"===\s*PAGE\s+(\d+)\s*===", re.IGNORECASE)
    parts = page_marker_re.split(raw_content)
    
    pages_out: List[Tuple[int, str, str]] = []
    if len(parts) > 1:
        # First part is header before page 1 (if any)
        header_text = parts[0].strip()
        idx_pair = 1
        while idx_pair < len(parts):
            try:
                p_num = int(parts[idx_pair])
            except ValueError:
                p_num = (idx_pair // 2) + 1
            p_text = parts[idx_pair + 1].strip() if (idx_pair + 1) < len(parts) else ""
            if p_num == 1 and header_text:
                p_text = f"{header_text}\n\n{p_text}".strip()
            pages_out.append((p_num, str(p_num), p_text))
            idx_pair += 2
        return pages_out

    # Check for form feed '\x0c'
    if "\x0c" in raw_content:
        ff_parts = raw_content.split("\x0c")
        for idx, part in enumerate(ff_parts, start=1):
            if part.strip():
                pages_out.append((idx, str(idx), part.strip()))
        if pages_out:
            return pages_out

    # Fallback: entire text is page 1
    return [(1, "1", raw_content.strip())]


def segment_page_into_spans(
    page_text: str,
    document_id: str,
    document_sha256: str,
    pdf_page_index: int,
    printed_page_label: Optional[str],
    product_scope: List[str],
    revision: str = "unknown",
) -> List[SourceSpan]:
    """Segments page text into discrete, verifiable SourceSpan evidence snippets (§5.1).
    
    Preserves section headers, bullet specifications, and distinct technical paragraphs.
    Handles empty text, bullets, numbered lists, plain paragraphs, and whitespace gracefully.
    """
    if not page_text or not page_text.strip():
        return []

    spans: List[SourceSpan] = []
    lines = page_text.splitlines()
    
    span_idx = 1
    current_block: List[str] = []

    def flush_block() -> None:
        nonlocal span_idx, current_block
        if current_block:
            block_text = " ".join(current_block).strip()
            if block_text:
                span_id = f"{document_id}:p{pdf_page_index:02d}:s{span_idx:02d}"
                spans.append(
                    SourceSpan(
                        span_id=span_id,
                        document_id=document_id,
                        document_sha256=document_sha256,
                        pdf_page_index=pdf_page_index,
                        printed_page_label=printed_page_label,
                        text=block_text,
                        product_scope=product_scope,
                        revision=revision,
                    )
                )
                span_idx += 1
            current_block = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            # Blank line marks paragraph boundary
            flush_block()
            continue

        # If line starts with a bullet ('-', '*', '•', '+') or numbered list item ('1.', '1)', etc.)
        is_bullet_or_number = line.startswith(("-", "*", "•", "+")) or bool(re.match(r"^\d+[\.\)]\s+", line))
        
        if is_bullet_or_number:
            flush_block()
            current_block = [line]
        else:
            current_block.append(line)

    flush_block()
    return spans


def parse_document(
    file_path: Path | str,
    doc_metadata: DocumentMetadata,
) -> Tuple[List[Dict[str, Any]], List[SourceSpan]]:
    """Parses a single document into pages and spans.
    
    Handles PDF and text files with graceful fallback when extraction fails or produces empty text.
    """
    p_path = Path(file_path)
    if not p_path.is_file():
        logger.warning(f"Document file does not exist: {file_path}")
        return [], []

    sha256_hash = compute_file_sha256(p_path)
    ext = p_path.suffix.lower()
    raw_pages: List[Tuple[int, str, str]] = []

    if ext == ".pdf":
        raw_pages = _extract_pages_from_pdf(p_path)
        # Check if PDF extraction returned empty list or all extracted pages have empty text
        has_text = any(p[2].strip() for p in raw_pages) if raw_pages else False
        if not raw_pages or not has_text:
            logger.info(f"PDF extraction yielded no text for {p_path}; attempting text fallback.")
            try:
                fallback_pages = _extract_pages_from_text(p_path)
                if fallback_pages and any(p[2].strip() for p in fallback_pages):
                    raw_pages = fallback_pages
            except Exception as te:
                logger.warning(f"Text fallback extraction failed for {p_path}: {te}")
        if not raw_pages:
            raw_pages = [(1, "1", "")]
    else:
        try:
            raw_pages = _extract_pages_from_text(p_path)
        except Exception as e:
            logger.warning(f"Failed to extract pages from text file {p_path}: {e}")
            raw_pages = [(1, "1", "")]

    pages_records: List[Dict[str, Any]] = []
    all_spans: List[SourceSpan] = []

    for p_idx, p_label, p_text in raw_pages:
        page_record = {
            "document_id": doc_metadata.document_id,
            "document_sha256": sha256_hash,
            "pdf_page_index": p_idx,
            "printed_page_label": p_label,
            "text": p_text,
            "product_ids": doc_metadata.product_ids,
            "revision": doc_metadata.revision,
            "source_kind": doc_metadata.source_kind.value if hasattr(doc_metadata.source_kind, "value") else str(doc_metadata.source_kind),
        }
        pages_records.append(page_record)

        spans = segment_page_into_spans(
            page_text=p_text,
            document_id=doc_metadata.document_id,
            document_sha256=sha256_hash,
            pdf_page_index=p_idx,
            printed_page_label=p_label,
            product_scope=doc_metadata.product_ids,
            revision=doc_metadata.revision,
        )
        all_spans.extend(spans)

    return pages_records, all_spans


def find_document_file(filename: str, raw_dir: Path | str) -> Optional[Path]:
    """Finds raw document file by filename, handling both .pdf and .txt fixtures."""
    r_dir = Path(raw_dir)
    if not r_dir.is_dir():
        logger.warning(f"Raw directory does not exist or is not a directory: {r_dir}")
        return None
    candidate = r_dir / filename
    if candidate.is_file():
        return candidate

    # Try replacement with .txt or .pdf or .md
    p = Path(filename)
    for alt_ext in [".txt", ".pdf", ".md"]:
        alt = r_dir / f"{p.stem}{alt_ext}"
        if alt.is_file():
            return alt

    return None


def ingest_documents(
    catalog_path: Optional[Path | str] = None,
    raw_dir: Optional[Path | str] = None,
    pages_dir: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Runs document ingestion pipeline across all catalog documents.
    
    Generates data/pages/pages.jsonl and data/pages/spans.jsonl.
    Handles non-existent raw directory or empty document lists gracefully.
    """
    cat_path = Path(catalog_path) if catalog_path is not None else DEFAULT_CATALOG_PATH
    r_dir = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    p_dir = Path(pages_dir) if pages_dir is not None else DEFAULT_PAGES_DIR

    p_dir.mkdir(parents=True, exist_ok=True)
    pages_file = p_dir / "pages.jsonl"
    spans_file = p_dir / "spans.jsonl"

    if not cat_path.is_file():
        raise FileNotFoundError(f"Catalog manifest not found: {cat_path}")

    if not r_dir.is_dir():
        logger.warning(f"Raw directory not found: {r_dir}")

    with open(cat_path, "r", encoding="utf-8") as f:
        catalog_raw = yaml.safe_load(f)

    if not isinstance(catalog_raw, dict):
        catalog_raw = {}

    products = catalog_raw.get("products", []) or []
    all_pages: List[Dict[str, Any]] = []
    all_spans: List[SourceSpan] = []
    seen_docs: set[str] = set()
    ingested_docs: set[str] = set()

    for prod in products:
        if not isinstance(prod, dict):
            continue
        prod_id = prod.get("product_id")
        docs = prod.get("documents", []) or []
        for doc_item in docs:
            # doc_item may be a dict or string
            if isinstance(doc_item, dict):
                doc_id = doc_item.get("document_id")
                filename = doc_item.get("filename")
                revision = doc_item.get("revision", "unknown")
                source_kind = doc_item.get("source_kind", "datasheet")
                product_ids = doc_item.get("product_ids", [prod_id] if prod_id else [])
            elif isinstance(doc_item, str):
                doc_id = str(doc_item)
                filename = f"{doc_id}.pdf"
                revision = "unknown"
                source_kind = "datasheet"
                product_ids = [prod_id] if prod_id else []
            else:
                continue

            if not doc_id or not filename:
                logger.warning(f"Skipping invalid doc item: {doc_item}")
                continue

            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

            raw_file = find_document_file(filename, r_dir)
            if not raw_file:
                logger.warning(f"Document file {filename} not found in {r_dir}")
                continue

            file_hash = compute_file_sha256(raw_file)
            doc_meta = DocumentMetadata(
                document_id=doc_id,
                product_ids=product_ids,
                filename=raw_file.name,
                sha256=file_hash,
                revision=revision,
                source_kind=SourceKind(source_kind) if source_kind in SourceKind.__members__ else SourceKind.datasheet,
            )

            pages_records, spans = parse_document(raw_file, doc_meta)
            all_pages.extend(pages_records)
            all_spans.extend(spans)
            ingested_docs.add(doc_id)

    # Write pages.jsonl
    with open(pages_file, "w", encoding="utf-8") as f:
        for p in all_pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Write spans.jsonl
    with open(spans_file, "w", encoding="utf-8") as f:
        for s in all_spans:
            f.write(s.to_json() + "\n")

    logger.info(f"Ingested {len(ingested_docs)} documents: {len(all_pages)} pages, {len(all_spans)} spans.")
    return {
        "documents_count": len(ingested_docs),
        "pages_count": len(all_pages),
        "spans_count": len(all_spans),
        "pages_file": str(pages_file),
        "spans_file": str(spans_file),
    }


def load_pages(pages_path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """Loads all page records from pages.jsonl.
    
    Returns an empty list if the file does not exist or is empty.
    Malformed JSON lines are skipped with a warning log without crashing.
    """
    p_file = Path(pages_path) if pages_path is not None else DEFAULT_PAGES_FILE
    if not p_file.is_file():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(p_file, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        records.append(data)
                    else:
                        logger.warning(f"Line {line_no} in {p_file} is not a valid JSON object: {line[:100]}")
                except Exception as e:
                    logger.warning(f"Skipping malformed JSON line {line_no} in {p_file}: {e}")
    except Exception as e:
        logger.warning(f"Error reading pages file {p_file}: {e}")
        return []
    return records


def load_spans(spans_path: Optional[Path | str] = None) -> List[SourceSpan]:
    """Loads all SourceSpan records from spans.jsonl.
    
    Returns an empty list if the file does not exist or is empty.
    Malformed JSON lines are skipped with a warning log without crashing.
    """
    s_file = Path(spans_path) if spans_path is not None else DEFAULT_SPANS_FILE
    if not s_file.is_file():
        return []
    spans: List[SourceSpan] = []
    try:
        with open(s_file, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    span = SourceSpan.from_json(line)
                    spans.append(span)
                except Exception as e:
                    logger.warning(f"Skipping malformed span line {line_no} in {s_file}: {e}")
    except Exception as e:
        logger.warning(f"Error reading spans file {s_file}: {e}")
        return []
    return spans


def get_document_page(
    document_id: str,
    page_index: int,
    pages_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieves specific page metadata and text by document ID and page index (1-based)."""
    try:
        target_idx = int(page_index)
    except (ValueError, TypeError):
        return None
    pages = load_pages(pages_path)
    for p in pages:
        if p.get("document_id") == document_id:
            try:
                if int(p.get("pdf_page_index", 0)) == target_idx:
                    return p
            except (ValueError, TypeError):
                continue
    return None


def get_document_page_spans(
    document_id: str,
    page_index: int,
    spans_path: Optional[Path | str] = None,
) -> List[SourceSpan]:
    """Retrieves all spans belonging to a specific document and page index (1-based)."""
    try:
        target_idx = int(page_index)
    except (ValueError, TypeError):
        return []
    spans = load_spans(spans_path)
    return [
        s for s in spans
        if s.document_id == document_id and getattr(s, "pdf_page_index", None) == target_idx
    ]

