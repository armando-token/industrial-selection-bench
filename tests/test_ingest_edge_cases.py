"""Edge case tests for document ingestion, span segmentation, and hashing.

Covers:
- Handling non-existent raw directory or empty document lists.
- Graceful fallback when PDF text extraction returns empty or fails.
- Handling empty/whitespace lines in text files.
- Robust load_pages and load_spans routines with missing files, empty files, and malformed JSON lines.
- compute_file_sha256 with missing files, empty files, and directories.
- segment_page_into_spans with bullets, numbered lists, plain paragraphs, empty text, and mixed content.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import yaml

from industrial_lab.schemas import DocumentMetadata, SourceKind, SourceSpan
from industrial_lab.ingest.documents import (
    _extract_pages_from_pdf,
    _extract_pages_from_text,
    compute_file_sha256,
    find_document_file,
    get_document_page,
    get_document_page_spans,
    ingest_documents,
    load_pages,
    load_spans,
    parse_document,
    segment_page_into_spans,
)


# ==============================================================================
# 1. compute_file_sha256 edge cases
# ==============================================================================

def test_compute_file_sha256_missing_file(tmp_path: Path) -> None:
    """compute_file_sha256 should return empty string "" if file does not exist."""
    missing = tmp_path / "does_not_exist.txt"
    assert compute_file_sha256(missing) == ""


def test_compute_file_sha256_directory(tmp_path: Path) -> None:
    """compute_file_sha256 should return empty string "" if path is a directory."""
    assert compute_file_sha256(tmp_path) == ""


def test_compute_file_sha256_empty_file(tmp_path: Path) -> None:
    """compute_file_sha256 on an empty 0-byte file returns standard sha256 of empty bytes."""
    empty_file = tmp_path / "empty.bin"
    empty_file.touch()
    expected = hashlib.sha256(b"").hexdigest()
    assert compute_file_sha256(empty_file) == expected


def test_compute_file_sha256_valid_file(tmp_path: Path) -> None:
    """compute_file_sha256 correctly computes hex digest for arbitrary bytes."""
    f = tmp_path / "valid.txt"
    content = b"Industrial Selection Lab verification"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert compute_file_sha256(f) == expected


# ==============================================================================
# 2. load_pages edge cases
# ==============================================================================

def test_load_pages_missing_file(tmp_path: Path) -> None:
    """load_pages returns empty list when file does not exist."""
    missing_file = tmp_path / "missing_pages.jsonl"
    assert load_pages(missing_file) == []


def test_load_pages_empty_file(tmp_path: Path) -> None:
    """load_pages returns empty list when file is empty (0 bytes)."""
    empty_file = tmp_path / "empty_pages.jsonl"
    empty_file.touch()
    assert load_pages(empty_file) == []


def test_load_pages_whitespace_only_file(tmp_path: Path) -> None:
    """load_pages returns empty list when file contains only whitespace and blank lines."""
    ws_file = tmp_path / "whitespace_pages.jsonl"
    ws_file.write_text("   \n\n\t  \r\n   \n", encoding="utf-8")
    assert load_pages(ws_file) == []


def test_load_pages_invalid_json_line(tmp_path: Path) -> None:
    """load_pages skips malformed JSON lines with warning instead of crashing."""
    bad_file = tmp_path / "bad_pages.jsonl"
    bad_file.write_text("not a valid json {{{{ \n", encoding="utf-8")
    assert load_pages(bad_file) == []


def test_load_pages_mixed_valid_and_invalid_lines(tmp_path: Path) -> None:
    """load_pages recovers valid records while ignoring malformed lines or non-object JSON."""
    mixed_file = tmp_path / "mixed_pages.jsonl"
    valid_record_1 = {"document_id": "DOC1", "pdf_page_index": 1, "text": "Page one text"}
    valid_record_2 = {"document_id": "DOC1", "pdf_page_index": 2, "text": "Page two text"}

    lines = [
        json.dumps(valid_record_1),
        "INVALID JSON LINE {{{",
        "",  # blank line
        "12345",  # JSON integer, not a dict
        '"just a string"',  # JSON string, not a dict
        json.dumps(valid_record_2),
        "{incomplete: json",
    ]
    mixed_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = load_pages(mixed_file)
    assert len(records) == 2
    assert records[0]["pdf_page_index"] == 1
    assert records[1]["pdf_page_index"] == 2


# ==============================================================================
# 3. load_spans edge cases
# ==============================================================================

def test_load_spans_missing_file(tmp_path: Path) -> None:
    """load_spans returns empty list when file does not exist."""
    missing_file = tmp_path / "missing_spans.jsonl"
    assert load_spans(missing_file) == []


def test_load_spans_empty_file(tmp_path: Path) -> None:
    """load_spans returns empty list when file is empty (0 bytes)."""
    empty_file = tmp_path / "empty_spans.jsonl"
    empty_file.touch()
    assert load_spans(empty_file) == []


def test_load_spans_whitespace_only_file(tmp_path: Path) -> None:
    """load_spans returns empty list when file contains only whitespace and blank lines."""
    ws_file = tmp_path / "whitespace_spans.jsonl"
    ws_file.write_text("   \n\n\t  \r\n   \n", encoding="utf-8")
    assert load_spans(ws_file) == []


def test_load_spans_invalid_json_line(tmp_path: Path) -> None:
    """load_spans skips malformed JSON lines with warning instead of crashing."""
    bad_file = tmp_path / "bad_spans.jsonl"
    bad_file.write_text("<<< invalid json >>>\n", encoding="utf-8")
    assert load_spans(bad_file) == []


def test_load_spans_mixed_valid_and_invalid_lines(tmp_path: Path) -> None:
    """load_spans recovers valid SourceSpan objects while skipping malformed or invalid schema lines."""
    span_1 = SourceSpan(
        span_id="D1:p01:s01",
        document_id="D1",
        document_sha256="abc123hash",
        pdf_page_index=1,
        printed_page_label="1",
        text="Valid span 1",
        product_scope=["P1"],
        revision="revA",
    )
    span_2 = SourceSpan(
        span_id="D1:p01:s02",
        document_id="D1",
        document_sha256="abc123hash",
        pdf_page_index=1,
        printed_page_label="1",
        text="Valid span 2",
        product_scope=["P1"],
        revision="revA",
    )

    mixed_file = tmp_path / "mixed_spans.jsonl"
    lines = [
        span_1.to_json(),
        "NOT JSON AT ALL",
        json.dumps({"document_id": "D1"}),  # missing required SourceSpan fields
        "",  # blank line
        span_2.to_json(),
        "{'single_quotes': True}",  # invalid JSON
    ]
    mixed_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    loaded = load_spans(mixed_file)
    assert len(loaded) == 2
    assert loaded[0].span_id == "D1:p01:s01"
    assert loaded[1].span_id == "D1:p01:s02"


# ==============================================================================
# 4. segment_page_into_spans edge cases
# ==============================================================================

def test_segment_page_into_spans_empty_and_whitespace() -> None:
    """segment_page_into_spans returns empty list for empty or whitespace-only text."""
    doc_id = "DOC1"
    sha = "hash123"

    assert segment_page_into_spans("", doc_id, sha, 1, "1", ["P1"]) == []
    assert segment_page_into_spans("   ", doc_id, sha, 1, "1", ["P1"]) == []
    assert segment_page_into_spans("\n\n\t  \r\n   \n", doc_id, sha, 1, "1", ["P1"]) == []


def test_segment_page_into_spans_bullets() -> None:
    """segment_page_into_spans splits distinct bullet items (dash, asterisk, bullet symbol, plus)."""
    text = (
        "- Supply Voltage: 24V DC nominal\n"
        "- Power Consumption: 15W max\n"
        "* Isolation: 1500V AC\n"
        "• Protection: Reverse polarity\n"
        "+ Auxiliary: 5V DC output"
    )
    spans = segment_page_into_spans(
        page_text=text,
        document_id="DOC1",
        document_sha256="hash123",
        pdf_page_index=1,
        printed_page_label="1",
        product_scope=["P1"],
        revision="rev1",
    )
    assert len(spans) == 5
    assert spans[0].text == "- Supply Voltage: 24V DC nominal"
    assert spans[0].span_id == "DOC1:p01:s01"
    assert spans[1].text == "- Power Consumption: 15W max"
    assert spans[1].span_id == "DOC1:p01:s02"
    assert spans[2].text == "* Isolation: 1500V AC"
    assert spans[3].text == "• Protection: Reverse polarity"
    assert spans[4].text == "+ Auxiliary: 5V DC output"


def test_segment_page_into_spans_numbered_lists() -> None:
    """segment_page_into_spans splits numbered list items like 1. and 2) correctly."""
    text = (
        "1. First step in installation.\n"
        "2. Second step in configuration.\n"
        "3. Third step in wiring.\n"
        "4) Alternative numbering style."
    )
    spans = segment_page_into_spans(
        page_text=text,
        document_id="DOC1",
        document_sha256="hash123",
        pdf_page_index=2,
        printed_page_label="2",
        product_scope=["P1"],
    )
    assert len(spans) == 4
    assert spans[0].text == "1. First step in installation."
    assert spans[1].text == "2. Second step in configuration."
    assert spans[2].text == "3. Third step in wiring."
    assert spans[3].text == "4) Alternative numbering style."


def test_segment_page_into_spans_plain_paragraphs() -> None:
    """segment_page_into_spans splits plain paragraphs separated by empty lines."""
    text = (
        "This is paragraph one.\n"
        "It spans two consecutive lines of prose.\n"
        "\n"
        "This is paragraph two.\n"
        "It also has multiple sentences.\n"
        "\n\n\n"  # multiple blank lines
        "This is paragraph three."
    )
    spans = segment_page_into_spans(
        page_text=text,
        document_id="DOC1",
        document_sha256="hash123",
        pdf_page_index=1,
        printed_page_label="1",
        product_scope=["P1"],
    )
    assert len(spans) == 3
    assert spans[0].text == "This is paragraph one. It spans two consecutive lines of prose."
    assert spans[1].text == "This is paragraph two. It also has multiple sentences."
    assert spans[2].text == "This is paragraph three."


def test_segment_page_into_spans_mixed_content() -> None:
    """segment_page_into_spans handles combinations of titles, paragraphs, and bullets."""
    text = (
        "Overview and Introduction\n"
        "This controller is designed for factory automation.\n"
        "\n"
        "Key Features:\n"
        "- High speed 24V inputs\n"
        "- Modbus RTU serial interface\n"
        "\n"
        "1. Mount on 35mm DIN rail\n"
        "2. Connect regulated 24V supply"
    )
    spans = segment_page_into_spans(
        page_text=text,
        document_id="DOC1",
        document_sha256="hash123",
        pdf_page_index=3,
        printed_page_label="3",
        product_scope=["P1"],
    )
    # 1: "Overview and Introduction This controller is designed for factory automation."
    # 2: "Key Features:"
    # 3: "- High speed 24V inputs"
    # 4: "- Modbus RTU serial interface"
    # 5: "1. Mount on 35mm DIN rail"
    # 6: "2. Connect regulated 24V supply"
    assert len(spans) == 6
    assert "Overview and Introduction" in spans[0].text
    assert spans[1].text == "Key Features:"
    assert spans[2].text == "- High speed 24V inputs"
    assert spans[3].text == "- Modbus RTU serial interface"
    assert spans[4].text == "1. Mount on 35mm DIN rail"
    assert spans[5].text == "2. Connect regulated 24V supply"


# ==============================================================================
# 5. Text / PDF extraction & fallback edge cases
# ==============================================================================

def test_extract_pages_from_text_empty_and_whitespace(tmp_path: Path) -> None:
    """_extract_pages_from_text handles empty or whitespace text files."""
    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_text("   \n\n\t  \n", encoding="utf-8")
    pages = _extract_pages_from_text(empty_txt)
    assert len(pages) == 1
    assert pages[0][0] == 1
    assert pages[0][2] == ""


def test_extract_pages_from_text_missing_file(tmp_path: Path) -> None:
    """_extract_pages_from_text returns empty list for missing file."""
    assert _extract_pages_from_text(tmp_path / "not_found.txt") == []


def test_extract_pages_from_pdf_missing_file(tmp_path: Path) -> None:
    """_extract_pages_from_pdf returns empty list for non-existent file."""
    assert _extract_pages_from_pdf(tmp_path / "missing.pdf") == []


def test_parse_document_mock_pdf_fallback(tmp_path: Path) -> None:
    """parse_document gracefully falls back to text extraction when a .pdf file is text-based."""
    fake_pdf = tmp_path / "test_doc.pdf"
    fake_pdf.write_text("=== PAGE 1 ===\nVoltage: 24V DC\n=== PAGE 2 ===\nCurrent: 5A\n", encoding="utf-8")

    meta = DocumentMetadata(
        document_id="D_TEST",
        product_ids=["P1"],
        filename=fake_pdf.name,
        sha256="testsha",
        revision="revA",
        source_kind=SourceKind.datasheet,
    )
    pages_records, spans = parse_document(fake_pdf, meta)
    assert len(pages_records) == 2
    assert pages_records[0]["pdf_page_index"] == 1
    assert "24V DC" in pages_records[0]["text"]
    assert pages_records[1]["pdf_page_index"] == 2
    assert "5A" in pages_records[1]["text"]
    assert len(spans) >= 2


def test_parse_document_empty_pdf(tmp_path: Path) -> None:
    """parse_document handles empty 0-byte .pdf file without crashing."""
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.touch()

    meta = DocumentMetadata(
        document_id="D_EMPTY",
        product_ids=["P1"],
        filename=empty_pdf.name,
        sha256="testsha",
    )
    pages_records, spans = parse_document(empty_pdf, meta)
    assert len(pages_records) == 1
    assert pages_records[0]["text"] == ""
    assert spans == []


# ==============================================================================
# 6. ingest_documents edge cases (non-existent raw dir, empty catalog)
# ==============================================================================

def test_find_document_file_missing_dir(tmp_path: Path) -> None:
    """find_document_file returns None when raw directory does not exist."""
    missing_dir = tmp_path / "no_such_dir"
    assert find_document_file("sample.pdf", missing_dir) is None


def test_ingest_documents_empty_catalog(tmp_path: Path) -> None:
    """ingest_documents handles empty catalog with no products without crashing."""
    cat_file = tmp_path / "empty_catalog.yaml"
    cat_file.write_text("products: []\n", encoding="utf-8")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pages_dir = tmp_path / "pages"

    res = ingest_documents(catalog_path=cat_file, raw_dir=raw_dir, pages_dir=pages_dir)
    assert res["documents_count"] == 0
    assert res["pages_count"] == 0
    assert res["spans_count"] == 0
    assert Path(res["pages_file"]).is_file()
    assert Path(res["spans_file"]).is_file()


def test_ingest_documents_missing_raw_dir(tmp_path: Path) -> None:
    """ingest_documents handles non-existent raw directory gracefully."""
    cat_file = tmp_path / "catalog.yaml"
    cat_data = {
        "products": [
            {
                "product_id": "P1",
                "documents": [{"document_id": "D1", "filename": "d1.pdf"}],
            }
        ]
    }
    with open(cat_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cat_data, f)

    non_existent_raw = tmp_path / "non_existent_raw"
    pages_dir = tmp_path / "pages"

    res = ingest_documents(catalog_path=cat_file, raw_dir=non_existent_raw, pages_dir=pages_dir)
    # File not found in raw dir -> 0 ingested documents, no unhandled exceptions
    assert res["documents_count"] == 0
    assert res["pages_count"] == 0
    assert res["spans_count"] == 0


# ==============================================================================
# 7. get_document_page and get_document_page_spans edge cases
# ==============================================================================

def test_get_document_page_non_existent(tmp_path: Path) -> None:
    """get_document_page returns None when document or page does not exist."""
    pages_file = tmp_path / "pages.jsonl"
    pages_file.touch()

    assert get_document_page("NON_EXISTENT", 1, pages_path=pages_file) is None
    # Invalid page number
    assert get_document_page("NON_EXISTENT", -99, pages_path=pages_file) is None


def test_get_document_page_spans_non_existent(tmp_path: Path) -> None:
    """get_document_page_spans returns empty list when document does not exist."""
    spans_file = tmp_path / "spans.jsonl"
    spans_file.touch()

    assert get_document_page_spans("NON_EXISTENT", 1, spans_path=spans_file) == []
    assert get_document_page_spans("NON_EXISTENT", -99, spans_path=spans_file) == []
