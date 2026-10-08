"""Unit tests for scripts/fetch_manuals.py.

STRICT BENCHMARK RULES:
- Strictly NO real network calls: mocked responses and monkeypatching only.
- Uses temporary directories for all file I/O.
- Verifies hash verification, CLI options, check-only mode, and exit code logic.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request

import pytest
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_manuals import (
    DEFAULT_USER_AGENT,
    compute_sha256,
    download_file,
    fetch_and_verify_documents,
    main,
    print_manual_instructions,
    print_summary_table,
)


def _sha256_of_bytes(data: bytes) -> str:
    """Helper to compute hex sha256 of byte string."""
    return hashlib.sha256(data).hexdigest()


class DummyHTTPResponse(io.BytesIO):
    """Mock HTTP response for urllib.request.urlopen."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)

    def __enter__(self) -> DummyHTTPResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def test_compute_sha256(tmp_path: Path) -> None:
    """Test compute_sha256 with known data and non-existent files."""
    test_file = tmp_path / "sample.pdf"
    content = b"PDF-1.4 test document content for sha verification"
    test_file.write_bytes(content)

    expected = _sha256_of_bytes(content)
    assert compute_sha256(test_file) == expected

    # Non-existent file should return empty string
    assert compute_sha256(tmp_path / "non_existent.pdf") == ""


def test_hash_verification_success_on_valid_file(tmp_path: Path) -> None:
    """Test that a file with matching SHA-256 produces status 'OK' and sha_match 'MATCH'."""
    content = b"Valid content for document A"
    expected_hash = _sha256_of_bytes(content)

    doc_file = tmp_path / "doc_a.pdf"
    doc_file.write_bytes(content)

    manifest_data = {
        "catalog_id": "test-catalog",
        "documents": [
            {
                "document_id": "DOC_A",
                "filename": "doc_a.pdf",
                "sha256": expected_hash,
                "download_url": None,
                "landing_page": "https://example.com/doc_a",
                "download_url_status": "landing_page",
                "notes": "Test document A notes",
                "title": "Doc A Title",
                "manufacturer": "Vendor A",
            }
        ],
    }
    manifest_path = tmp_path / "sources.yaml"
    manifest_path.write_text(yaml.dump(manifest_data), encoding="utf-8")

    results = fetch_and_verify_documents(
        manifest_path=manifest_path,
        dest_dir=tmp_path,
        check_only=True,
    )

    assert len(results) == 1
    res = results[0]
    assert res["document_id"] == "DOC_A"
    assert res["status"] == "OK"
    assert res["sha_match"] == "MATCH"
    assert res["actual_sha"] == expected_hash


def test_hash_verification_failure_on_mismatched_content(tmp_path: Path) -> None:
    """Test that mismatched content produces status 'MISMATCH' and sha_match 'MISMATCH'."""
    content = b"Corrupted or modified document content"
    expected_hash = _sha256_of_bytes(b"Original official content")

    doc_file = tmp_path / "doc_b.pdf"
    doc_file.write_bytes(content)

    manifest_data = {
        "catalog_id": "test-catalog",
        "documents": [
            {
                "document_id": "DOC_B",
                "filename": "doc_b.pdf",
                "sha256": expected_hash,
                "download_url": "https://example.com/doc_b.pdf",
                "landing_page": "https://example.com/doc_b",
                "download_url_status": "verified",
                "notes": "Test doc B",
                "title": "Doc B Title",
                "manufacturer": "Vendor B",
            }
        ],
    }
    manifest_path = tmp_path / "sources.yaml"
    manifest_path.write_text(yaml.dump(manifest_data), encoding="utf-8")

    results = fetch_and_verify_documents(
        manifest_path=manifest_path,
        dest_dir=tmp_path,
        check_only=True,
    )

    assert len(results) == 1
    res = results[0]
    assert res["document_id"] == "DOC_B"
    assert res["status"] == "MISMATCH"
    assert res["sha_match"] == "MISMATCH"
    assert res["actual_sha"] == _sha256_of_bytes(content)
    assert res["actual_sha"] != expected_hash


def test_check_only_behavior_with_missing_and_present_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test --check-only never calls network and correctly distinguishes OK, MISMATCH, MISSING, and MANUAL."""
    # Guard: raise error if urllib.request.urlopen is ever called during check-only
    def fail_on_network_call(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Network call attempted in check-only mode!")

    monkeypatch.setattr(urllib.request, "urlopen", fail_on_network_call)

    # Prepare files in tmp_path
    ok_content = b"Document 1 OK"
    ok_hash = _sha256_of_bytes(ok_content)
    (tmp_path / "doc1.pdf").write_bytes(ok_content)

    mismatch_content = b"Document 2 bad content"
    (tmp_path / "doc2.pdf").write_bytes(mismatch_content)

    manifest_data = {
        "catalog_id": "test-catalog",
        "documents": [
            {
                "document_id": "DOC_1",
                "filename": "doc1.pdf",
                "sha256": ok_hash,
                "download_url": "https://example.com/doc1.pdf",
                "download_url_status": "verified",
            },
            {
                "document_id": "DOC_2",
                "filename": "doc2.pdf",
                "sha256": _sha256_of_bytes(b"Document 2 expected"),
                "download_url": "https://example.com/doc2.pdf",
                "download_url_status": "verified",
            },
            {
                "document_id": "DOC_3",
                "filename": "doc3.pdf",
                "sha256": _sha256_of_bytes(b"Document 3 expected"),
                "download_url": "https://example.com/doc3.pdf",
                "download_url_status": "unverified",
            },
            {
                "document_id": "DOC_4",
                "filename": "doc4.pdf",
                "sha256": _sha256_of_bytes(b"Document 4 expected"),
                "download_url": None,
                "landing_page": "https://example.com/doc4",
                "download_url_status": "landing_page",
            },
        ],
    }
    manifest_path = tmp_path / "sources.yaml"
    manifest_path.write_text(yaml.dump(manifest_data), encoding="utf-8")

    results = fetch_and_verify_documents(
        manifest_path=manifest_path,
        dest_dir=tmp_path,
        check_only=True,
    )

    statuses = {r["document_id"]: (r["status"], r["sha_match"]) for r in results}
    assert statuses["DOC_1"] == ("OK", "MATCH")
    assert statuses["DOC_2"] == ("MISMATCH", "MISMATCH")
    assert statuses["DOC_3"] == ("MISSING", "MISSING")
    assert statuses["DOC_4"] == ("MANUAL", "MISSING")


def test_download_mock_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test download_file with mocked responses and retries."""
    content = b"Downloaded PDF payload from mock server"
    correct_hash = _sha256_of_bytes(content)

    requested_headers: dict[str, str] = {}

    def mock_urlopen_success(req: urllib.request.Request, timeout: int = 30) -> DummyHTTPResponse:
        requested_headers.update(req.headers)
        return DummyHTTPResponse(content)

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_success)

    target_file = tmp_path / "downloaded.pdf"

    # Test 1: Successful download and verified hash
    success = download_file(
        url="https://example.com/test.pdf",
        dest_path=target_file,
        expected_sha256=correct_hash,
        user_agent=DEFAULT_USER_AGENT,
        timeout=10,
        max_retries=1,
    )
    assert success is True
    assert target_file.is_file()
    assert target_file.read_bytes() == content
    assert requested_headers.get("User-agent") == DEFAULT_USER_AGENT

    # Test 2: Download succeeds but hash mismatches
    mismatch_target = tmp_path / "mismatch.pdf"
    success_mismatch = download_file(
        url="https://example.com/test.pdf",
        dest_path=mismatch_target,
        expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        user_agent=DEFAULT_USER_AGENT,
        timeout=10,
        max_retries=1,
    )
    assert success_mismatch is False
    assert mismatch_target.is_file()  # Left at destination for MISMATCH report

    # Test 3: Download failure with retries
    call_count = 0

    def mock_urlopen_failure(req: urllib.request.Request, timeout: int = 30) -> Any:
        nonlocal call_count
        call_count += 1
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen_failure)
    monkeypatch.setattr("time.sleep", lambda s: None)  # Prevent actual sleep in test

    fail_target = tmp_path / "fail.pdf"
    success_fail = download_file(
        url="https://example.com/fail.pdf",
        dest_path=fail_target,
        expected_sha256=correct_hash,
        max_retries=2,
    )
    assert success_fail is False
    assert not fail_target.exists()
    assert call_count == 3  # Initial + 2 retries


def test_correct_exit_code_logic_zero_only_when_all_five_match(tmp_path: Path) -> None:
    """Test exit code logic: 0 ONLY when all 5 files exist and their SHA-256 hashes match."""
    # Build 5 dummy documents
    docs_spec = []
    for i in range(1, 6):
        data = f"Document content {i}".encode("utf-8")
        h = _sha256_of_bytes(data)
        filename = f"manual_{i}.pdf"
        docs_spec.append((f"DOC_{i}", filename, h, data))

    def make_manifest(doc_tuples: list[tuple[str, str, str, bytes]]) -> Path:
        manifest_data = {
            "catalog_id": "test-5-docs",
            "documents": [
                {
                    "document_id": d_id,
                    "filename": fname,
                    "sha256": exp_sha,
                    "download_url": None,
                    "landing_page": f"https://example.com/{d_id}",
                    "download_url_status": "landing_page",
                    "title": f"Doc {d_id}",
                }
                for (d_id, fname, exp_sha, _) in doc_tuples
            ],
        }
        m_path = tmp_path / f"manifest_{len(doc_tuples)}.yaml"
        m_path.write_text(yaml.dump(manifest_data), encoding="utf-8")
        return m_path

    manifest_5 = make_manifest(docs_spec)

    dest_dir = tmp_path / "raw_dest"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Case 1: Missing all files -> non-zero exit code (1)
    code = main(["--manifest", str(manifest_5), "--dest", str(dest_dir), "--check-only"])
    assert code == 1

    # Case 2: Write 4 matching files, 1 file still missing -> non-zero exit code (1)
    for i in range(4):
        _, fname, _, data = docs_spec[i]
        (dest_dir / fname).write_bytes(data)

    code = main(["--manifest", str(manifest_5), "--dest", str(dest_dir), "--check-only"])
    assert code == 1

    # Case 3: Write 5th file but with corrupted content -> non-zero exit code (1)
    _, fname_5, _, _ = docs_spec[4]
    (dest_dir / fname_5).write_bytes(b"corrupted 5th file content")

    code = main(["--manifest", str(manifest_5), "--dest", str(dest_dir), "--check-only"])
    assert code == 1

    # Case 4: Write correct 5th file -> all 5 match -> exit code 0!
    _, fname_5, _, data_5 = docs_spec[4]
    (dest_dir / fname_5).write_bytes(data_5)

    code = main(["--manifest", str(manifest_5), "--dest", str(dest_dir), "--check-only"])
    assert code == 0

    # Case 5: Manifest with only 4 documents, all 4 match -> exit code 1 (must be all 5)
    manifest_4 = make_manifest(docs_spec[:4])
    code = main(["--manifest", str(manifest_4), "--dest", str(dest_dir), "--check-only"])
    assert code == 1

    # Case 6: Non-existent manifest -> exit code 1
    code = main(["--manifest", str(tmp_path / "missing.yaml"), "--dest", str(dest_dir)])
    assert code == 1


def test_table_and_manual_instructions_formatting(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test that summary table and manual download instructions print expected text."""
    results = [
        {
            "document_id": "D_TEST_1",
            "filename": "test1.pdf",
            "status": "OK",
            "sha_match": "MATCH",
            "expected_sha": "abc123",
            "actual_sha": "abc123",
            "download_url": "https://example.com/dl",
            "landing_page": "https://example.com/page",
            "download_url_status": "verified",
            "notes": "Verified download link",
            "title": "Test Title 1",
            "manufacturer": "Vendor 1",
            "path": str(tmp_path / "test1.pdf"),
        },
        {
            "document_id": "D_TEST_2",
            "filename": "test2.pdf",
            "status": "MANUAL",
            "sha_match": "MISSING",
            "expected_sha": "def456",
            "actual_sha": "",
            "download_url": None,
            "landing_page": "https://example.com/manual_page",
            "download_url_status": "landing_page",
            "notes": "Must request from manufacturer",
            "title": "Test Title 2",
            "manufacturer": "Vendor 2",
            "path": str(tmp_path / "test2.pdf"),
        },
    ]

    print_summary_table(results)
    out = capsys.readouterr().out
    assert "Document ID" in out
    assert "Filename" in out
    assert "Status" in out
    assert "SHA-256 Match" in out
    assert "D_TEST_1" in out
    assert "OK" in out
    assert "MATCH" in out
    assert "D_TEST_2" in out
    assert "MANUAL" in out

    print_manual_instructions(results, dest_dir=tmp_path)
    inst_out = capsys.readouterr().out
    assert "MANUAL DOWNLOAD INSTRUCTIONS" in inst_out
    assert "D_TEST_2 - Test Title 2" in inst_out
    assert "Vendor Landing Page: https://example.com/manual_page" in inst_out
    assert "Must request from manufacturer" in inst_out
    # D_TEST_1 was OK, so it shouldn't be in manual instructions
    assert "D_TEST_1" not in inst_out
