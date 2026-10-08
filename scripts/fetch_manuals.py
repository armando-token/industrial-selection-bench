#!/usr/bin/env python3
"""fetch_manuals.py - Benchmark manual source downloader and hash verification tool.

COPYRIGHT AND REDISTRIBUTION NOTICE:
====================================
The manufacturer documents (PDF manuals, datasheets, and installation guides)
used in this benchmark are NOT redistributed in this repository due to copyright
restrictions. For example, Horner Automation's user manual explicitly states:
"No part of this publication may be reproduced without the prior agreement and
written permission of Horner APG, LLC."

To ensure compliance with intellectual property laws and vendor licenses, users
and researchers must download each document directly from official manufacturer
sources and verify the file integrity against the SHA-256 hashes published in:
    data/manifests/manual_sources.yaml

This script automates the download of documents with direct vendor URLs,
verifies file integrity using cryptographic SHA-256 checksums, and provides
detailed manual retrieval instructions for documents requiring vendor navigation
or direct inquiry.

PIPELINE REBUILD WORKFLOW:
==========================
Once all 5 raw documents are fetched and verified in data/raw/ (or custom --dest),
you must rebuild the benchmark dataset (pages, facts, knowledge graph, and search
indexes) using the following existing CLI commands in src/industrial_lab/cli.py:

1. Validate Raw Documents and Catalog Integrity:
   $ python -m industrial_lab.cli validate-inputs \\
       --catalog data/manifests/catalog.yaml \\
       --data-dir data

   Validates that catalog.yaml is structurally sound, all 3 real products and 5
   documents are present in data/raw/, and document SHA-256 hashes match the
   frozen benchmark specifications.

2. Ingest Document Pages and Text Spans:
   $ python -m industrial_lab.cli ingest \\
       --catalog data/manifests/catalog.yaml \\
       --stage pages \\
       --raw-dir data/raw \\
       --pages-dir data/pages

   Extracts text and page boundaries from raw PDF documents into data/pages/pages.jsonl
   and data/pages/spans.jsonl.

3. Extract Candidate Facts from Spans:
   $ python -m industrial_lab.cli ingest \\
       --catalog data/manifests/catalog.yaml \\
       --stage facts

   Runs automated fact extraction across page spans to generate candidate facts
   at data/facts/facts.auto.jsonl.

4. Export Facts for Human Expert Review:
   $ python -m industrial_lab.cli review-export \\
       --output data/review/pending.json \\
       --facts-file data/facts/facts.auto.jsonl

   Exports unapproved/unreviewed facts into a structured review queue for human
   curation and verification against source spans.

5. Import Human-Approved Facts:
   $ python -m industrial_lab.cli review-import \\
       --file data/review/approved.json \\
       --output data/facts/facts.reviewed.jsonl \\
       --catalog data/manifests/catalog.yaml

   Validates and imports reviewed facts into data/facts/facts.reviewed.jsonl,
   enforcing strict schema adherence and product ID validation.

6. Build Knowledge Store and Relational Graph:
   $ python -m industrial_lab.cli build-knowledge \\
       --reviewed-facts data/facts/facts.reviewed.jsonl

   Compiles the SQLite/JSON knowledge store and NetworkX property graph used by
   the TypeSafe AI / JEV engine.

7. Build Hybrid RAG Retrieval Index:
   $ python -m industrial_lab.cli build-rag \\
       --data-dir data

   Indexes page texts into the hybrid BM25 and vector embedding index at
   data/rag/index for RAG+LLM retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("fetch_manuals")

# Default settings
DEFAULT_USER_AGENT = (
    "IndustrialSelectionBench/0.1.0 (research benchmark; manual fetcher)"
)
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 2
DEFAULT_MANIFEST = Path("data/manifests/manual_sources.yaml")
DEFAULT_DEST = Path("data/raw")
EXPECTED_DOCUMENT_COUNT = 5


def compute_sha256(filepath: Path | str) -> str:
    """Compute the hexadecimal SHA-256 checksum of a file.

    Args:
        filepath: Path to the target file.

    Returns:
        Hexadecimal SHA-256 hash string, or empty string if file does not exist.
    """
    p = Path(filepath)
    if not p.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def download_file(
    url: str,
    dest_path: Path,
    expected_sha256: str,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> bool:
    """Download a file with retries and verify its SHA-256 hash.

    Args:
        url: Remote HTTP/HTTPS URL.
        dest_path: Target path on local disk.
        expected_sha256: Expected hexadecimal SHA-256 hash.
        user_agent: Custom User-Agent header string.
        timeout: Socket timeout in seconds.
        max_retries: Maximum number of retries upon error (excluding initial attempt).

    Returns:
        True if download succeeded and hash matches, False otherwise.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent},
    )

    total_attempts = 1 + max(0, max_retries)
    for attempt in range(1, total_attempts + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)...", url, attempt, total_attempts)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with open(tmp_path, "wb") as out_f:
                    while chunk := response.read(65536):
                        out_f.write(chunk)

            # Compute and check hash of downloaded temporary file
            actual_sha = compute_sha256(tmp_path)
            if expected_sha256 and actual_sha != expected_sha256.lower():
                logger.warning(
                    "Hash mismatch for downloaded %s: expected %s, got %s",
                    dest_path.name,
                    expected_sha256,
                    actual_sha,
                )
                # Keep file at dest_path so verification correctly reports MISMATCH
                tmp_path.replace(dest_path)
                return False

            # Success: rename temp file to target destination
            tmp_path.replace(dest_path)
            logger.info("Successfully fetched %s (hash verified).", dest_path.name)
            return True

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            logger.warning(
                "Attempt %d/%d failed to download %s: %s",
                attempt,
                total_attempts,
                url,
                exc,
            )
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if attempt < total_attempts:
                backoff = float(attempt)
                time.sleep(backoff)

    return False


def fetch_and_verify_documents(
    manifest_path: Path,
    dest_dir: Path,
    check_only: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    user_agent: str = DEFAULT_USER_AGENT,
) -> List[Dict[str, Any]]:
    """Process all documents listed in the manifest and verify their checksums.

    Args:
        manifest_path: Path to manual_sources.yaml manifest.
        dest_dir: Target directory where documents reside or will be downloaded.
        check_only: If True, do not download files; only verify local files.
        timeout: Download timeout in seconds.
        max_retries: Max retries for download attempts.
        user_agent: User-Agent header string.

    Returns:
        List of dicts representing verification results for each document.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = yaml.safe_load(f) or {}

    documents = manifest_data.get("documents", [])
    results: List[Dict[str, Any]] = []

    for doc in documents:
        doc_id = doc.get("document_id", "UNKNOWN")
        filename = doc.get("filename", "")
        expected_sha = (doc.get("sha256") or "").lower()
        download_url = doc.get("download_url")
        landing_page = doc.get("landing_page")
        url_status = doc.get("download_url_status", "")
        notes = doc.get("notes", "")
        title = doc.get("title", "")
        manufacturer = doc.get("manufacturer", "")

        file_path = dest_dir / filename if filename else None

        # Check if file already exists locally
        if file_path and file_path.is_file():
            actual_sha = compute_sha256(file_path)
            if expected_sha and actual_sha == expected_sha:
                status = "OK"
                sha_match = "MATCH"
            else:
                status = "MISMATCH"
                sha_match = "MISMATCH"
        else:
            # File is missing locally
            actual_sha = ""
            if not check_only and download_url:
                # Attempt automated download
                success = download_file(
                    url=download_url,
                    dest_path=file_path,
                    expected_sha256=expected_sha,
                    user_agent=user_agent,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                if success:
                    status = "OK"
                    sha_match = "MATCH"
                    actual_sha = expected_sha
                else:
                    if file_path and file_path.is_file():
                        actual_sha = compute_sha256(file_path)
                        status = "MISMATCH"
                        sha_match = "MISMATCH"
                    else:
                        status = "MISSING"
                        sha_match = "MISSING"
            else:
                # Either in check_only mode or no direct download URL
                if not download_url or url_status in ("landing_page", "not_found"):
                    status = "MANUAL"
                    sha_match = "MISSING"
                else:
                    status = "MISSING"
                    sha_match = "MISSING"

        results.append({
            "document_id": doc_id,
            "filename": filename,
            "status": status,
            "sha_match": sha_match,
            "expected_sha": expected_sha,
            "actual_sha": actual_sha,
            "download_url": download_url,
            "landing_page": landing_page,
            "download_url_status": url_status,
            "notes": notes,
            "title": title,
            "manufacturer": manufacturer,
            "path": str(file_path) if file_path else "",
        })

    return results


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print an aligned ASCII table summarizing document statuses."""
    id_w = max(len("Document ID"), max((len(r["document_id"]) for r in results), default=24))
    fn_w = max(len("Filename"), max((len(r["filename"]) for r in results), default=40))
    st_w = max(len("Status"), max((len(r["status"]) for r in results), default=8))
    sh_w = max(len("SHA-256 Match"), max((len(r["sha_match"]) for r in results), default=13))

    sep = f"+-{'-' * id_w}-+-{'-' * fn_w}-+-{'-' * st_w}-+-{'-' * sh_w}-+"
    header = (
        f"| {'Document ID':<{id_w}} "
        f"| {'Filename':<{fn_w}} "
        f"| {'Status':<{st_w}} "
        f"| {'SHA-256 Match':<{sh_w}} |"
    )

    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        row = (
            f"| {r['document_id']:<{id_w}} "
            f"| {r['filename']:<{fn_w}} "
            f"| {r['status']:<{st_w}} "
            f"| {r['sha_match']:<{sh_w}} |"
        )
        print(row)
    print(sep + "\n")


def print_manual_instructions(
    results: List[Dict[str, Any]],
    dest_dir: Path,
) -> None:
    """Print detailed manual retrieval instructions for documents not yet verified."""
    unresolved = [r for r in results if r["status"] != "OK"]
    if not unresolved:
        print("All documents are present and verified matching official SHA-256 hashes.\n")
        return

    print("=" * 80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 80)
    print(
        f"{len(unresolved)} of {len(results)} document(s) require manual download or attention:\n"
    )

    for idx, r in enumerate(unresolved, start=1):
        target_path = Path(r["path"]) if r["path"] else (dest_dir / r["filename"])
        print(f"[{idx}] {r['document_id']} - {r['title']}")
        print(f"    Manufacturer:        {r['manufacturer']}")
        print(f"    Target Filename:     {r['filename']}")
        print(f"    Target Location:     {target_path}")
        print(f"    Current Status:      {r['status']}")
        print(f"    Expected SHA-256:    {r['expected_sha']}")
        if r["actual_sha"]:
            print(f"    Actual SHA-256:      {r['actual_sha']} (MISMATCH)")
        if r["download_url"]:
            print(f"    Vendor Download URL: {r['download_url']}")
        if r["landing_page"]:
            print(f"    Vendor Landing Page: {r['landing_page']}")
        if r["download_url_status"]:
            print(f"    Manifest Status:     {r['download_url_status']}")
        if r["notes"]:
            print(f"    Notes:               {r['notes']}")
        print(
            f"    Action Required:     Download the exact document revision from the manufacturer,\n"
            f"                         place it at '{target_path}',\n"
            f"                         and re-run this script to verify its SHA-256 checksum.\n"
        )
    print("=" * 80 + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch benchmark PDF manuals and verify cryptographic SHA-256 checksums.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Just verify files already in destination directory without downloading anything.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory for raw files (default: {DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to manual_sources.yaml (default: {DEFAULT_MANIFEST}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for fetch_manuals.

    Exit code rules:
        0: ONLY if all 5 benchmark files exist and their SHA-256 hashes match.
        Non-zero (1): If any file is missing or has a hash mismatch.
    """
    args = parse_args(argv)

    print("Industrial Selection Bench - Manual Fetcher & Integrity Verifier")
    print(f"Manifest:    {args.manifest}")
    print(f"Destination: {args.dest}")
    print(f"Check only:  {args.check_only}")

    try:
        results = fetch_and_verify_documents(
            manifest_path=args.manifest,
            dest_dir=args.dest,
            check_only=args.check_only,
        )
    except Exception as exc:
        logger.error("Failed to process manual sources manifest: %s", exc)
        return 1

    print_summary_table(results)
    print_manual_instructions(results, dest_dir=args.dest)

    # Exit code: 0 ONLY if all 5 files exist and their SHA-256 hashes match.
    # Non-zero exit code if any file is missing or has a hash mismatch.
    total_docs = len(results)
    matched_docs = sum(1 for r in results if r["status"] == "OK")

    if total_docs == EXPECTED_DOCUMENT_COUNT and matched_docs == EXPECTED_DOCUMENT_COUNT:
        logger.info(
            "All %d documents successfully verified with matching SHA-256.",
            EXPECTED_DOCUMENT_COUNT,
        )
        return 0

    logger.warning(
        "Verification incomplete: %d/%d documents verified OK.",
        matched_docs,
        EXPECTED_DOCUMENT_COUNT,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
