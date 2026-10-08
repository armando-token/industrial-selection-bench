"""Global pytest configuration for industrial-selection-lab.

Ensures tests run strictly offline (LAB_DRY_RUN=1) by default unless explicitly
overridden, preventing accidental provider calls during local test execution.
"""

from __future__ import annotations

import os

from pathlib import Path
import pytest

# Set LAB_DRY_RUN=1 by default to ensure local tests never touch live external APIs
os.environ.setdefault("LAB_DRY_RUN", "1")


def _has_non_empty_file(path_str: str) -> bool:
    p = Path(path_str)
    return p.exists() and p.is_file() and p.stat().st_size > 0


requires_real_manuals = pytest.mark.skipif(
    not _has_non_empty_file("data/pages/pages.jsonl"),
    reason="needs locally rebuilt manual data; run scripts/fetch_manuals.py + ingest (manuals are not redistributed)",
)

requires_real_facts = pytest.mark.skipif(
    not _has_non_empty_file("data/facts/facts.reviewed.jsonl"),
    reason="needs locally extracted facts; run ingest + review-import",
)

requires_real_pdfs = pytest.mark.skipif(
    not (
        Path("data/raw/horner_x4_user_manual_MAN1137_HE-X4A_HE-X4R.pdf").exists()
        and Path("data/raw/horner_x4_user_manual_MAN1137_HE-X4A_HE-X4R.pdf").stat().st_size > 1000
    ),
    reason="needs downloaded manufacturer PDFs; run scripts/fetch_manuals.py",
)

requires_runs = pytest.mark.skipif(
    not Path("runs/latest_report").exists(),
    reason="needs local run artifacts; run benchmark first",
)
