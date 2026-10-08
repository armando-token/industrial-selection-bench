"""Tests for report generation edge cases adhering to MEGAPLAN.md §25.

Covers:
- Wilson score confidence interval boundary conditions (0/0, 0/10, 10/10, negative).
- calculate_percentile edge cases (empty list, None/NaN, 1 element, 2 elements, clamp).
- Empty run directories and runs with 0 responses.
- Runs where all requests failed with provider_error or timeout.
- HTML and Markdown rendering resilience when cost or latency metrics are zero or None.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from industrial_lab.observability.spans import TelemetryRecord
from industrial_lab.report.build import (
    EngineMetrics,
    ReportBuilder,
    ReportData,
    calculate_percentile,
    generate_reports,
    wilson_score_interval,
)


# ==============================================================================
# 1. Wilson Score Interval Edge Cases
# ==============================================================================

def test_wilson_score_interval_zero_total() -> None:
    """Test 0/0 and non-positive totals return (0.0, 0.0)."""
    assert wilson_score_interval(0, 0) == (0.0, 0.0)
    assert wilson_score_interval(5, 0) == (0.0, 0.0)
    assert wilson_score_interval(0, -10) == (0.0, 0.0)
    assert wilson_score_interval(-2, -5) == (0.0, 0.0)


def test_wilson_score_interval_zero_successes() -> None:
    """Test 0 successes out of 10 cases (0/10)."""
    low, high = wilson_score_interval(0, 10)
    assert low == 0.0
    # For 0/10 at z=1.96, Wilson upper bound is ~27.8%
    assert 20.0 < high < 35.0


def test_wilson_score_interval_all_successes() -> None:
    """Test 10 successes out of 10 cases (10/10)."""
    low, high = wilson_score_interval(10, 10)
    # For 10/10 at z=1.96, Wilson lower bound is ~72.2%
    assert 65.0 < low < 80.0
    assert high == 100.0


def test_wilson_score_interval_clamping() -> None:
    """Test negative successes or successes exceeding total are clamped gracefully."""
    # Negative successes clamped to 0
    assert wilson_score_interval(-5, 10) == wilson_score_interval(0, 10)
    # Successes > total clamped to total
    assert wilson_score_interval(15, 10) == wilson_score_interval(10, 10)


# ==============================================================================
# 2. calculate_percentile Edge Cases
# ==============================================================================

def test_calculate_percentile_empty_list() -> None:
    """Test empty list returns 0.0."""
    assert calculate_percentile([], 50.0) == 0.0
    assert calculate_percentile([], 95.0) == 0.0
    assert calculate_percentile([], 0.0) == 0.0


def test_calculate_percentile_none_and_nan() -> None:
    """Test lists containing None or NaN values are filtered safely."""
    assert calculate_percentile([None, float("nan")], 50.0) == 0.0
    assert calculate_percentile([100.0, None, 200.0], 50.0) == 150.0


def test_calculate_percentile_single_element() -> None:
    """Test single element returns exact value across all percentiles."""
    val = 42.5
    assert calculate_percentile([val], 0.0) == val
    assert calculate_percentile([val], 50.0) == val
    assert calculate_percentile([val], 95.0) == val
    assert calculate_percentile([val], 100.0) == val


def test_calculate_percentile_two_elements() -> None:
    """Test two elements interpolate correctly."""
    vals = [10.0, 20.0]
    assert calculate_percentile(vals, 0.0) == 10.0
    assert calculate_percentile(vals, 50.0) == 15.0
    assert calculate_percentile(vals, 100.0) == 20.0
    # 95th percentile: 10 + 0.95 * 10 = 19.5
    assert calculate_percentile(vals, 95.0) == pytest.approx(19.5)


def test_calculate_percentile_clamping() -> None:
    """Test percentiles outside [0, 100] are clamped without IndexError."""
    vals = [10.0, 20.0, 30.0]
    assert calculate_percentile(vals, -20.0) == 10.0
    assert calculate_percentile(vals, 150.0) == 30.0


# ==============================================================================
# 3. Generating Reports on Mock Run Directory with 0 Responses
# ==============================================================================

def test_generate_reports_empty_directory(tmp_path: Path) -> None:
    """Test generating reports on a completely empty run directory."""
    empty_run_dir = tmp_path / "empty_run"
    empty_run_dir.mkdir()
    out_dir = tmp_path / "empty_out"

    md_path, html_path = generate_reports(run_dir=empty_run_dir, output_dir=out_dir)

    assert md_path.exists()
    assert html_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")

    assert "# Industrial Selection Lab — Benchmark Report" in md_content
    assert "<!DOCTYPE html>" in html_content
    assert "empty_run" in md_content
    assert "empty_run" in html_content


def test_generate_reports_zero_responses_empty_telemetry(tmp_path: Path) -> None:
    """Test generating reports when telemetry.jsonl exists but has 0 lines."""
    run_dir = tmp_path / "run_zero_telemetry"
    run_dir.mkdir()
    (run_dir / "telemetry.jsonl").write_text("", encoding="utf-8")
    out_dir = tmp_path / "out_zero_telemetry"

    md_path, html_path = generate_reports(run_dir=run_dir, output_dir=out_dir)

    assert md_path.exists()
    assert html_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")

    assert "0/0 (0.0%)" in md_content
    assert "0/0" in html_content


def test_generate_reports_with_manifest_only(tmp_path: Path) -> None:
    """Test generating reports when only run_manifest.json exists with 0 responses."""
    run_dir = tmp_path / "run_manifest_only"
    run_dir.mkdir()
    manifest = {
        "run_id": "run_manifest_only",
        "engines": ["structured_jev", "rag_llm"],
        "status": "aborted",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out_dir = tmp_path / "out_manifest_only"

    md_path, html_path = generate_reports(run_dir=run_dir, output_dir=out_dir)

    assert md_path.exists()
    assert html_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "structured_jev" in md_content
    assert "rag_llm" in md_content


# ==============================================================================
# 4. Generating Reports on Run Directory where All Requests Failed
# ==============================================================================

def test_generate_reports_all_requests_failed(tmp_path: Path) -> None:
    """Test generating reports where all requests failed with provider_error or timeout."""
    run_dir = tmp_path / "run_all_errors"
    run_dir.mkdir()

    records = [
        # Request 1: structured_jev failed with provider_error, cost is None
        TelemetryRecord(
            run_id="run_all_errors",
            request_id="req_001",
            case_id="case_T001",
            engine="structured_jev",
            repeat=1,
            start_utc="2026-10-02T12:00:00Z",
            elapsed_ms=15.0,
            failure_code="provider_error",
            cost_status="unknown",
            cost_usd=None,
        ),
        # Request 2: structured_jev timed out, cost is 0.0
        TelemetryRecord(
            run_id="run_all_errors",
            request_id="req_002",
            case_id="case_T002",
            engine="structured_jev",
            repeat=1,
            start_utc="2026-10-02T12:00:01Z",
            elapsed_ms=30000.0,
            failure_code="timeout",
            cost_status="exact",
            cost_usd=0.0,
        ),
        # Request 3: rag_llm failed with provider_error, cost was billed
        TelemetryRecord(
            run_id="run_all_errors",
            request_id="req_003",
            case_id="case_T001",
            engine="rag_llm",
            repeat=1,
            start_utc="2026-10-02T12:00:02Z",
            elapsed_ms=120.0,
            failure_code="provider_error",
            cost_status="exact",
            cost_usd=0.0025,
        ),
        # Request 4: scrape_llm timed out, cost is None
        TelemetryRecord(
            run_id="run_all_errors",
            request_id="req_004",
            case_id="case_T001",
            engine="scrape_llm",
            repeat=1,
            start_utc="2026-10-02T12:00:03Z",
            elapsed_ms=60000.0,
            failure_code="timeout",
            cost_status="unknown",
            cost_usd=None,
        ),
    ]

    tel_file = run_dir / "telemetry.jsonl"
    with tel_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")

    # Mock scoring where 0 tasks succeeded
    scoring = {
        "engines": {
            "structured_jev": {
                "total_cases": 2,
                "successful_tasks": 0,
                "false_approvals": 0,
                "incompatible_cases": 1,
                "excessive_abstentions": 0,
                "resolvable_cases": 1,
                "preparation_compute_cost_usd": 0.0,
            },
            "rag_llm": {
                "total_cases": 1,
                "successful_tasks": 0,
                "false_approvals": 0,
                "incompatible_cases": 1,
                "excessive_abstentions": 0,
                "resolvable_cases": 0,
                "preparation_compute_cost_usd": 0.0,
            },
            "scrape_llm": {
                "total_cases": 1,
                "successful_tasks": 0,
                "false_approvals": 0,
                "incompatible_cases": 1,
                "excessive_abstentions": 0,
                "resolvable_cases": 0,
                "preparation_compute_cost_usd": 0.0,
            },
        }
    }
    (run_dir / "scoring.json").write_text(json.dumps(scoring), encoding="utf-8")

    out_dir = tmp_path / "out_all_errors"
    md_path, html_path = generate_reports(run_dir=run_dir, output_dir=out_dir)

    assert md_path.exists()
    assert html_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    html_content = html_path.read_text(encoding="utf-8")

    # Success rate must be 0.0%
    assert "0/2 (0.0%)" in md_content
    assert "0/1 (0.0%)" in md_content
    # USD/tarea correcta should handle 0 successes (infinity or Unknown)
    assert "∞ (0 éxitos)" in md_content or "Unknown" in md_content
    assert "∞ (0 éxitos)" in html_content or "Unknown" in html_content
    # HTML must render without crashing on None or 0 cost
    assert "<!DOCTYPE html>" in html_content


def test_generate_reports_from_normalized_responses_only(tmp_path: Path) -> None:
    """Test report generation reconstructing telemetry from responses_normalized.jsonl."""
    run_dir = tmp_path / "run_norm_only"
    run_dir.mkdir()

    norm_entries = [
        {
            "run_id": "run_norm_only",
            "request_id": "req_1",
            "engine": "structured_jev",
            "case_id": "T001",
            "repeat": 1,
            "elapsed_ms": 25.4,
            "response": {"execution_status": "provider_error"},
            "created_at_utc": "2026-10-02T12:00:00Z",
        },
        {
            "run_id": "run_norm_only",
            "request_id": "req_2",
            "engine": "structured_jev",
            "case_id": "T002",
            "repeat": 1,
            "elapsed_ms": 10.2,
            "response": {"execution_status": "completed"},
            "created_at_utc": "2026-10-02T12:00:01Z",
        },
    ]

    norm_file = run_dir / "responses_normalized.jsonl"
    with norm_file.open("w", encoding="utf-8") as f:
        for entry in norm_entries:
            f.write(json.dumps(entry) + "\n")

    out_dir = tmp_path / "out_norm_only"
    md_p, html_p = generate_reports(run_dir=run_dir, output_dir=out_dir)

    assert md_p.exists()
    assert html_p.exists()
    assert "structured_jev" in md_p.read_text(encoding="utf-8")


# ==============================================================================
# 5. Direct Rendering Resilience (HTML & Markdown with Zero/None Metrics)
# ==============================================================================

def test_html_rendering_none_and_zero_metrics() -> None:
    """Ensure HTML report rendering does not crash when cost or latencies are zero or None."""
    data = ReportData(run_id="run_boundary_rendering")

    # Engine with None total cost and empty latencies
    data.engines["structured_jev"] = EngineMetrics(
        engine_id="structured_jev",
        display_name="A: structured_jev",
        total_cases=0,
        total_requests=0,
        successful_tasks=0,
        latencies_ms=[],
        total_cost_usd=None,
        cost_status="unknown",
        preparation_compute_cost_usd=0.0,
        preparation_human_review_min=0.0,
    )

    # Engine with 0.0 total cost and 0.0 latencies
    data.engines["rag_llm"] = EngineMetrics(
        engine_id="rag_llm",
        display_name="B: rag_llm",
        total_cases=5,
        total_requests=5,
        successful_tasks=0,
        latencies_ms=[0.0, 0.0],
        total_cost_usd=0.0,
        cost_status="exact",
        preparation_compute_cost_usd=0.0,
        preparation_human_review_min=0.0,
    )

    # Ablation with None cost
    data.ablations["structured_llm"] = EngineMetrics(
        engine_id="structured_llm",
        display_name="Ablation: structured_llm",
        total_cases=1,
        total_requests=1,
        successful_tasks=0,
        latencies_ms=[],
        total_cost_usd=None,
        cost_status="unknown",
    )

    builder = ReportBuilder(data)

    # Neither call should raise TypeError or ValueError
    html = builder.generate_html()
    md = builder.generate_markdown()

    assert "<!DOCTYPE html>" in html
    assert "Unknown" in html
    assert "$0.000" in html
    assert "# Industrial Selection Lab" in md


def test_html_rendering_completely_empty_engines() -> None:
    """Ensure HTML rendering does not crash when engines dict is completely empty."""
    data = ReportData(run_id="run_no_engines", engines={}, ablations={})
    builder = ReportBuilder(data)

    html = builder.generate_html()
    md = builder.generate_markdown()

    assert "<!DOCTYPE html>" in html
    assert "Sin datos de ejecución registrados" in html
    assert "Sin ablaciones metodológicas registradas" in html
    assert "# Industrial Selection Lab" in md
