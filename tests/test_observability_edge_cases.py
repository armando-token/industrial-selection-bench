"""Edge-case tests for observability and cost accounting modules.

Adheres to MEGAPLAN.md §14.1, §14.2, §16.2, and §16.3.
Verifies robust handling of:
- Unknown models, zero tokens, negative tokens, None values, and missing pricing files.
- Nested span hierarchy, monotonic timing, error propagation, empty span names,
  and safe telemetry export formatting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from industrial_lab.observability.costs import (
    AccountingRecord,
    CallCostResult,
    CostCalculator,
    ModelRate,
    calculate_cost,
)
from industrial_lab.observability.spans import (
    CostStatus,
    RateStatus,
    SpanManager,
    SpanRecord,
    StandardSpan,
    TelemetryRecord,
    UsageStatus,
)


# ==============================================================================
# CostCalculator Edge Case Tests
# ==============================================================================


class TestCostCalculatorEdgeCases:
    """Edge case tests for CostCalculator and calculate_cost."""

    @pytest.fixture
    def calculator(self) -> CostCalculator:
        """Create CostCalculator instance with default configuration."""
        return CostCalculator()

    def test_unknown_model_returns_zero_or_fallback(self, calculator: CostCalculator) -> None:
        """Ensure unknown model names return 0.0 or default fallback cost without throwing."""
        # Method on calculator returning 0.0 by default
        cost_default = calculator.calculate_cost(
            "completely_unknown_vendor_model_v999",
            uncached_input_tokens=5000,
            billed_output_tokens=1000,
        )
        assert cost_default == 0.0

        # Method with explicit fallback_cost
        cost_with_fallback = calculator.calculate_cost(
            "unknown-model-xyz",
            100,
            50,
            fallback_cost=0.025,
        )
        assert cost_with_fallback == 0.025

        # calculate_call_cost on unknown model returns CallCostResult without raising
        res = calculator.calculate_call_cost(
            "unknown-model-xyz",
            uncached_input_tokens=100,
            billed_output_tokens=50,
        )
        assert isinstance(res, CallCostResult)
        assert res.model_id == "unknown-model-xyz"
        assert res.cost_status in ("unknown", CostStatus.UNKNOWN.value)
        assert res.cost_usd is None
        assert "not found" in (res.warning or "")

        # Module-level calculate_cost function
        mod_cost = calculate_cost("unregistered_model", 200, 100)
        assert mod_cost == 0.0

    def test_zero_tokens_returns_zero_cost(self, calculator: CostCalculator) -> None:
        """Verify zero tokens return 0.0 without throwing exceptions."""
        # Known model with zero tokens
        cost_known = calculator.calculate_cost("gpt-4o-mini", 0, 0)
        assert cost_known == 0.0

        cost_named = calculator.calculate_cost(
            "gpt-4o-mini",
            uncached_input_tokens=0,
            billed_output_tokens=0,
            cached_input_tokens=0,
        )
        assert cost_named == 0.0

        # Unknown model with zero tokens
        cost_unknown = calculator.calculate_cost("unknown-model", 0, 0)
        assert cost_unknown == 0.0

        # Call cost result with zero tokens is exact $0.00
        call_res = calculator.calculate_call_cost(
            "gpt-4o-mini",
            uncached_input_tokens=0,
            billed_output_tokens=0,
        )
        assert call_res.cost_usd == 0.0
        assert call_res.cost_status in ("exact", CostStatus.CALCULATED.value)

        # Module level function
        assert calculate_cost("gpt-4o-mini", 0, 0) == 0.0

    def test_negative_tokens_handled_safely(self, calculator: CostCalculator) -> None:
        """Verify negative token counts are clamped to zero without negative costs or exceptions."""
        cost_neg = calculator.calculate_cost(
            "gpt-4o-mini",
            uncached_input_tokens=-500,
            billed_output_tokens=-100,
        )
        assert cost_neg == 0.0

        call_res = calculator.calculate_call_cost(
            "gpt-4o-mini",
            uncached_input_tokens=-1000,
            cached_input_tokens=-200,
            billed_output_tokens=-50,
        )
        assert call_res.cost_usd == 0.0
        assert call_res.uncached_input_tokens == 0
        assert call_res.cached_input_tokens == 0
        assert call_res.billed_output_tokens == 0

    def test_missing_pricing_file_uses_fallbacks_and_empty_rates_safe(self, tmp_path: Path) -> None:
        """Verify missing or invalid pricing.yaml falls back gracefully and empty rates do not throw."""
        non_existent = tmp_path / "non_existent_pricing.yaml"
        calc_fallback = CostCalculator(pricing_path=non_existent)

        # Baseline fallbacks should be populated
        assert len(calc_fallback.rates) > 0
        assert "gpt-4o-mini" in calc_fallback.rates
        cost = calc_fallback.calculate_cost("gpt-4o-mini", uncached_input_tokens=1000, billed_output_tokens=500)
        assert cost > 0.0

        # Completely empty rates test
        calc_fallback.rates.clear()
        calc_fallback.aliases.clear()
        assert len(calc_fallback.rates) == 0

        # Should safely return 0.0 (or default fallback cost) with empty rates
        assert calc_fallback.calculate_cost("gpt-4o-mini", 100, 50) == 0.0
        assert calc_fallback.calculate_cost("gpt-4o-mini", 100, 50, fallback_cost=0.01) == 0.01

        # Corrupted YAML file test
        corrupt_yaml = tmp_path / "corrupt_pricing.yaml"
        corrupt_yaml.write_text("invalid: [yaml: {unclosed", encoding="utf-8")
        calc_corrupt = CostCalculator(pricing_path=corrupt_yaml)
        assert len(calc_corrupt.rates) > 0  # Reverted to fallback baseline

    def test_none_values_and_malformed_inputs_handled(self, calculator: CostCalculator) -> None:
        """Verify None values and malformed inputs do not raise unhandled exceptions."""
        # All None arguments
        assert calculator.calculate_cost(None, None, None) == 0.0
        assert calculator.calculate_cost("gpt-4o-mini", None, None) == 0.0
        assert calculate_cost(None) == 0.0

        # calculate_call_cost with None model and tokens
        res_none = calculator.calculate_call_cost(None, uncached_input_tokens=None, billed_output_tokens=None)
        assert res_none.cost_usd is None
        assert res_none.cost_status in ("unknown", CostStatus.UNKNOWN.value)

        # Edge cases on cost_per_correct_task (REPAIR_PLAN §9: None on <= 0, never 0)
        assert calculator.calculate_cost_per_correct_task(None, 10) is None
        assert calculator.calculate_cost_per_correct_task(10.0, 0) is None
        assert calculator.calculate_cost_per_correct_task(10.0, -3) is None

        # Edge cases on break_even
        # delta_online <= 0
        assert calculator.calculate_break_even(100.0, 50.0, 1.0, 1.0) is None
        assert calculator.calculate_break_even(100.0, 50.0, 2.0, 1.0) is None
        # delta_prep <= 0 (A already cheaper or equal in prep)
        assert calculator.calculate_break_even(50.0, 100.0, 0.5, 1.0) == 0.0


# ==============================================================================
# SpanManager Edge Case Tests
# ==============================================================================


class TestSpanManagerEdgeCases:
    """Edge case tests for SpanManager error handling, nesting, and export formatting."""

    def test_error_propagation_and_recording(self) -> None:
        """Verify errors inside spans are recorded and propagated without suppression."""
        mgr = SpanManager(case_id="edge_err_case")

        # Uncaught exception bubbles up
        with pytest.raises(RuntimeError, match="unhandled catastrophic error"):
            with mgr.span("outer_operation") as outer:
                outer.set_tag("step", 1)
                with mgr.span("inner_operation") as inner:
                    inner.set_tag("step", 2)
                    raise RuntimeError("unhandled catastrophic error")

        # Active stack is clean (not left corrupted)
        assert mgr.active_span is None

        # Both outer and inner spans are recorded and marked as error
        spans = mgr.spans
        assert len(spans) == 2

        inner_span = mgr.find_spans("inner_operation")[0]
        outer_span = mgr.find_spans("outer_operation")[0]

        assert inner_span.status == "error"
        assert "unhandled catastrophic error" in (inner_span.error or "")
        assert inner_span.duration_ms is not None and inner_span.duration_ms >= 0

        assert outer_span.status == "error"
        assert "unhandled catastrophic error" in (outer_span.error or "")
        assert outer_span.duration_ms is not None and outer_span.duration_ms >= 0

    def test_caught_exception_in_child_does_not_poison_parent(self) -> None:
        """Verify that an exception caught inside a parent span records child error but allows parent success."""
        mgr = SpanManager()

        with mgr.span("parent_job") as parent:
            try:
                with mgr.span("faulty_child") as child:
                    child.set_tag("attempt", 1)
                    raise TimeoutError("connection timed out")
            except TimeoutError:
                parent.set_tag("recovered", True)

        assert mgr.active_span is None
        parent_span = mgr.find_spans("parent_job")[0]
        child_span = mgr.find_spans("faulty_child")[0]

        assert child_span.status == "error"
        assert "connection timed out" in (child_span.error or "")

        assert parent_span.status == "ok"
        assert parent_span.tags["recovered"] is True

    def test_nested_spans_hierarchy_and_timing(self) -> None:
        """Verify nested spans maintain correct parent IDs, containment timing, and critical path."""
        mgr = SpanManager(engine="test_engine", case_id="nested_timing")

        with mgr.span(StandardSpan.REQUEST_TOTAL.value) as root:
            time.sleep(0.01)

            with mgr.span(StandardSpan.TECHNICAL_RETRIEVAL.value) as child1:
                time.sleep(0.01)

                with mgr.span(StandardSpan.PDF_PARSE.value) as grandchild:
                    grandchild.record_bytes(4096)
                    time.sleep(0.01)

            with mgr.span(StandardSpan.MODEL_REQUEST.value) as child2:
                child2.record_usage(input_tokens=100, output_tokens=50)
                time.sleep(0.01)

        spans = mgr.spans
        assert len(spans) == 4

        root_s = mgr.find_spans(StandardSpan.REQUEST_TOTAL.value)[0]
        child1_s = mgr.find_spans(StandardSpan.TECHNICAL_RETRIEVAL.value)[0]
        grandchild_s = mgr.find_spans(StandardSpan.PDF_PARSE.value)[0]
        child2_s = mgr.find_spans(StandardSpan.MODEL_REQUEST.value)[0]

        # Hierarchy validation
        assert root_s.parent_id is None
        assert child1_s.parent_id == root_s.span_id
        assert grandchild_s.parent_id == child1_s.span_id
        assert child2_s.parent_id == root_s.span_id

        # Timing containment
        assert root_s.start_ns <= child1_s.start_ns <= grandchild_s.start_ns
        assert grandchild_s.end_ns is not None and child1_s.end_ns is not None
        assert grandchild_s.end_ns <= child1_s.end_ns
        assert child1_s.end_ns <= (root_s.end_ns or float("inf"))

        # Durations are positive and monotonic
        assert root_s.duration_ms is not None and root_s.duration_ms > 0
        assert child1_s.duration_ms is not None and child1_s.duration_ms > 0
        assert grandchild_s.duration_ms is not None and grandchild_s.duration_ms > 0
        assert child2_s.duration_ms is not None and child2_s.duration_ms > 0

        assert root_s.duration_ms >= child1_s.duration_ms
        assert child1_s.duration_ms >= grandchild_s.duration_ms

        # Critical path calculation
        crit_path = mgr.critical_path_ms()
        assert crit_path > 0
        assert crit_path >= root_s.duration_ms

    def test_empty_or_none_span_names_handled_safely(self) -> None:
        """Verify empty, whitespace, and None span names are replaced with 'unnamed_span' safely."""
        mgr = SpanManager()

        with mgr.span("") as s1:
            s1.set_tag("k", "v1")

        with mgr.span(None) as s2:  # type: ignore[arg-type]
            s2.set_tag("k", "v2")

        with mgr.span("   ") as s3:
            s3.set_tag("k", "v3")

        spans = mgr.spans
        assert len(spans) == 3
        for s in spans:
            assert s.name == "unnamed_span"
            assert s.duration_ms is not None and s.duration_ms >= 0

    def test_telemetry_export_formatting_and_missing_fields(self, tmp_path: Path) -> None:
        """Verify telemetry record export handles missing fields, json/jsonl writing, and custom metadata."""
        mgr = SpanManager()

        # Build telemetry record with no spans, missing fields, None parameters
        record = mgr.build_telemetry_record(
            cost_usd=None,
            cost_status="unknown",
            failure_code=None,
            attempts=1,
            metadata={"arbitrary_key": "val", "non_serializable": set([1, 2, 3])},
        )

        assert isinstance(record, TelemetryRecord)
        assert record.case_id == "unassigned"
        assert record.engine == "unknown"
        assert record.input_tokens is None
        assert record.output_tokens is None
        assert record.spans == []

        # Export formatting to JSON string handles non-serializable sets via fallback
        json_str = record.to_json(indent=2)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["case_id"] == "unassigned"
        assert "arbitrary_key" in parsed["metadata"]

        # Direct manager export to JSON file
        json_file = tmp_path / "telemetry_out.json"
        res_rec = mgr.export_telemetry(file_path=json_file)
        assert json_file.exists()
        assert isinstance(res_rec, TelemetryRecord)

        # Direct manager export to JSONL file
        jsonl_file = tmp_path / "telemetry_out.jsonl"
        mgr.export_telemetry(file_path=jsonl_file)
        mgr.export_telemetry(file_path=jsonl_file)
        lines = jsonl_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            line_data = json.loads(line)
            assert "run_id" in line_data
            assert "request_id" in line_data

        # Export as json string
        as_json_str = mgr.export_telemetry(as_json=True)
        assert isinstance(as_json_str, str)
        assert "run_id" in json.loads(as_json_str)

    def test_unclosed_active_span_auto_closed_on_export(self) -> None:
        """Verify an active span that was not exited manually is closed gracefully during export."""
        mgr = SpanManager()
        span_ctx = mgr.span("dangling_span")
        span_ctx.__enter__()

        # active span is still open
        assert mgr.active_span is not None

        # Exporting telemetry auto-closes active spans without crashing
        rec = mgr.build_telemetry_record()
        assert mgr.active_span is None
        assert len(rec.spans) == 1
        assert rec.spans[0].name == "dangling_span"
        assert rec.spans[0].end_ns is not None
        assert rec.spans[0].duration_ms is not None

    def test_record_usage_and_bytes_sanitization(self) -> None:
        """Verify record_usage and record_bytes safely ignore bad types or negative numbers."""
        mgr = SpanManager()

        with mgr.span("metric_test") as s:
            # Negative numbers clamped to 0
            s.record_usage(input_tokens=-100, output_tokens=-50, cached_tokens=-20)
            s.record_bytes(-1024)

            # Valid numbers incremented
            s.record_usage(input_tokens=200, output_tokens=80, cached_tokens=40)
            s.record_bytes(2048)

            # Invalid string/types don't crash
            s.record_usage(input_tokens="invalid", output_tokens=None)  # type: ignore[arg-type]
            s.record_bytes("not_an_int")  # type: ignore[arg-type]

        span = mgr.spans[0]
        assert span.input_tokens == 200
        assert span.output_tokens == 80
        assert span.cached_tokens == 40
        assert span.bytes_transferred == 2048


# ==============================================================================
# REPAIR_PLAN Section 9 and 10 Edge Cases
# ==============================================================================


class TestObservabilitySection9And10EdgeCases:
    """Edge cases specifically targeting Section 9 and 10 requirements."""

    def test_missing_usage_with_zero_fallback_never_zero_dollars(self) -> None:
        """Section 9: If provider usage is missing, cost must NEVER be $0.00 even if fallback_cost=0.0."""
        calc = CostCalculator()
        res = calc.calculate_call_cost("typesafe/jev", uncached_input_tokens=None, billed_output_tokens=None, fallback_cost=0.0)
        assert res.cost_usd is None
        assert res.cost_status in ("unknown", CostStatus.UNKNOWN.value)

    def test_missing_usage_with_positive_fallback_is_estimated(self) -> None:
        """Section 9: Explicit positive fallback marks cost_status as 'estimated'."""
        calc = CostCalculator()
        res = calc.calculate_call_cost("typesafe/jev", uncached_input_tokens=None, billed_output_tokens=None, fallback_cost=0.05)
        assert res.cost_usd == 0.05
        assert res.cost_status in ("estimated", CostStatus.ESTIMATED.value)

    def test_unverified_rate_always_produces_estimated_status(self) -> None:
        """Section 9: Unverified rate (verified: false) produces cost_status='estimated', never 'exact'."""
        calc = CostCalculator()
        calc.register_rate(
            ModelRate(
                model_id="typesafe/jev",
                provider="typesafe",
                rate_input=1.0,
                rate_cached=0.25,
                verified=False,
                rate_status=RateStatus.ASSUMED.value,
                notes="synthetic test rate",
            )
        )
        res_jev = calc.calculate_call_cost("typesafe/jev", uncached_input_tokens=1000, billed_output_tokens=0)
        assert res_jev.cost_status in ("estimated", CostStatus.ESTIMATED.value)
        assert res_jev.cost_usd is not None and res_jev.cost_usd > 0

        res_gemma = calc.calculate_call_cost("google.gemma-4-31b", uncached_input_tokens=1000, billed_output_tokens=200)
        assert res_gemma.cost_status in ("estimated", CostStatus.ESTIMATED.value)
        assert res_gemma.cost_usd is not None and res_gemma.cost_usd > 0

    def test_scenario_lookup_unknown_scenario_falls_back(self) -> None:
        """Pricing scenario with unknown scenario name falls back to model_id rate safely."""
        calc = CostCalculator()
        res = calc.calculate_call_cost("gpt-4o-mini", uncached_input_tokens=1000, billed_output_tokens=100, scenario="non_existent_scenario")
        assert res.cost_status in ("exact", CostStatus.CALCULATED.value)
        assert res.cost_usd is not None

    def test_accounting_with_historical_422_and_errors(self) -> None:
        """Section 9: Accounting design captures 422 errors and retries accurately."""
        calc = CostCalculator()

        # Telemetry record with HTTP 422 failure
        mgr = SpanManager(request_id="hist-422-01", case_id="Q1", engine="structured_jev")
        with mgr.span(StandardSpan.REQUEST_TOTAL.value):
            with mgr.span(StandardSpan.MODEL_REQUEST.value) as s:
                s.set_tag("http_status", 422)
                s.set_error("JEV 422 Unprocessable Entity")
        rec_failed = mgr.build_telemetry_record(attempts=1)
        assert rec_failed.http_statuses == [422]
        assert rec_failed.failure_code == "JEV 422 Unprocessable Entity"
        assert rec_failed.execution_status == "PROVIDER_ERROR"

        # Accounting with 5 historical 422 calls
        acc = calc.compute_accounting([rec_failed], include_historical_422=True, historical_422_count=5)
        assert acc.logical_request_count == 1
        assert acc.provider_call_count == 6  # 1 in record + 5 historical
        assert acc.status_422_count == 6      # 1 in record + 5 historical
        assert acc.error_call_count == 6
        assert acc.historical_422_calls == 5
        assert len(acc.notes) > 0

    def test_stage_spans_preserve_timeouts_and_budgets(self) -> None:
        """Section 10: Preserving timeout and budget_exceeded stage statuses."""
        mgr = SpanManager(request_id="stage_budget_01", case_id="Q4", engine="scrape_llm")
        with mgr.span(StandardSpan.REQUEST_TOTAL.value):
            with mgr.span(StandardSpan.TOOL_EXECUTION.value) as s:
                s.set_status("budget_exceeded")
                s.set_tag("tool_rounds", 8)

        rec = mgr.build_telemetry_record()
        assert rec.execution_status == "BUDGET_EXCEEDED"
        assert rec.tool_calls == 1
        assert rec.provider_calls == 0
