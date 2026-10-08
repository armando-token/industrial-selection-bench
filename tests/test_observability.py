"""Tests for observability and cost accounting modules.

Adheres to MEGAPLAN.md §14.1, §14.2, §16.2, and §16.3.
"""

import asyncio
import time
from pathlib import Path
import pytest

from industrial_lab.observability.costs import (
    AccountingRecord,
    AggregatedCostResult,
    CallCostResult,
    CostCalculator,
    ModelRate,
)
from industrial_lab.observability.spans import (
    CostStatus,
    RateStatus,
    STANDARD_SPANS,
    SpanManager,
    SpanRecord,
    StandardSpan,
    TelemetryRecord,
    UsageStatus,
)


def test_standard_spans_coverage() -> None:
    """Verify that all standard spans from MEGAPLAN.md §14.1 are present."""
    expected = [
        "request_total",
        "query_interpretation",
        "catalog_load",
        "technical_retrieval",
        "html_fetch",
        "pdf_download",
        "pdf_parse",
        "embedding_query",
        "model_request",
        "rule_evaluation",
        "evidence_resolution",
        "commerce_read",
        "quote_generation",
        "render_response",
    ]
    for s in expected:
        assert s in STANDARD_SPANS
        assert StandardSpan(s) is not None


def test_hierarchical_spans_and_monotonic_durations() -> None:
    """Test span nesting, parent_id assignment, and monotonic timing."""
    mgr = SpanManager(
        run_id="run_test",
        request_id="req_test",
        case_id="case_01",
        engine="structured_jev",
    )

    with mgr.span(StandardSpan.REQUEST_TOTAL.value) as root_ctx:
        root_ctx.set_tag("client", "test_runner")
        time.sleep(0.01)

        with mgr.span(StandardSpan.QUERY_INTERPRETATION.value) as child1:
            child1.set_tag("method", "regex_rules")
            time.sleep(0.005)

        with mgr.span(StandardSpan.MODEL_REQUEST.value) as child2:
            child2.record_usage(input_tokens=150, output_tokens=30, cached_tokens=50)
            child2.record_bytes(1024)
            child2.set_tag("model", "typesafe/jev")
            time.sleep(0.005)

    spans = mgr.spans
    assert len(spans) == 3

    root_span = mgr.find_spans(StandardSpan.REQUEST_TOTAL.value)[0]
    child1_span = mgr.find_spans(StandardSpan.QUERY_INTERPRETATION.value)[0]
    child2_span = mgr.find_spans(StandardSpan.MODEL_REQUEST.value)[0]

    assert root_span.parent_id is None
    assert child1_span.parent_id == root_span.span_id
    assert child2_span.parent_id == root_span.span_id

    # Durations must be monotonic and positive
    assert root_span.duration_ms is not None and root_span.duration_ms > 0
    assert child1_span.duration_ms is not None and child1_span.duration_ms > 0
    assert child2_span.duration_ms is not None and child2_span.duration_ms > 0
    assert root_span.duration_ms >= (child1_span.duration_ms + child2_span.duration_ms) * 0.8

    # ISO 8601 UTC checks
    assert "T" in root_span.start_utc and "+" in root_span.start_utc or "Z" in root_span.start_utc
    assert root_span.end_utc is not None


@pytest.mark.asyncio
async def test_async_span_context() -> None:
    """Verify async context manager protocol on spans."""
    mgr = SpanManager(engine="rag_llm")

    async with mgr.span(StandardSpan.REQUEST_TOTAL.value):
        async with mgr.span(StandardSpan.TECHNICAL_RETRIEVAL.value) as s:
            await asyncio.sleep(0.01)
            s.set_tag("hits", 5)

    spans = mgr.spans
    assert len(spans) == 2
    retrieval_span = mgr.find_spans(StandardSpan.TECHNICAL_RETRIEVAL.value)[0]
    assert retrieval_span.tags["hits"] == 5
    assert retrieval_span.duration_ms is not None and retrieval_span.duration_ms >= 5.0


def test_span_error_handling() -> None:
    """Verify exceptions inside span are recorded without suppression."""
    mgr = SpanManager()

    with pytest.raises(ValueError, match="simulated failure"):
        with mgr.span("pdf_download") as s:
            s.set_tag("url", "https://example.com/manual.pdf")
            raise ValueError("simulated failure")

    span = mgr.find_spans("pdf_download")[0]
    assert span.status == "error"
    assert "simulated failure" in (span.error or "")


def test_telemetry_record_construction(tmp_path: Path) -> None:
    """Test TelemetryRecord matching §14.2 fields and serialization."""
    mgr = SpanManager(
        run_id="run_100",
        request_id="req_200",
        case_id="case_sensor_01",
        engine="structured_jev",
        repeat=2,
    )

    with mgr.span("request_total"):
        with mgr.span("model_request") as s:
            s.record_usage(input_tokens=800, output_tokens=0, cached_tokens=200)
            s.set_tag("model", "typesafe/jev")
            s.set_tag("http_status", 200)

    record = mgr.build_telemetry_record(
        cost_usd=0.00425,
        cost_status="exact",
        scenario_id="scenario_dev_01",
        metadata={"reviewer": "automated"},
    )

    assert record.run_id == "run_100"
    assert record.request_id == "req_200"
    assert record.case_id == "case_sensor_01"
    assert record.engine == "structured_jev"
    assert record.repeat == 2
    assert record.input_tokens == 800
    assert record.output_tokens == 0
    assert record.cached_tokens == 200
    assert record.cost_usd == 0.00425
    assert record.cost_status == CostStatus.CALCULATED.value
    assert record.usage_status == UsageStatus.PROVIDER_REPORTED.value
    assert record.http_statuses == [200]
    assert record.model_ids == ["typesafe/jev"]

    # Serialization test
    json_path = tmp_path / "telemetry.json"
    record.save_json(json_path)
    assert json_path.exists()

    jsonl_path = tmp_path / "telemetry.jsonl"
    record.append_to_jsonl(jsonl_path)
    record.append_to_jsonl(jsonl_path)
    assert len(jsonl_path.read_text().strip().splitlines()) == 2


@pytest.fixture
def synthetic_pricing_yaml(tmp_path: Path) -> Path:
    pricing_file = tmp_path / "synthetic_pricing.yaml"
    pricing_file.write_text(
        """
scenarios:
  lab_card_rate:
    model_id: lab_card_rate
    provider: typesafe
    rate_input: 1.0
    rate_cached: 0.25
    rate_output: 0.0
    rate_other: 0.0
    verified: false
    rate_status: ASSUMED
    source_url: "https://example.com/synthetic-pricing"
    notes: "synthetic test rate - unverified lab card rate"
  published_typesafe_jev:
    model_id: published_typesafe_jev
    provider: typesafe
    rate_input: 1.0
    rate_cached: 0.0
    rate_output: 0.0
    rate_other: 0.0
    verified: false
    rate_status: PUBLIC_RATE
    source_url: "https://example.com/synthetic-pricing"
    notes: "synthetic test rate - published blog rate"
providers:
  typesafe:
    models:
      systemone-preview-v1:
        rate_input: 1.0
        rate_cached: 0.25
        rate_output: 0.0
        rate_other: 0.0
        verified: false
        rate_status: ASSUMED
        source_url: "https://example.com/synthetic-pricing"
        notes: "synthetic test rate"
      typesafe/jev:
        rate_input: 1.0
        rate_cached: 0.25
        rate_output: 0.0
        rate_other: 0.0
        verified: false
        rate_status: ASSUMED
        source_url: "https://example.com/synthetic-pricing"
        notes: "synthetic test rate"
      jev-preview:
        rate_input: 1.0
        rate_cached: 0.25
        rate_output: 0.0
        rate_other: 0.0
        verified: false
        rate_status: ASSUMED
        source_url: "https://example.com/synthetic-pricing"
        notes: "synthetic test rate"
      jev-1.13.0:
        rate_input: 1.0
        rate_cached: 0.25
        rate_output: 0.0
        rate_other: 0.0
        verified: false
        rate_status: ASSUMED
        source_url: "https://example.com/synthetic-pricing"
        notes: "synthetic test rate"
  openai:
    models:
      gpt-4o-mini:
        rate_input: 0.150
        rate_cached: 0.075
        rate_output: 0.600
        rate_other: 0.0
        verified: true
        rate_status: ACCOUNT_CONFIRMED
        source_url: "https://openai.com/api/pricing/"
  anthropic:
    models:
      claude-3-5-sonnet:
        rate_input: 3.00
        rate_cached: 0.30
        rate_output: 15.00
        rate_other: 0.0
        verified: true
        rate_status: ACCOUNT_CONFIRMED
        source_url: "https://www.anthropic.com/pricing"
  google:
    models:
      gemini-1.5-flash:
        rate_input: 0.075
        rate_cached: 0.01875
        rate_output: 0.300
        rate_other: 0.0
        verified: true
        rate_status: ACCOUNT_CONFIRMED
        source_url: "https://ai.google.dev/pricing"
  bedrock-mantle:
    models:
      google.gemma-4-31b:
        rate_input: 0.14
        rate_cached: 0.035
        rate_output: 0.40
        rate_other: 0.0
        verified: false
        rate_status: ASSUMED
        source_url: "https://aws.amazon.com/bedrock/pricing/"
        notes: "synthetic test rate"
""",
        encoding="utf-8",
    )
    return pricing_file


def test_cost_calculator_pricing_yaml_loading(synthetic_pricing_yaml: Path) -> None:
    """Test loading pricing rates from synthetic pricing yaml."""
    calc = CostCalculator(synthetic_pricing_yaml)
    assert "systemone-preview-v1" in calc.rates
    assert "typesafe/jev" in calc.rates
    assert "gpt-4o-mini" in calc.rates
    assert "claude-3-5-sonnet" in calc.rates
    assert "gemini-1.5-flash" in calc.rates

    jev_rate = calc.get_rate("typesafe/jev")
    assert jev_rate is not None
    assert jev_rate.rate_input == 1.0
    assert jev_rate.rate_cached == 0.25
    assert jev_rate.rate_output == 0.0


def test_typesafe_fallback_rates_unknown(tmp_path: Path) -> None:
    """Verify fallback rates for TypeSafe models default to UNKNOWN rate status and zero rates."""
    calc = CostCalculator(tmp_path / "nonexistent_pricing.yaml")
    for model_id in ["systemone-preview-v1", "typesafe/jev", "jev-preview", "jev-1.13.0"]:
        rate = calc.get_rate(model_id)
        assert rate is not None
        assert rate.rate_status == RateStatus.UNKNOWN.value
        assert rate.rate_input == 0.0
        assert rate.rate_cached == 0.0
        assert rate.rate_output == 0.0
        assert rate.rate_other == 0.0
        assert rate.source_url == ""
        assert rate.notes == "Unknown / unconfigured rate"


def test_cost_calculation_formula_exact() -> None:
    """Test cost calculation adheres strictly to MEGAPLAN §16.2 formula."""
    calc = CostCalculator(Path("configs/pricing.yaml"))

    # Formula:
    # cost = uncached_in * rate_in / 1e6 + cached_in * rate_cached / 1e6 + out * rate_out / 1e6
    # GPT-4o-mini: input 0.150, cached 0.075, output 0.600
    # uncached = 10,000, cached = 5,000, out = 2,000
    # cost = 10,000 * 0.150/1e6 + 5,000 * 0.075/1e6 + 2,000 * 0.600/1e6
    #      = 0.00150 + 0.000375 + 0.00120 = 0.003075
    res = calc.calculate_call_cost(
        "gpt-4o-mini",
        uncached_input_tokens=10000,
        cached_input_tokens=5000,
        billed_output_tokens=2000,
    )
    assert res.cost_status in ("exact", CostStatus.CALCULATED.value)
    assert res.cost_usd is not None
    assert abs(res.cost_usd - 0.003075) < 1e-8


def test_cost_calculation_missing_tokens_never_zero() -> None:
    """Ensure missing usage results in 'unknown' status and never silently returns $0.00."""
    calc = CostCalculator(Path("configs/pricing.yaml"))

    # Missing input tokens
    res1 = calc.calculate_call_cost("gpt-4o-mini", uncached_input_tokens=None, billed_output_tokens=100)
    assert res1.cost_status in ("unknown", CostStatus.UNKNOWN.value)
    assert res1.cost_usd is None
    assert "Missing usage/token count" in (res1.warning or "")

    # Missing output tokens
    res2 = calc.calculate_call_cost("gpt-4o-mini", uncached_input_tokens=500, billed_output_tokens=None)
    assert res2.cost_status in ("unknown", CostStatus.UNKNOWN.value)
    assert res2.cost_usd is None

    # Aggregated with one unknown call reports lower bound and partial/unknown status
    res_ok = calc.calculate_call_cost("gpt-4o-mini", uncached_input_tokens=1000, billed_output_tokens=100)
    agg = calc.aggregate_costs([res_ok, res1])
    assert agg.cost_status == "partial"
    assert agg.lower_bound_usd > 0
    assert "incomplete" in (agg.warning or "")


def test_cost_per_correct_task_and_amortization() -> None:
    """Test cost per correct task (§16.2), amortized cost (§16.3), and break-even."""
    calc = CostCalculator()

    # Normal success
    c_per_task = calc.calculate_cost_per_correct_task(10.0, 25)
    assert c_per_task == 0.40

    # Zero successes must yield None / undefined (REPAIR_PLAN.md §9, never 0 or $0.00)
    c_zero = calc.calculate_cost_per_correct_task(10.0, 0)
    assert c_zero is None

    # Amortized cost: total_cost(N) = prep + N * mean_online
    # prep = 100, online = 0.01
    # N=100 -> total = 100 + 1 = 101 -> amortized = 1.01
    am100 = calc.calculate_amortized_cost(100.0, 0.01, 100)
    assert am100 == 1.01

    # Break-even:
    # A: prep = 200, online = 0.01
    # B: prep = 20, online = 0.05
    # (200 - 20) / (0.05 - 0.01) = 180 / 0.04 = 4500
    be = calc.calculate_break_even(200.0, 20.0, 0.01, 0.05)
    assert be == 4500.0

    # If A has higher online cost, break-even is impossible
    be_none = calc.calculate_break_even(200.0, 20.0, 0.05, 0.01)
    assert be_none is None


def test_pricing_scenarios_and_unverified_jev_rates(synthetic_pricing_yaml: Path) -> None:
    """Verify Section 9 pricing requirements: unverified JEV rate, scenarios, and Bedrock Gemma."""
    calc = CostCalculator(synthetic_pricing_yaml)

    # JEV rate must NOT be confirmed/verified (verified: false)
    jev_rate = calc.get_rate("typesafe/jev")
    assert jev_rate is not None
    assert jev_rate.verified is False
    assert jev_rate.rate_input == 1.0
    assert jev_rate.rate_cached == 0.25
    assert jev_rate.rate_output == 0.0

    # Pricing scenarios
    # Scenario a: Lab card rate
    lab_scen = calc.get_scenario("lab_card_rate")
    assert lab_scen is not None
    assert lab_scen.verified is False
    assert lab_scen.rate_input == 1.00
    assert lab_scen.rate_cached == 0.25
    assert lab_scen.rate_output == 0.00
    assert "synthetic test rate" in lab_scen.notes

    # Scenario b: Published TypeSafe JEV rate
    pub_scen = calc.get_scenario("published_typesafe_jev")
    assert pub_scen is not None
    assert pub_scen.verified is False
    assert pub_scen.rate_input == 1.0
    assert pub_scen.rate_output == 0.00
    assert "synthetic test rate" in pub_scen.notes
    assert pub_scen.source_url == "https://example.com/synthetic-pricing"

    # REPAIR_PLAN §9 calculation verification with synthetic rate:
    # 12,912 tokens with published scenario (1.0 per 1M) = 12912 * 1.0 / 1e6 = 0.012912
    pub_cost = calc.calculate_call_cost(
        "typesafe/jev",
        uncached_input_tokens=12912,
        billed_output_tokens=0,
        scenario="published_typesafe_jev",
    )
    assert pub_cost.cost_status in ("estimated", CostStatus.ESTIMATED.value)  # Unverified published rate yields estimated
    assert pub_cost.cost_usd is not None
    assert abs(pub_cost.cost_usd - 0.012912) < 1e-8

    # Bedrock Mantle / Gemma rate is verified: false (cost_status: estimated)
    gemma_rate = calc.get_rate("google.gemma-4-31b")
    assert gemma_rate is not None
    assert gemma_rate.verified is False
    gemma_call = calc.calculate_call_cost("google.gemma-4-31b", uncached_input_tokens=1000, billed_output_tokens=200)
    assert gemma_call.cost_status in ("estimated", CostStatus.ESTIMATED.value)
    assert gemma_call.cost_usd is not None and gemma_call.cost_usd > 0


def test_missing_provider_usage_never_zero_cost_or_tokens() -> None:
    """Verify Section 9: missing provider usage must NEVER be converted to 0 tokens or $0.00 cost."""
    calc = CostCalculator(Path("configs/pricing.yaml"))

    # Missing tokens on call calculation
    call_missing = calc.calculate_call_cost("typesafe/jev", uncached_input_tokens=None, billed_output_tokens=None)
    assert call_missing.cost_usd is None
    assert call_missing.cost_status in ("unknown", CostStatus.UNKNOWN.value)
    assert call_missing.uncached_input_tokens is None
    assert call_missing.billed_output_tokens is None

    # Aggregated with missing usage calls
    call_ok = calc.calculate_call_cost("gpt-4o-mini", uncached_input_tokens=1000, billed_output_tokens=100)
    agg = calc.aggregate_costs([call_ok, call_missing])
    assert agg.cost_status == "partial"
    assert agg.uncached_input_tokens is None  # Total tokens cannot pretend to be known
    assert agg.billed_output_tokens is None
    assert agg.total_cost_usd is not None and agg.total_cost_usd > 0  # Lower bound reported, never $0.00

    # When all calls have missing usage, total_cost_usd is None (never $0.00)
    agg_all_missing = calc.aggregate_costs([call_missing])
    assert agg_all_missing.total_cost_usd is None
    assert agg_all_missing.cost_status in ("unknown", CostStatus.UNKNOWN.value)

    # SpanManager token aggregation with missing usage
    mgr = SpanManager()
    with mgr.span("request_total"):
        with mgr.span("model_request") as s1:
            s1.record_usage(input_tokens=500, output_tokens=100)
        with mgr.span("model_request") as s2:
            # Failed or unreturned usage on provider call
            s2.set_tag("provider_call", True)
            pass

    in_tok, out_tok, _ = mgr.aggregate_tokens()
    # Missing provider usage must NEVER be converted to 0 tokens
    assert in_tok is None
    assert out_tok is None


def test_accounting_tracking_and_historical_422() -> None:
    """Verify Section 9 accounting design: logical_request_count, provider_call_count, tool_call_count, retry_count, and historical 422 calls."""
    calc = CostCalculator()

    # Create telemetry records representing 2 logical requests:
    # Request 1: 1 JEV call (provider call)
    mgr1 = SpanManager(request_id="req-1", case_id="Q1", engine="structured_jev")
    with mgr1.span(StandardSpan.REQUEST_TOTAL.value):
        with mgr1.span(StandardSpan.QUERY_INTERPRETATION.value):
            pass
        with mgr1.span(StandardSpan.MODEL_REQUEST.value) as s:
            s.record_usage(input_tokens=1000, output_tokens=0)
            s.set_tag("model", "typesafe/jev")
            s.set_tag("http_status", 200)

    rec1 = mgr1.build_telemetry_record()
    assert rec1.logical_request_count == 1
    assert rec1.provider_calls == 1
    assert rec1.tool_calls == 0
    assert rec1.retry_count == 0

    # Request 2: Scrape LLM with 3 tool calls and 2 model calls
    mgr2 = SpanManager(request_id="req-2", case_id="Q2", engine="scrape_llm")
    with mgr2.span(StandardSpan.REQUEST_TOTAL.value):
        with mgr2.span(StandardSpan.MODEL_REQUEST.value) as s:
            s.record_usage(input_tokens=500, output_tokens=50)
            s.set_tag("model", "google.gemma-4-31b")
        with mgr2.span(StandardSpan.HTML_FETCH.value):
            pass
        with mgr2.span(StandardSpan.PDF_PARSE.value):
            pass
        with mgr2.span(StandardSpan.MODEL_REQUEST.value) as s:
            s.record_usage(input_tokens=800, output_tokens=100)
            s.set_tag("model", "google.gemma-4-31b")

    rec2 = mgr2.build_telemetry_record(attempts=2)
    assert rec2.logical_request_count == 1
    assert rec2.provider_calls == 2
    assert rec2.tool_calls == 2
    assert rec2.retry_count == 1

    # Compute run accounting across records, accounting for 5 historical 422 calls
    accounting = calc.compute_accounting([rec1, rec2], include_historical_422=True, historical_422_count=5)
    assert accounting.logical_request_count == 2
    # Provider calls: rec1(1) + rec2(2) + historical_422(5) = 8
    assert accounting.provider_call_count == 8
    assert accounting.tool_call_count == 2
    assert accounting.retry_count == 1
    assert accounting.status_422_count == 5
    assert accounting.historical_422_calls == 5
    # Historical 422 calls had missing usage, so overall cost_status cannot be exact
    assert accounting.cost_status in ("partial", "estimated", "unknown", CostStatus.ESTIMATED.value, CostStatus.UNKNOWN.value)


def test_monotonic_clock_query_to_validation_span() -> None:
    """Verify Section 10: monotonic clock spans from query receipt to validated answer response."""
    mgr = SpanManager(request_id="mono_test_01", case_id="Q1")

    # Root span begins at query receipt
    with mgr.span(StandardSpan.REQUEST_TOTAL.value) as root:
        time.sleep(0.005)

        # Stage 1: query interpretation
        with mgr.span(StandardSpan.QUERY_INTERPRETATION.value) as stage1:
            time.sleep(0.002)

        # Stage 2: rule evaluation
        with mgr.span(StandardSpan.RULE_EVALUATION.value) as stage2:
            time.sleep(0.002)

        # Stage 3: model request
        with mgr.span(StandardSpan.MODEL_REQUEST.value) as stage3:
            stage3.record_usage(input_tokens=200, output_tokens=50)
            time.sleep(0.003)

        # Stage 4: response validation
        with mgr.span(StandardSpan.RESPONSE_VALIDATION.value) as stage4:
            time.sleep(0.002)

    root_s = mgr.find_spans(StandardSpan.REQUEST_TOTAL.value)[0]
    stage1_s = mgr.find_spans(StandardSpan.QUERY_INTERPRETATION.value)[0]
    stage2_s = mgr.find_spans(StandardSpan.RULE_EVALUATION.value)[0]
    stage3_s = mgr.find_spans(StandardSpan.MODEL_REQUEST.value)[0]
    stage4_s = mgr.find_spans(StandardSpan.RESPONSE_VALIDATION.value)[0]

    # Monotonic timing checks via time.perf_counter_ns
    assert root_s.start_ns <= stage1_s.start_ns
    assert stage1_s.end_ns <= stage2_s.start_ns
    assert stage2_s.end_ns <= stage3_s.start_ns
    assert stage3_s.end_ns <= stage4_s.start_ns
    assert stage4_s.end_ns <= root_s.end_ns

    # Root span duration covers the full lifecycle to validation
    assert root_s.duration_ms is not None and root_s.duration_ms >= (
        stage1_s.duration_ms + stage2_s.duration_ms + stage3_s.duration_ms + stage4_s.duration_ms
    ) * 0.9


def test_stage_spans_and_failure_preservation() -> None:
    """Verify Section 10: stage spans preserve failure codes, statuses, and durations."""
    mgr = SpanManager(request_id="fail_test_01", case_id="Q3")

    with pytest.raises(RuntimeError, match="JEV API 422: missing criteria"):
        with mgr.span(StandardSpan.REQUEST_TOTAL.value):
            with mgr.span(StandardSpan.QUERY_INTERPRETATION.value) as s1:
                s1.set_tag("rule", "pass")
            with mgr.span(StandardSpan.MODEL_REQUEST.value) as s2:
                s2.set_tag("http_status", 422)
                s2.set_tag("model", "typesafe/jev")
                raise RuntimeError("JEV API 422: missing criteria")

    rec = mgr.build_telemetry_record()
    assert rec.execution_status == "PROVIDER_ERROR"
    assert "JEV API 422: missing criteria" in (rec.failure_code or "")
    assert rec.http_statuses == [422]

    # Verify failed stage span preserved duration and error message
    failed_span = mgr.find_spans(StandardSpan.MODEL_REQUEST.value)[0]
    assert failed_span.status == "error"
    assert "JEV API 422: missing criteria" in (failed_span.error or "")
    assert failed_span.duration_ms is not None and failed_span.duration_ms >= 0


def test_breakdown_spans_and_accurate_telemetry_reporting():
    """Verify F14 and F16: full hierarchical breakdown spans and accurate status reporting."""
    mgr = SpanManager(
        run_id="run_f14_f16",
        request_id="req_f14_01",
        case_id="case_sensor_01",
        engine="rag_llm",
        repeat=1,
    )

    breakdown = mgr.record_breakdown_spans(
        total_duration_ms=150.0,
        engine="rag_llm",
        http_status=200,
        input_tokens=1500,
        output_tokens=300,
        cached_tokens=100,
    )

    span_names = [s.name for s in breakdown]
    assert StandardSpan.REQUEST_TOTAL.value in span_names
    assert StandardSpan.INTERPRET_QUERY.value in span_names
    assert StandardSpan.RAG_RETRIEVAL.value in span_names
    assert StandardSpan.RULES_EVALUATE.value in span_names
    assert "provider_call_1" in span_names
    assert "tool_call_1" in span_names
    assert StandardSpan.RENDER_RESPONSE.value in span_names
    assert StandardSpan.VALIDATE_RESPONSE.value in span_names

    tel = mgr.build_telemetry_record(
        cost_usd=0.00045,
        cost_status=CostStatus.CALCULATED.value,
        usage_status=UsageStatus.PROVIDER_REPORTED.value,
        rate_status=RateStatus.ACCOUNT_CONFIRMED.value,
    )

    assert tel.provider_calls == 1
    assert tel.tool_calls == 1
    assert tel.http_statuses == [200]
    assert tel.input_tokens == 1500
    assert tel.output_tokens == 300
    assert tel.cached_tokens == 100
    assert tel.usage_status == UsageStatus.PROVIDER_REPORTED.value
    assert tel.rate_status == RateStatus.ACCOUNT_CONFIRMED.value
    assert tel.cost_status == CostStatus.CALCULATED.value


def test_telemetry_estimated_rate_forces_estimated_cost_status():
    """Verify F15: if usage or rate is estimated, cost_status MUST be ESTIMATED, never exact/CALCULATED."""
    mgr = SpanManager(request_id="req_est_01")
    with mgr.span("request_total"):
        with mgr.span("provider_call_1") as s:
            s.record_usage(input_tokens=1000, output_tokens=200)
            s.set_http_status(200)

    # If rate is ASSUMED/unverified, cost_status must be forced to ESTIMATED even if CALCULATED is passed
    tel = mgr.build_telemetry_record(
        cost_usd=0.005,
        cost_status=CostStatus.CALCULATED.value,
        usage_status=UsageStatus.PROVIDER_REPORTED.value,
        rate_status=RateStatus.ASSUMED.value,
    )
    assert tel.cost_status == CostStatus.ESTIMATED.value

