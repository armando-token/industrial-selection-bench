"""Comprehensive unit tests for Industrial Selection Lab Benchmark package.

Tests:
- Scoring Evaluator (§15): task_success conditions, secondary metrics, aggregation.
- StatisticsAnalyzer (§17): paired cluster bootstrap, 95% CIs, Holm adjustment, outcome labels.
- BenchmarkRunner (§18): balanced permutations, scenario installation, gold stripping,
  modes (validate, smoke, official, replay), file persistence.
"""

import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from industrial_lab.benchmark.runner import BenchmarkRunner, RunManifest
from industrial_lab.benchmark.scoring import (
    BenchmarkSummary,
    CaseScoreRecord,
    EngineMetricsSummary,
    Evaluator,
    safe_rate,
)
from industrial_lab.benchmark.statistics import (
    StatisticsAnalyzer,
    adjust_p_values_holm,
    produce_statistics_json,
    produce_summary_csv,
)
from industrial_lab.observability.spans import SpanRecord, TelemetryRecord
from industrial_lab.schemas import (
    BenchmarkCase,
    CheckResult,
    CheckStatus,
    CommercialData,
    DecisionOrigin,
    ExecutionStatus,
    GoldCase,
    QueryRequest,
    QueryResponse,
    Quote,
    QuoteLine,
    QuoteStatus,
    Requirement,
    RequirementKind,
    TechnicalVerdict,
)


# ==============================================================================
# Helper Fixtures
# ==============================================================================

def make_test_case(
    case_id: str = "T001",
    family_id: str = "FAM-01",
    acceptable: List[List[str]] = [["P1"]],
    verdict: TechnicalVerdict = TechnicalVerdict.COMPATIBLE,
    required_checks: List[Dict[str, Any]] = [{"requirement_id": "R1", "status": "PASS"}],
    required_evidence_groups: List[Dict[str, Any]] = [{"group_id": "G1", "acceptable_spans": ["DOC1:p1:s1"]}],
) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        scenario_family_id=family_id,
        scenario_id="S0001",
        split="test",
        mode="M1",
        query_text="Consulta tecnica de prueba para P1",
        gold=GoldCase(
            acceptable_selections=acceptable,
            technical_verdict=verdict,
            required_checks=required_checks,
            required_evidence_groups=required_evidence_groups,
            commerce_revision="rev-test-01",
            human_review_status="approved",
        ),
    )


def make_test_response(
    request_id: str = "req_1",
    engine: str = "structured_jev",
    selected: List[str] = ["P1"],
    verdict: TechnicalVerdict = TechnicalVerdict.COMPATIBLE,
    status: ExecutionStatus = ExecutionStatus.completed,
    checks: List[CheckResult] = None,
    quote: Quote = None,
) -> QueryResponse:
    if checks is None:
        checks = [
            CheckResult(
                requirement_id="R1",
                status=CheckStatus.PASS,
                evidence_ids=["DOC1:p1:s1"],
                decision_origin=DecisionOrigin.rule,
            )
        ]
    return QueryResponse(
        request_id=request_id,
        engine=engine,
        engine_version="test-v1",
        execution_status=status,
        catalog_version="v1",
        knowledge_version="v1",
        selected_product_ids=selected,
        technical_verdict=verdict,
        checks=checks,
        quote=quote,
        summary="Test response summary",
    )


# ==============================================================================
# 1. Scoring & Evaluator Tests (§15)
# ==============================================================================

def test_safe_rate():
    """Verify safe rate calculation and zero-denominator handling per §15.2."""
    assert safe_rate(10, 20) == 0.5
    assert safe_rate(0, 5) == 0.0
    assert safe_rate(5, 0) is None
    assert safe_rate(0, 0) is None


def test_task_success_all_conditions_satisfied():
    """Verify task_success passes when all 6 conditions are simultaneously met."""
    case = make_test_case()
    resp = make_test_response()

    evaluator = Evaluator(valid_spans={"DOC1:p1:s1"})
    score = evaluator.evaluate_case(case, resp)

    assert score.task_success is True
    assert score.execution_completed is True
    assert score.selection_correct is True
    assert score.verdict_correct is True
    assert score.required_checks_passed is True
    assert score.evidence_valid_and_supported is True
    assert score.commerce_consistent is True
    assert len(score.failure_reasons) == 0


def test_task_success_fails_on_execution_status():
    """Condition 1: task_success fails if execution_status != completed."""
    case = make_test_case()
    resp = make_test_response(status=ExecutionStatus.provider_error)

    evaluator = Evaluator(valid_spans={"DOC1:p1:s1"})
    score = evaluator.evaluate_case(case, resp)

    assert score.task_success is False
    assert score.execution_completed is False
    assert any("Execution status is 'provider_error'" in r for r in score.failure_reasons)


def test_task_success_fails_on_incorrect_selection():
    """Condition 2: task_success fails if selected_product_ids not in acceptable_selections."""
    case = make_test_case(acceptable=[["P1"]])
    # Wrong product selected
    resp = make_test_response(selected=["P2"])

    evaluator = Evaluator(valid_spans={"DOC1:p1:s1"})
    score = evaluator.evaluate_case(case, resp)

    assert score.task_success is False
    assert score.selection_correct is False
    assert any("Selected products ['P2'] not in acceptable" in r for r in score.failure_reasons)


def test_task_success_order_independent_selection():
    """Condition 2: order of multiple products in selection must not matter (§13.4)."""
    case = make_test_case(acceptable=[["P1", "P2"]])
    # Response has ["P2", "P1"] (different order)
    resp = make_test_response(selected=["P2", "P1"])

    evaluator = Evaluator(valid_spans={"DOC1:p1:s1"})
    score = evaluator.evaluate_case(case, resp)

    assert score.selection_correct is True
    assert score.task_success is True


def test_task_success_fails_on_verdict_mismatch():
    """Condition 3: task_success fails if technical_verdict != gold.technical_verdict."""
    case = make_test_case(verdict=TechnicalVerdict.COMPATIBLE)
    resp = make_test_response(verdict=TechnicalVerdict.INCOMPATIBLE)

    evaluator = Evaluator(valid_spans={"DOC1:p1:s1"})
    score = evaluator.evaluate_case(case, resp)

    assert score.task_success is False
    assert score.verdict_correct is False
    assert score.false_rejection is True


def test_task_success_fails_on_missing_required_check():
    """Condition 4: task_success fails if required check is missing."""
    case = make_test_case(required_checks=[{"requirement_id": "REQ-CRITICAL", "status": "PASS"}])
    # Response only checks R1, missing REQ-CRITICAL
    resp = make_test_response()

    evaluator = Evaluator(valid_spans={"DOC1:p1:s1"})
    score = evaluator.evaluate_case(case, resp)

    assert score.task_success is False
    assert score.required_checks_passed is False
    assert any("Missing required check for requirement 'REQ-CRITICAL'" in r for r in score.failure_reasons)


def test_task_success_fails_on_invalid_evidence_span():
    """Condition 5: task_success fails if cited span does not exist in valid corpus."""
    case = make_test_case()
    # Response cites non-existent span
    checks = [
        CheckResult(
            requirement_id="R1",
            status=CheckStatus.PASS,
            evidence_ids=["DOC_HALLUCINATED:p99:s99"],
            decision_origin=DecisionOrigin.llm,
        )
    ]
    resp = make_test_response(checks=checks)

    evaluator = Evaluator(valid_spans={"DOC1:p1:s1"})
    score = evaluator.evaluate_case(case, resp)

    assert score.task_success is False
    assert score.evidence_valid_and_supported is False
    assert any("Invalid or non-existent evidence spans" in r for r in score.failure_reasons)


def test_task_success_approving_check_must_have_evidence():
    """Condition 5: Check with PASS must cite non-empty evidence (§15.1)."""
    case = make_test_case(required_evidence_groups=[])
    # Approving check with empty evidence
    checks = [
        CheckResult(
            requirement_id="R1",
            status=CheckStatus.PASS,
            evidence_ids=[],
            decision_origin=DecisionOrigin.rule,
        )
    ]
    resp = make_test_response(checks=checks)

    evaluator = Evaluator(valid_spans=set())
    score = evaluator.evaluate_case(case, resp)

    assert score.task_success is False
    assert score.evidence_valid_and_supported is False
    assert any("approved with PASS but has no evidence_ids" in r for r in score.failure_reasons)


def test_task_success_fails_on_inconsistent_commerce():
    """Condition 6: Quote status must be requires_technical_review on non-compatible verdicts (§6.2)."""
    case = make_test_case(
        acceptable=[],
        verdict=TechnicalVerdict.INCOMPATIBLE,
        required_checks=[],
        required_evidence_groups=[],
    )
    # Quote with status 'preliminary' when verdict is INCOMPATIBLE violates §6.2
    quote = Quote(
        status=QuoteStatus.preliminary,
        lines=[QuoteLine(product_id="P1", quantity=1, unit_price_minor=1000, line_total_minor=1000)],
        subtotal_minor=1000,
        tax_minor=0,
        total_minor=1000,
        revision="rev-01",
        expires_at_simulated="2026-12-31T00:00:00Z",
    )
    resp = make_test_response(
        selected=[],
        verdict=TechnicalVerdict.INCOMPATIBLE,
        checks=[],
        quote=quote,
    )

    evaluator = Evaluator()
    score = evaluator.evaluate_case(case, resp)

    assert score.task_success is False
    assert score.commerce_consistent is False
    assert any("Quote status must be 'requires_technical_review'" in r for r in score.failure_reasons)


def test_secondary_metrics_safety_false_approval():
    """Verify false_approval detection when engine approves incompatible case (§15.2)."""
    case = make_test_case(
        acceptable=[],
        verdict=TechnicalVerdict.INCOMPATIBLE,
        required_checks=[],
        required_evidence_groups=[],
    )
    # Engine incorrectly approves
    resp = make_test_response(
        selected=["P1"],
        verdict=TechnicalVerdict.COMPATIBLE,
    )

    evaluator = Evaluator()
    score = evaluator.evaluate_case(case, resp)

    assert score.false_approval is True
    assert score.task_success is False


def test_secondary_metrics_abstention_rates():
    """Verify correct and excessive abstention classification (§15.2)."""
    # 1. Correct abstention: gold is UNKNOWN, engine abstains
    case_unk = make_test_case(
        acceptable=[],
        verdict=TechnicalVerdict.INSUFFICIENT_EVIDENCE,
        required_checks=[],
        required_evidence_groups=[],
    )
    resp_unk = make_test_response(
        selected=[],
        verdict=TechnicalVerdict.INSUFFICIENT_EVIDENCE,
        checks=[],
    )
    evaluator = Evaluator()
    score_unk = evaluator.evaluate_case(case_unk, resp_unk)
    assert score_unk.is_abstention is True
    assert score_unk.correct_abstention is True
    assert score_unk.excessive_abstention is False

    # 2. Excessive abstention: gold is resoluble (COMPATIBLE), engine abstains
    case_comp = make_test_case(verdict=TechnicalVerdict.COMPATIBLE)
    score_comp = evaluator.evaluate_case(case_comp, resp_unk)
    assert score_comp.is_abstention is True
    assert score_comp.correct_abstention is False
    assert score_comp.excessive_abstention is True


def test_evaluator_aggregate_summary():
    """Verify metric aggregation into BenchmarkSummary across multiple cases."""
    case1 = make_test_case(case_id="C1", family_id="F1")
    case2 = make_test_case(
        case_id="C2",
        family_id="F2",
        verdict=TechnicalVerdict.INCOMPATIBLE,
        acceptable=[],
        required_checks=[],
        required_evidence_groups=[],
    )

    resp1 = make_test_response(request_id="r1", engine="structured_jev")
    resp2 = make_test_response(
        request_id="r2",
        engine="structured_jev",
        selected=[],
        verdict=TechnicalVerdict.INCOMPATIBLE,
        checks=[],
    )

    evaluator = Evaluator(valid_spans={"DOC1:p1:s1"})
    s1 = evaluator.evaluate_case(case1, resp1)
    s2 = evaluator.evaluate_case(case2, resp2)

    tel1 = TelemetryRecord(
        run_id="run1",
        request_id="r1",
        case_id="C1",
        engine="structured_jev",
        start_utc="2026-10-02T12:00:00Z",
        elapsed_ms=10.0,
        cost_usd=0.01,
        cost_status="exact",
    )
    tel2 = TelemetryRecord(
        run_id="run1",
        request_id="r2",
        case_id="C2",
        engine="structured_jev",
        start_utc="2026-10-02T12:00:01Z",
        elapsed_ms=15.0,
        cost_usd=0.01,
        cost_status="exact",
    )

    summary = evaluator.aggregate(
        score_records=[s1, s2],
        telemetry_records=[tel1, tel2],
        run_id="run1",
    )

    assert summary.total_cases == 2
    assert "structured_jev" in summary.engine_summaries
    s = summary.engine_summaries["structured_jev"]
    assert s.cases == 2
    assert s.completed_cases == 2
    assert s.task_success.numerator == 2
    assert s.task_success.rate == 1.0
    assert s.latency_mean_ms == 12.5
    assert s.total_cost_usd == 0.02


# ==============================================================================
# 2. Statistics & Bootstrap Analysis Tests (§17)
# ==============================================================================

def test_adjust_p_values_holm():
    """Verify Holm-Bonferroni step-down adjustment (§17.2)."""
    raw_p = {
        "test_a": 0.01,
        "test_b": 0.04,
        "test_c": 0.03,
    }
    # Sorted: test_a: 0.01 (x3 -> 0.03), test_c: 0.03 (x2 -> 0.06), test_b: 0.04 (x1 -> 0.04 -> max(0.06, 0.04)=0.06)
    adj = adjust_p_values_holm(raw_p)
    assert pytest.approx(adj["test_a"], 0.001) == 0.03
    assert pytest.approx(adj["test_c"], 0.001) == 0.06
    assert pytest.approx(adj["test_b"], 0.001) == 0.06
    # Enforces monotonicity
    assert adj["test_a"] <= adj["test_c"] <= adj["test_b"]


def test_statistics_analyzer_paired_cluster_bootstrap():
    """Verify cluster bootstrap by scenario_family_id and confidence intervals (§17.2)."""
    # Create paired records for 4 families, 2 engines
    records: List[CaseScoreRecord] = []
    telemetry: List[TelemetryRecord] = []

    for i in range(1, 9):
        fam_id = f"FAM-0{((i - 1) % 4) + 1}"
        case_id = f"C{i}"
        req_a = f"req_a_{i}"
        req_b = f"req_b_{i}"

        # Engine A has 100% success, 10ms latency, $0.01 cost
        rec_a = CaseScoreRecord(
            run_id="run_stat",
            request_id=req_a,
            case_id=case_id,
            scenario_family_id=fam_id,
            scenario_id="S0001",
            engine="structured_jev",
            execution_status=ExecutionStatus.completed,
            task_success=True,
            execution_completed=True,
            selection_correct=True,
            verdict_correct=True,
        )
        tel_a = TelemetryRecord(
            run_id="run_stat",
            request_id=req_a,
            case_id=case_id,
            engine="structured_jev",
            start_utc="2026-10-02T12:00:00Z",
            elapsed_ms=10.0,
            cost_usd=0.01,
            cost_status="exact",
        )

        # Engine B has 50% success (even indices fail), 50ms latency, $0.05 cost
        rec_b = CaseScoreRecord(
            run_id="run_stat",
            request_id=req_b,
            case_id=case_id,
            scenario_family_id=fam_id,
            scenario_id="S0001",
            engine="rag_llm",
            execution_status=ExecutionStatus.completed,
            task_success=(i % 2 == 1),
            execution_completed=True,
            selection_correct=(i % 2 == 1),
            verdict_correct=(i % 2 == 1),
        )
        tel_b = TelemetryRecord(
            run_id="run_stat",
            request_id=req_b,
            case_id=case_id,
            engine="rag_llm",
            start_utc="2026-10-02T12:00:00Z",
            elapsed_ms=50.0,
            cost_usd=0.05,
            cost_status="exact",
        )

        records.extend([rec_a, rec_b])
        telemetry.extend([tel_a, tel_b])

    analyzer = StatisticsAnalyzer(
        bootstrap_resamples=500,  # Fast for unit tests
        confidence_level=0.95,
        seed=42,
    )
    report = analyzer.analyze(records, telemetry, run_id="run_stat")

    assert report.num_families == 4
    assert report.num_cases == 8
    assert "structured_jev_vs_rag_llm" in report.paired_comparisons

    comp = report.paired_comparisons["structured_jev_vs_rag_llm"]
    # Point diff in success: 1.0 - 0.5 = 0.5
    assert comp.task_success.point_difference == 0.5
    assert comp.task_success.ci_lower >= 0.0
    assert comp.task_success.ci_upper > 0.0
    # Speedup: 50 / 10 = 5.0x
    assert comp.latency_speedup.point_difference == 5.0
    assert comp.latency_target_met is True
    # Cost reduction: (0.05 - 0.01)/0.05 = 80% reduction
    assert comp.cost_reduction_ratio.point_difference == pytest.approx(0.80)
    assert comp.cost_target_met is True
    assert comp.success_non_inferior is True


def test_statistics_file_outputs():
    """Verify summary.csv and statistics.json file generation (§18.4)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_path = tmp_path / "summary.csv"
        json_path = tmp_path / "statistics.json"

        # Make sample summary
        summary = BenchmarkSummary(
            run_id="run_file_test",
            mode="official",
            total_cases=1,
            total_records=1,
            engines=["structured_jev"],
            engine_summaries={},
        )
        produce_summary_csv(csv_path, summary)
        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert "engine" in headers
            assert "task_success_rate" in headers
            assert "latency_p50_ms" in headers

        # Make sample report
        analyzer = StatisticsAnalyzer(bootstrap_resamples=100)
        report = analyzer.analyze([], [], run_id="run_file_test")
        produce_statistics_json(json_path, report)
        assert json_path.exists()
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["run_id"] == "run_file_test"
            assert "outcome_label" in data


def test_zero_denominator_produces_na_in_summary_csv():
    """Verify MEGAPLAN §15.2 requirement: zero denominators produce 'N/A' rather than ZeroDivisionError."""
    # Create record where completed=False, gold is COMPATIBLE, no evidence, no commerce
    rec = CaseScoreRecord(
        run_id="run_zero_den",
        request_id="req_zd_1",
        case_id="C_ZD1",
        scenario_family_id="FAM_ZD",
        scenario_id="S_ZD",
        engine="engine_zd",
        execution_status=ExecutionStatus.provider_error,
        execution_completed=False,
        task_success=False,
        selection_correct=False,
        verdict_correct=False,
        required_checks_passed=False,
        evidence_valid_and_supported=False,
        commerce_consistent=False,
        false_approval=False,
        false_rejection=False,
        is_abstention=False,
        correct_abstention=False,
        excessive_abstention=False,
        is_resolutive=False,
        evidence_validity_num=0,
        evidence_validity_den=0,
        evidence_support_num=0,
        evidence_support_den=0,
        evidence_coverage_num=0,
        evidence_coverage_den=0,
        commerce_num=0,
        commerce_den=0,
        details={"gold_verdict": TechnicalVerdict.COMPATIBLE.value},
    )

    # Telemetry with unknown cost and empty latencies
    tel = TelemetryRecord(
        run_id="run_zero_den",
        request_id="req_zd_1",
        case_id="C_ZD1",
        engine="engine_zd",
        cost_status="unknown",
        cost_usd=None,
    )

    evaluator = Evaluator()
    summary = evaluator.aggregate(score_records=[rec], telemetry_records=[tel], run_id="run_zero_den")

    s = summary.engine_summaries["engine_zd"]
    assert s.cases == 1
    assert s.completed_cases == 0

    # All ratios with zero denominators must have rate = None
    assert s.verdict_accuracy_completed.rate is None
    assert s.false_approval_rate.rate is None  # no non-compatible gold cases
    assert s.abstention_rate.rate is None     # 0 completed cases
    assert s.correct_abstention_rate.rate is None  # no unknown gold cases
    assert s.selective_precision.rate is None  # 0 completed non-abstaining cases
    assert s.evidence_validity.rate is None
    assert s.evidence_support.rate is None
    assert s.evidence_coverage.rate is None
    assert s.commercial_accuracy.rate is None
    assert s.latency_p50_ms is None
    assert s.latency_p95_ms is None
    assert s.total_cost_usd is None
    assert s.cost_per_correct_task_usd is None

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "summary.csv"
        produce_summary_csv(csv_path, summary)
        assert csv_path.exists()

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            row = next(reader)
            row_dict = dict(zip(headers, row))

            # Verify every zero-denominator metric is represented as "N/A"
            assert row_dict["verdict_accuracy"] == "N/A"
            assert row_dict["false_approval_rate"] == "N/A"
            assert row_dict["abstention_rate"] == "N/A"
            assert row_dict["correct_abstention_rate"] == "N/A"
            assert row_dict["selective_precision"] == "N/A"
            assert row_dict["evidence_validity"] == "N/A"
            assert row_dict["evidence_support"] == "N/A"
            assert row_dict["evidence_coverage"] == "N/A"
            assert row_dict["commercial_accuracy"] == "N/A"
            assert row_dict["latency_p50_ms"] == "N/A"
            assert row_dict["latency_p90_ms"] == "N/A"
            assert row_dict["latency_p95_ms"] == "N/A"
            assert row_dict["latency_mean_ms"] == "N/A"
            assert row_dict["total_cost_usd"] == "N/A"
            assert row_dict["mean_cost_usd"] == "N/A"
            assert row_dict["cost_per_correct_task_usd"] == "N/A"
            assert row_dict["cost_status"] == "unknown"


def test_zero_success_produces_none_cost_per_correct_task():
    """Verify MEGAPLAN §16.2 & REPAIR3_PLAN §14: when total cost > 0 but n_success == 0, cost_per_correct_task is None / 'N/A'."""
    rec = CaseScoreRecord(
        run_id="run_inf",
        request_id="req_inf_1",
        case_id="C_INF1",
        scenario_family_id="FAM_INF",
        scenario_id="S_INF",
        engine="engine_inf",
        execution_status=ExecutionStatus.completed,
        execution_completed=True,
        task_success=False,
    )
    tel = TelemetryRecord(
        run_id="run_inf",
        request_id="req_inf_1",
        case_id="C_INF1",
        engine="engine_inf",
        cost_usd=0.05,
        cost_status="CALCULATED",
    )
    evaluator = Evaluator()
    summary = evaluator.aggregate(score_records=[rec], telemetry_records=[tel], run_id="run_inf")
    s = summary.engine_summaries["engine_inf"]
    assert s.total_cost_usd == 0.05
    assert s.cost_per_correct_task_usd is None

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "summary.csv"
        produce_summary_csv(csv_path, summary)
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            row = next(reader)
            row_dict = dict(zip(headers, row))
            assert row_dict["cost_per_correct_task_usd"] == "N/A"


def test_single_family_small_sample_disables_statistical_inference():
    """Verify F17: single-family or small sample (k<=1, n<5) disables statistical inference."""
    records: List[CaseScoreRecord] = []
    telemetry: List[TelemetryRecord] = []

    # 1 single family, 4 cases (k=1, n=4)
    for i in range(4):
        case_id = f"C{i+1}"
        records.append(
            CaseScoreRecord(
                run_id="run_small",
                request_id=f"req_a_{i}",
                case_id=case_id,
                scenario_family_id="FAM_SINGLE",
                scenario_id="S001",
                engine="structured_jev",
                execution_status=ExecutionStatus.completed,
                task_success=True,
                execution_completed=True,
            )
        )
        telemetry.append(
            TelemetryRecord(
                run_id="run_small",
                request_id=f"req_a_{i}",
                case_id=case_id,
                engine="structured_jev",
                elapsed_ms=10.0,
                cost_usd=0.001,
            )
        )
        records.append(
            CaseScoreRecord(
                run_id="run_small",
                request_id=f"req_b_{i}",
                case_id=case_id,
                scenario_family_id="FAM_SINGLE",
                scenario_id="S001",
                engine="rag_llm",
                execution_status=ExecutionStatus.completed,
                task_success=True,
                execution_completed=True,
            )
        )
        telemetry.append(
            TelemetryRecord(
                run_id="run_small",
                request_id=f"req_b_{i}",
                case_id=case_id,
                engine="rag_llm",
                elapsed_ms=50.0,
                cost_usd=0.01,
            )
        )

    analyzer = StatisticsAnalyzer(seed=42)
    report = analyzer.analyze(records, telemetry, run_id="run_small")

    assert report.num_families == 1
    assert report.num_cases == 4
    comp = report.paired_comparisons["structured_jev_vs_rag_llm"]

    # Inference must be disabled
    assert comp.task_success.ci_lower is None
    assert comp.task_success.ci_upper is None
    assert comp.task_success.p_value is None
    assert comp.task_success.p_value_holm is None
    assert comp.task_success.is_statistically_significant is None
    assert comp.task_success.details.get("inference_disabled") is True

    assert comp.latency_speedup.p_value is None
    assert comp.cost_reduction_ratio.p_value is None

    # Favorable outcome can NEVER be declared with disabled inference
    assert comp.overall_favorable_a is False
    assert report.outcome_label == "Inconclusive"


def test_zero_successes_never_overall_favorable_a():
    """Verify F17: an engine with zero successes can NEVER be declared overall_favorable_a."""
    records: List[CaseScoreRecord] = []
    telemetry: List[TelemetryRecord] = []

    # 4 families, 12 cases (k=4, n=12) - large enough sample
    for i in range(12):
        fam_id = f"FAM-0{(i % 4) + 1}"
        case_id = f"C{i+1}"

        # Engine A (structured_jev): 0% success (0 successes), but very fast (5ms) and cheap ($0.001)
        records.append(
            CaseScoreRecord(
                run_id="run_zerosucc",
                request_id=f"req_a_{i}",
                case_id=case_id,
                scenario_family_id=fam_id,
                scenario_id="S001",
                engine="structured_jev",
                execution_status=ExecutionStatus.completed,
                task_success=False,
                execution_completed=True,
            )
        )
        telemetry.append(
            TelemetryRecord(
                run_id="run_zerosucc",
                request_id=f"req_a_{i}",
                case_id=case_id,
                engine="structured_jev",
                elapsed_ms=5.0,
                cost_usd=0.001,
            )
        )

        # Engine B (rag_llm): also 0% success, but slow (50ms)
        records.append(
            CaseScoreRecord(
                run_id="run_zerosucc",
                request_id=f"req_b_{i}",
                case_id=case_id,
                scenario_family_id=fam_id,
                scenario_id="S001",
                engine="rag_llm",
                execution_status=ExecutionStatus.completed,
                task_success=False,
                execution_completed=True,
            )
        )
        telemetry.append(
            TelemetryRecord(
                run_id="run_zerosucc",
                request_id=f"req_b_{i}",
                case_id=case_id,
                engine="rag_llm",
                elapsed_ms=50.0,
                cost_usd=0.02,
            )
        )

    analyzer = StatisticsAnalyzer(seed=42)
    report = analyzer.analyze(records, telemetry, run_id="run_zerosucc")

    comp = report.paired_comparisons["structured_jev_vs_rag_llm"]
    # Even if latency speedup target is met (10x faster), overall_favorable_a MUST be False because 0 successes!
    assert comp.latency_target_met is True
    assert comp.overall_favorable_a is False


def test_bootstrap_resampling_10000_megaplan_spec():
    """Verify MEGAPLAN §17 and §21: 10,000 bootstrap resamples, 95% CIs, and paired tests."""
    from industrial_lab.benchmark.statistics import (
        DEFAULT_BOOTSTRAP_RESAMPLES,
        DEFAULT_CONFIDENCE_LEVEL,
    )

    # 1. Verify default constants strictly match MEGAPLAN §21
    assert DEFAULT_BOOTSTRAP_RESAMPLES == 10000
    assert DEFAULT_CONFIDENCE_LEVEL == 0.95

    # 2. Build paired cluster observations across 4 scenario families
    records: List[CaseScoreRecord] = []
    telemetry: List[TelemetryRecord] = []

    for i in range(12):
        fam_id = f"FAM-0{(i % 4) + 1}"
        case_id = f"C{i+1}"
        req_a = f"req_a_{i}"
        req_b = f"req_b_{i}"

        # Engine A (structured_jev): 100% success, 12ms latency, $0.005 cost
        records.append(
            CaseScoreRecord(
                run_id="run_boot10k",
                request_id=req_a,
                case_id=case_id,
                scenario_family_id=fam_id,
                scenario_id="S001",
                engine="structured_jev",
                execution_status=ExecutionStatus.completed,
                task_success=True,
                execution_completed=True,
                selection_correct=True,
                verdict_correct=True,
            )
        )
        telemetry.append(
            TelemetryRecord(
                run_id="run_boot10k",
                request_id=req_a,
                case_id=case_id,
                engine="structured_jev",
                elapsed_ms=12.0,
                cost_usd=0.005,
                cost_status="exact",
            )
        )

        # Engine B (rag_llm): 50% success, 60ms latency, $0.025 cost
        records.append(
            CaseScoreRecord(
                run_id="run_boot10k",
                request_id=req_b,
                case_id=case_id,
                scenario_family_id=fam_id,
                scenario_id="S001",
                engine="rag_llm",
                execution_status=ExecutionStatus.completed,
                task_success=(i % 2 == 0),
                execution_completed=True,
                selection_correct=(i % 2 == 0),
                verdict_correct=(i % 2 == 0),
            )
        )
        telemetry.append(
            TelemetryRecord(
                run_id="run_boot10k",
                request_id=req_b,
                case_id=case_id,
                engine="rag_llm",
                elapsed_ms=60.0,
                cost_usd=0.025,
                cost_status="exact",
            )
        )

    analyzer = StatisticsAnalyzer(seed=12345)  # uses default 10,000 resamples
    assert analyzer.bootstrap_resamples == 10000
    assert analyzer.confidence_level == 0.95

    report = analyzer.analyze(records, telemetry, run_id="run_boot10k")

    assert report.bootstrap_resamples == 10000
    assert report.confidence_level == 0.95
    assert report.num_families == 4
    assert report.num_cases == 12
    assert "structured_jev_vs_rag_llm" in report.paired_comparisons

    comp = report.paired_comparisons["structured_jev_vs_rag_llm"]

    # 95% Confidence Interval check
    assert comp.task_success.bootstrap_resamples == 10000
    assert comp.task_success.confidence_level == 0.95
    assert comp.task_success.point_difference == 0.5
    assert comp.task_success.ci_lower <= comp.task_success.ci_upper
    assert comp.task_success.ci_lower >= 0.0

    # Latency speedup: 60 / 12 = 5.0x
    assert comp.latency_speedup.point_difference == 5.0
    assert comp.latency_speedup.ci_lower <= comp.latency_speedup.ci_upper
    assert comp.latency_target_met is True

    # Cost reduction: (0.025 - 0.005) / 0.025 = 80% reduction
    assert comp.cost_reduction_ratio.point_difference == pytest.approx(0.80)
    assert comp.cost_reduction_ratio.ci_lower <= comp.cost_reduction_ratio.ci_upper
    assert comp.cost_target_met is True

    # Holm-Bonferroni adjusted p-values
    assert 0.0 <= comp.task_success.p_value <= 1.0
    assert 0.0 <= comp.task_success.p_value_holm <= 1.0
    assert comp.task_success.p_value_holm >= comp.task_success.p_value



# ==============================================================================
# 3. BenchmarkRunner Tests (§18)
# ==============================================================================

def test_runner_balanced_permutations():
    """Verify that runner generates all 6 balanced permutations for 3 engines (§18.1)."""
    engines = ["structured_jev", "rag_llm", "scrape_llm"]
    perms = BenchmarkRunner.get_all_engine_permutations(engines)
    assert len(perms) == 6
    assert len(set(tuple(p) for p in perms)) == 6

    runner = BenchmarkRunner(config_path=Path("non_existent_file.yaml"))
    runner.engines = engines

    # Verify balanced rotation across cases and repeats
    seen_perms = []
    for rep in range(1, 4):
        for case_idx in range(6):
            order = runner.get_balanced_order(f"C{case_idx}", case_idx, rep, engines)
            seen_perms.append(tuple(order))

    # All 6 permutations must appear in the schedule
    assert len(set(seen_perms)) == 6


def test_runner_strips_gold_field_leakage_protection():
    """CRITICAL: Verify create_public_request strictly strips gold field (§18.1)."""
    case = make_test_case(case_id="T100")
    runner = BenchmarkRunner(config_path=Path("non_existent_file.yaml"))

    request = runner.create_public_request(case, engine="structured_jev", repeat=1)

    req_dict = request.model_dump()
    assert "gold" not in req_dict
    assert "acceptable_selections" not in str(req_dict)
    assert "required_evidence_groups" not in str(req_dict)
    assert request.catalog_version == runner.catalog_version
    assert request.query_text == case.query_text


def test_runner_mode_validate():
    """Verify validate mode executes without calling inference models (§18.3)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir),
        )
        manifest = runner.run(mode="validate")
        assert manifest.status == "completed"
        assert manifest.mode == "validate"
        assert manifest.cases_count > 0
        assert manifest.total_scheduled_requests == 0


def test_runner_mode_smoke_end_to_end():
    """Verify smoke mode executes full pipeline and produces all required files (§18.3, §18.4)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir),
            allow_mock_fallback=True,
        )
        # Smoke runs 1 repetition on up to 2 cases
        manifest = runner.run(mode="smoke")

        assert manifest.status == "completed"
        assert manifest.completed_requests > 0

        # Check all required files (§18.4)
        run_dir = runner.run_dir
        assert (run_dir / "requests.jsonl").exists()
        assert (run_dir / "responses_raw.jsonl").exists()
        assert (run_dir / "responses_normalized.jsonl").exists()
        assert (run_dir / "spans.jsonl").exists()
        assert (run_dir / "run_manifest.json").exists()
        assert (run_dir / "order_schedule.json").exists()
        assert (run_dir / "scoring.jsonl").exists()
        assert (run_dir / "summary.csv").exists()
        assert (run_dir / "statistics.json").exists()

        # Verify requests.jsonl records do not leak gold
        with (run_dir / "requests.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                assert "gold" not in record["request"]


def test_runner_mode_replay():
    """Verify replay mode reconstructs scoring and stats without engine calls (§18.3)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir),
            allow_mock_fallback=True,
        )
        # 1. Run smoke to populate files
        runner.run(mode="smoke")

        # 2. Replay from the populated run directory
        replay_runner = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            run_id=runner.run_id,
            output_dir=Path(tmpdir),
            allow_mock_fallback=False,  # Engine calls forbidden in replay
        )
        replay_manifest = replay_runner.run(mode="replay")

        assert replay_manifest.status == "completed"
        assert (replay_runner.run_dir / "summary.csv").exists()
        assert (replay_runner.run_dir / "statistics.json").exists()


def test_runner_get_engine_structured_jev_never_mocked(monkeypatch: pytest.MonkeyPatch):
    """CRITICAL (§0.1 #7): structured_jev must NEVER fall back to a mock engine.

    Without TYPESAFE_API_KEY, it must execute via StructuredJevEngine returning provider_error 30.
    """
    for key in (
        "TYPESAFE_API_KEY",
        "LLM_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "JEV_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir),
            allow_mock_fallback=True,  # Even with allow_mock_fallback=True!
        )
        engine_fn = runner._get_engine("structured_jev")

        req = QueryRequest(
            request_id="req_test_jev",
            engine="structured_jev",
            query_text="Siemens Sinamics S120 7.5kW",
            mode="M1",
        )
        raw_resp, resp = engine_fn(req)

        assert resp.execution_status == ExecutionStatus.provider_error
        assert "30" in resp.summary or "JEV blocked" in resp.summary
        assert resp.engine_version != "mock-v1.0"
        assert resp.technical_verdict != TechnicalVerdict.COMPATIBLE
        assert "Mock response from structured_jev" not in str(resp.summary)


def test_runner_get_engine_structured_rules_supported():
    """Verify structured_rules is supported as an ablation engine in _get_engine."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir),
        )
        engine_fn = runner._get_engine("structured_rules")

        req = QueryRequest(
            request_id="req_test_rules",
            engine="structured_rules",
            query_text="Siemens Sinamics S120",
            mode="M1",
        )
        raw_resp, resp = engine_fn(req)

        assert resp.engine == "structured_rules"
        assert resp.execution_status == ExecutionStatus.completed


def test_runner_get_engine_dryrun_rag_and_scrape(monkeypatch: pytest.MonkeyPatch):
    """Verify rag_llm and scrape_llm in dry-run mode (LAB_DRY_RUN=1) execute _execute_dry_run()."""
    monkeypatch.setenv("LAB_DRY_RUN", "1")
    for key in (
        "TYPESAFE_API_KEY",
        "LLM_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "JEV_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir),
            allow_mock_fallback=False,  # Must not use mock engine!
        )

        # 1. RAG LLM
        rag_fn = runner._get_engine("rag_llm")
        req_rag = QueryRequest(
            request_id="req_rag_dry",
            engine="rag_llm",
            query_text="Siemens Sinamics S120",
            mode="M1",
        )
        raw_rag, resp_rag = rag_fn(req_rag)
        assert resp_rag.execution_status == ExecutionStatus.completed
        assert resp_rag.engine_version != "mock-v1.0"

        # 2. Scrape LLM
        scrape_fn = runner._get_engine("scrape_llm")
        req_scrape = QueryRequest(
            request_id="req_scrape_dry",
            engine="scrape_llm",
            query_text="Siemens Sinamics S120",
            mode="M1",
        )
        raw_scrape, resp_scrape = scrape_fn(req_scrape)
        assert resp_scrape.execution_status == ExecutionStatus.completed
        assert resp_scrape.engine_version != "mock-v1.0"


def test_runner_dryrun_prefixes_run_id_and_sets_manifest_metadata(monkeypatch: pytest.MonkeyPatch):
    """Verify LAB_DRY_RUN=1 or mode='dry-run' prefixes run_id with dryrun_ and sets manifest fields."""
    monkeypatch.setenv("LAB_DRY_RUN", "1")

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir),
            run_id="custom_experiment_test",
        )
        assert runner.run_id.startswith("dryrun_")

        manifest = runner.run(mode="dry-run")
        assert manifest.run_id.startswith("dryrun_")
        assert manifest.data_origin == "synthetic_fixture"
        assert manifest.is_official is False
        assert manifest.metadata.get("data_origin") == "synthetic_fixture"
        assert manifest.metadata.get("is_official") is False
        assert manifest.metadata.get("dry_run") is True

    # Also test mode='dry-run' without LAB_DRY_RUN in env
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir2:
        runner2 = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir2),
            run_id="unprefixed_run",
        )
        assert not runner2.run_id.startswith("dryrun_")

        manifest2 = runner2.run(mode="dry-run")
        assert runner2.run_id.startswith("dryrun_")
        assert manifest2.run_id.startswith("dryrun_")
        assert manifest2.data_origin == "synthetic_fixture"
        assert manifest2.is_official is False
        assert manifest2.metadata.get("data_origin") == "synthetic_fixture"
        assert manifest2.metadata.get("is_official") is False


# ==============================================================================
# 4. Workstream D Audit Tests (§21, §22, §18)
# ==============================================================================

def test_runner_split_dev_and_test_filtering():
    """Verify --split dev and --split test correctly filter cases and set manifest (§21, §22)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Test split="dev"
        runner_dev = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir),
            allow_mock_fallback=True,
        )
        manifest_dev = runner_dev.run(mode="smoke", split_override="dev")
        assert manifest_dev.split == "dev"
        assert (runner_dev.run_dir / "run_manifest.json").exists()

        # Check requests.jsonl recorded dev cases
        dev_cases = []
        with (runner_dev.run_dir / "requests.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                dev_cases.append(rec["case_id"])
        assert len(dev_cases) > 0
        assert all(c_id.startswith("DEV") for c_id in dev_cases)

        # 2. Test split="test"
        runner_test = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir),
            allow_mock_fallback=True,
        )
        manifest_test = runner_test.run(mode="smoke", split_override="test")
        assert manifest_test.split == "test"

        test_cases = []
        with (runner_test.run_dir / "requests.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                test_cases.append(rec["case_id"])
        assert len(test_cases) > 0
        assert all(c_id.startswith("TEST") for c_id in test_cases)


def test_runner_repetition_looping_and_schedule():
    """Verify repetition looping schedules across repeats and shuffles cases per repeat (§18.1)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir),
            allow_mock_fallback=True,
        )
        runner.engines = ["structured_rules"]

        repeats = 3
        manifest = runner.run(mode="dev", repeats_override=repeats, split_override="dev")

        assert manifest.repetitions == repeats
        assert manifest.completed_requests == manifest.cases_count * repeats * len(runner.engines)

        # Verify order_schedule.json contains all 3 repeats
        schedule_file = runner.run_dir / "order_schedule.json"
        assert schedule_file.exists()
        schedule = json.loads(schedule_file.read_text(encoding="utf-8"))
        scheduled_repeats = {entry["repeat"] for entry in schedule}
        assert scheduled_repeats == {1, 2, 3}


def test_runner_request_deadline_enforcement():
    """Verify request_deadline_seconds enforcement marks ExecutionStatus.timeout per MEGAPLAN §21 & §5.6."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=Path(tmpdir),
        )
        runner.request_deadline_seconds = 0.1  # 100ms tight deadline
        runner.engines = ["slow_engine"]

        def slow_engine_mock(req: QueryRequest) -> Tuple[Any, QueryResponse]:
            import time
            time.sleep(0.5)  # Intentionally exceed deadline
            return {}, QueryResponse(
                request_id=req.request_id,
                engine="slow_engine",
                engine_version="v1",
                execution_status=ExecutionStatus.completed,
                catalog_version=req.catalog_version,
                knowledge_version=req.knowledge_version,
                selected_product_ids=[],
                technical_verdict=TechnicalVerdict.COMPATIBLE,
                checks=[],
            )

        runner.register_engine("slow_engine", slow_engine_mock)

        manifest = runner.run(mode="smoke", split_override="dev")

        assert manifest.failed_requests > 0

        # Check responses_normalized.jsonl records timeout status
        norm_file = runner.run_dir / "responses_normalized.jsonl"
        with norm_file.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) > 0
        assert records[0]["response"]["execution_status"] == ExecutionStatus.timeout
        assert "timed out" in records[0]["response"]["summary"].lower()

        # Check spans.jsonl records timeout status
        span_file = runner.run_dir / "spans.jsonl"
        with span_file.open("r", encoding="utf-8") as f:
            spans = [json.loads(line) for line in f if line.strip()]
        assert spans[0]["status"] == "timeout"


def test_runner_lazy_directory_creation():
    """Verify runner does NOT create empty run directories on init or if early validation fails (§18.1)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Instantiation must NOT create directory
        runner = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=tmp_path,
        )
        assert not runner.run_dir.exists()
        assert len(list(tmp_path.iterdir())) == 0

        # 2. Early validation failure must NOT leave run directory
        invalid_case = BenchmarkCase(
            case_id="INVALID_01",
            scenario_family_id="FAM_01",
            scenario_id="S0001",
            split="dev",
            mode="M1",
            query_text="Invalid query without gold",
            gold=None,  # Intentionally missing gold to fail validate
        )

        runner_invalid = BenchmarkRunner(
            config_path=Path("configs/experiment.yaml"),
            output_dir=tmp_path,
        )
        with pytest.raises(AssertionError):
            runner_invalid.run(mode="validate", cases_override=[invalid_case])

        # Must NOT leave run directory behind on disk
        assert not runner_invalid.run_dir.exists()
        assert len(list(tmp_path.iterdir())) == 0


def test_runner_paired_execution_order_ab_ba_two_engines():
    """Verify paired execution order for 2 engines alternates AB and BA per MEGAPLAN §22 / §18.1."""
    engines = ["structured_jev", "structured_llm"]
    perms = BenchmarkRunner.get_all_engine_permutations(engines)

    # Must be exactly AB and BA
    assert len(perms) == 2
    assert perms == [["structured_jev", "structured_llm"], ["structured_llm", "structured_jev"]]

    runner = BenchmarkRunner(config_path=Path("non_existent_file.yaml"))
    runner.engines = engines

    # Case 0, Repeat 1 -> Permutation 0 (AB)
    order_c0_r1 = runner.get_balanced_order("C0", 0, 1, engines)
    assert order_c0_r1 == ["structured_jev", "structured_llm"]

    # Case 1, Repeat 1 -> Permutation 1 (BA)
    order_c1_r1 = runner.get_balanced_order("C1", 1, 1, engines)
    assert order_c1_r1 == ["structured_llm", "structured_jev"]

    # Case 0, Repeat 2 -> Permutation 1 (BA)
    order_c0_r2 = runner.get_balanced_order("C0", 0, 2, engines)
    assert order_c0_r2 == ["structured_llm", "structured_jev"]

    # Case 1, Repeat 2 -> Permutation 0 (AB)
    order_c1_r2 = runner.get_balanced_order("C1", 1, 2, engines)
    assert order_c1_r2 == ["structured_jev", "structured_llm"]


def test_runner_concurrency_controls():
    """Verify concurrency setting adheres to MEGAPLAN §21 execution: concurrency: 1."""
    runner = BenchmarkRunner(config_path=Path("configs/experiment.yaml"))
    assert runner.concurrency == 1
    assert runner.repetitions == 3
    assert runner.request_deadline_seconds == 60.0
    assert runner.provider_timeout_seconds == 30.0
