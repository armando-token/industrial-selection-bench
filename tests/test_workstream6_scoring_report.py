"""Unit tests for Workstream 6: Gold dataset contract, scorer, and single-source report generation.

Adheres strictly to REPAIR3_PLAN.md §10, §11, §15, and fixes F08, F09, F18:
1. F08: DATASET_SCHEMA_ERROR on null/missing requirement_id; clean matching by ID/property/product.
2. F09: Single-source scoring metrics; content_fully_correct and operationally_valid decoupling.
3. F18: false_approval_rate returns None / 'N/A' when denominator is 0, NEVER 0.0%.
4. Informational queries (FACT_LOOKUP, VARIANT_COMPARISON) have selection_accuracy = None (N/A).
5. ReportBuilder single-source consistency across report.md, summary.csv, scorecard.json, comparison.md, INFORME_FINAL.md.
"""

import json
from pathlib import Path
import pytest

from industrial_lab.benchmark.scoring import (
    Evaluator,
    CaseScoreRecord,
    validate_dataset,
)
from industrial_lab.observability.spans import TelemetryRecord
from industrial_lab.report.build import ReportBuilder, ReportData, EngineMetrics
from industrial_lab.schemas import (
    BenchmarkCase,
    CheckResult,
    CheckStatus,
    DatasetSchemaError,
    DecisionOrigin,
    ExecutionStatus,
    Fact,
    GoldCase,
    QueryResponse,
    TaskType,
    TechnicalVerdict,
)


# ==============================================================================
# 1. F08: Dataset Contract & Schema Validation Tests
# ==============================================================================

def test_f08_gold_case_raises_dataset_schema_error_on_missing_requirement_id():
    """Verify that a GoldCase with null or missing requirement_id raises DatasetSchemaError with DATASET_SCHEMA_ERROR."""
    with pytest.raises((DatasetSchemaError, ValueError), match="DATASET_SCHEMA_ERROR"):
        GoldCase(
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            required_checks=[{"check_name": "voltage_match", "status": "PASS"}],  # missing requirement_id
        )

    with pytest.raises((DatasetSchemaError, ValueError), match="DATASET_SCHEMA_ERROR"):
        GoldCase(
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            required_checks=[{"requirement_id": "", "status": "PASS"}],  # empty requirement_id
        )

    with pytest.raises((DatasetSchemaError, ValueError), match="DATASET_SCHEMA_ERROR"):
        GoldCase(
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            expected_items=[{"check_name": "temp_check"}],  # missing requirement_id in expected_items
        )


def test_f08_validate_dataset_enforces_contract():
    """Verify validate_dataset catches invalid cases and passes valid cases."""
    valid_case = BenchmarkCase(
        case_id="VALID_01",
        scenario_family_id="FAM_01",
        scenario_id="S01",
        query_text="Test query",
        gold=GoldCase(
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            required_checks=[{"requirement_id": "REQ_01", "status": "PASS"}],
        ),
    )
    validate_dataset([valid_case])  # must not raise

    # Case with invalid check in gold dict
    case_invalid = BenchmarkCase.model_construct(
        case_id="INVALID_01",
        scenario_family_id="FAM_01",
        scenario_id="S01",
        query_text="Test query",
        gold=GoldCase.model_construct(
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            required_checks=[{"check_name": "legacy_name"}],  # no requirement_id
        ),
    )
    with pytest.raises(DatasetSchemaError, match="DATASET_SCHEMA_ERROR"):
        validate_dataset([case_invalid])


def test_f08_clean_requirement_matching_exact_and_property():
    """Verify clean requirement matching in Evaluator.find_matching_check across exact ID, property, and facts."""
    evaluator = Evaluator()

    # Response with checks
    resp = QueryResponse(
        request_id="req_01",
        engine="rag_llm",
        execution_status=ExecutionStatus.completed,
        technical_verdict=TechnicalVerdict.COMPATIBLE,
        checks=[
            CheckResult(
                requirement_id="REQ_SUPPLY_VOLTAGE",
                status=CheckStatus.PASS,
                evidence_ids=["DOC1:p1:s1"],
                decision_origin=DecisionOrigin.rule,
                product_id="P_PLC",
            ),
            CheckResult(
                requirement_id="REQ_COMM_PROTOCOL",
                status=CheckStatus.PASS,
                evidence_ids=["DOC1:p2:s1"],
                decision_origin=DecisionOrigin.llm,
                product_id="P_SENSOR",
            ),
        ],
    )

    # 1. Exact requirement_id match
    match1 = evaluator.find_matching_check({"requirement_id": "REQ_SUPPLY_VOLTAGE"}, resp)
    assert match1 is not None
    assert match1.requirement_id == "REQ_SUPPLY_VOLTAGE"

    # 2. Normalized / core requirement_id match (case-insensitive, hyphen/underscore)
    match2 = evaluator.find_matching_check({"requirement_id": "req-supply-voltage"}, resp)
    assert match2 is not None
    assert match2.requirement_id == "REQ_SUPPLY_VOLTAGE"

    # 3. Product-specific match when multiple exist
    resp_multi = QueryResponse(
        request_id="req_multi",
        engine="rag_llm",
        execution_status=ExecutionStatus.completed,
        technical_verdict=TechnicalVerdict.COMPATIBLE,
        checks=[
            CheckResult(
                requirement_id="REQ_VOLTAGE",
                status=CheckStatus.FAIL,
                product_id="P_WRONG",
                decision_origin=DecisionOrigin.rule,
            ),
            CheckResult(
                requirement_id="REQ_VOLTAGE",
                status=CheckStatus.PASS,
                product_id="P_RIGHT",
                evidence_ids=["DOC1:p1:s1"],
                decision_origin=DecisionOrigin.rule,
            ),
        ],
    )
    match3 = evaluator.find_matching_check(
        {"requirement_id": "REQ_VOLTAGE", "target_product_id": "P_RIGHT", "status": "PASS"},
        resp_multi,
    )
    assert match3 is not None
    assert match3.product_id == "P_RIGHT"
    assert match3.status == CheckStatus.PASS

    # 4. Fallback to facts in natural modality
    resp_facts = QueryResponse(
        request_id="req_facts",
        engine="scrape_llm",
        execution_status=ExecutionStatus.completed,
        technical_verdict=TechnicalVerdict.COMPATIBLE,
        checks=[],
        facts=[
            Fact(
                fact_id="fact_01",
                product_id="P_THT",
                property="communication_protocol",
                value="Modbus-RTU",
                evidence_ids=["DOC2:p1:s1"],
            )
        ],
    )
    match4 = evaluator.find_matching_check(
        {"requirement_id": "REQ_SENSOR_PROTOCOL", "property": "communication_protocol", "product_id": "P_THT"},
        resp_facts,
    )
    assert match4 is not None
    assert match4.status == CheckStatus.PASS
    assert match4.evidence_ids == ["DOC2:p1:s1"]


# ==============================================================================
# 2. F09: Metric Decoupling (content_fully_correct, operationally_valid, task_success)
# ==============================================================================

def test_f09_metrics_decoupling_completed_and_correct():
    """Case 1: Fully completed execution and correct content -> task_success = True."""
    evaluator = Evaluator(valid_spans={"DOC1:p1:s1"})
    case = BenchmarkCase(
        case_id="C_OK",
        scenario_family_id="F1",
        scenario_id="S1",
        query_text="Select compatible controller",
        gold=GoldCase(
            acceptable_selections=[["P1"]],
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            required_checks=[{"requirement_id": "REQ_V", "status": "PASS"}],
            required_evidence_groups=[["DOC1:p1:s1"]],
        ),
    )
    resp = QueryResponse(
        request_id="r_ok",
        engine="structured_jev",
        execution_status=ExecutionStatus.completed,
        selected_product_ids=["P1"],
        technical_verdict=TechnicalVerdict.COMPATIBLE,
        checks=[
            CheckResult(
                requirement_id="REQ_V",
                status=CheckStatus.PASS,
                evidence_ids=["DOC1:p1:s1"],
                decision_origin=DecisionOrigin.rule,
            )
        ],
    )
    score = evaluator.evaluate_case(case, resp)
    assert score.operationally_valid is True
    assert score.content_fully_correct is True
    assert score.task_success is True


def test_f09_metrics_decoupling_budget_exceeded_with_correct_content():
    """Case 2 (C Q3 scenario): Budget exceeded (operational failure) but content is correct.

    content_fully_correct = True, operationally_valid = False -> task_success = False.
    Eliminates contradiction: explains why factual grade was correct while primary success was 0.
    """
    evaluator = Evaluator(valid_spans={"DOC1:p1:s1"})
    case = BenchmarkCase(
        case_id="C_BUDGET",
        scenario_family_id="F1",
        scenario_id="S1",
        query_text="Informational query on heater",
        task_type=TaskType.FACT_LOOKUP,
        gold=GoldCase(
            acceptable_selections=[],
            selection_required=False,
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            required_checks=[{"requirement_id": "REQ_HEATER_V", "status": "PASS"}],
            required_evidence_groups=[["DOC1:p1:s1"]],
        ),
    )
    # Model reached budget limit
    resp = QueryResponse(
        request_id="r_budget",
        engine="scrape_llm",
        execution_status=ExecutionStatus.budget_exceeded,
        selected_product_ids=[],
        technical_verdict=TechnicalVerdict.COMPATIBLE,
        checks=[
            CheckResult(
                requirement_id="REQ_HEATER_V",
                status=CheckStatus.PASS,
                evidence_ids=["DOC1:p1:s1"],
                decision_origin=DecisionOrigin.llm,
            )
        ],
    )
    score = evaluator.evaluate_case(case, resp)
    assert score.operationally_valid is False
    assert score.content_fully_correct is True
    assert score.task_success is False  # task_success requires both!


def test_f09_metrics_decoupling_completed_with_incorrect_content():
    """Case 3: Execution completed but wrong technical content.

    content_fully_correct = False, operationally_valid = True -> task_success = False.
    """
    evaluator = Evaluator()
    case = BenchmarkCase(
        case_id="C_WRONG",
        scenario_family_id="F1",
        scenario_id="S1",
        query_text="Selection query",
        gold=GoldCase(
            acceptable_selections=[["P1"]],
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            required_checks=[{"requirement_id": "REQ_V", "status": "PASS"}],
        ),
    )
    resp = QueryResponse(
        request_id="r_wrong",
        engine="rag_llm",
        execution_status=ExecutionStatus.completed,
        selected_product_ids=["P2"],  # wrong selection!
        technical_verdict=TechnicalVerdict.INCOMPATIBLE,  # wrong verdict!
        checks=[
            CheckResult(
                requirement_id="REQ_V",
                status=CheckStatus.FAIL,
                decision_origin=DecisionOrigin.rule,
            )
        ],
    )
    score = evaluator.evaluate_case(case, resp)
    assert score.operationally_valid is True
    assert score.content_fully_correct is False
    assert score.task_success is False


# ==============================================================================
# 3. F18: Safety Metric — False Approval Rate Null Handling
# ==============================================================================

def test_f18_false_approval_rate_none_when_zero_incompatible_cases():
    """Verify false_approval_rate is None (N/A) when there are 0 non-compatible gold cases, NEVER 0.0%."""
    evaluator = Evaluator()
    # A case where gold IS COMPATIBLE
    case = BenchmarkCase(
        case_id="C_COMPAT",
        scenario_family_id="F1",
        scenario_id="S1",
        query_text="Compatible case",
        gold=GoldCase(
            acceptable_selections=[["P1"]],
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            required_checks=[{"requirement_id": "REQ_1", "status": "PASS"}],
        ),
    )
    resp = QueryResponse(
        request_id="r1",
        engine="structured_jev",
        execution_status=ExecutionStatus.completed,
        selected_product_ids=["P1"],
        technical_verdict=TechnicalVerdict.COMPATIBLE,
        checks=[
            CheckResult(
                requirement_id="REQ_1",
                status=CheckStatus.PASS,
                evidence_ids=["DOC1:p1:s1"],
                decision_origin=DecisionOrigin.rule,
            )
        ],
    )
    score = evaluator.evaluate_case(case, resp)
    summary = evaluator.aggregate([score])
    engine_summary = summary.engine_summaries["structured_jev"]

    # Incompatible denominator is 0 -> false_approval_rate.rate MUST be None
    assert engine_summary.false_approval_rate.denominator == 0
    assert engine_summary.false_approval_rate.rate is None

    # In EngineMetrics (report builder)
    m = EngineMetrics(
        engine_id="structured_jev",
        display_name="Engine A",
        total_cases=1,
        incompatible_cases=0,
        false_approvals=0,
    )
    assert m.false_approval_rate_pct is None


# ==============================================================================
# 4. Informational Queries: selection_accuracy = None (N/A)
# ==============================================================================

def test_informational_queries_have_selection_accuracy_none():
    """Verify that FACT_LOOKUP and VARIANT_COMPARISON have selection_accuracy = None (N/A)."""
    evaluator = Evaluator(valid_spans={"DOC1:p1:s1"})
    case = BenchmarkCase(
        case_id="Q2_LOOKUP",
        scenario_family_id="F1",
        scenario_id="S1",
        query_text="What protocol does the sensor speak?",
        task_type=TaskType.FACT_LOOKUP,
        gold=GoldCase(
            acceptable_selections=[],
            selection_required=False,
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            required_checks=[{"requirement_id": "REQ_PROT", "status": "PASS"}],
            required_evidence_groups=[["DOC1:p1:s1"]],
        ),
    )
    resp = QueryResponse(
        request_id="r_lookup",
        engine="scrape_llm",
        execution_status=ExecutionStatus.completed,
        selected_product_ids=[],  # No selection
        technical_verdict=TechnicalVerdict.COMPATIBLE,
        checks=[
            CheckResult(
                requirement_id="REQ_PROT",
                status=CheckStatus.PASS,
                evidence_ids=["DOC1:p1:s1"],
                decision_origin=DecisionOrigin.llm,
            )
        ],
    )
    score = evaluator.evaluate_case(case, resp)
    assert score.details["selection_applicable"] is False

    summary = evaluator.aggregate([score])
    eng_summary = summary.engine_summaries["scrape_llm"]
    # Selection accuracy should NOT be 1.0 or 0.0, but None (NOT_APPLICABLE)
    assert eng_summary.selection_accuracy.denominator == 0
    assert eng_summary.selection_accuracy.rate is None


# ==============================================================================
# 5. Single-Source Report Deliverables Consistency
# ==============================================================================

def test_single_source_report_generation_consistency(tmp_path: Path):
    """Verify ReportBuilder generates report.md, report.html, summary.csv, scorecard.json, comparison.md, and INFORME_FINAL.md without discrepancies."""
    data = ReportData(
        run_id="run_single_source_test",
        metadata={"is_official": False, "mode": "test"},
    )
    data.engines["test_engine"] = EngineMetrics(
        engine_id="test_engine",
        display_name="Engine Test",
        total_cases=4,
        total_requests=4,
        successful_tasks=2,
        content_fully_correct_tasks=3,
        operationally_valid_tasks=2,
        false_approvals=0,
        incompatible_cases=0,  # 0 non-compatibles
        excessive_abstentions=0,
        resolvable_cases=4,
        latencies_ms=[100.0, 150.0, 200.0, 250.0],
        total_cost_usd=0.015,
        cost_status="exact",
    )

    builder = ReportBuilder(data)
    md_path, html_path = builder.build_and_save(tmp_path)

    # 1. Verify all 6 deliverables exist
    assert md_path.exists()
    assert html_path.exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "scorecard.json").exists()
    assert (tmp_path / "comparison.md").exists()
    assert (tmp_path / "INFORME_FINAL.md").exists()

    # 2. Check summary.csv
    csv_content = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "task_success_rate" in csv_content
    assert "content_fully_correct_rate" in csv_content
    assert "operationally_valid_rate" in csv_content
    assert "0.5000" in csv_content  # 2/4 = 0.5000 task_success_rate
    assert "0.7500" in csv_content  # 3/4 = 0.7500 content_fully_correct_rate
    assert "N/A" in csv_content     # false_approval_rate is N/A because incompatible_cases = 0

    # 3. Check scorecard.json
    scorecard = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    eng_card = scorecard["engines"]["test_engine"]
    assert eng_card["successful_tasks"] == 2
    assert eng_card["task_success_rate_pct"] == 50.0
    assert eng_card["content_fully_correct_tasks"] == 3
    assert eng_card["content_fully_correct_rate_pct"] == 75.0
    assert eng_card["operationally_valid_tasks"] == 2
    assert eng_card["false_approval_rate_pct"] is None  # N/A

    # 4. Check comparison.md
    comp_content = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "3/4 (75.0%)" in comp_content  # Content fully correct
    assert "2/4 (50.0%)" in comp_content  # Operationally valid and Task success
    assert "N/A (sin casos no compatibles)" in comp_content

    # 5. Check INFORME_FINAL.md
    informe_content = (tmp_path / "INFORME_FINAL.md").read_text(encoding="utf-8")
    assert "3/4 (75.0%)" in informe_content
    assert "2/4 (50.0%)" in informe_content
    assert "N/A" in informe_content
