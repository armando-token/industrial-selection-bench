"""Unit tests for src/industrial_lab/schemas.py."""

import json
import pytest
from pydantic import ValidationError

from industrial_lab.schemas import (
    AnswerStatus,
    CheckResult,
    CheckStatus,
    CommercialData,
    DecisionOrigin,
    ExecutionStatus,
    ExtractionStatus,
    Fact,
    Port,
    QueryRequest,
    QueryResponse,
    Quote,
    QuoteLine,
    QuoteStatus,
    Relation,
    RelationType,
    Requirement,
    RequirementKind,
    SelectionStatus,
    SourceSpan,
    TaskType,
    TechnicalVerdict,
    aggregate_verdict,
    get_check_composite_key,
)


def test_enums():
    """Verify all required enum members and values."""
    assert ExtractionStatus.auto_extracted.value == "auto_extracted"
    assert ExtractionStatus.reviewed.value == "reviewed"
    assert ExtractionStatus.rejected.value == "rejected"
    assert ExtractionStatus.conflicting.value == "conflicting"

    assert RelationType.HAS_PORT.value == "HAS_PORT"
    assert RelationType.REQUIRES.value == "REQUIRES"
    assert RelationType.EXCLUDES.value == "EXCLUDES"
    assert RelationType.SUPPORTS_SIGNAL.value == "SUPPORTS_SIGNAL"
    assert RelationType.SUPPORTS_PROTOCOL.value == "SUPPORTS_PROTOCOL"
    assert RelationType.REQUIRES_ACCESSORY.value == "REQUIRES_ACCESSORY"
    assert RelationType.CONDITION_APPLIES_TO.value == "CONDITION_APPLIES_TO"
    assert RelationType.COMPATIBLE_WITH.value == "COMPATIBLE_WITH"

    assert RequirementKind.exact_property.value == "exact_property"
    assert RequirementKind.semantic_use_case.value == "semantic_use_case"
    assert RequirementKind.pair_compatibility.value == "pair_compatibility"
    assert RequirementKind.operating_condition.value == "operating_condition"
    assert RequirementKind.commercial.value == "commercial"

    assert CheckStatus.PASS.value == "PASS"
    assert CheckStatus.FAIL.value == "FAIL"
    assert CheckStatus.UNKNOWN.value == "UNKNOWN"

    assert TechnicalVerdict.COMPATIBLE.value == "COMPATIBLE"
    assert TechnicalVerdict.INCOMPATIBLE.value == "INCOMPATIBLE"
    assert TechnicalVerdict.INSUFFICIENT_EVIDENCE.value == "INSUFFICIENT_EVIDENCE"
    assert TechnicalVerdict.NOT_APPLICABLE.value == "NOT_APPLICABLE"
    assert TechnicalVerdict.UNDETERMINED.value == "UNDETERMINED"

    assert ExecutionStatus.OK.value == "OK"
    assert ExecutionStatus.completed.value == "completed"
    assert ExecutionStatus.PROVIDER_ERROR.value == "PROVIDER_ERROR"
    assert ExecutionStatus.provider_error.value == "provider_error"
    assert ExecutionStatus.SCHEMA_ERROR.value == "SCHEMA_ERROR"
    assert ExecutionStatus.schema_error.value == "schema_error"
    assert ExecutionStatus.TIMEOUT.value == "TIMEOUT"
    assert ExecutionStatus.timeout.value == "timeout"
    assert ExecutionStatus.BUDGET_EXCEEDED.value == "BUDGET_EXCEEDED"
    assert ExecutionStatus.budget_exceeded.value == "budget_exceeded"
    assert ExecutionStatus.INTERNAL_ERROR.value == "INTERNAL_ERROR"
    assert ExecutionStatus.internal_error.value == "internal_error"
    assert ExecutionStatus.invalid_output.value == "invalid_output"
    assert ExecutionStatus.dependency_error.value == "dependency_error"

    # AnswerStatus (§6, REPAIR3_PLAN §6.2)
    assert AnswerStatus.COMPLETE.value == "COMPLETE"
    assert AnswerStatus.PARTIAL.value == "PARTIAL"
    assert AnswerStatus.INSUFFICIENT_EVIDENCE.value == "INSUFFICIENT_EVIDENCE"
    assert AnswerStatus.UNAVAILABLE.value == "UNAVAILABLE"

    # SelectionStatus (§6, REPAIR3_PLAN §6.2)
    assert SelectionStatus.SATISFIED.value == "SATISFIED"
    assert SelectionStatus.UNSATISFIED.value == "UNSATISFIED"
    assert SelectionStatus.UNDETERMINED.value == "UNDETERMINED"
    assert SelectionStatus.NOT_APPLICABLE.value == "NOT_APPLICABLE"

    assert DecisionOrigin.rule.value == "rule"
    assert DecisionOrigin.jev.value == "jev"
    assert DecisionOrigin.llm.value == "llm"
    assert DecisionOrigin.combined.value == "combined"

    assert QuoteStatus.preliminary.value == "preliminary"
    assert QuoteStatus.requires_technical_review.value == "requires_technical_review"
    assert QuoteStatus.unavailable.value == "unavailable"
    assert QuoteStatus.expired.value == "expired"


def test_extra_forbid():
    """Verify that models reject unknown / extra fields."""
    with pytest.raises(ValidationError):
        SourceSpan(
            span_id="D1:p1:s01",
            document_id="D1",
            document_sha256="abc",
            pdf_page_index=1,
            text="text",
            revision="r1",
            unauthorized_field="illegal",
        )

    with pytest.raises(ValidationError):
        Fact(
            fact_id="F1",
            product_id="P1",
            property="voltage",
            value=24,
            document_revision="r1",
            extra_field="boom",
        )

    with pytest.raises(ValidationError):
        Requirement(
            requirement_id="R1",
            kind=RequirementKind.exact_property,
            operator="eq",
            target=24,
            foo="bar",
        )


def test_source_span_serialization():
    """Verify SourceSpan creation, serialization, and deserialization."""
    span = SourceSpan(
        span_id="D1:p12:s03",
        document_id="D1",
        document_sha256="a" * 64,
        pdf_page_index=12,
        printed_page_label="10",
        text="Supply voltage: 24 VDC nominal",
        bbox=[10.5, 20.0, 100.2, 50.8],
        table_id=None,
        row_header=None,
        column_header=None,
        product_scope=["P1"],
        revision="rev2",
    )
    data = span.to_dict()
    assert data["span_id"] == "D1:p12:s03"
    assert data["pdf_page_index"] == 12

    json_str = span.to_json()
    loaded = SourceSpan.from_json(json_str)
    assert loaded == span


def test_fact_model():
    """Verify Fact model with complex value and conditions."""
    fact = Fact(
        fact_id="F0001",
        product_id="P1",
        port_id="port_power",
        property="supply_voltage_nominal",
        value=24,
        unit="VDC",
        conditions=[{"ambient_temp_max_c": 55}],
        variant_scope=["P1-STD"],
        evidence_ids=["D1:p12:s03"],
        extraction_status=ExtractionStatus.reviewed,
        reviewed_by="engineer_alice",
        document_revision="rev2",
    )
    assert fact.value == 24
    assert fact.extraction_status == ExtractionStatus.reviewed


def test_port_and_relation():
    """Verify Port and Relation models."""
    port = Port(
        port_id="P1:DI0",
        product_id="P1",
        direction="in",
        physical_interface="terminal_block",
        signal_kind="digital_24v",
        protocol=None,
        role="digital_input",
        supported_ranges=[{"voltage_min": 15, "voltage_max": 30}],
        wiring_conditions=["sink_or_source"],
        evidence_ids=["D1:p14:s01"],
    )
    assert port.direction == "in"

    rel = Relation(
        relation_id="REL001",
        source_id="P1",
        target_id="P2",
        relation_type=RelationType.REQUIRES,
        conditions=[{"mandatory": True}],
        evidence_ids=["D1:p20:s05"],
    )
    assert rel.relation_type == RelationType.REQUIRES


def test_query_request_defaults():
    """Verify QueryRequest default values."""
    req = QueryRequest(
        request_id="req-123",
        query_text="Is sensor P1 compatible with controller P2?",
        engine="structured_jev",
    )
    assert req.mode == "M1"
    assert req.scenario_id == "S0001"
    assert req.catalog_version == "v1"
    assert req.knowledge_version == "v1"
    assert req.include_quote is False
    assert req.render_mode == "template"


def test_quote_and_lines_computation():
    """Verify Quote and QuoteLine total computation."""
    line1 = QuoteLine(
        product_id="P1",
        quantity=2,
        unit_price_minor=1500,  # 15.00
    )
    assert line1.line_total_minor == 3000

    line2 = QuoteLine(
        product_id="P2",
        quantity=1,
        unit_price_minor=4500,
        line_total_minor=4500,
    )

    quote = Quote(
        status=QuoteStatus.preliminary,
        lines=[line1, line2],
        currency="USD",
        revision="commerce-v1",
        expires_at_simulated="2026-12-31T23:59:59Z",
    )
    assert quote.subtotal_minor == 7500
    assert quote.tax_minor == 0
    assert quote.total_minor == 7500


def test_commercial_data():
    """Verify CommercialData model."""
    comm = CommercialData(
        product_id="P1",
        price_minor=25000,
        currency="USD",
        stock_available=12,
        revision="commerce-v1",
        observed_at="2026-10-02T12:00:00Z",
    )
    assert comm.is_simulated is True
    assert comm.stock_available == 12


def test_query_response():
    """Verify QueryResponse full schema validation."""
    resp = QueryResponse(
        request_id="req-123",
        engine="structured_jev",
        engine_version="git-sha-test",
        catalog_version="v1",
        knowledge_version="v1",
        technical_verdict=TechnicalVerdict.COMPATIBLE,
        checks=[
            CheckResult(
                requirement_id="R1",
                status=CheckStatus.PASS,
                evidence_ids=["D1:p12:s03"],
                reason_code=None,
                decision_origin=DecisionOrigin.rule,
            )
        ],
        summary="Compatible with tested requirements.",
    )
    assert resp.schema_version == "3"
    assert resp.execution_status == ExecutionStatus.OK
    assert resp.technical_verdict == TechnicalVerdict.COMPATIBLE
    assert resp.content_coverage_complete is False
    assert len(resp.checks) == 1


# ==============================================================================
# Tests for Three-Valued Verdict Aggregation (§5.5)
# ==============================================================================

def test_aggregate_verdict_all_pass():
    reqs = [
        Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target=24, hard=True),
        Requirement(requirement_id="R2", kind=RequirementKind.operating_condition, operator="gte", target=-10, hard=True),
    ]
    checks = [
        CheckResult(requirement_id="R1", status=CheckStatus.PASS, decision_origin=DecisionOrigin.rule),
        CheckResult(requirement_id="R2", status=CheckStatus.PASS, decision_origin=DecisionOrigin.jev),
    ]
    assert aggregate_verdict(checks, reqs) == TechnicalVerdict.COMPATIBLE


def test_aggregate_verdict_hard_fail_overrides_unknown():
    """MEGAPLAN §5.5 & §24.1: FAIL + UNKNOWN -> INCOMPATIBLE."""
    reqs = [
        Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target=24, hard=True),
        Requirement(requirement_id="R2", kind=RequirementKind.operating_condition, operator="gte", target=-10, hard=True),
    ]
    checks = [
        CheckResult(requirement_id="R1", status=CheckStatus.FAIL, decision_origin=DecisionOrigin.rule),
        CheckResult(requirement_id="R2", status=CheckStatus.UNKNOWN, decision_origin=DecisionOrigin.rule),
    ]
    assert aggregate_verdict(checks, reqs) == TechnicalVerdict.INCOMPATIBLE


def test_aggregate_verdict_unknown_gives_insufficient_evidence():
    """MEGAPLAN §5.5: PASS + UNKNOWN (no FAIL) -> INSUFFICIENT_EVIDENCE."""
    reqs = [
        Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target=24, hard=True),
        Requirement(requirement_id="R2", kind=RequirementKind.operating_condition, operator="gte", target=-10, hard=True),
    ]
    checks = [
        CheckResult(requirement_id="R1", status=CheckStatus.PASS, decision_origin=DecisionOrigin.rule),
        CheckResult(requirement_id="R2", status=CheckStatus.UNKNOWN, decision_origin=DecisionOrigin.rule),
    ]
    assert aggregate_verdict(checks, reqs) == TechnicalVerdict.INSUFFICIENT_EVIDENCE


def test_aggregate_verdict_soft_requirement_fail_does_not_break_compatibility():
    """Soft requirements (hard=False) should not force INCOMPATIBLE if all hard pass."""
    reqs = [
        Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target=24, hard=True),
        Requirement(requirement_id="R2", kind=RequirementKind.commercial, operator="lte", target=1000, hard=False),
    ]
    checks = [
        CheckResult(requirement_id="R1", status=CheckStatus.PASS, decision_origin=DecisionOrigin.rule),
        CheckResult(requirement_id="R2", status=CheckStatus.FAIL, decision_origin=DecisionOrigin.rule),
    ]
    assert aggregate_verdict(checks, reqs) == TechnicalVerdict.COMPATIBLE


def test_aggregate_verdict_missing_check_for_hard_requirement():
    """If a hard requirement is not evaluated in checks, evidence is missing -> INSUFFICIENT_EVIDENCE."""
    reqs = [
        Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target=24, hard=True),
        Requirement(requirement_id="R2", kind=RequirementKind.exact_property, operator="eq", target="modbus", hard=True),
    ]
    checks = [
        CheckResult(requirement_id="R1", status=CheckStatus.PASS, decision_origin=DecisionOrigin.rule),
    ]
    assert aggregate_verdict(checks, reqs) == TechnicalVerdict.INSUFFICIENT_EVIDENCE


def test_aggregate_verdict_checks_only():
    """Checks provided without explicit requirements list (all treated as hard)."""
    checks_pass = [
        CheckResult(requirement_id="R1", status=CheckStatus.PASS, decision_origin=DecisionOrigin.rule),
    ]
    assert aggregate_verdict(checks_pass) == TechnicalVerdict.COMPATIBLE

    checks_fail = [
        CheckResult(requirement_id="R1", status=CheckStatus.FAIL, decision_origin=DecisionOrigin.rule),
    ]
    assert aggregate_verdict(checks_fail) == TechnicalVerdict.INCOMPATIBLE

    checks_unknown = [
        CheckResult(requirement_id="R1", status=CheckStatus.UNKNOWN, decision_origin=DecisionOrigin.rule),
    ]
    assert aggregate_verdict(checks_unknown) == TechnicalVerdict.INSUFFICIENT_EVIDENCE


def test_check_result_composite_identity():
    """REPAIR3_PLAN §6.2: Checks identified by composite key (req_id, role_id, prod_id, var_id)."""
    chk = CheckResult(
        requirement_id="R_VOLT",
        status=CheckStatus.PASS,
        decision_origin=DecisionOrigin.rule,
        role_id="MAIN_PLC",
        product_id="HORNER_X4",
        variant_id="HE-X4A",
    )
    assert chk.role_id == "MAIN_PLC"
    assert chk.product_id == "HORNER_X4"
    assert chk.variant_id == "HE-X4A"
    assert chk.check_key == ("R_VOLT", "MAIN_PLC", "HORNER_X4", "HE-X4A")
    assert chk.composite_key == ("R_VOLT", "MAIN_PLC", "HORNER_X4", "HE-X4A")
    assert get_check_composite_key(chk) == ("R_VOLT", "MAIN_PLC", "HORNER_X4", "HE-X4A")

    # Optional defaults to None
    chk_default = CheckResult(
        requirement_id="R_VOLT",
        status=CheckStatus.UNKNOWN,
        decision_origin=DecisionOrigin.rule,
    )
    assert chk_default.role_id is None
    assert chk_default.product_id is None
    assert chk_default.variant_id is None
    assert chk_default.check_key == ("R_VOLT", None, None, None)


def test_schema_version_migration_and_deserialization():
    """REPAIR3_PLAN §6.1: Default schema_version '3' while accepting '2' on deserialization."""
    # 1. Default when creating fresh
    resp_fresh = QueryResponse(
        request_id="req-v3",
        engine="test_engine",
    )
    assert resp_fresh.schema_version == "3"

    # 2. Deserializing legacy version 2 payload
    v2_json = json.dumps({
        "schema_version": "2",
        "request_id": "req-v2",
        "engine": "legacy_engine",
        "execution_status": "completed",
        "answer_status": "COMPLETE",
        "selection_status": "SATISFIED",
        "technical_verdict": "COMPATIBLE",
    })
    resp_v2 = QueryResponse.model_validate_json(v2_json)
    assert resp_v2.schema_version == "2"
    assert resp_v2.engine == "legacy_engine"

    # 3. Deserializing version 3 payload
    v3_json = json.dumps({
        "schema_version": "3",
        "request_id": "req-v3",
        "engine": "v3_engine",
        "execution_status": "OK",
        "answer_status": "COMPLETE",
        "selection_status": "NOT_APPLICABLE",
    })
    resp_v3 = QueryResponse.model_validate_json(v3_json)
    assert resp_v3.schema_version == "3"

    # 4. Unsupported schema_version rejected
    with pytest.raises(ValidationError):
        QueryResponse(
            schema_version="99",
            request_id="req-bad",
            engine="bad_engine",
        )


def test_technical_verdict_nullable_and_informational():
    """REPAIR3_PLAN §6.2: technical_verdict nullable or NOT_APPLICABLE for informational lookups."""
    # 1. Explicitly nullable
    resp_null = QueryResponse(
        request_id="req-null",
        engine="test_engine",
        technical_verdict=None,
    )
    assert resp_null.technical_verdict is None

    # 2. FACT_LOOKUP task defaults to NOT_APPLICABLE
    resp_info = QueryResponse(
        request_id="req-fact",
        engine="test_engine",
        task_type=TaskType.FACT_LOOKUP,
    )
    assert resp_info.technical_verdict == TechnicalVerdict.NOT_APPLICABLE
    assert resp_info.selection_status == SelectionStatus.NOT_APPLICABLE

    # 3. aggregate_verdict on FACT_LOOKUP returns NOT_APPLICABLE
    verdict = aggregate_verdict([], [], task_type=TaskType.FACT_LOOKUP)
    assert verdict == TechnicalVerdict.NOT_APPLICABLE


def test_operational_error_never_incompatible():
    """REPAIR3_PLAN §6.2: Operational errors (SCHEMA_ERROR, etc.) must NEVER produce INCOMPATIBLE."""
    # Attempting to set INCOMPATIBLE with SCHEMA_ERROR gets coerced to UNDETERMINED
    resp_schema_err = QueryResponse(
        request_id="req-schema-err",
        engine="rag_llm",
        execution_status=ExecutionStatus.SCHEMA_ERROR,
        technical_verdict=TechnicalVerdict.INCOMPATIBLE,
    )
    assert resp_schema_err.execution_status == ExecutionStatus.SCHEMA_ERROR
    assert resp_schema_err.technical_verdict != TechnicalVerdict.INCOMPATIBLE
    assert resp_schema_err.technical_verdict == TechnicalVerdict.UNDETERMINED
    assert resp_schema_err.selection_status == SelectionStatus.UNDETERMINED
    assert resp_schema_err.answer_status == AnswerStatus.UNAVAILABLE

    # INTERNAL_ERROR also never INCOMPATIBLE
    resp_internal_err = QueryResponse(
        request_id="req-internal-err",
        engine="structured_jev",
        execution_status=ExecutionStatus.INTERNAL_ERROR,
        technical_verdict=TechnicalVerdict.INCOMPATIBLE,
    )
    assert resp_internal_err.technical_verdict == TechnicalVerdict.UNDETERMINED
    assert resp_internal_err.selection_status == SelectionStatus.UNDETERMINED


def test_content_coverage_complete_flag():
    """REPAIR3_PLAN §6.2: Add content_coverage_complete (bool) to QueryResponse."""
    resp_default = QueryResponse(
        request_id="req-cov",
        engine="test_engine",
    )
    assert resp_default.content_coverage_complete is False

    resp_covered = QueryResponse(
        request_id="req-cov-2",
        engine="test_engine",
        content_coverage_complete=True,
    )
    assert resp_covered.content_coverage_complete is True
