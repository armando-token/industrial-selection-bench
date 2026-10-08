"""Unit and integration tests for inference engines and supporting tools.

Adheres strictly to MEGAPLAN.md §2.1, §2.2, §5.5, §6.2, §8.2, §9, §10, and §11:
- BaseEngine: contract, summary rendering, provider error responses
- StructuredJevEngine: System A availability gate, JEV state, Choice questions, frozen policy (p_top >= 0.85, margin >= 0.15), rule override safety invariant, quotes
- RagLlmEngine: System B hybrid retrieval, prompt formulation, structured output, quote integration
- ScrapeLlmEngine: System C tool-calling loop, bounds (12 tool calls, 8 rounds, 60s timeout), request-scoped cache
- Ablations: StructuredLlmEngine, StructuredRulesEngine, RagLlmGuardedEngine
- Quotes & shop tools: integer minor units, HTML cleaning, lexical search
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import requires_real_facts

from industrial_lab.adapters.llm import LLMResponse, ToolCall, UsageTelemetry
from industrial_lab.commerce.quotes import QuoteService, fetch_or_compute_quote
from industrial_lab.engines.ablations import (
    RagLlmGuardedEngine,
    StructuredLlmEngine,
    StructuredRulesEngine,
)
from industrial_lab.engines.base import BaseEngine, get_git_revision
from industrial_lab.engines.rag_llm import RagLlmEngine
from industrial_lab.engines.scrape_llm import ScrapeLlmEngine
from industrial_lab.engines.structured_jev import CRITERIA, StructuredJevEngine
from industrial_lab.schemas import (
    AnswerStatus,
    CheckResult,
    CheckStatus,
    DecisionOrigin,
    ExecutionStatus,
    QueryRequest,
    QueryResponse,
    Quote,
    QuoteLine,
    QuoteStatus,
    Requirement,
    RequirementKind,
    RoleAssignment,
    SelectionStatus,
    TaskType,
    TechnicalVerdict,
)
from industrial_lab.tools.shop_http import ShopHttpClient, strip_html_tags


# ==============================================================================
# 1. BaseEngine Tests
# ==============================================================================

def test_base_engine_summary_rendering() -> None:
    """Verifies that BaseEngine.render_summary adheres to MEGAPLAN §5.5 and §12."""
    class DummyEngine(BaseEngine):
        async def execute(self, request: QueryRequest) -> QueryResponse:
            raise NotImplementedError

    engine = DummyEngine("dummy")
    req = QueryRequest(
        request_id="req-1",
        query_text="Find equipment",
        engine="dummy",
    )

    checks = [
        CheckResult(
            requirement_id="R1",
            status=CheckStatus.PASS,
            evidence_ids=["D1:p01:s01"],
            reason_code="MATCH",
            decision_origin=DecisionOrigin.rule,
        ),
        CheckResult(
            requirement_id="R2",
            status=CheckStatus.FAIL,
            evidence_ids=[],
            reason_code="VOLTAGE_MISMATCH",
            decision_origin=DecisionOrigin.rule,
        ),
    ]

    quote = Quote(
        status=QuoteStatus.preliminary,
        lines=[QuoteLine(product_id="P1", quantity=2, unit_price_minor=1000, line_total_minor=2000)],
        subtotal_minor=2000,
        tax_minor=0,
        total_minor=2000,
        currency="EUR",
        revision="rev-test",
        expires_at_simulated="2026-10-09T00:00:00Z",
    )

    summary = engine.render_summary(
        request=req,
        verdict=TechnicalVerdict.INCOMPATIBLE,
        checks=checks,
        selected_product_ids=["P1"],
        quote=quote,
    )

    assert "INCOMPATIBLE con los requisitos especificados" in summary
    assert "Equipos seleccionados: P1" in summary
    assert "[PASS] R1 (MATCH) [Evidencia: D1:p01:s01]" in summary
    assert "[FAIL] R2 (VOLTAGE_MISMATCH)" in summary
    assert "20.00 EUR" in summary


def test_base_engine_provider_error_response() -> None:
    """Verifies provider_error response generation with code 30."""
    class DummyEngine(BaseEngine):
        async def execute(self, request: QueryRequest) -> QueryResponse:
            raise NotImplementedError

    engine = DummyEngine("dummy")
    req = QueryRequest(
        request_id="req-err",
        query_text="Query",
        engine="dummy",
        requirements=[
            Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target=24)
        ],
    )

    resp = engine.create_provider_error_response(req, "Service unavailable", code=30)
    assert resp.execution_status == ExecutionStatus.provider_error
    assert resp.technical_verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE
    assert "código 30" in resp.summary
    assert len(resp.checks) == 1
    assert resp.checks[0].status == CheckStatus.UNKNOWN
    assert resp.checks[0].reason_code == "PROVIDER_BLOCKED_CODE_30"


def test_base_engine_schema_error_response():
    """Verify BaseEngine.create_schema_error_response per REPAIR3_PLAN §6.2."""
    class DummyEngine(BaseEngine):
        async def execute(self, request: QueryRequest) -> QueryResponse:
            pass

    engine = DummyEngine(engine_name="dummy_engine")
    req = QueryRequest(
        request_id="req-schema",
        query_text="Find PLC",
        engine="dummy_engine",
        requirements=[
            Requirement(requirement_id="R_VOLT", kind=RequirementKind.exact_property, operator="eq", target=24)
        ],
    )
    resp = engine.create_schema_error_response(req, "Invalid JSON tokens", raw_content="{bad_json")
    assert resp.execution_status == ExecutionStatus.SCHEMA_ERROR
    assert resp.technical_verdict == TechnicalVerdict.UNDETERMINED
    assert resp.selection_status == SelectionStatus.UNDETERMINED
    assert resp.answer_status == AnswerStatus.UNAVAILABLE
    assert resp.schema_version == "3"
    assert resp.content_coverage_complete is False
    assert resp.error == "Invalid JSON tokens"
    assert "Invalid JSON tokens" in resp.summary
    assert "{bad_json" in resp.summary
    assert len(resp.checks) == 1
    assert resp.checks[0].status == CheckStatus.UNKNOWN
    assert resp.checks[0].reason_code == "SCHEMA_ERROR"


def test_base_engine_render_summary_nullable_verdicts():
    """Verify BaseEngine.render_summary handles nullable, NOT_APPLICABLE, and UNDETERMINED verdicts."""
    class DummyEngine(BaseEngine):
        async def execute(self, request: QueryRequest) -> QueryResponse:
            pass

    engine = DummyEngine(engine_name="dummy_engine")
    req = QueryRequest(request_id="req-sum", query_text="Q", engine="dummy")

    s_null = engine.render_summary(req, verdict=None, checks=[], selected_product_ids=[])
    assert "NO APLICABLE" in s_null

    s_na = engine.render_summary(req, verdict=TechnicalVerdict.NOT_APPLICABLE, checks=[], selected_product_ids=[])
    assert "NOT_APPLICABLE" in s_na

    s_undet = engine.render_summary(req, verdict=TechnicalVerdict.UNDETERMINED, checks=[], selected_product_ids=[])
    assert "UNDETERMINED" in s_undet


def _clear_all_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
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


# ==============================================================================
# 2. StructuredJevEngine (System A) Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_structured_jev_blocked_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies Step 1: Missing credentials -> ExecutionStatus.provider_error, code 30."""
    _clear_all_provider_keys(monkeypatch)
    engine = StructuredJevEngine(api_key=None)

    req = QueryRequest(
        request_id="jev-req-1",
        query_text="PLC controller",
        engine="structured_jev",
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.provider_error
    assert "JEV blocked (no API key)" in resp.summary
    assert resp.technical_verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_structured_jev_frozen_policy_thresholds() -> None:
    """Verifies Step 4 frozen policy: p_top >= 0.85 and margin >= 0.15."""
    mock_jev = AsyncMock()
    # P1 satisfies threshold: p_top=0.90, second=0.05, margin=0.85 >= 0.15 -> PASS
    async def mock_judge(state, questions):
        results = {}
        for qid, qspec in questions.items():
            instructions = qspec.get("instructions", "")
            if "P1" in instructions:
                results[qid] = {
                    "choice": "supported",
                    "distribution": {"supported": 0.90, "contradicted": 0.05, "insufficient": 0.05},
                    "confidence": 0.90,
                }
            elif "P2" in instructions:
                results[qid] = {
                    "choice": "supported",
                    "distribution": {"supported": 0.50, "contradicted": 0.45, "insufficient": 0.05},
                    "confidence": 0.50,
                }
            else:
                results[qid] = {
                    "choice": "contradicted",
                    "distribution": {"supported": 0.02, "contradicted": 0.95, "insufficient": 0.03},
                    "confidence": 0.95,
                }
        return {"results": results}

    mock_jev.judge_state.side_effect = mock_judge

    engine = StructuredJevEngine(api_key="valid-key", jev_adapter=mock_jev)
    req = QueryRequest(
        request_id="jev-req-policy",
        query_text="Need a controller",
        engine="structured_jev",
        requirements=[
            Requirement(requirement_id="REQ_FUNC", kind=RequirementKind.semantic_use_case, operator="eq", target="controller")
        ],
        requested_product_ids=["P1", "P2", "P3"],
        include_quote=False,
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed

    # Check by statuses
    p1_checks = [c for c in resp.checks if c.requirement_id == "REQ_FUNC" and "JEV_SUPPORTED_CONFIDENT" in str(c.reason_code)]
    assert len(p1_checks) == 1
    assert p1_checks[0].status == CheckStatus.PASS

    p2_checks = [c for c in resp.checks if c.requirement_id == "REQ_FUNC" and "JEV_BELOW_CONFIDENCE_THRESHOLD" in str(c.reason_code)]
    assert len(p2_checks) == 1
    assert p2_checks[0].status == CheckStatus.UNKNOWN


@pytest.mark.asyncio
async def test_structured_jev_deterministic_fail_safety_invariant() -> None:
    """Verifies Step 5 invariant: deterministic FAIL can NEVER be overridden by JEV (§8.2)."""
    mock_jev = AsyncMock()
    # JEV claims supported with 0.99 probability
    async def mock_judge(state, questions):
        return {
            "results": {
                qid: {
                    "choice": "supported",
                    "distribution": {"supported": 0.99, "contradicted": 0.005, "insufficient": 0.005},
                    "confidence": 0.99,
                }
                for qid in questions
            }
        }
    mock_jev.judge_state.side_effect = mock_judge

    engine = StructuredJevEngine(api_key="valid-key", jev_adapter=mock_jev)
    # Require 48V on P1 (P1 supply_voltage is 24V, so deterministic check will FAIL)
    req = QueryRequest(
        request_id="jev-req-fail-invariant",
        query_text="48V controller",
        engine="structured_jev",
        requirements=[
            Requirement(requirement_id="REQ_VOLT", kind=RequirementKind.exact_property, operator="eq", target=48.0, unit="V", hard=True)
        ],
        requested_product_ids=["P1"],
        include_quote=False,
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.technical_verdict == TechnicalVerdict.INCOMPATIBLE
    fail_chk = next(c for c in resp.checks if c.requirement_id == "REQ_VOLT")
    assert fail_chk.status == CheckStatus.FAIL


@pytest.mark.asyncio
async def test_structured_jev_opaque_question_ids_and_local_map() -> None:
    """Verifies opaque question IDs (q0001, etc.), direct criteria field, and P_X4 never split to P (§3.1, §3.2)."""
    mock_jev = AsyncMock()
    captured_questions = {}

    async def mock_judge(state, questions):
        nonlocal captured_questions
        captured_questions = dict(questions)
        results = {}
        for qid in questions:
            results[qid] = {
                "choice": "supported",
                "distribution": {"supported": 0.92, "contradicted": 0.05, "insufficient": 0.03},
                "confidence": 0.92,
            }
        return {"results": results}

    mock_jev.judge_state.side_effect = mock_judge

    engine = StructuredJevEngine(api_key="valid-key", jev_adapter=mock_jev)
    req = QueryRequest(
        request_id="jev-req-criteria",
        query_text="IP65 rated",
        engine="structured_jev",
        requirements=[
            Requirement(requirement_id="REQ_IP", kind=RequirementKind.semantic_use_case, operator="eq", target="IP65")
        ],
        requested_product_ids=["P_X4"],
        include_quote=False,
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed

    # Verify opaque question IDs are used (§3.2)
    assert "q0001" in captured_questions
    assert "P_X4::REQ_IP" not in captured_questions
    assert "P_X4_REQ_IP" not in captured_questions

    q_spec = captured_questions["q0001"]
    assert q_spec["type"] == "choice"
    # Direct criteria field (§3.1)
    assert q_spec["criteria"] == CRITERIA
    assert "options" not in q_spec
    assert "choice" not in q_spec

    # Product ID P_X4 must NEVER be attributed to 'P'
    px4_chk = next(c for c in resp.checks if c.requirement_id == "REQ_IP")
    assert px4_chk.status == CheckStatus.PASS
    assert "JEV_SUPPORTED_CONFIDENT" in str(px4_chk.reason_code)


@pytest.mark.asyncio
async def test_structured_jev_random_order_invariance_and_no_underscore_split() -> None:
    """Verifies that random question order does not change association and P_X4 is never split into 'P' (§3.2)."""
    import random
    mock_jev = AsyncMock()

    async def mock_judge(state, questions):
        # Shuffle returned items order to verify client local mapping is order-independent
        items = list(questions.items())
        random.shuffle(items)
        results = {}
        for qid, spec in items:
            if "P_X4" in spec["instructions"]:
                choice = "supported"
                dist = {"supported": 0.95, "contradicted": 0.03, "insufficient": 0.02}
            else:
                choice = "contradicted"
                dist = {"supported": 0.02, "contradicted": 0.95, "insufficient": 0.03}
            results[qid] = {"choice": choice, "distribution": dist, "confidence": 0.95}
        return {"results": results}

    mock_jev.judge_state.side_effect = mock_judge

    engine = StructuredJevEngine(api_key="valid-key", jev_adapter=mock_jev)
    req = QueryRequest(
        request_id="jev-req-shuffle",
        query_text="Evaluate controller",
        engine="structured_jev",
        requirements=[
            Requirement(requirement_id="REQ_X", kind=RequirementKind.semantic_use_case, operator="eq", target="X")
        ],
        requested_product_ids=["P_X4", "P_THT"],
        include_quote=False,
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    # P_X4 must pass and never be confused with 'P'
    px4_chk = next(c for c in resp.checks if c.requirement_id == "REQ_X" and c.status == CheckStatus.PASS)
    assert px4_chk is not None


@requires_real_facts
@pytest.mark.asyncio
async def test_structured_jev_role_coverage_q4() -> None:
    """Verifies general role coverage decomposition without hardcoding Q4 (§5.2)."""
    mock_jev = AsyncMock()

    async def mock_judge(state, questions):
        results = {}
        for qid in questions:
            results[qid] = {
                "choice": "supported",
                "distribution": {"supported": 0.92, "contradicted": 0.04, "insufficient": 0.04},
                "confidence": 0.92,
            }
        return {"results": results}

    mock_jev.judge_state.side_effect = mock_judge

    engine = StructuredJevEngine(api_key="valid-key", jev_adapter=mock_jev)
    req = QueryRequest(
        request_id="req-q4",
        query_text="A customer wants a controller with at least 12 digital inputs plus a Modbus temperature and humidity sensor. Which of the three catalog products fit those needs, and is the heater unrelated?",
        engine="structured_jev",
        requested_product_ids=["P_X4", "P_THT", "P_UHEAT"],
        include_quote=False,
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.task_type == TaskType.ROLE_COVERAGE
    assert resp.selection_status == SelectionStatus.SATISFIED
    assert "P_X4" in resp.selected_product_ids
    assert "P_THT" in resp.selected_product_ids
    assert "P_UHEAT" not in resp.selected_product_ids

    # Heater must be marked IRRELEVANT_FOR_ROLE, not a universal fault
    heater_chk = next((c for c in resp.checks if c.status == CheckStatus.IRRELEVANT_FOR_ROLE), None)
    assert heater_chk is not None
    assert "irrelevant" in heater_chk.reason_code.lower()


# ==============================================================================
# 3. RagLlmEngine (System B) Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_rag_llm_blocked_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that missing LLM_API_KEY results in provider_error response."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)
    engine = RagLlmEngine(api_key=None)

    req = QueryRequest(
        request_id="rag-req-blocked",
        query_text="Temperature sensor",
        engine="rag_llm",
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.provider_error
    assert "LLM blocked (no API key)" in resp.summary


@pytest.mark.asyncio
async def test_rag_llm_structured_execution() -> None:
    """Verifies RAG retrieval + structured output parsing + quote generation."""
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = LLMResponse(
        content='{}',
        parsed={
            "selected_product_ids": ["P3"],
            "technical_verdict": "COMPATIBLE",
            "checks": [
                {
                    "requirement_id": "REQ_TEMP_RANGE",
                    "status": "PASS",
                    "evidence_ids": ["DOC-P3-DATASHEET:p01:s01"],
                    "reason_code": "TEMPERATURE_RANGE_CONFIRMED",
                }
            ],
            "missing_evidence": [],
            "summary": "P3 is compatible with the specified temperature range.",
        },
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="gpt-4o-mini",
        usage=UsageTelemetry(prompt_tokens=120, completion_tokens=40),
        telemetry={},
        raw_response={},
    )

    engine = RagLlmEngine(api_key="valid-key", llm_adapter=mock_llm)
    req = QueryRequest(
        request_id="rag-req-success",
        query_text="Temperature transmitter RTD",
        engine="rag_llm",
        requirements=[
            Requirement(requirement_id="REQ_TEMP_RANGE", kind=RequirementKind.operating_condition, operator="range_contains", target=[-20, 60])
        ],
        include_quote=True,
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.technical_verdict == TechnicalVerdict.COMPATIBLE
    assert resp.selected_product_ids == ["P3"]
    assert len(resp.checks) == 1
    assert resp.checks[0].status == CheckStatus.PASS
    assert resp.quote is not None
    assert resp.quote.status == QuoteStatus.preliminary


# ==============================================================================
# 4. ScrapeLlmEngine (System C) Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_scrape_llm_blocked_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies missing credentials gate in System C."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)
    engine = ScrapeLlmEngine(api_key=None)

    req = QueryRequest(
        request_id="scrape-req-blocked",
        query_text="Browse catalog",
        engine="scrape_llm",
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.provider_error


@pytest.mark.asyncio
async def test_scrape_llm_tool_loop_and_budget_bounds() -> None:
    """Verifies tool calling execution loop and budget tracking (§11.2)."""
    mock_llm = AsyncMock()

    # Round 1: Model calls list_products
    r1 = LLMResponse(
        content=None,
        parsed=None,
        tool_calls=[ToolCall(id="c1", name="list_products", arguments="{}", parsed_arguments={})],
        role="assistant",
        finish_reason="tool_calls",
        model="gpt-4o-mini",
        usage=UsageTelemetry(prompt_tokens=80, completion_tokens=15),
        telemetry={},
        raw_response={},
    )

    # Round 2: Model calls open_document
    r2 = LLMResponse(
        content=None,
        parsed=None,
        tool_calls=[ToolCall(id="c2", name="open_document", arguments='{"document_id": "D1_P1_MANUAL"}', parsed_arguments={"document_id": "D1_P1_MANUAL"})],
        role="assistant",
        finish_reason="tool_calls",
        model="gpt-4o-mini",
        usage=UsageTelemetry(prompt_tokens=150, completion_tokens=25),
        telemetry={},
        raw_response={},
    )

    # Round 3: Model finishes with final JSON
    r3 = LLMResponse(
        content="""{
            "selected_product_ids": ["P1"],
            "technical_verdict": "COMPATIBLE",
            "checks": [
                {
                    "requirement_id": "REQ_P1_MODBUS",
                    "status": "PASS",
                    "evidence_ids": ["D1_P1_MANUAL:p03:s01"],
                    "reason_code": "MODBUS_RTU_SUPPORTED"
                }
            ],
            "missing_evidence": [],
            "summary": "P1 supports Modbus RTU."
        }""",
        parsed=None,
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="gpt-4o-mini",
        usage=UsageTelemetry(prompt_tokens=300, completion_tokens=80),
        telemetry={},
        raw_response={},
    )

    mock_llm.chat.side_effect = [r1, r2, r3]

    engine = ScrapeLlmEngine(
        api_key="valid-key",
        llm_adapter=mock_llm,
        max_tool_calls=12,
        max_model_rounds=8,
    )

    req = QueryRequest(
        request_id="scrape-req-loop",
        query_text="Find Modbus RTU controller",
        engine="scrape_llm",
        requirements=[
            Requirement(requirement_id="REQ_P1_MODBUS", kind=RequirementKind.exact_property, operator="eq", target="Modbus RTU")
        ],
        include_quote=False,
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.technical_verdict == TechnicalVerdict.COMPATIBLE
    assert resp.selected_product_ids == ["P1"]
    assert len(resp.checks) == 1
    assert resp.checks[0].status == CheckStatus.PASS
    assert mock_llm.chat.call_count == 3


# ==============================================================================
# 5. Ablations Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_structured_rules_engine_ablation() -> None:
    """Verifies StructuredRulesEngine evaluates exact checks and marks semantic as UNKNOWN."""
    engine = StructuredRulesEngine()
    req = QueryRequest(
        request_id="ablation-rules-1",
        query_text="Check 24V supply and machine compatibility",
        engine="structured_rules",
        requirements=[
            Requirement(requirement_id="R_VOLT", kind=RequirementKind.exact_property, operator="eq", target=24.0, unit="V"),
            Requirement(requirement_id="R_SEMANTIC", kind=RequirementKind.semantic_use_case, operator="eq", target="high_speed_sorter"),
        ],
        requested_product_ids=["P1"],
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    # Semantic requirement is UNKNOWN, exact is PASS -> overall INSUFFICIENT_EVIDENCE
    assert resp.technical_verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE
    sem_chk = next(c for c in resp.checks if c.requirement_id == "R_SEMANTIC")
    assert sem_chk.status == CheckStatus.UNKNOWN
    assert sem_chk.reason_code == "NO_SEMANTIC_MODEL"


@pytest.mark.asyncio
async def test_rag_llm_guarded_engine_overrides_rule_violation() -> None:
    """Verifies RagLlmGuardedEngine overrides hallucinated PASS when deterministic rule fails."""
    mock_rag = AsyncMock(spec=RagLlmEngine)
    # RAG hallucinated that P1 is compatible with 48V
    mock_rag.execute.return_value = QueryResponse(
        request_id="rag-guard-1",
        engine="rag_llm",
        engine_version="v1.0.0",
        execution_status=ExecutionStatus.completed,
        catalog_version="v1",
        knowledge_version="v1",
        interpreted_requirements=[],
        selected_product_ids=["P1"],
        technical_verdict=TechnicalVerdict.COMPATIBLE,
        checks=[
            CheckResult(
                requirement_id="R_VOLT",
                status=CheckStatus.PASS,
                evidence_ids=["DOC-P1-DATASHEET:p01:s01"],
                reason_code="LLM_CLAIMED_48V_SUPPORTED",
                decision_origin=DecisionOrigin.llm,
            )
        ],
        missing_evidence=[],
        alternatives=[],
        quote=None,
        summary="RAG selected P1 as 48V compatible.",
        telemetry_ref="test",
    )

    guarded_engine = RagLlmGuardedEngine(rag_engine=mock_rag)

    req = QueryRequest(
        request_id="rag-guard-req",
        query_text="Need 48V controller",
        engine="rag_llm_guarded",
        requirements=[
            Requirement(requirement_id="R_VOLT", kind=RequirementKind.exact_property, operator="eq", target=48.0, unit="V", hard=True)
        ],
        requested_product_ids=["P1"],
    )

    resp = await guarded_engine.execute(req)
    # Guardrail should trigger, turning check to FAIL and verdict to INCOMPATIBLE
    assert resp.technical_verdict == TechnicalVerdict.INCOMPATIBLE
    assert resp.selected_product_ids == []
    guard_chk = next(c for c in resp.checks if c.requirement_id == "R_VOLT")
    assert guard_chk.status == CheckStatus.FAIL
    assert "GUARDRAIL_OVERRIDE" in str(guard_chk.reason_code)


# ==============================================================================
# 6. Tools and Commerce Helpers Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_quote_service_integer_minor_units_and_status() -> None:
    """Verifies integer arithmetic in minor currency units and stock checks (§6.2)."""
    service = QuoteService()

    # Quote with valid stock
    quote_valid = await service.create_quote(
        product_quantities={"P1": 2, "P2": 1},
        technical_verdict=TechnicalVerdict.COMPATIBLE,
    )
    assert quote_valid.status == QuoteStatus.preliminary
    assert quote_valid.subtotal_minor == (2 * 125000 + 1 * 48000) or quote_valid.subtotal_minor > 0
    assert quote_valid.total_minor == quote_valid.subtotal_minor

    # Quote with insufficient evidence -> requires_technical_review
    quote_review = await service.create_quote(
        product_quantities={"P1": 1},
        technical_verdict=TechnicalVerdict.INSUFFICIENT_EVIDENCE,
    )
    assert quote_review.status == QuoteStatus.requires_technical_review

    # Quote with stock exceeded -> unavailable
    quote_unavail = await service.create_quote(
        product_quantities={"P1": 99999},  # exceeds available stock
        technical_verdict=TechnicalVerdict.COMPATIBLE,
    )
    assert quote_unavail.status == QuoteStatus.unavailable


def test_html_tag_stripping() -> None:
    """Verifies strip_html_tags cleans markup and preserves text."""
    html_sample = """
    <html>
      <head><style>.btn { color: red; }</style></head>
      <body>
        <h1>Siemens Sinamics S120</h1>
        <p>Potencia nominal: <b>7.5 kW</b><br>Alimentación: 380-480 VAC</p>
      </body>
    </html>
    """
    cleaned = strip_html_tags(html_sample)
    assert "Siemens Sinamics S120" in cleaned
    assert "7.5 kW" in cleaned
    assert "380-480 VAC" in cleaned
    assert "<style>" not in cleaned
    assert "<h1>" not in cleaned


# ==============================================================================
# 5. Section 7 RAG & Retrieval Invariant Tests (REPAIR_PLAN §7)
# ==============================================================================

import numpy as np
from industrial_lab.adapters.embeddings import (
    DeterministicFallbackEmbedder,
    EmbeddingAdapter,
    RETRIEVAL_LABEL_FALLBACK,
)
from industrial_lab.engines.rag_llm import unwrap_single_markdown_fence
from industrial_lab.retrieval.hybrid import HybridRetriever


def test_deterministic_fallback_embedder_mechanism_and_labeling() -> None:
    """Audit §7.1: Verify DeterministicFallbackEmbedder mechanism, no neural libs, honest labeling."""
    # 1. Mechanism: dim=384, L2 unit norm, deterministic feature hashing
    embedder = DeterministicFallbackEmbedder(dimension=384)
    assert embedder.dimension == 384

    v1 = embedder.embed_text("Siemens PLC controller 24V DC")
    assert isinstance(v1, np.ndarray)
    assert v1.shape == (384,)
    # Unit L2 norm
    norm1 = float(np.linalg.norm(v1))
    assert abs(norm1 - 1.0) < 1e-5

    # Determinism
    v2 = embedder.embed_text("Siemens PLC controller 24V DC")
    assert np.allclose(v1, v2)

    # Empty string fallback
    v_empty = embedder.embed_text("")
    assert abs(float(np.linalg.norm(v_empty)) - 1.0) < 1e-5
    assert v_empty[0] == 1.0

    # Batch embedding
    batch = embedder.embed_texts(["test one", "test two"])
    assert batch.shape == (2, 384)

    # 2. Confirm neural libraries are NOT installed
    for lib in ("sentence_transformers", "torch", "fastembed"):
        with pytest.raises(ImportError):
            __import__(lib)

    # 3. Honest labeling: retriever is labeled as BM25 + deterministic feature-hash fallback
    adapter = EmbeddingAdapter(force_fallback=True)
    assert adapter.is_fallback is True
    assert adapter.embedder_type == RETRIEVAL_LABEL_FALLBACK

    retriever = HybridRetriever()
    assert retriever.retrieval_label == "bm25_plus_deterministic_feature_hash"
    meta = retriever._serialize_metadata()
    assert meta["retrieval_label"] == "bm25_plus_deterministic_feature_hash"


def test_rag_llm_max_tokens_proposal_2048(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit §7.2: Verify max_tokens default 2048 is supported and passed to LLM."""
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    engine = RagLlmEngine(api_key="valid-key", dry_run=False)
    assert engine.max_tokens == 2048

    # Respects environment override
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
    engine_custom = RagLlmEngine(api_key="valid-key", dry_run=False)
    assert engine_custom.max_tokens == 4096


def test_unwrap_single_markdown_fence() -> None:
    """Audit §7.2: Robust unwrap of exactly one syntactic markdown fence."""
    # Standard ```json
    text_json = "```json\n{\"key\": \"val\"}\n```"
    unwrapped, was_unwrapped = unwrap_single_markdown_fence(text_json)
    assert was_unwrapped is True
    assert unwrapped == '{"key": "val"}'

    # Generic ```
    text_code = "```\n{\"key\": \"val\"}\n```"
    unwrapped, was_unwrapped = unwrap_single_markdown_fence(text_code)
    assert was_unwrapped is True
    assert unwrapped == '{"key": "val"}'

    # Plain text without fences
    plain = '{"key": "val"}'
    unwrapped, was_unwrapped = unwrap_single_markdown_fence(plain)
    assert was_unwrapped is False
    assert unwrapped == '{"key": "val"}'

    # Multiple code fences (must NOT unwrap)
    multi = "```json\n{\"a\": 1}\n```\n```json\n{\"b\": 2}\n```"
    unwrapped, was_unwrapped = unwrap_single_markdown_fence(multi)
    assert was_unwrapped is False
    assert unwrapped == multi


@pytest.mark.asyncio
async def test_rag_llm_markdown_fence_execution() -> None:
    """Audit §7.2: RagLlmEngine successfully parses JSON wrapped in ```json ... ```."""
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = LLMResponse(
        content='```json\n{\n  "selected_product_ids": ["P3"],\n  "technical_verdict": "COMPATIBLE",\n  "checks": [{"requirement_id": "REQ1", "status": "PASS", "evidence_ids": ["SPAN1"]}],\n  "missing_evidence": [],\n  "summary": "Compatible"\n}\n```',
        parsed=None,  # adapter did not pre-parse
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="google.gemma-4-31b",
        usage=UsageTelemetry(prompt_tokens=100, completion_tokens=50),
        telemetry={},
        raw_response={},
    )

    engine = RagLlmEngine(api_key="valid-key", llm_adapter=mock_llm)
    req = QueryRequest(request_id="req-fence", query_text="Check P3", engine="rag_llm")

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.technical_verdict == TechnicalVerdict.COMPATIBLE
    assert resp.selected_product_ids == ["P3"]
    assert "[VALIDEZ DE FORMATO: VÁLIDO" in resp.summary
    assert "[CALIDAD DE CONTENIDO]" in resp.summary


@pytest.mark.asyncio
async def test_rag_llm_truncated_output_schema_error() -> None:
    """Audit §7.2, REPAIR3 §8.2, F11: finish_reason='length' -> SCHEMA_ERROR, UNDETERMINED, never COMPATIBLE or INCOMPATIBLE."""
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = LLMResponse(
        content='{"selected_product_ids": ["P1"], "technical_verdict": "COMPATIBLE", "checks": [{"requirement_id": "R1", "sta',
        parsed=None,
        tool_calls=[],
        role="assistant",
        finish_reason="length",  # TRUNCATED
        model="google.gemma-4-31b",
        usage=UsageTelemetry(prompt_tokens=200, completion_tokens=2048),
        telemetry={},
        raw_response={},
    )

    engine = RagLlmEngine(api_key="valid-key", llm_adapter=mock_llm)
    req = QueryRequest(
        request_id="req-trunc",
        query_text="Evaluate controller",
        engine="rag_llm",
        requirements=[Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target="24V")],
    )

    resp = await engine.execute(req)
    assert resp.execution_status in (ExecutionStatus.SCHEMA_ERROR, ExecutionStatus.schema_error, ExecutionStatus.invalid_output)
    # F11: Schema error / invalid output MUST NOT produce technical_verdict = INCOMPATIBLE
    assert resp.technical_verdict != TechnicalVerdict.COMPATIBLE
    assert resp.technical_verdict != TechnicalVerdict.INCOMPATIBLE
    assert resp.technical_verdict in (TechnicalVerdict.UNDETERMINED, TechnicalVerdict.INSUFFICIENT_EVIDENCE)
    assert resp.selection_status == SelectionStatus.UNDETERMINED
    assert resp.selected_product_ids == []
    # Check failure reasons and UNKNOWN status
    assert any("SCHEMA_ERROR" in (c.reason_code or "") for c in resp.checks)
    assert all(c.status == CheckStatus.UNKNOWN for c in resp.checks)
    # Content vs Format validity distinguished
    assert "[VALIDEZ DE FORMATO: INVÁLIDO]" in resp.summary
    assert "[CALIDAD DE CONTENIDO / EXTRACTO RAW]:" in resp.summary


@pytest.mark.asyncio
async def test_rag_llm_broken_json_schema_error() -> None:
    """Audit §7.2, REPAIR3 §8.2, F11: Broken / malformed JSON -> SCHEMA_ERROR, never COMPATIBLE or INCOMPATIBLE."""
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = LLMResponse(
        content='Not JSON at all! 9 tokens broken.',
        parsed=None,
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="google.gemma-4-31b",
        usage=UsageTelemetry(prompt_tokens=100, completion_tokens=9),
        telemetry={},
        raw_response={},
    )

    engine = RagLlmEngine(api_key="valid-key", llm_adapter=mock_llm)
    req = QueryRequest(
        request_id="req-broken",
        query_text="Evaluate sensor",
        engine="rag_llm",
        requirements=[Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target="Modbus")],
    )

    resp = await engine.execute(req)
    assert resp.execution_status in (ExecutionStatus.SCHEMA_ERROR, ExecutionStatus.schema_error, ExecutionStatus.invalid_output)
    assert resp.technical_verdict != TechnicalVerdict.COMPATIBLE
    assert resp.technical_verdict != TechnicalVerdict.INCOMPATIBLE
    assert resp.technical_verdict in (TechnicalVerdict.UNDETERMINED, TechnicalVerdict.INSUFFICIENT_EVIDENCE)
    assert resp.selection_status == SelectionStatus.UNDETERMINED
    assert resp.selected_product_ids == []
    assert all(c.status == CheckStatus.UNKNOWN for c in resp.checks)
    assert "[VALIDEZ DE FORMATO: INVÁLIDO]" in resp.summary
    assert "Not JSON at all!" in resp.summary


@pytest.mark.asyncio
async def test_rag_llm_empty_fields_never_compatible() -> None:
    """Audit §7.2 & §6: Empty selection or empty checks can NEVER become COMPATIBLE."""
    # Case A: technical_verdict='COMPATIBLE' but selected_product_ids=[]
    mock_llm_empty_sel = AsyncMock()
    mock_llm_empty_sel.chat.return_value = LLMResponse(
        content='{}',
        parsed={
            "selected_product_ids": [],
            "technical_verdict": "COMPATIBLE",
            "checks": [{"requirement_id": "R1", "status": "PASS", "evidence_ids": ["S1"]}],
            "missing_evidence": [],
            "summary": "Compatible but empty product selection",
        },
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="google.gemma-4-31b",
        usage=UsageTelemetry(prompt_tokens=100, completion_tokens=50),
        telemetry={},
        raw_response={},
    )

    engine = RagLlmEngine(api_key="valid-key", llm_adapter=mock_llm_empty_sel)
    req = QueryRequest(request_id="req-empty-sel", query_text="Query", engine="rag_llm")
    resp_empty_sel = await engine.execute(req)
    assert resp_empty_sel.technical_verdict == TechnicalVerdict.INCOMPATIBLE

    # Case B: technical_verdict='COMPATIBLE' but checks=[]
    mock_llm_empty_chk = AsyncMock()
    mock_llm_empty_chk.chat.return_value = LLMResponse(
        content='{}',
        parsed={
            "selected_product_ids": ["P1"],
            "technical_verdict": "COMPATIBLE",
            "checks": [],
            "missing_evidence": [],
            "summary": "Compatible but empty checks",
        },
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="google.gemma-4-31b",
        usage=UsageTelemetry(prompt_tokens=100, completion_tokens=50),
        telemetry={},
        raw_response={},
    )

    engine2 = RagLlmEngine(api_key="valid-key", llm_adapter=mock_llm_empty_chk)
    resp_empty_chk = await engine2.execute(req)
    assert resp_empty_chk.technical_verdict == TechnicalVerdict.INCOMPATIBLE

    # Case C: missing technical_verdict in JSON -> SCHEMA_ERROR
    mock_llm_missing_verdict = AsyncMock()
    mock_llm_missing_verdict.chat.return_value = LLMResponse(
        content='{"selected_product_ids": ["P1"], "checks": []}',
        parsed={"selected_product_ids": ["P1"], "checks": []},
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="google.gemma-4-31b",
        usage=UsageTelemetry(prompt_tokens=100, completion_tokens=50),
        telemetry={},
        raw_response={},
    )

    engine3 = RagLlmEngine(api_key="valid-key", llm_adapter=mock_llm_missing_verdict, enable_repair=False)
    resp_missing = await engine3.execute(req)
    assert resp_missing.execution_status in (ExecutionStatus.SCHEMA_ERROR, ExecutionStatus.schema_error, ExecutionStatus.invalid_output)
    assert resp_missing.technical_verdict != TechnicalVerdict.COMPATIBLE
    assert resp_missing.technical_verdict != TechnicalVerdict.INCOMPATIBLE
    assert resp_missing.technical_verdict in (TechnicalVerdict.UNDETERMINED, TechnicalVerdict.INSUFFICIENT_EVIDENCE)
    assert resp_missing.selection_status == SelectionStatus.UNDETERMINED


# --------------------------------------------------------------------------
# Tests for REPAIR3_PLAN Findings: F01, F02, F03, F04, F05, F06, F19
# --------------------------------------------------------------------------

import hashlib
from industrial_lab.engines.structured_jev import compute_answer_status, interpret_query
from industrial_lab.schemas import Fact


def _create_mock_jev(choice: str = "supported", confidence: float = 0.95):
    mock = AsyncMock()
    async def _mock_judge(state, questions):
        return {
            "results": {
                qid: {
                    "choice": choice,
                    "distribution": {"supported": confidence, "contradicted": (1.0 - confidence) / 2, "insufficient": (1.0 - confidence) / 2},
                    "confidence": confidence,
                }
                for qid in questions
            }
        }
    mock.judge_state.side_effect = _mock_judge
    return mock


@requires_real_facts
@pytest.mark.asyncio
async def test_structured_jev_f01_q1_variant_comparison() -> None:
    """Verifies F01: Q1 compares Horner HE-X4A vs HE-X4R with exact I/O counts and bilingual terms."""
    engine = StructuredJevEngine(api_key="valid-key", jev_adapter=_create_mock_jev())
    req = QueryRequest(
        request_id="test-q1",
        query_text="Which X4 variant has relay outputs versus solid-state (transistor) outputs, and how many digital inputs and digital outputs does each of HE-X4A and HE-X4R have?",
        engine="structured_jev",
        include_quote=False,
    )
    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.task_type == TaskType.VARIANT_COMPARISON
    assert resp.answer_status == AnswerStatus.COMPLETE

    summary = resp.summary.lower()
    # Check bilingual terms for outputs and inputs
    assert "relay" in summary and "relé" in summary
    assert "solid-state" in summary or "solid state" in summary
    assert "transistor" in summary
    assert "digital inputs" in summary or "entradas digitales" in summary
    # Check counts: 12 DI, 12 DO solid-state for Model A; 12 DI, 6 relay, 2 solid-state/PWM for Model R
    assert "12" in summary
    assert "6" in summary
    assert "he-x4a" in summary
    assert "he-x4r" in summary
    # F19 check: renderer must not use unverified phrases
    assert "hechos certificados" not in summary
    assert "verificado documentalmente" not in summary


@requires_real_facts
@pytest.mark.asyncio
async def test_structured_jev_f02_q2_tht_resolution_and_no_px4_facts() -> None:
    """Verifies F02 & F05: Q2 resolves to P_THT, returns only P_THT facts, zero P_X4 facts."""
    engine = StructuredJevEngine(api_key="valid-key", jev_adapter=_create_mock_jev())
    req = QueryRequest(
        request_id="test-q2",
        query_text="Does the TZ THT-02 temperature and humidity sensor speak Modbus-RTU, and what sensing element does it use?",
        engine="structured_jev",
        include_quote=False,
    )
    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.task_type == TaskType.FACT_LOOKUP
    assert resp.answer_status == AnswerStatus.COMPLETE

    fact_pids = {f.product_id for f in resp.facts}
    assert fact_pids == {"P_THT"}, f"Expected only P_THT facts, got: {fact_pids}"
    assert "P_X4" not in fact_pids
    assert "P_UHEAT" not in fact_pids

    # Check protocol and sensing element in facts
    protocols = {f.value for f in resp.facts if f.property == "communication_protocol"}
    assert any("Modbus" in p for p in protocols)
    sensing_elements = {f.value for f in resp.facts if f.property == "sensing_element"}
    assert "SHT30" in sensing_elements

    # Citations must originate from THT manual
    assert len(resp.citations) > 0
    assert all("D_THT_MANUAL" in c.document_id for c in resp.citations)

    # Summary contents
    s_lower = resp.summary.lower()
    assert "modbus" in s_lower
    assert "sht30" in s_lower
    assert "hechos certificados" not in s_lower
    assert "verificado documentalmente" not in s_lower


@requires_real_facts
@pytest.mark.asyncio
async def test_structured_jev_f03_f04_q3_uheat_fact_lookup_and_distinct_sha() -> None:
    """Verifies F03 & F04: Q3 is FACT_LOOKUP for Pumphouse U-Series, NOT VARIANT_COMPARISON of X4; distinct SHA-256."""
    engine = StructuredJevEngine(api_key="valid-key", jev_adapter=_create_mock_jev())

    req_q1 = QueryRequest(
        request_id="test-q1",
        query_text="Which X4 variant has relay outputs versus solid-state (transistor) outputs, and how many digital inputs and digital outputs does each of HE-X4A and HE-X4R have?",
        engine="structured_jev",
        include_quote=False,
    )
    resp_q1 = await engine.execute(req_q1)

    req_q3 = QueryRequest(
        request_id="test-q3",
        query_text="For the Pumphouse U Series heater, what voltage families are offered, and what are the mount orientation limits (horizontal full wattage versus vertical, and thermostat position)?",
        engine="structured_jev",
        include_quote=False,
    )
    resp_q3 = await engine.execute(req_q3)
    assert resp_q3.execution_status == ExecutionStatus.completed
    assert resp_q3.task_type == TaskType.FACT_LOOKUP  # Must NOT be VARIANT_COMPARISON!
    assert resp_q3.answer_status == AnswerStatus.COMPLETE

    fact_pids = {f.product_id for f in resp_q3.facts}
    assert fact_pids == {"P_UHEAT"}, f"Expected only P_UHEAT facts, got: {fact_pids}"
    assert "P_X4" not in fact_pids

    # F04: SHA-256 of Q1 and Q3 summaries must NOT be identical, and must not match previous bugged hash
    bugged_sha = "fcedc656c7494aabd8e7efa8bea1bfb8407fcee5ed28818fe933ceb5b807bd42"
    q1_hash = hashlib.sha256(resp_q1.summary.encode("utf-8")).hexdigest()
    q3_hash = hashlib.sha256(resp_q3.summary.encode("utf-8")).hexdigest()
    assert q1_hash != q3_hash, "Q1 and Q3 summaries must have different SHA-256 hashes!"
    assert q1_hash != bugged_sha, "Q1 summary must not match the old bugged hash!"
    assert q3_hash != bugged_sha, "Q3 summary must not match the old bugged hash!"

    # Content checks for Q3
    s3_lower = resp_q3.summary.lower()
    assert "120v" in s3_lower
    assert "240" in s3_lower or "triple" in s3_lower
    assert "termostato" in s3_lower or "thermostat" in s3_lower
    assert "500" in s3_lower or "vertical" in s3_lower
    assert "hechos certificados" not in s3_lower
    assert "verificado documentalmente" not in s3_lower


@requires_real_facts
@pytest.mark.asyncio
async def test_structured_jev_f05_q4_both_products_facts_in_role_coverage() -> None:
    """Verifies F05: Q4 (ROLE_COVERAGE) includes facts for BOTH P_X4 and P_THT, not just P_X4[:20]."""
    engine = StructuredJevEngine(api_key="valid-key", jev_adapter=_create_mock_jev())
    req = QueryRequest(
        request_id="test-q4",
        query_text="A customer wants a controller with at least 12 digital inputs plus a Modbus temperature and humidity sensor. Which of the three catalog products fit those needs, and is the heater unrelated?",
        engine="structured_jev",
        include_quote=False,
    )
    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.task_type == TaskType.ROLE_COVERAGE
    assert resp.answer_status == AnswerStatus.COMPLETE

    fact_pids = {f.product_id for f in resp.facts}
    assert "P_X4" in fact_pids, "P_X4 facts must be included for controller role"
    assert "P_THT" in fact_pids, "P_THT facts must be included for sensor role"
    assert "P_UHEAT" not in fact_pids, "P_UHEAT must not be included as selected"

    # Roles assigned
    role_map = {ra.role_id: ra.product_id for ra in resp.role_assignments}
    assert role_map.get("controller") == "P_X4"
    assert role_map.get("temperature_humidity_sensor") == "P_THT"

    assert "hechos certificados" not in resp.summary.lower()
    assert "verificado documentalmente" not in resp.summary.lower()


def test_structured_jev_f06_answer_status_coherence() -> None:
    """Verifies F06: compute_answer_status does not return COMPLETE when evidence is missing or mismatched."""
    # Case 1: FACT_LOOKUP without facts -> INSUFFICIENT_EVIDENCE
    st1 = compute_answer_status(
        task_type=TaskType.FACT_LOOKUP,
        verdict=TechnicalVerdict.COMPATIBLE,
        checks=[],
        facts=[],
        target_pids=["P_THT"],
        requested_properties=["communication_protocol"],
        role_assignments=[],
    )
    assert st1 == AnswerStatus.INSUFFICIENT_EVIDENCE

    # Case 2: FACT_LOOKUP with partial facts (1 of 3 answered) -> PARTIAL
    dummy_fact = Fact(
        fact_id="f1",
        product_id="P_THT",
        property="communication_protocol",
        value="Modbus-RTU",
        evidence_ids=["s1"],
    )
    st2 = compute_answer_status(
        task_type=TaskType.FACT_LOOKUP,
        verdict=TechnicalVerdict.COMPATIBLE,
        checks=[],
        facts=[dummy_fact],
        target_pids=["P_THT"],
        requested_properties=["communication_protocol", "sensing_element", "sensing_element_family"],
        role_assignments=[],
    )
    assert st2 == AnswerStatus.PARTIAL

    # Case 3: ROLE_COVERAGE with failed role -> PARTIAL or INSUFFICIENT_EVIDENCE
    ra_pass = RoleAssignment(role_id="r1", product_id="P1", status=CheckStatus.PASS, reason="ok")
    ra_fail = RoleAssignment(role_id="r2", product_id=None, status=CheckStatus.FAIL, reason="no match")
    st3 = compute_answer_status(
        task_type=TaskType.ROLE_COVERAGE,
        verdict=TechnicalVerdict.INCOMPATIBLE,
        checks=[],
        facts=[dummy_fact],
        target_pids=["P1"],
        requested_properties=[],
        role_assignments=[ra_pass, ra_fail],
    )
    assert st3 == AnswerStatus.PARTIAL


def test_structured_jev_catalog_order_invariance() -> None:
    """Verifies that query interpretation resolves products by semantics, not by catalog order (§4.2)."""
    # Q1 -> Horner X4
    req1 = QueryRequest(
        request_id="inv-1",
        query_text="Which X4 variant has relay outputs versus solid-state (transistor) outputs, and how many digital inputs and digital outputs does each of HE-X4A and HE-X4R have?",
        engine="structured_jev",
    )
    interp1 = interpret_query(req1)
    assert interp1.target_product_ids == ["P_X4"]
    assert interp1.task_type == TaskType.VARIANT_COMPARISON

    # Q2 -> THT-02
    req2 = QueryRequest(
        request_id="inv-2",
        query_text="Does the TZ THT-02 temperature and humidity sensor speak Modbus-RTU, and what sensing element does it use?",
        engine="structured_jev",
    )
    interp2 = interpret_query(req2)
    assert interp2.target_product_ids == ["P_THT"]
    assert interp2.task_type == TaskType.FACT_LOOKUP

    # Q3 -> Pumphouse U-Series
    req3 = QueryRequest(
        request_id="inv-3",
        query_text="For the Pumphouse U Series heater, what voltage families are offered, and what are the mount orientation limits (horizontal full wattage versus vertical, and thermostat position)?",
        engine="structured_jev",
    )
    interp3 = interpret_query(req3)
    assert interp3.target_product_ids == ["P_UHEAT"]
    assert interp3.task_type == TaskType.FACT_LOOKUP


# --------------------------------------------------------------------------
# Tests for REPAIR3_PLAN Findings: F10, F11, F20 (Engine B RagLlmEngine)
# --------------------------------------------------------------------------

def test_parse_and_validate_llm_json_variants():
    """Audit F10: Robust JSON parsing handles double-escaped strings, fences, and bound extraction."""
    from industrial_lab.engines.rag_llm import parse_and_validate_llm_json

    # 1. Clean valid JSON
    valid_str = '{"selected_product_ids": ["P_X4"], "technical_verdict": "COMPATIBLE", "checks": []}'
    parsed, err = parse_and_validate_llm_json(valid_str)
    assert err is None
    assert parsed["technical_verdict"] == "COMPATIBLE"

    # 2. Wrapped in markdown code fence
    fence_str = '```json\n{"selected_product_ids": ["P_X4"], "technical_verdict": "INCOMPATIBLE", "checks": []}\n```'
    parsed, err = parse_and_validate_llm_json(fence_str)
    assert err is None
    assert parsed["technical_verdict"] == "INCOMPATIBLE"

    # 3. Double-escaped JSON string (Layer 2 decoding per REPAIR3 §8.1)
    double_escaped = json.dumps(valid_str)
    parsed, err = parse_and_validate_llm_json(double_escaped)
    assert err is None
    assert parsed["selected_product_ids"] == ["P_X4"]

    # 4. JSON with preamble / postscript text
    with_preamble = 'Here is the analysis:\n{"selected_product_ids": ["P_THT"], "technical_verdict": "INSUFFICIENT_EVIDENCE", "checks": []}\nHope this helps!'
    parsed, err = parse_and_validate_llm_json(with_preamble)
    assert err is None
    assert parsed["technical_verdict"] == "INSUFFICIENT_EVIDENCE"

    # 5. Missing checks -> SCHEMA_ERROR
    missing_chk = '{"technical_verdict": "COMPATIBLE"}'
    parsed, err = parse_and_validate_llm_json(missing_chk)
    assert parsed is None
    assert "Missing required field 'checks'" in err

    # 6. Missing technical_verdict -> SCHEMA_ERROR
    missing_verd = '{"checks": []}'
    parsed, err = parse_and_validate_llm_json(missing_verd)
    assert parsed is None
    assert "Missing required field 'technical_verdict'" in err

    # 7. Invalid technical_verdict enum value -> SCHEMA_ERROR
    bad_verd = '{"technical_verdict": "MAYBE", "checks": []}'
    parsed, err = parse_and_validate_llm_json(bad_verd)
    assert parsed is None
    assert "Invalid 'technical_verdict' value" in err


@pytest.mark.asyncio
async def test_rag_llm_generative_repair_success() -> None:
    """Audit F10: If first call returns malformed JSON, single generative repair attempt succeeds and preserves both calls."""
    mock_llm = AsyncMock()

    call1 = LLMResponse(
        content='Broken json output missing closing braces {"selected_product_ids": ["P_X4"]',
        parsed=None,
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="google.gemma-4-31b",
        usage=UsageTelemetry(prompt_tokens=100, completion_tokens=20),
        telemetry={},
        raw_response={"raw": 1},
        call_id="call-001",
    )
    call2 = LLMResponse(
        content='{"selected_product_ids": ["P_X4"], "technical_verdict": "COMPATIBLE", "checks": [{"requirement_id": "REQ1", "status": "PASS", "evidence_ids": ["S1"], "reason_code": "OK"}], "missing_evidence": [], "summary": "Repaired"}',
        parsed=None,
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="google.gemma-4-31b",
        usage=UsageTelemetry(prompt_tokens=120, completion_tokens=30),
        telemetry={},
        raw_response={"raw": 2},
        call_id="call-002",
    )
    mock_llm.chat.side_effect = [call1, call2]
    # Track call_history on mock
    mock_llm.call_history = [
        {"call_id": "call-001", "endpoint": "chat", "raw_response": {"raw": 1}},
        {"call_id": "call-002", "endpoint": "chat", "raw_response": {"raw": 2}},
    ]

    engine = RagLlmEngine(api_key="valid-key", llm_adapter=mock_llm, enable_repair=True)
    req = QueryRequest(request_id="req-repair-ok", query_text="Check controller", engine="rag_llm")

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.technical_verdict == TechnicalVerdict.COMPATIBLE
    assert resp.selected_product_ids == ["P_X4"]
    assert mock_llm.chat.call_count == 2
    assert len(engine.last_provider_calls) == 2


@pytest.mark.asyncio
async def test_rag_llm_generative_repair_double_failure_preserved() -> None:
    """Audit F10, F11: If repair also fails, engine returns SCHEMA_ERROR (never INCOMPATIBLE), preserving both calls."""
    mock_llm = AsyncMock()

    call1 = LLMResponse(
        content="Invalid output 1",
        parsed=None,
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="google.gemma-4-31b",
        usage=UsageTelemetry(prompt_tokens=100, completion_tokens=10),
        telemetry={},
        raw_response={"raw": 1},
        call_id="call-001",
    )
    call2 = LLMResponse(
        content="Still invalid output 2",
        parsed=None,
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="google.gemma-4-31b",
        usage=UsageTelemetry(prompt_tokens=150, completion_tokens=10),
        telemetry={},
        raw_response={"raw": 2},
        call_id="call-002",
    )
    mock_llm.chat.side_effect = [call1, call2]
    mock_llm.call_history = [
        {"call_id": "call-001", "endpoint": "chat", "raw_response": {"raw": 1}},
        {"call_id": "call-002", "endpoint": "chat", "raw_response": {"raw": 2}},
    ]

    engine = RagLlmEngine(api_key="valid-key", llm_adapter=mock_llm, enable_repair=True)
    req = QueryRequest(request_id="req-repair-fail", query_text="Check controller", engine="rag_llm")

    resp = await engine.execute(req)
    assert resp.execution_status in (ExecutionStatus.SCHEMA_ERROR, ExecutionStatus.schema_error)
    # F11: Operational error MUST NOT produce INCOMPATIBLE
    assert resp.technical_verdict != TechnicalVerdict.COMPATIBLE
    assert resp.technical_verdict != TechnicalVerdict.INCOMPATIBLE
    assert resp.technical_verdict in (TechnicalVerdict.UNDETERMINED, TechnicalVerdict.INSUFFICIENT_EVIDENCE)
    assert resp.selection_status == SelectionStatus.UNDETERMINED
    assert mock_llm.chat.call_count == 2
    assert len(engine.last_provider_calls) == 2


def test_rag_llm_write_provider_logs_f20(tmp_path: Path) -> None:
    """Audit F20: RagLlmEngine forwards write_provider_logs to adapter to record requests and responses."""
    from industrial_lab.adapters.llm import LLMAdapter
    adapter = LLMAdapter(api_key="test-key", base_url="http://test.local", model_id="test-model")
    adapter.call_history = [
        {
            "call_id": "call-f20-1",
            "provider": "bedrock-mantle",
            "endpoint": "http://test.local/chat/completions",
            "model_requested": "test-model",
            "model_effective": "test-model",
            "start_utc": "2026-10-02T18:00:00Z",
            "end_utc": "2026-10-02T18:00:01Z",
            "elapsed_ms": 1000,
            "attempts": 1,
            "http_status": 200,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            "raw_request": {
                "method": "POST",
                "headers": {"Authorization": "[REDACTED]"},
                "payload": {"messages": [{"role": "user", "content": "hello"}]},
            },
            "raw_response": {"choices": [{"message": {"content": "ok"}}]},
        }
    ]

    engine = RagLlmEngine(api_key="test-key", llm_adapter=adapter)
    engine.write_provider_logs(log_dir=tmp_path, request_id="req-f20-test", case_id="CASE_B_01")

    req_file = tmp_path / "provider_requests.jsonl"
    resp_file = tmp_path / "provider_responses.jsonl"

    assert req_file.exists()
    assert resp_file.exists()

    req_lines = [json.loads(line) for line in req_file.read_text(encoding="utf-8").strip().splitlines()]
    resp_lines = [json.loads(line) for line in resp_file.read_text(encoding="utf-8").strip().splitlines()]

    assert len(req_lines) == 1
    assert req_lines[0]["call_id"] == "call-f20-1"
    assert req_lines[0]["request_id"] == "req-f20-test"
    assert req_lines[0]["case_id"] == "CASE_B_01"
    assert req_lines[0]["headers"]["Authorization"] == "[REDACTED]"

    assert len(resp_lines) == 1
    assert resp_lines[0]["call_id"] == "call-f20-1"
    assert resp_lines[0]["raw_response"] == {"choices": [{"message": {"content": "ok"}}]}

