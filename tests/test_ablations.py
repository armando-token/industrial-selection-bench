"""Comprehensive contractual and unit tests for ablation engines (MEGAPLAN §2.2).

Covers Workstream C (G5 Ablation Engines):
1. StructuredRulesEngine (§2.2 - H6):
   - Operates purely on deterministic DSL rules without any external LLM keys.
   - Evaluates exact checks, marks semantic checks as UNKNOWN (NO_SEMANTIC_MODEL).
   - Aggregates verdicts and produces valid QueryResponse.
2. StructuredLlmEngine (§2.2 - H4):
   - Shares exact same state, candidates, rules, and semantic question generator as System A.
   - Returns provider_error (code 30) when API key is missing and dry_run=False.
   - Supports dry-run mode (dry_run=True or LAB_DRY_RUN=1) executing offline on synthetic fixtures.
   - Dispatches semantic judgment to LLM with JSON formatted answers and confidence thresholds.
   - Strictly enforces that deterministic rule FAIL cannot be overridden by LLM semantic judgment.
3. RagLlmGuardedEngine (§2.2):
   - Applies deterministic rules engine as guardrail over System B (RAG) output.
   - Overrides hallucinated PASS to FAIL upon rule violation and updates verdict/quote.
   - Passes through valid recommendations when rules pass.
   - Strictly separates raw vs guarded results via last_raw_response.
   - Handles provider error pass-through and dry-run execution.
4. BenchmarkRunner integration:
   - Runner correctly dispatches structured_rules, structured_llm, and rag_llm_guarded.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from tests.conftest import requires_real_facts

from industrial_lab.adapters.llm import LLMAdapter, LLMResponse, UsageTelemetry
from industrial_lab.benchmark.runner import BenchmarkRunner, _run_async
from industrial_lab.engines.ablations import (
    RagLlmGuardedEngine,
    StructuredLlmEngine,
    StructuredRulesEngine,
)
from industrial_lab.engines.rag_llm import RagLlmEngine
from industrial_lab.engines.structured_jev import StructuredJevEngine
from industrial_lab.schemas import (
    CheckResult,
    CheckStatus,
    DecisionOrigin,
    ExecutionStatus,
    QueryRequest,
    QueryResponse,
    Requirement,
    RequirementKind,
    SelectionStatus,
    TaskType,
    TechnicalVerdict,
)


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
# 1. StructuredRulesEngine Tests (§2.2 - H6)
# ==============================================================================

@pytest.mark.asyncio
async def test_structured_rules_pure_deterministic_without_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies StructuredRulesEngine operates purely on deterministic DSL rules with zero API keys."""
    # Ensure all API keys are cleared
    _clear_all_provider_keys(monkeypatch)

    engine = StructuredRulesEngine()

    req = QueryRequest(
        request_id="rules-test-001",
        query_text="PLC controller with 24V supply and high-speed sorting",
        engine="structured_rules",
        requirements=[
            Requirement(
                requirement_id="supply_voltage_nominal_v",
                kind=RequirementKind.exact_property,
                operator="eq",
                target=24.0,
                unit="V",
                hard=True,
            ),
            Requirement(
                requirement_id="REQ_SEMANTIC_SPEED",
                kind=RequirementKind.semantic_use_case,
                operator="eq",
                target="high_speed_sorting",
                hard=True,
            ),
        ],
        requested_product_ids=["P1"],
    )

    resp = await engine.execute(req)

    assert resp.execution_status == ExecutionStatus.completed
    assert resp.engine == "structured_rules"
    # All checks in structured_rules must originate from rules
    assert all(c.decision_origin == DecisionOrigin.rule for c in resp.checks)

    # Exact property should be evaluated by rules engine
    volt_chk = next(c for c in resp.checks if c.requirement_id == "supply_voltage_nominal_v")
    assert volt_chk.status == CheckStatus.PASS

    # Semantic requirement must be UNKNOWN with NO_SEMANTIC_MODEL
    sem_chk = next(c for c in resp.checks if c.requirement_id == "REQ_SEMANTIC_SPEED")
    assert sem_chk.status == CheckStatus.UNKNOWN
    assert sem_chk.reason_code == "NO_SEMANTIC_MODEL"

    # Overall verdict should be INSUFFICIENT_EVIDENCE due to semantic UNKNOWN
    assert resp.technical_verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE
    assert "REQ_SEMANTIC_SPEED" in resp.missing_evidence


@pytest.mark.asyncio
async def test_structured_rules_m1_semantic_query_marks_unknown() -> None:
    """Verifies in M1 mode (natural language only), structured_rules marks semantic function as UNKNOWN."""
    engine = StructuredRulesEngine()
    req = QueryRequest(
        request_id="rules-m1-002",
        query_text="Sensor de temperatura para horno",
        engine="structured_rules",
        mode="M1",
        requested_product_ids=["P2"],
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.technical_verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE


# ==============================================================================
# 2. StructuredLlmEngine Tests (§2.2 - H4)
# ==============================================================================

@pytest.mark.asyncio
async def test_structured_llm_blocked_without_api_key_when_not_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """StructuredLlmEngine returns provider_error (code 30) when key is missing and not dry-run."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)

    engine = StructuredLlmEngine(api_key=None, dry_run=False)
    req = QueryRequest(
        request_id="sllm-blocked-1",
        query_text="Controller 24V",
        engine="structured_llm",
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.provider_error
    assert "LLM blocked" in resp.summary or "30" in resp.summary


@pytest.mark.asyncio
async def test_structured_llm_dry_run_executes_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """StructuredLlmEngine in dry-run mode completes successfully without API key."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)

    engine = StructuredLlmEngine(api_key=None, dry_run=True)
    assert engine.dry_run is True

    req = QueryRequest(
        request_id="sllm-dryrun-1",
        query_text="PLC controller 24V",
        engine="structured_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_VOLT",
                kind=RequirementKind.exact_property,
                operator="eq",
                target=24.0,
                unit="V",
            ),
            Requirement(
                requirement_id="REQ_SEMANTIC",
                kind=RequirementKind.semantic_use_case,
                operator="eq",
                target="industrial automation",
            ),
        ],
        requested_product_ids=["P1"],
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert "[DRY-RUN / SYNTHETIC FIXTURE" in resp.summary
    assert resp.telemetry_ref == f"struct-llm-dryrun-{req.request_id}"
    assert len(resp.checks) > 0


@pytest.mark.asyncio
async def test_structured_llm_environment_variable_activates_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting LAB_DRY_RUN=1 enables dry run mode automatically in StructuredLlmEngine."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LAB_DRY_RUN", "1")

    engine = StructuredLlmEngine(api_key=None)
    assert engine.dry_run is True

    req = QueryRequest(
        request_id="sllm-env-dryrun",
        query_text="Temperature sensor",
        engine="structured_llm",
        requested_product_ids=["P2"],
    )
    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.telemetry_ref == f"struct-llm-dryrun-{req.request_id}"


@pytest.mark.asyncio
async def test_structured_llm_shares_state_and_dispatches_to_llm() -> None:
    """Verifies StructuredLlmEngine formulates choice questions and dispatches semantic judgment to LLM."""
    mock_adapter = AsyncMock(spec=LLMAdapter)
    mock_resp = MagicMock()
    mock_resp.parsed = {
        "answers": {
            "P1::REQ_SEMANTIC": {
                "choice": "supported",
                "confidence": 0.95,
                "reason": "Verified in datasheet",
            }
        }
    }
    mock_adapter.chat.return_value = mock_resp

    engine = StructuredLlmEngine(llm_adapter=mock_adapter)

    req = QueryRequest(
        request_id="sllm-dispatch-1",
        query_text="High performance sorting",
        engine="structured_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_VOLT",
                kind=RequirementKind.exact_property,
                operator="eq",
                target=24.0,
                unit="V",
            ),
            Requirement(
                requirement_id="REQ_SEMANTIC",
                kind=RequirementKind.semantic_use_case,
                operator="eq",
                target="high performance sorting",
            ),
        ],
        requested_product_ids=["P1"],
    )

    resp = await engine.execute(req)

    # Verify adapter.chat was called
    assert mock_adapter.chat.called
    call_args = mock_adapter.chat.call_args[0][0]
    sys_msg = call_args[0]["content"]
    user_msg = call_args[1]["content"]

    # Verify atomic question formulation matching System A
    assert "P1::REQ_SEMANTIC" in user_msg or "P1_REQ_SEMANTIC" in user_msg
    assert "supported" in user_msg and "contradicted" in user_msg and "insufficient" in user_msg
    assert "ESTADO TÉCNICO COMPACTO" in user_msg

    # Verify checks
    assert resp.execution_status == ExecutionStatus.completed
    sem_chk = next(c for c in resp.checks if c.requirement_id == "REQ_SEMANTIC")
    assert sem_chk.status == CheckStatus.PASS
    assert sem_chk.decision_origin in (DecisionOrigin.llm, DecisionOrigin.combined)


@pytest.mark.asyncio
async def test_structured_llm_rule_fail_never_overridden_by_llm() -> None:
    """Verifies critical safety invariant: deterministic rule FAIL can NEVER be overridden by LLM PASS."""
    mock_adapter = AsyncMock(spec=LLMAdapter)
    # LLM hallucinates that P1 passes 48V requirement
    mock_resp = MagicMock()
    mock_resp.parsed = {
        "answers": {
            "P1_REQ_VOLT": {
                "choice": "supported",
                "confidence": 0.99,
                "reason": "LLM says yes",
            }
        }
    }
    mock_adapter.chat.return_value = mock_resp

    engine = StructuredLlmEngine(llm_adapter=mock_adapter)

    req = QueryRequest(
        request_id="sllm-fail-safety",
        query_text="Need 48V power supply",
        engine="structured_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_VOLT",
                kind=RequirementKind.exact_property,
                operator="eq",
                target=48.0,  # P1 has 24V, rule evaluates to FAIL
                unit="V",
                hard=True,
            )
        ],
        requested_product_ids=["P1"],
    )

    resp = await engine.execute(req)

    # Rule FAIL must not be overridden
    volt_chk = next(c for c in resp.checks if c.requirement_id == "REQ_VOLT")
    assert volt_chk.status == CheckStatus.FAIL
    assert resp.technical_verdict == TechnicalVerdict.INCOMPATIBLE
    assert resp.selected_product_ids == []


# ==============================================================================
# 3. RagLlmGuardedEngine Tests (§2.2)
# ==============================================================================

@pytest.mark.asyncio
async def test_rag_llm_guarded_overrides_rule_violation() -> None:
    """Verifies deterministic guardrail overrides hallucinated PASS when rule fails."""
    mock_rag = AsyncMock(spec=RagLlmEngine)
    mock_rag.execute.return_value = QueryResponse(
        request_id="rag-guard-test-1",
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
                requirement_id="REQ_VOLT",
                status=CheckStatus.PASS,
                evidence_ids=["DOC-P1-MANUAL:p01:s01"],
                reason_code="RAG_HALLUCINATED_48V",
                decision_origin=DecisionOrigin.llm,
            )
        ],
        missing_evidence=[],
        alternatives=[],
        quote=None,
        summary="RAG selected P1 as compatible.",
        telemetry_ref="rag-test",
    )

    guarded_engine = RagLlmGuardedEngine(rag_engine=mock_rag)

    req = QueryRequest(
        request_id="rag-guard-req-1",
        query_text="Need 48V supply",
        engine="rag_llm_guarded",
        requirements=[
            Requirement(
                requirement_id="REQ_VOLT",
                kind=RequirementKind.exact_property,
                operator="eq",
                target=48.0,  # P1 has 24V, deterministic rule fails
                unit="V",
                hard=True,
            )
        ],
        requested_product_ids=["P1"],
    )

    resp = await guarded_engine.execute(req)

    # 1. Guardrail must trigger
    assert resp.technical_verdict == TechnicalVerdict.INCOMPATIBLE
    assert resp.selected_product_ids == []
    guard_chk = next(c for c in resp.checks if c.requirement_id == "REQ_VOLT")
    assert guard_chk.status == CheckStatus.FAIL
    assert "GUARDRAIL_OVERRIDE" in str(guard_chk.reason_code)
    assert "[GUARDRAIL ACTIVADO]" in resp.summary

    # 2. Separation of raw vs guarded results
    assert guarded_engine.last_raw_response is not None
    assert guarded_engine.last_raw_response.technical_verdict == TechnicalVerdict.COMPATIBLE
    assert guarded_engine.last_raw_response.selected_product_ids == ["P1"]
    assert resp.technical_verdict != guarded_engine.last_raw_response.technical_verdict


@pytest.mark.asyncio
async def test_rag_llm_guarded_passes_when_rules_agree() -> None:
    """Verifies RagLlmGuardedEngine preserves COMPATIBLE when deterministic rules pass."""
    mock_rag = AsyncMock(spec=RagLlmEngine)
    mock_rag.execute.return_value = QueryResponse(
        request_id="rag-guard-test-2",
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
                requirement_id="REQ_VOLT",
                status=CheckStatus.PASS,
                evidence_ids=["DOC-P1-MANUAL:p01:s01"],
                reason_code="RAG_24V_CORRECT",
                decision_origin=DecisionOrigin.llm,
            )
        ],
        missing_evidence=[],
        alternatives=[],
        quote=None,
        summary="RAG selected P1 correctly.",
        telemetry_ref="rag-test",
    )

    guarded_engine = RagLlmGuardedEngine(rag_engine=mock_rag)

    req = QueryRequest(
        request_id="rag-guard-req-2",
        query_text="Need 24V supply",
        engine="rag_llm_guarded",
        requirements=[
            Requirement(
                requirement_id="REQ_VOLT",
                kind=RequirementKind.exact_property,
                operator="eq",
                target=24.0,  # P1 has 24V -> PASS
                unit="V",
                hard=True,
            )
        ],
        requested_product_ids=["P1"],
    )

    resp = await guarded_engine.execute(req)

    assert resp.technical_verdict == TechnicalVerdict.COMPATIBLE
    assert resp.selected_product_ids == ["P1"]
    assert "[GUARDRAIL ACTIVADO]" not in resp.summary
    assert guarded_engine.last_raw_response is not None


@pytest.mark.asyncio
async def test_rag_llm_guarded_dry_run_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies RagLlmGuardedEngine in dry-run mode executes successfully without API keys."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LAB_DRY_RUN", "1")

    engine = RagLlmGuardedEngine()
    assert engine.dry_run is True

    req = QueryRequest(
        request_id="rag-guarded-dryrun",
        query_text="PLC controller 24V",
        engine="rag_llm_guarded",
        requirements=[
            Requirement(
                requirement_id="REQ_VOLT",
                kind=RequirementKind.exact_property,
                operator="eq",
                target=24.0,
                unit="V",
            )
        ],
        requested_product_ids=["P1"],
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.telemetry_ref == f"rag-guarded-dryrun-{req.request_id}"
    assert engine.last_raw_response is not None


# ==============================================================================
# 4. BenchmarkRunner Dispatch Verification
# ==============================================================================

def test_runner_dispatches_all_three_ablations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies BenchmarkRunner dispatches structured_rules, structured_llm, and rag_llm_guarded."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LAB_DRY_RUN", "1")

    runner = BenchmarkRunner(
        config_path=Path("configs/experiment.yaml"),
        output_dir=tmp_path,
        allow_mock_fallback=False,
    )

    # 1. structured_rules
    fn_rules = runner._get_engine("structured_rules")
    req_rules = QueryRequest(request_id="r1", engine="structured_rules", query_text="PLC 24V")
    _, resp_rules = fn_rules(req_rules)
    assert resp_rules.execution_status == ExecutionStatus.completed
    assert resp_rules.engine == "structured_rules"

    # 2. structured_llm (dry-run under LAB_DRY_RUN=1)
    fn_sllm = runner._get_engine("structured_llm")
    req_sllm = QueryRequest(request_id="r2", engine="structured_llm", query_text="PLC 24V")
    _, resp_sllm = fn_sllm(req_sllm)
    assert resp_sllm.execution_status == ExecutionStatus.completed
    assert resp_sllm.engine == "structured_llm"
    assert resp_sllm.telemetry_ref.startswith("struct-llm-dryrun-")

    # 3. rag_llm_guarded (dry-run under LAB_DRY_RUN=1)
    fn_guarded = runner._get_engine("rag_llm_guarded")
    req_guarded = QueryRequest(request_id="r3", engine="rag_llm_guarded", query_text="PLC 24V")
    _, resp_guarded = fn_guarded(req_guarded)
    assert resp_guarded.execution_status == ExecutionStatus.completed
    assert resp_guarded.engine == "rag_llm_guarded"


# ==============================================================================
# 5. Product ID Underscore Safety & Distinguishability (§3.2, §1, §11)
# ==============================================================================

@pytest.mark.asyncio
async def test_structured_llm_product_id_with_underscore_not_truncated() -> None:
    """Verifies candidate IDs with underscores (P_X4, P_THT, P_UHEAT) are never truncated to 'P' (§3.2)."""
    mock_adapter = AsyncMock(spec=LLMAdapter)
    mock_resp = MagicMock()
    # Test both :: format and _ format
    mock_resp.parsed = {
        "answers": {
            "P_X4::REQ_SEMANTIC": {
                "choice": "supported",
                "confidence": 0.95,
                "reason": "Horner X4 supports logic",
            },
            "P_THT_REQ_SEMANTIC": {
                "choice": "supported",
                "confidence": 0.92,
                "reason": "TZ THT sensor supports humidity",
            },
        }
    }
    mock_adapter.chat.return_value = mock_resp

    engine = StructuredLlmEngine(llm_adapter=mock_adapter)

    req = QueryRequest(
        request_id="underscore-id-test",
        query_text="Industrial controller and sensor",
        engine="structured_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_SEMANTIC",
                kind=RequirementKind.semantic_use_case,
                operator="eq",
                target="automation",
            )
        ],
        requested_product_ids=["P_X4", "P_THT"],
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed

    # Ensure neither product was truncated to "P"
    req_ids = [c.requirement_id for c in resp.checks]
    assert all(r == "REQ_SEMANTIC" for r in req_ids)
    assert "X4_REQ_SEMANTIC" not in req_ids
    assert "THT_REQ_SEMANTIC" not in req_ids


@pytest.mark.asyncio
async def test_engines_distinguishability_a_d_e() -> None:
    """Verifies that Controls D & E and System A are strictly distinguishable by decision origin and mechanism (§1, §13).

    - System A (structured_jev): decision_origin = DecisionOrigin.jev
    - System D (structured_llm): decision_origin = DecisionOrigin.llm
    - System E (structured_rules): decision_origin = DecisionOrigin.rule, semantic UNKNOWN (NO_SEMANTIC_MODEL)
    """
    sem_req = Requirement(
        requirement_id="REQ_SEM",
        kind=RequirementKind.semantic_use_case,
        operator="eq",
        target="automation",
    )

    # 1. System E: Rules only, zero model calls
    engine_e = StructuredRulesEngine()
    req_e = QueryRequest(
        request_id="req-dist-e",
        query_text="Automation controller",
        engine="structured_rules",
        requirements=[sem_req],
        requested_product_ids=["P1"],
    )
    resp_e = await engine_e.execute(req_e)
    assert resp_e.engine == "structured_rules"
    chk_e = next(c for c in resp_e.checks if c.requirement_id == "REQ_SEM")
    assert chk_e.status == CheckStatus.UNKNOWN
    assert chk_e.decision_origin == DecisionOrigin.rule
    assert chk_e.reason_code == "NO_SEMANTIC_MODEL"
    assert chk_e.model_probabilities is None

    # 2. System D: LLM judgment slot (Gemma slot)
    mock_llm_adapter = AsyncMock(spec=LLMAdapter)
    mock_resp = MagicMock()
    mock_resp.parsed = {
        "answers": {
            "P1::REQ_SEM": {"choice": "supported", "confidence": 0.94, "reason": "Gemma decision"}
        }
    }
    mock_llm_adapter.chat.return_value = mock_resp
    engine_d = StructuredLlmEngine(llm_adapter=mock_llm_adapter)
    req_d = QueryRequest(
        request_id="req-dist-d",
        query_text="Automation controller",
        engine="structured_llm",
        requirements=[sem_req],
        requested_product_ids=["P1"],
    )
    resp_d = await engine_d.execute(req_d)
    assert resp_d.engine == "structured_llm"
    chk_d = next(c for c in resp_d.checks if c.requirement_id == "REQ_SEM")
    assert chk_d.decision_origin in (DecisionOrigin.llm, DecisionOrigin.combined)
    assert chk_d.status == CheckStatus.PASS

    # Origins must be distinguishable: System E is purely rules without model, System D has model judgment
    assert chk_e.decision_origin == DecisionOrigin.rule
    assert chk_e.decision_origin != chk_d.decision_origin


@pytest.mark.asyncio
async def test_ablation_engines_refuse_gold_in_request() -> None:
    """Verifies that engines refuse gold leakage in QueryRequest per §11 and §18.1."""
    engine_d = StructuredLlmEngine(dry_run=True)
    engine_e = StructuredRulesEngine()
    engine_guarded = RagLlmGuardedEngine(dry_run=True)

    req = QueryRequest(
        request_id="gold-leak-test",
        query_text="Check controller",
        engine="structured_llm",
    )
    # Simulate gold leakage into the request
    object.__setattr__(req, "gold", {"acceptable_selections": [["P1"]]})

    with pytest.raises(AssertionError, match="Engine safety violation: gold leaked in request"):
        await engine_d.execute(req)

    with pytest.raises(AssertionError, match="Engine safety violation: gold leaked in request"):
        await engine_e.execute(req)

    with pytest.raises(AssertionError, match="Engine safety violation: gold leaked in request"):
        await engine_guarded.execute(req)


# ==============================================================================
# 6. Comprehensive Attribution & Runner Verification (§15, F07)
# ==============================================================================

@requires_real_facts
@pytest.mark.asyncio
async def test_q4_attribution_all_rules_across_a_d_e() -> None:
    """Verifies §15 & F07: Q4 is fully resolved by deterministic rules across A, D, and E.

    All 4 checks originate from rules (DecisionOrigin.rule).
    Because no model judgment was used for any final check, model_decision_used == False across all three.
    """
    mock_jev = AsyncMock()
    mock_jev.judge_state.return_value = {"results": {}}
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = MagicMock(parsed={"answers": {}})

    engine_a = StructuredJevEngine(api_key="valid-key", jev_adapter=mock_jev)
    engine_d = StructuredLlmEngine(llm_adapter=mock_llm)
    engine_e = StructuredRulesEngine()

    req = QueryRequest(
        request_id="req-q4-attribution",
        query_text="A customer wants a controller with at least 12 digital inputs plus a Modbus temperature and humidity sensor. Which of the three catalog products fit those needs, and is the heater unrelated?",
        engine="structured_jev",
        requested_product_ids=["P_X4", "P_THT", "P_UHEAT"],
        include_quote=False,
    )

    for eng, label in [(engine_a, "structured_jev"), (engine_d, "structured_llm"), (engine_e, "structured_rules")]:
        resp = await eng.execute(req)
        assert resp.execution_status == ExecutionStatus.completed
        assert resp.task_type == TaskType.ROLE_COVERAGE
        assert resp.technical_verdict == TechnicalVerdict.COMPATIBLE
        assert resp.selection_status == SelectionStatus.SATISFIED
        assert "P_X4" in resp.selected_product_ids
        assert "P_THT" in resp.selected_product_ids
        assert "P_UHEAT" not in resp.selected_product_ids

        # Exactly 4 checks in role coverage: 1 controller + 2 sensor + 1 heater relevance
        assert len(resp.checks) == 4
        # Every check must originate strictly from rules
        for c in resp.checks:
            assert c.decision_origin == DecisionOrigin.rule, f"Engine {label} check {c.requirement_id} origin must be rule"

        # Crucial invariant: when model decision was not used, model_decision_used MUST be False
        assert resp.model_decision_used is False, f"Engine {label} model_decision_used must be False for Q4"

        # Heater must be marked IRRELEVANT_FOR_ROLE
        heater_chk = next(c for c in resp.checks if c.requirement_id == "REQ_HEATER_ROLE_COVERAGE")
        assert heater_chk.status == CheckStatus.IRRELEVANT_FOR_ROLE


@pytest.mark.asyncio
async def test_semantic_query_attribution_a_d_e() -> None:
    """Verifies §15 & F07: Semantic requirements utilize model judgment in A and D, but return UNKNOWN in E.

    - System A (JEV): model_decision_used == True, decision_origin in (jev, combined)
    - System D (LLM): model_decision_used == True, decision_origin in (llm, combined)
    - System E (Rules): model_decision_used == False, decision_origin == rule, reason_code == NO_SEMANTIC_MODEL
    """
    mock_jev = AsyncMock()
    mock_jev.judge_state.return_value = {
        "results": {
            "q0001": {
                "choice": "supported",
                "distribution": {"supported": 0.95, "contradicted": 0.03, "insufficient": 0.02},
                "confidence": 0.95,
            }
        }
    }

    mock_llm = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.parsed = {
        "answers": {
            "P1::REQ_SEM": {"choice": "supported", "confidence": 0.95, "reason": "Verified by LLM"}
        }
    }
    mock_llm.chat.return_value = mock_resp

    engine_a = StructuredJevEngine(api_key="valid-key", jev_adapter=mock_jev)
    engine_d = StructuredLlmEngine(llm_adapter=mock_llm)
    engine_e = StructuredRulesEngine()

    req = QueryRequest(
        request_id="sem-attr-test",
        query_text="Food packaging certification",
        engine="structured_jev",
        requirements=[
            Requirement(
                requirement_id="REQ_SEM",
                kind=RequirementKind.semantic_use_case,
                operator="eq",
                target="food packaging certified",
            )
        ],
        requested_product_ids=["P1"],
        include_quote=False,
    )

    # 1. System A
    resp_a = await engine_a.execute(req)
    assert resp_a.model_decision_used is True
    chk_a = next(c for c in resp_a.checks if c.requirement_id == "REQ_SEM")
    assert chk_a.decision_origin in (DecisionOrigin.jev, DecisionOrigin.combined)
    assert chk_a.status == CheckStatus.PASS

    # 2. System D
    resp_d = await engine_d.execute(req)
    assert resp_d.model_decision_used is True
    chk_d = next(c for c in resp_d.checks if c.requirement_id == "REQ_SEM")
    assert chk_d.decision_origin in (DecisionOrigin.llm, DecisionOrigin.combined)
    assert chk_d.status == CheckStatus.PASS

    # 3. System E
    resp_e = await engine_e.execute(req)
    assert resp_e.model_decision_used is False
    chk_e = next(c for c in resp_e.checks if c.requirement_id == "REQ_SEM")
    assert chk_e.decision_origin == DecisionOrigin.rule
    assert chk_e.status == CheckStatus.UNKNOWN
    assert chk_e.reason_code == "NO_SEMANTIC_MODEL"


def test_runner_dispatches_all_five_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies BenchmarkRunner dispatches all 5 engines (A, B, C, D, E) and schedules them in balanced order."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LAB_DRY_RUN", "1")

    runner = BenchmarkRunner(
        config_path=Path("configs/experiment.yaml"),
        output_dir=tmp_path,
        allow_mock_fallback=False,
    )

    # Register deterministic mock for structured_jev to avoid blocked error without network calls
    mock_jev = AsyncMock()
    mock_jev.judge_state.return_value = {"results": {}}
    mock_jev_engine = StructuredJevEngine(api_key="valid-key", jev_adapter=mock_jev)
    runner.register_engine("structured_jev", lambda req: ({"mock": True}, _run_async(mock_jev_engine.execute(req))))

    all_engines = ["structured_jev", "rag_llm", "scrape_llm", "structured_llm", "structured_rules"]

    for eng_name in all_engines:
        fn = runner._get_engine(eng_name)
        req = QueryRequest(request_id=f"run-{eng_name}", engine=eng_name, query_text="PLC controller 24V")
        _, resp = fn(req)
        assert resp.execution_status == ExecutionStatus.completed, f"{eng_name} failed: {resp.summary}"
        assert resp.engine == eng_name

    # Verify balanced permutation schedule across all 5 engines
    perms = runner.get_all_engine_permutations(all_engines)
    assert len(perms) == 120  # 5! = 120
    order_1 = runner.get_balanced_order("case_1", 0, 1, all_engines)
    order_2 = runner.get_balanced_order("case_1", 1, 1, all_engines)
    assert len(order_1) == 5
    assert set(order_1) == set(all_engines)
    assert order_1 != order_2


