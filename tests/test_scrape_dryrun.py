"""Tests for ScrapeLlmEngine dry-run mode and synthetic fixture simulation.

Verifies:
- Default mode without API key returns provider_error with code 30.
- dry_run=True executes without API key, returns ExecutionStatus.completed,
  contains the "[DRY-RUN / SYNTHETIC FIXTURE - NON-OFFICIAL]" label in summary
  and dry-run reason codes, and sets telemetry_ref.
- LAB_DRY_RUN environment variable activates dry run.
"""

from __future__ import annotations

import os
import pytest

from tests.conftest import requires_real_pdfs

from industrial_lab.engines.scrape_llm import ScrapeLlmEngine
from industrial_lab.schemas import (
    CheckStatus,
    DecisionOrigin,
    ExecutionStatus,
    QueryRequest,
    Requirement,
    RequirementKind,
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


@pytest.mark.asyncio
async def test_scrape_llm_default_mode_blocked_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default mode without API key returns provider_error with code 30."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)

    engine = ScrapeLlmEngine(api_key=None, dry_run=False)
    assert engine.dry_run is False

    req = QueryRequest(
        request_id="req-blocked-1",
        query_text="Browse catalog for PLC controllers",
        engine="scrape_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_VOLT_24V",
                kind=RequirementKind.exact_property,
                operator="eq",
                target=24.0,
            )
        ],
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.provider_error
    assert "código 30" in resp.summary or "30" in resp.summary
    assert len(resp.checks) == 1
    assert resp.checks[0].reason_code == "PROVIDER_BLOCKED_CODE_30"
    assert resp.technical_verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_scrape_llm_dry_run_flag_executes_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    """dry_run=True executes without API key, returning completed status and required markers."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)

    engine = ScrapeLlmEngine(api_key=None, dry_run=True)
    assert engine.dry_run is True

    req = QueryRequest(
        request_id="req-dry-flag",
        query_text="Find P1 controller with Modbus RTU support",
        engine="scrape_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_MODBUS",
                kind=RequirementKind.exact_property,
                operator="eq",
                target="Modbus RTU",
            )
        ],
        requested_product_ids=["P1"],
        include_quote=True,
    )

    resp = await engine.execute(req)

    # 1. Execution status
    assert resp.execution_status == ExecutionStatus.completed

    # 2. Mandatory prefix in summary
    assert "[DRY-RUN / SYNTHETIC FIXTURE - NON-OFFICIAL]" in resp.summary
    expected_prefix = (
        "[DRY-RUN / SYNTHETIC FIXTURE - NON-OFFICIAL] scrape_llm executed in dry-run mode on synthetic test fixtures. "
        "This is NOT an official benchmark outcome for System C. Technical verdict:"
    )
    assert expected_prefix in resp.summary

    # 3. Telemetry reference
    assert resp.telemetry_ref == f"scrape-dryrun-{req.request_id}"

    # 4. Check results and reasons
    assert len(resp.checks) == 1
    chk = resp.checks[0]
    assert chk.requirement_id == "REQ_MODBUS"
    assert chk.decision_origin == DecisionOrigin.llm
    assert chk.reason_code == "[DRY-RUN / SYNTHETIC FIXTURE] REQ_MODBUS_EVALUATED_ONLINE_TOOL_MOCK"
    assert chk.status == CheckStatus.PASS
    assert len(chk.evidence_ids) > 0

    # 5. Technical verdict and quote
    assert resp.technical_verdict == TechnicalVerdict.COMPATIBLE
    assert resp.selected_product_ids == ["P1"]
    assert resp.quote is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("env_val", ["1", "true", "yes", "True", "YES"])
async def test_scrape_llm_dry_run_env_var_activates(monkeypatch: pytest.MonkeyPatch, env_val: str) -> None:
    """LAB_DRY_RUN environment variable activates dry run without explicit constructor flag."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LAB_DRY_RUN", env_val)

    engine = ScrapeLlmEngine(api_key=None)
    assert engine.dry_run is True

    req = QueryRequest(
        request_id=f"req-dry-env-{env_val}",
        query_text="Explore P2 power supply",
        engine="scrape_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_P2_OUT",
                kind=RequirementKind.exact_property,
                operator="eq",
                target="24V DC",
            )
        ],
        requested_product_ids=["P2"],
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert "[DRY-RUN / SYNTHETIC FIXTURE - NON-OFFICIAL]" in resp.summary
    assert resp.telemetry_ref == f"scrape-dryrun-{req.request_id}"
    assert len(resp.checks) == 1
    assert resp.checks[0].reason_code == "[DRY-RUN / SYNTHETIC FIXTURE] REQ_P2_OUT_EVALUATED_ONLINE_TOOL_MOCK"
    assert resp.checks[0].decision_origin == DecisionOrigin.llm


@pytest.mark.asyncio
async def test_scrape_llm_dry_run_missing_evidence_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulated inspection produces UNKNOWN and INSUFFICIENT_EVIDENCE when target spec is absent."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LAB_DRY_RUN", "1")

    engine = ScrapeLlmEngine()

    req = QueryRequest(
        request_id="req-dry-missing",
        query_text="Check nonexistent property on P1",
        engine="scrape_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_ABSENT",
                kind=RequirementKind.exact_property,
                operator="eq",
                target="NonexistentSpecialProtocolX999",
                hard=True,
            )
        ],
        requested_product_ids=["P1"],
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert "[DRY-RUN / SYNTHETIC FIXTURE - NON-OFFICIAL]" in resp.summary
    assert resp.technical_verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE
    assert len(resp.checks) == 1
    assert resp.checks[0].status == CheckStatus.UNKNOWN
    assert "REQ_ABSENT" in resp.missing_evidence
    assert resp.checks[0].reason_code == "[DRY-RUN / SYNTHETIC FIXTURE] REQ_ABSENT_EVALUATED_ONLINE_TOOL_MOCK"


@pytest.mark.asyncio
async def test_scrape_tools_verification() -> None:
    """Verify standard tool names, no answer_Q1 tools, and no hints from gold."""
    from industrial_lab.tools.shop_http import ShopHttpClient

    client = ShopHttpClient()
    tools = client.get_tool_definitions()
    tool_names = [t["function"]["name"] for t in tools]

    expected_tools = [
        "list_products",
        "fetch_product_page",
        "open_document",
        "read_document_pages",
        "find_document_text",
        "get_commerce",
        "create_quote",
    ]
    assert tool_names == expected_tools
    assert "answer_q1" not in [n.lower() for n in tool_names]
    assert not any("answer_" in n.lower() for n in tool_names)

    # Check for gold hints in descriptions
    for t in tools:
        desc = t["function"]["description"].lower()
        assert "gold" not in desc
        assert "ground truth" not in desc
        assert "q1" not in desc
        assert "q2" not in desc
        assert "q3" not in desc
        assert "q4" not in desc


@requires_real_pdfs
@pytest.mark.asyncio
async def test_shop_http_pdf_caching_per_request() -> None:
    """Ensure that PDF parsing is cached within request_cache so multiple searches/reads do not re-parse."""
    from typing import Any
    from industrial_lab.tools.shop_http import ShopHttpClient

    request_cache: dict[str, Any] = {}
    client = ShopHttpClient(request_cache=request_cache)

    # First open
    open_1 = await client.open_document("D_THT_MANUAL")
    assert open_1["status"] == "opened"
    assert open_1["page_count"] == 9

    # Second open of same document ID
    open_2 = await client.open_document("D_THT_MANUAL")
    assert open_2["status"] == "already_opened"

    # Open by filename alias
    open_alias = await client.open_document("tz_tht02_temp_humidity_sensor_user_manual.pdf")
    assert open_alias["status"] == "already_opened"

    # read_document_pages hits cache
    read_res = await client.read_document_pages("D_THT_MANUAL", [1, 2])
    assert read_res["read_pages_count"] == 2

    # find_document_text hits cache
    find_res = await client.find_document_text("D_THT_MANUAL", "Modbus")
    assert find_res["match_count"] > 0


@pytest.mark.asyncio
async def test_scrape_llm_budget_configuration() -> None:
    """Verify bounded budget: 7 tool rounds + 1 final round (max 8 rounds total) and max 12 tool calls (REPAIR3_PLAN §9)."""
    engine = ScrapeLlmEngine()
    assert engine.max_tool_rounds == 7
    assert engine.max_model_rounds == 8
    assert engine.max_tool_calls == 12
    assert engine.reserve_final_model_round is True
    assert engine.final_max_output_tokens == 2048
    assert engine.cross_request_document_cache is False


@pytest.mark.asyncio
async def test_scrape_llm_records_tool_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify tool traces record round, tool_name, sanitized_args, result_summary, duration, finish_reason."""
    _clear_all_provider_keys(monkeypatch)
    engine = ScrapeLlmEngine(dry_run=True)
    req = QueryRequest(
        request_id="req-trace-test",
        query_text="Find P1 controller",
        engine="scrape_llm",
        requirements=[Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target="Sinamics")],
    )
    await engine.execute(req)

    traces = engine.last_tool_traces
    assert len(traces) >= 3
    for tr in traces:
        assert "round" in tr
        assert "tool_name" in tr
        assert "sanitized_args" in tr
        assert "result_summary" in tr
        assert "duration" in tr
        assert "finish_reason" in tr


@pytest.mark.asyncio
async def test_scrape_llm_live_tool_trace_sanitization() -> None:
    """Verify live tool execution sanitizes sensitive arguments in traces."""
    from unittest.mock import AsyncMock
    from industrial_lab.adapters.llm import LLMResponse, ToolCall, UsageTelemetry

    mock_adapter = AsyncMock()
    calls = 0

    def chat_mock(messages, tools=None, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return LLMResponse(
                content=None,
                parsed=None,
                tool_calls=[ToolCall(id="tc1", type="function", name="list_products", arguments='{"auth_token": "secret-123"}')],
                role="assistant",
                finish_reason="tool_calls",
                model="test-model",
                usage=UsageTelemetry(),
                telemetry={},
                raw_response={},
            )
        return LLMResponse(
            content='{"selected_product_ids": ["P_X4"], "technical_verdict": "COMPATIBLE", "checks": []}',
            parsed=None,
            tool_calls=[],
            role="assistant",
            finish_reason="stop",
            model="test-model",
            usage=UsageTelemetry(),
            telemetry={},
            raw_response={},
        )

    mock_adapter.chat.side_effect = chat_mock
    engine = ScrapeLlmEngine(llm_adapter=mock_adapter, dry_run=False)
    req = QueryRequest(
        request_id="req-live-trace",
        query_text="Explore products",
        engine="scrape_llm",
    )
    await engine.execute(req)

    assert len(engine.last_tool_traces) == 1
    tr = engine.last_tool_traces[0]
    assert tr["tool_name"] == "list_products"
    assert tr["sanitized_args"].get("auth_token") == "[REDACTED]"


@pytest.mark.asyncio
async def test_scrape_llm_budget_exceeded_preserves_checks_and_content_f12_f13() -> None:
    """F12/F13: System C does NOT demote PASS checks to UNKNOWN upon budget_exceeded (REPAIR3_PLAN §9).

    Operational status is ExecutionStatus.budget_exceeded while content_coverage_complete is True.
    """
    from unittest.mock import AsyncMock
    from industrial_lab.adapters.llm import LLMResponse, ToolCall, UsageTelemetry
    from industrial_lab.schemas import AnswerStatus

    mock_adapter = AsyncMock()

    def chat_mock(messages, tools=None, **kwargs):
        if tools:
            return LLMResponse(
                content=None,
                parsed=None,
                tool_calls=[ToolCall(id="tc1", type="function", name="list_products", arguments="{}")],
                role="assistant",
                finish_reason="tool_calls",
                model="test-model",
                usage=UsageTelemetry(),
                telemetry={},
                raw_response={},
            )
        return LLMResponse(
            content='{"selected_product_ids": ["P_X4"], "technical_verdict": "COMPATIBLE", "checks": [{"requirement_id": "R1", "status": "PASS", "evidence_ids": ["DOC-1:p01:s01"], "reason_code": "VOLTAGE_MATCH"}], "summary": "Voltage matches requirement."}',
            parsed=None,
            tool_calls=[],
            role="assistant",
            finish_reason="stop",
            model="test-model",
            usage=UsageTelemetry(),
            telemetry={},
            raw_response={},
        )

    mock_adapter.chat.side_effect = chat_mock
    engine = ScrapeLlmEngine(llm_adapter=mock_adapter, max_tool_calls=2, dry_run=False)
    req = QueryRequest(
        request_id="req-safety-budget",
        query_text="Validate requirement under low budget",
        engine="scrape_llm",
        requirements=[Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target=10)],
    )
    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.budget_exceeded
    assert len(resp.checks) == 1
    assert resp.checks[0].status == CheckStatus.PASS
    assert resp.checks[0].reason_code == "VOLTAGE_MATCH"
    assert resp.checks[0].reason_code != "DEMOTED_DUE_TO_BUDGET_EXCEEDED"
    assert resp.content_coverage_complete is True
    assert resp.answer_status == AnswerStatus.COMPLETE
    assert "Voltage matches" in resp.summary


@pytest.mark.asyncio
async def test_scrape_llm_q3_content_preservation_f13() -> None:
    """F13: In C Q3, summary contains voltage families and mount restrictions from documents.

    Do NOT treat content as incomplete solely because tool rounds reached limit.
    """
    import json
    from unittest.mock import AsyncMock
    from industrial_lab.adapters.llm import LLMResponse, ToolCall, UsageTelemetry
    from industrial_lab.schemas import AnswerStatus

    mock_adapter = AsyncMock()

    def chat_mock(messages, tools=None, **kwargs):
        if tools:
            return LLMResponse(
                content=None,
                parsed=None,
                tool_calls=[ToolCall(id="tc1", type="function", name="list_products", arguments="{}")],
                role="assistant",
                finish_reason="tool_calls",
                model="test-model",
                usage=UsageTelemetry(),
                telemetry={},
                raw_response={},
            )
        return LLMResponse(
            content=json.dumps({
                "selected_product_ids": ["P_UHEAT"],
                "technical_verdict": "COMPATIBLE",
                "checks": [
                    {
                        "requirement_id": "voltage_families",
                        "status": "PASS",
                        "evidence_ids": ["D_UHEAT_DATASHEET:p01:s01"],
                        "reason_code": "Offers 120V and triple-rated 240/208/120V families."
                    },
                    {
                        "requirement_id": "mount_orientation_limits",
                        "status": "PASS",
                        "evidence_ids": ["D_UHEAT_DATASHEET:p01:s01"],
                        "reason_code": "Horizontal mount allows full wattage; vertical mount is limited to up to 500W."
                    }
                ],
                "summary": "The Pumphouse U Series heater is available in 120V and triple-rated 240/208/120V voltage families. Mounting is permitted horizontally for full wattage or vertically for units up to 500W."
            }),
            parsed=None,
            tool_calls=[],
            role="assistant",
            finish_reason="stop",
            model="test-model",
            usage=UsageTelemetry(),
            telemetry={},
            raw_response={},
        )

    mock_adapter.chat.side_effect = chat_mock
    engine = ScrapeLlmEngine(llm_adapter=mock_adapter, max_tool_calls=2, dry_run=False)
    req = QueryRequest(
        request_id="req-c-q3",
        query_text="What voltage families and mount limits apply to Pumphouse U Series?",
        engine="scrape_llm",
        requirements=[
            Requirement(requirement_id="voltage_families", kind=RequirementKind.exact_property, operator="eq", target="120V"),
            Requirement(requirement_id="mount_orientation_limits", kind=RequirementKind.exact_property, operator="eq", target="horizontal"),
        ],
    )
    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.budget_exceeded
    assert resp.content_coverage_complete is True
    assert resp.answer_status == AnswerStatus.COMPLETE
    assert len(resp.checks) == 2
    assert all(c.status == CheckStatus.PASS for c in resp.checks)
    assert "voltage families" in resp.summary
    assert "Mounting is permitted" in resp.summary


@pytest.mark.asyncio
async def test_scrape_llm_cross_request_doc_caching_configuration() -> None:
    """Verify cross-request document cache configuration (REPAIR3_PLAN §9)."""
    # 1. Cold cache by default
    engine_cold = ScrapeLlmEngine(cross_request_document_cache=False)
    assert engine_cold.cross_request_document_cache is False

    # 2. Enabled cross-request cache
    engine_shared = ScrapeLlmEngine(cross_request_document_cache=True)
    assert engine_shared.cross_request_document_cache is True
    assert engine_shared._instance_doc_cache == {}


@pytest.mark.asyncio
async def test_scrape_llm_safety_timeout_never_compatible() -> None:
    """Local safety: timeout never becomes COMPATIBLE."""
    import asyncio
    from unittest.mock import AsyncMock
    from industrial_lab.adapters.llm import LLMResponse, ToolCall, UsageTelemetry

    mock_adapter = AsyncMock()

    async def chat_mock(*args, **kwargs):
        await asyncio.sleep(0.05)
        return LLMResponse(
            content='{"selected_product_ids": ["P_X4"], "technical_verdict": "COMPATIBLE", "checks": [{"requirement_id": "R1", "status": "PASS"}]}',
            parsed=None,
            tool_calls=[ToolCall(id="tc1", type="function", name="list_products", arguments="{}")],
            role="assistant",
            finish_reason="tool_calls",
            model="test-model",
            usage=UsageTelemetry(),
            telemetry={},
            raw_response={},
        )

    mock_adapter.chat = chat_mock
    engine = ScrapeLlmEngine(llm_adapter=mock_adapter, timeout_seconds=0.01, dry_run=False)
    req = QueryRequest(
        request_id="req-safety-timeout",
        query_text="Test timeout safety",
        engine="scrape_llm",
        requirements=[Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target=10)],
    )
    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.timeout
    assert resp.technical_verdict != TechnicalVerdict.COMPATIBLE
    assert resp.technical_verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_scrape_llm_safety_broken_json_never_compatible() -> None:
    """Local safety: broken JSON output never becomes COMPATIBLE."""
    from unittest.mock import AsyncMock
    from industrial_lab.adapters.llm import LLMResponse, UsageTelemetry

    mock_adapter = AsyncMock()
    mock_adapter.chat.return_value = LLMResponse(
        content='This is invalid json { "technical_verdict": "COMPATIBLE"',
        parsed=None,
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="test-model",
        usage=UsageTelemetry(),
        telemetry={},
        raw_response={},
    )
    engine = ScrapeLlmEngine(llm_adapter=mock_adapter, dry_run=False)
    req = QueryRequest(
        request_id="req-safety-broken",
        query_text="Test broken json safety",
        engine="scrape_llm",
        requirements=[Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target=10)],
    )
    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.invalid_output
    assert resp.technical_verdict != TechnicalVerdict.COMPATIBLE
    assert resp.technical_verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE


@pytest.mark.asyncio
async def test_scrape_llm_safety_empty_output_never_compatible() -> None:
    """Local safety: empty output never becomes COMPATIBLE."""
    from unittest.mock import AsyncMock
    from industrial_lab.adapters.llm import LLMResponse, UsageTelemetry

    mock_adapter = AsyncMock()
    mock_adapter.chat.return_value = LLMResponse(
        content="",
        parsed=None,
        tool_calls=[],
        role="assistant",
        finish_reason="stop",
        model="test-model",
        usage=UsageTelemetry(),
        telemetry={},
        raw_response={},
    )
    engine = ScrapeLlmEngine(llm_adapter=mock_adapter, dry_run=False)
    req = QueryRequest(
        request_id="req-safety-empty",
        query_text="Test empty output safety",
        engine="scrape_llm",
        requirements=[Requirement(requirement_id="R1", kind=RequirementKind.exact_property, operator="eq", target=10)],
    )
    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.invalid_output
    assert resp.technical_verdict != TechnicalVerdict.COMPATIBLE
    assert resp.technical_verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE


def test_system_c_does_not_read_a_facts_or_b_index() -> None:
    """C must not read A's private facts or B's prepared index."""
    import inspect
    from industrial_lab.engines import scrape_llm
    from industrial_lab.tools import shop_http

    src_scrape = inspect.getsource(scrape_llm)
    src_shop = inspect.getsource(shop_http)

    for term in ("facts.reviewed", "data/facts", "BM25Index", "retriever", "data/index"):
        assert term not in src_scrape, f"System C scrape_llm leaked access to {term}"
        assert term not in src_shop, f"ShopHttpClient leaked access to {term}"


@requires_real_pdfs
def test_shop_serves_real_products_and_hashes_match() -> None:
    """Audit shop fixtures: verify 3 real products and PDF hashes match data/manifests/catalog.yaml."""
    import hashlib
    from industrial_lab.shop import fixtures

    catalog = fixtures.load_catalog()
    product_ids = [p["product_id"] for p in catalog.get("products", [])]
    assert product_ids == ["P_X4", "P_THT", "P_UHEAT"]

    for p in catalog.get("products", []):
        for d in p.get("documents", []):
            doc_id = d["document_id"]
            expected_hash = d["sha256"]
            pdf_bytes = fixtures.get_document_bytes(doc_id)
            actual_hash = hashlib.sha256(pdf_bytes).hexdigest()
            assert actual_hash == expected_hash, f"Hash mismatch for {doc_id}"
