"""Contractual tests for RagLlmEngine dry-run mode on synthetic fixtures.

Verifies:
1. Default mode without API key returns provider_error (code 30).
2. dry_run=True parameter executes without API key, returns ExecutionStatus.completed,
   contains the '[DRY-RUN / SYNTHETIC FIXTURE - NON-OFFICIAL]' banner in summary and
   '[DRY-RUN / SYNTHETIC FIXTURE]' in reason codes, and sets telemetry_ref to 'rag-dryrun-{request_id}'.
3. LAB_DRY_RUN=1 (and 'true', 'yes') activates dry-run mode.
"""

from __future__ import annotations

import os
import pytest
from tests.conftest import requires_real_manuals

from industrial_lab.engines.rag_llm import RagLlmEngine
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
async def test_rag_llm_default_mode_blocked_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default mode without API key must return provider_error with code 30."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)

    engine = RagLlmEngine(api_key=None, dry_run=False)
    assert not engine.dry_run

    req = QueryRequest(
        request_id="rag-req-blocked-default",
        query_text="Temperature sensor 4-20mA",
        engine="rag_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_TEMP",
                kind=RequirementKind.operating_condition,
                operator="range_contains",
                target=[-20, 60],
                hard=True,
            )
        ],
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.provider_error
    assert "código 30" in resp.summary or "code 30" in resp.summary or "LLM blocked" in resp.summary
    assert any("PROVIDER_BLOCKED_CODE_30" in (c.reason_code or "") for c in resp.checks)


@requires_real_manuals
@pytest.mark.asyncio
async def test_rag_llm_dry_run_explicit_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    """dry_run=True executes without API key, completes, and sets banners and telemetry_ref."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)

    engine = RagLlmEngine(api_key=None, dry_run=True)
    assert engine.dry_run is True

    req = QueryRequest(
        request_id="dryrun-req-001",
        query_text="FX-CPU-24V industrial controller 24V supply",
        engine="rag_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_VOLTAGE",
                kind=RequirementKind.exact_property,
                operator="eq",
                target="24V",
                unit="V",
                hard=True,
                source_user_text="supply voltage 24V DC",
            )
        ],
        include_quote=True,
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.telemetry_ref == f"rag-dryrun-{req.request_id}"

    # Verify summary mandatory prefix
    mandatory_prefix = (
        "[DRY-RUN / SYNTHETIC FIXTURE - NON-OFFICIAL] rag_llm executed in dry-run mode on "
        "synthetic test fixtures. This is NOT an official benchmark outcome for System B. "
        "Technical verdict:"
    )
    assert resp.summary.startswith(mandatory_prefix)
    assert "[DRY-RUN / SYNTHETIC FIXTURE - NON-OFFICIAL]" in resp.summary

    # Verify checks and reasons
    assert len(resp.checks) == 1
    chk = resp.checks[0]
    assert chk.requirement_id == "REQ_VOLTAGE"
    assert chk.decision_origin == DecisionOrigin.llm
    assert chk.status == CheckStatus.PASS
    assert chk.reason_code == "[DRY-RUN / SYNTHETIC FIXTURE] REQ_VOLTAGE_EVALUATED_OFFLINE"
    assert "[DRY-RUN / SYNTHETIC FIXTURE]" in (chk.reason_code or "")
    assert len(chk.evidence_ids) > 0

    # Reasons are also included in the breakdown within the rendered summary
    assert "[DRY-RUN / SYNTHETIC FIXTURE] REQ_VOLTAGE_EVALUATED_OFFLINE" in resp.summary

    # Verdict and quote
    assert resp.technical_verdict == TechnicalVerdict.COMPATIBLE
    assert len(resp.selected_product_ids) > 0
    assert resp.quote is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("env_val", ["1", "true", "yes", "TRUE", "YES"])
async def test_rag_llm_dry_run_environment_variable(monkeypatch: pytest.MonkeyPatch, env_val: str) -> None:
    """LAB_DRY_RUN environment variable activates dry-run mode."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LAB_DRY_RUN", env_val)

    engine = RagLlmEngine(api_key=None)
    assert engine.dry_run is True

    req = QueryRequest(
        request_id=f"dryrun-env-{env_val}",
        query_text="FX-TS-420MA temperature transmitter Pt100 RTD",
        engine="rag_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_OUTPUT",
                kind=RequirementKind.operating_condition,
                operator="eq",
                target="4-20mA",
                hard=True,
            )
        ],
        include_quote=False,
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.telemetry_ref == f"rag-dryrun-{req.request_id}"
    assert "[DRY-RUN / SYNTHETIC FIXTURE - NON-OFFICIAL]" in resp.summary
    assert any("[DRY-RUN / SYNTHETIC FIXTURE]" in (c.reason_code or "") for c in resp.checks)


@pytest.mark.asyncio
async def test_rag_llm_dry_run_dynamic_env_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    """LAB_DRY_RUN set dynamically at execution time also enables dry run."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)

    engine = RagLlmEngine(api_key=None, dry_run=False)
    assert engine.dry_run is False

    monkeypatch.setenv("LAB_DRY_RUN", "1")
    req = QueryRequest(
        request_id="dryrun-dynamic-env",
        query_text="Power supply FX-PSU-24V-5A",
        engine="rag_llm",
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.telemetry_ref == "rag-dryrun-dryrun-dynamic-env"
    assert "[DRY-RUN / SYNTHETIC FIXTURE - NON-OFFICIAL]" in resp.summary


@pytest.mark.asyncio
async def test_rag_llm_dry_run_unmatched_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requirements without matching retrieved evidence are marked UNKNOWN and yield INSUFFICIENT_EVIDENCE."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LAB_DRY_RUN", "1")

    engine = RagLlmEngine(api_key=None)

    req = QueryRequest(
        request_id="dryrun-unmatched",
        query_text="Controller FX-CPU-24V",
        engine="rag_llm",
        requirements=[
            Requirement(
                requirement_id="REQ_SUPERCONDUCTOR",
                kind=RequirementKind.exact_property,
                operator="eq",
                target="999999_unobtainium_spec",
                hard=True,
            )
        ],
    )

    resp = await engine.execute(req)
    assert resp.execution_status == ExecutionStatus.completed
    assert resp.technical_verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE
    assert len(resp.checks) == 1
    chk = resp.checks[0]
    assert chk.status == CheckStatus.UNKNOWN
    assert chk.reason_code == "[DRY-RUN / SYNTHETIC FIXTURE] REQ_SUPERCONDUCTOR_EVALUATED_OFFLINE"
    assert "REQ_SUPERCONDUCTOR" in resp.missing_evidence
    assert resp.selected_product_ids == []
