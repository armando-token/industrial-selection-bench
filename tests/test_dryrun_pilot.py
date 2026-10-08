"""Tests for dry-run benchmark integrity adhering to MEGAPLAN.md §18 and G0/G1 freeze requirements.

Verifies:
1. Running BenchmarkRunner with LAB_DRY_RUN=1 sets data_origin: synthetic_fixture in manifest and metadata.
2. structured_rules executes successfully without API keys (pure deterministic rules).
3. structured_jev is never silently substituted with a mock when credentials are empty.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import pytest

from tests.conftest import requires_real_facts

from industrial_lab.benchmark.runner import BenchmarkRunner, RunManifest
from industrial_lab.engines.ablations import StructuredRulesEngine
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
    TechnicalVerdict,
)


# ==============================================================================
# 1. LAB_DRY_RUN=1 Sets data_origin: synthetic_fixture
# ==============================================================================

def test_benchmark_runner_dry_run_sets_data_origin_in_validate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies running BenchmarkRunner with LAB_DRY_RUN=1 sets data_origin: synthetic_fixture in validate mode."""
    monkeypatch.setenv("LAB_DRY_RUN", "1")

    runner = BenchmarkRunner(
        config_path=Path("configs/experiment.yaml"),
        output_dir=tmp_path,
    )
    assert runner.run_id.startswith("dryrun_")

    manifest = runner.run(mode="validate")

    # Verify manifest object
    assert manifest.data_origin == "synthetic_fixture"
    assert manifest.metadata.get("data_origin") == "synthetic_fixture"
    assert manifest.metadata.get("dry_run") is True

    # Verify persisted run_manifest.json on disk
    manifest_file = runner.run_dir / "run_manifest.json"
    assert manifest_file.exists()
    disk_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert disk_manifest.get("data_origin") == "synthetic_fixture"
    assert disk_manifest.get("metadata", {}).get("data_origin") == "synthetic_fixture"
    assert disk_manifest.get("metadata", {}).get("dry_run") is True


def test_benchmark_runner_dry_run_sets_data_origin_in_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies running BenchmarkRunner with LAB_DRY_RUN=1 sets data_origin: synthetic_fixture in smoke mode."""
    monkeypatch.setenv("LAB_DRY_RUN", "1")

    runner = BenchmarkRunner(
        config_path=Path("configs/experiment.yaml"),
        output_dir=tmp_path,
        allow_mock_fallback=True,
    )
    manifest = runner.run(mode="smoke")

    assert manifest.data_origin == "synthetic_fixture"
    assert manifest.metadata.get("data_origin") == "synthetic_fixture"
    assert manifest.metadata.get("dry_run") is True

    manifest_file = runner.run_dir / "run_manifest.json"
    assert manifest_file.exists()
    disk_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert disk_manifest.get("data_origin") == "synthetic_fixture"


@requires_real_facts
def test_benchmark_runner_official_sets_data_origin_official(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies running BenchmarkRunner without LAB_DRY_RUN sets data_origin: official."""
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)
    monkeypatch.setattr("industrial_lab.shop.fixtures.load_catalog", lambda: {"data_origin": "real_manual", "products": []})

    runner = BenchmarkRunner(
        config_path=Path("configs/experiment.yaml"),
        output_dir=tmp_path,
    )
    assert not runner.run_id.startswith("dryrun_")

    manifest = runner.run(mode="validate")

    assert manifest.data_origin == "official"
    assert manifest.metadata.get("data_origin") == "official"
    assert manifest.metadata.get("dry_run") is False

    manifest_file = runner.run_dir / "run_manifest.json"
    disk_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert disk_manifest.get("data_origin") == "official"


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
# 2. structured_rules Executes Successfully Without API Keys
# ==============================================================================

def test_structured_rules_direct_execution_without_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies StructuredRulesEngine executes successfully without any API keys."""
    # Strip all API keys
    _clear_all_provider_keys(monkeypatch)

    engine = StructuredRulesEngine()

    # M1 request without explicit requirements
    req_m1 = QueryRequest(
        request_id="req_rules_m1",
        engine="structured_rules",
        query_text="Seleccionar variador de frecuencia para motor 5kW",
        mode="M1",
        catalog_version="v1",
        knowledge_version="v1",
    )
    resp_m1 = asyncio.run(engine.execute(req_m1))

    assert resp_m1.execution_status == ExecutionStatus.completed
    assert resp_m1.engine == "structured_rules"
    assert resp_m1.technical_verdict in (
        TechnicalVerdict.COMPATIBLE,
        TechnicalVerdict.INCOMPATIBLE,
        TechnicalVerdict.INSUFFICIENT_EVIDENCE,
    )

    # M2 request with explicit technical requirements
    req_m2 = QueryRequest(
        request_id="req_rules_m2",
        engine="structured_rules",
        query_text="Verificar requisitos técnicos de alimentación y temperatura",
        mode="M2",
        requirements=[
            Requirement(
                requirement_id="REQ_VOLTAGE",
                kind=RequirementKind.exact_property,
                operator="lte",
                target="480",
                hard=True,
                source_user_text="Tensión máxima 480V",
            ),
        ],
        catalog_version="v1",
        knowledge_version="v1",
    )
    resp_m2 = asyncio.run(engine.execute(req_m2))

    assert resp_m2.execution_status == ExecutionStatus.completed
    assert resp_m2.engine == "structured_rules"
    assert len(resp_m2.checks) > 0
    # Decisions in structured_rules must originate from rules
    assert all(c.decision_origin == DecisionOrigin.rule for c in resp_m2.checks)


def test_structured_rules_dispatch_via_benchmark_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies BenchmarkRunner dispatches and runs structured_rules without API keys."""
    _clear_all_provider_keys(monkeypatch)

    runner = BenchmarkRunner(
        config_path=Path("configs/experiment.yaml"),
        output_dir=tmp_path,
    )
    engine_fn = runner._get_engine("structured_rules")

    req = QueryRequest(
        request_id="req_runner_rules",
        engine="structured_rules",
        query_text="Controlador PLC para celda robotizada",
        mode="M1",
    )
    raw_resp, norm_resp = engine_fn(req)

    assert norm_resp.execution_status == ExecutionStatus.completed
    assert norm_resp.engine == "structured_rules"
    assert norm_resp.engine_version != "mock-v1.0"


def test_structured_rules_pipeline_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies BenchmarkRunner runs a full smoke cycle with structured_rules alone."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LAB_DRY_RUN", "1")

    runner = BenchmarkRunner(
        config_path=Path("configs/experiment.yaml"),
        output_dir=tmp_path,
    )
    runner.engines = ["structured_rules"]

    manifest = runner.run(mode="smoke")

    assert manifest.status == "completed"
    assert manifest.completed_requests > 0
    assert manifest.failed_requests == 0


# ==============================================================================
# 3. structured_jev Is Never Silently Substituted With a Mock
# ==============================================================================

def test_structured_jev_never_substituted_with_mock_when_keys_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verifies structured_jev returns provider_error (blocked) and is NEVER silently substituted with mock_engine."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)

    # Note allow_mock_fallback=True: even with fallback enabled, structured_jev must NOT be mocked
    runner = BenchmarkRunner(
        config_path=Path("configs/experiment.yaml"),
        output_dir=tmp_path,
        allow_mock_fallback=True,
    )
    engine_fn = runner._get_engine("structured_jev")

    # Query specifically triggers mock matching if mock were active ("p1", "siemens")
    req = QueryRequest(
        request_id="req_jev_no_mock",
        engine="structured_jev",
        query_text="Equipo Siemens P1 para línea de ensamblaje",
        mode="M1",
    )
    raw_resp, resp = engine_fn(req)

    # Must NOT be the mock engine
    assert resp.engine_version != "mock-v1.0"
    assert "Mock response from structured_jev" not in str(resp.summary)
    assert resp.selected_product_ids != ["P1"]

    # Must be provider_error indicating System A is blocked (§0.1 #7)
    assert resp.execution_status == ExecutionStatus.provider_error
    assert "JEV blocked (no API key)" in str(resp.summary) or "30" in str(resp.summary)


def test_structured_jev_never_substituted_with_mock_under_dryrun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verifies structured_jev is never mocked even under LAB_DRY_RUN=1."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.setenv("LAB_DRY_RUN", "1")

    runner = BenchmarkRunner(
        config_path=Path("configs/experiment.yaml"),
        output_dir=tmp_path,
        allow_mock_fallback=True,
    )
    engine_fn = runner._get_engine("structured_jev")

    req = QueryRequest(
        request_id="req_jev_dryrun",
        engine="structured_jev",
        query_text="Consulta Siemens P1 bajo dry run",
        mode="M1",
    )
    raw_resp, resp = engine_fn(req)

    assert resp.engine_version != "mock-v1.0"
    assert resp.execution_status == ExecutionStatus.provider_error
    assert "Mock response from structured_jev" not in str(resp.summary)


def test_deterministic_mock_engine_helper_never_mocks_structured_jev(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verifies _create_deterministic_mock_engine('structured_jev') routes to real blocked engine, not mock."""
    _clear_all_provider_keys(monkeypatch)
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)

    runner = BenchmarkRunner(output_dir=tmp_path)
    mock_fn = runner._create_deterministic_mock_engine("structured_jev")

    req = QueryRequest(
        request_id="req_direct_mock_check",
        engine="structured_jev",
        query_text="Siemens P1 compatibility query",
        mode="M1",
    )
    raw_resp, resp = mock_fn(req)

    # Must NOT produce mock_engine output
    assert resp.engine_version != "mock-v1.0"
    assert resp.execution_status == ExecutionStatus.provider_error
    assert resp.technical_verdict != TechnicalVerdict.COMPATIBLE


def test_structured_jev_engine_class_step1_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies StructuredJevEngine class itself adheres strictly to Step 1 availability check."""
    _clear_all_provider_keys(monkeypatch)

    engine = StructuredJevEngine()
    req = QueryRequest(
        request_id="req_jev_class",
        engine="structured_jev",
        query_text="Direct execution test",
        mode="M1",
    )
    resp = asyncio.run(engine.execute(req))

    assert resp.execution_status == ExecutionStatus.provider_error
    assert "30" in str(resp.summary)
    assert "JEV blocked (no API key)" in str(resp.summary)
