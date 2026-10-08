"""Contractual tests for CLI commands and exit codes adhering to MEGAPLAN.md §22.

Standard Exit Codes (§22):
- 0: SUCCESS
- 10: INVALID_CONFIG
- 20: PENDING_DATA
- 30: PROVIDER_BLOCKED
- 40: BUDGET_EXCEEDED
- 50: INTEGRITY_FAILURE
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch
import pytest
from click.testing import CliRunner

from tests.conftest import requires_real_facts
from industrial_lab.cli import ExitCode, cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_help(runner: CliRunner) -> None:
    """CLI provides help and lists all required commands (§22)."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == ExitCode.SUCCESS
    required_commands = [
        "validate-inputs",
        "provider-preflight",
        "smoke-providers",
        "ingest",
        "review-export",
        "review-import",
        "build-knowledge",
        "build-rag",
        "serve-shop",
        "serve-api",
        "smoke",
        "benchmark",
        "freeze",
        "score",
        "report",
        "replay",
        "test-local",
    ]
    for cmd in required_commands:
        assert cmd in result.output, f"Command '{cmd}' not found in CLI help output"


def test_cli_validate_inputs(runner: CliRunner) -> None:
    """validate-inputs checks catalog, documents, hashes, scenarios."""
    result = runner.invoke(cli, ["validate-inputs"])
    assert result.exit_code == ExitCode.SUCCESS
    assert "VALIDATING INPUTS" in result.output
    assert "Products Count: 3" in result.output


def test_cli_provider_preflight_blocked_warning(runner: CliRunner) -> None:
    """provider-preflight without TYPESAFE_API_KEY logs System A blocked."""
    with patch.dict(os.environ, {}, clear=True):
        if "TYPESAFE_API_KEY" in os.environ:
            del os.environ["TYPESAFE_API_KEY"]
        result = runner.invoke(cli, ["provider-preflight"])
        assert result.exit_code == ExitCode.SUCCESS
        assert "[SYSTEM A: JEV] BLOCKED" in result.output
        assert "PREFLIGHT WARNING" in result.output


def test_cli_provider_preflight_require_all_exits_30(runner: CliRunner) -> None:
    """provider-preflight --require-all exits with code 30 when key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        if "TYPESAFE_API_KEY" in os.environ:
            del os.environ["TYPESAFE_API_KEY"]
        result = runner.invoke(cli, ["provider-preflight", "--require-all"])
        assert result.exit_code == ExitCode.PROVIDER_BLOCKED
        assert result.exit_code == 30
        assert "PREFLIGHT FAILED" in result.output


def test_cli_smoke_offline(runner: CliRunner) -> None:
    """smoke test verifies fixtures and rules engine."""
    result = runner.invoke(cli, ["smoke"])
    assert result.exit_code == ExitCode.SUCCESS
    assert "SMOKE TEST PASSED" in result.output


def test_cli_smoke_real_providers_exits_30_when_blocked(runner: CliRunner) -> None:
    """smoke --real-providers exits with code 30 when credentials missing."""
    with patch.dict(os.environ, {}, clear=True):
        if "TYPESAFE_API_KEY" in os.environ:
            del os.environ["TYPESAFE_API_KEY"]
        result = runner.invoke(cli, ["smoke", "--real-providers"])
        assert result.exit_code == ExitCode.PROVIDER_BLOCKED
        assert result.exit_code == 30


def test_cli_ingest_pages_and_facts(runner: CliRunner) -> None:
    """ingest runs pages and facts stages."""
    r_pages = runner.invoke(cli, ["ingest", "--stage", "pages"])
    assert r_pages.exit_code == ExitCode.SUCCESS
    assert "Pages extraction complete" in r_pages.output

    r_facts = runner.invoke(cli, ["ingest", "--stage", "facts"])
    assert r_facts.exit_code == ExitCode.SUCCESS
    assert "Fact extraction complete" in r_facts.output


def test_cli_review_lifecycle(runner: CliRunner, tmp_path: Path) -> None:
    """review-export exports pending facts, review-import imports approved facts."""
    export_file = tmp_path / "pending_test.json"
    r_export = runner.invoke(cli, ["review-export", "--output", str(export_file)])
    assert r_export.exit_code == ExitCode.SUCCESS
    assert export_file.exists()

    # Import back into isolated test output file
    import_out = tmp_path / "imported_reviewed.jsonl"
    r_import = runner.invoke(cli, ["review-import", "--file", str(export_file), "--output", str(import_out)])
    assert r_import.exit_code == ExitCode.SUCCESS
    assert "Successfully imported" in r_import.output
    assert import_out.exists()


def test_cli_build_knowledge_and_rag(runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """build-knowledge builds store and graph; build-rag builds hybrid index."""
    import shutil
    data_tmp = tmp_path / "data"
    data_tmp.mkdir()
    (data_tmp / "facts").mkdir()
    facts_src = Path("data/facts/facts.reviewed.jsonl")
    if not (facts_src.exists() and facts_src.stat().st_size > 0):
        facts_src = Path("data/synthetic_fixtures/facts/facts.reviewed.jsonl")
    shutil.copy(facts_src, data_tmp / "facts" / "facts.reviewed.jsonl")

    (data_tmp / "pages").mkdir()
    pages_src = Path("data/pages/pages.jsonl")
    if not (pages_src.exists() and pages_src.stat().st_size > 0):
        pages_src = Path("data/synthetic_fixtures/pages/pages.jsonl")
    shutil.copy(pages_src, data_tmp / "pages" / "pages.jsonl")

    (data_tmp / "manifests").mkdir()
    cat_src = Path("data/manifests/catalog.yaml")
    if not cat_src.exists():
        cat_src = Path("data/synthetic_fixtures/manifests/catalog.yaml")
    shutil.copy(cat_src, data_tmp / "manifests" / "catalog.yaml")
    monkeypatch.setenv("LAB_DATA_ROOT", str(data_tmp))

    r_know = runner.invoke(cli, ["build-knowledge", "--reviewed-facts", str(data_tmp / "facts" / "facts.reviewed.jsonl")])
    assert r_know.exit_code == ExitCode.SUCCESS
    assert "Knowledge store & graph compiled successfully" in r_know.output

    r_rag = runner.invoke(cli, ["build-rag", "--data-dir", str(data_tmp)])
    assert r_rag.exit_code == ExitCode.SUCCESS
    assert "Hybrid RAG index built successfully" in r_rag.output


@requires_real_facts
def test_cli_freeze_and_benchmark_lifecycle(runner: CliRunner, tmp_path: Path) -> None:
    """freeze seals configuration; benchmark runs dev split; score, report, replay execute."""
    freeze_manifest = tmp_path / "freeze_test.json"
    r_freeze = runner.invoke(cli, ["freeze", "--output", str(freeze_manifest)])
    assert r_freeze.exit_code == ExitCode.SUCCESS
    assert freeze_manifest.exists()

    # Run benchmark on dev split with structured_rules engine
    runs_dir = tmp_path / "runs"
    r_bench = runner.invoke(
        cli,
        [
            "benchmark",
            "--split", "dev",
            "--repetitions", "1",
            "--engines", "structured_rules",
            "--output-dir", str(runs_dir),
        ],
    )
    assert r_bench.exit_code == ExitCode.SUCCESS
    assert "Benchmark completed successfully" in r_bench.output

    # Find the created run_id
    created_runs = list(runs_dir.iterdir())
    assert len(created_runs) >= 1
    run_id = created_runs[0].name

    # Score run
    r_score = runner.invoke(cli, ["score", "--run-id", run_id, "--runs-dir", str(runs_dir)])
    assert r_score.exit_code == ExitCode.SUCCESS

    # Report run
    r_rep = runner.invoke(cli, ["report", "--run-id", run_id, "--runs-dir", str(runs_dir)])
    assert r_rep.exit_code == ExitCode.SUCCESS
    assert "Reports generated successfully" in r_rep.output

    # Replay run without model calls
    r_replay = runner.invoke(cli, ["replay", "--run-id", run_id, "--runs-dir", str(runs_dir)])
    assert r_replay.exit_code == ExitCode.SUCCESS
    assert "Replay report reconstructed without model calls" in r_replay.output
    replay_out = runs_dir / run_id / "replay"
    assert (replay_out / "summary.csv").exists()
    assert (replay_out / "scoring.jsonl").exists()
    assert (replay_out / "scores.json").exists()
    assert (replay_out / "statistics.json").exists()
    assert (replay_out / "report.md").exists()
    assert (replay_out / "report.html").exists()


def test_cli_test_local_command(runner: CliRunner) -> None:
    """test-local command executes offline tests without remote API calls."""
    result = runner.invoke(cli, ["test-local", "--group", "contract", "-q"])
    assert result.exit_code == ExitCode.SUCCESS
    assert "RUNNING LOCAL OFFLINE TESTS (group: contract)" in result.output
    assert "ZERO remote provider calls permitted" in result.output
    assert "PASSED successfully (0 external API calls)" in result.output


def test_cli_report_demo(runner: CliRunner, tmp_path: Path) -> None:
    """report --demo generates synthetic demo report without run directory."""
    out_dir = tmp_path / "demo_reports"
    result = runner.invoke(cli, ["report", "--demo", "--output-dir", str(out_dir)])
    assert result.exit_code == ExitCode.SUCCESS
    assert "[DEMO / SYNTHETIC / NON-OFFICIAL] Demo report generated from synthetic test fixtures." in result.output
    assert "Reports generated successfully" in result.output
    assert (out_dir / "report.md").exists()
    assert (out_dir / "report.html").exists()


def test_cli_report_requires_run_id_when_not_demo(runner: CliRunner) -> None:
    """report without --demo requires --run-id."""
    result = runner.invoke(cli, ["report"])
    assert result.exit_code != ExitCode.SUCCESS
    assert "Missing option '--run-id'" in result.output


def test_cli_benchmark_dry_run_structured_jev_blocked(runner: CliRunner, tmp_path: Path) -> None:
    """benchmark --dry-run sets dryrun_ prefix, disclaimer logging, and reports structured_jev as BLOCKED (not mocked)."""
    import json
    runs_dir = tmp_path / "runs"
    with patch.dict(os.environ, {}, clear=True):
        if "TYPESAFE_API_KEY" in os.environ:
            del os.environ["TYPESAFE_API_KEY"]
        if "JEV_API_KEY" in os.environ:
            del os.environ["JEV_API_KEY"]

        result = runner.invoke(
            cli,
            [
                "benchmark",
                "--dry-run",
                "--split", "dev",
                "--repetitions", "1",
                "--engines", "structured_jev",
                "--output-dir", str(runs_dir),
            ],
        )

        assert result.exit_code == ExitCode.SUCCESS
        assert "[DRY-RUN / NON-OFFICIAL]" in result.output
        assert "data_origin: synthetic_fixture" in result.output
        assert "[SYSTEM A: JEV] BLOCKED (provider_error code 30)" in result.output
        assert "structured_jev will NOT be substituted by a mock" in result.output

        created_runs = list(runs_dir.iterdir())
        assert len(created_runs) == 1
        run_folder = created_runs[0]
        assert run_folder.name.startswith("dryrun_")

        # Verify normalized responses record provider_error and not mock completion
        norm_file = run_folder / "responses_normalized.jsonl"
        assert norm_file.exists()
        with open(norm_file, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) > 0
        for entry in lines:
            resp = entry["response"]
            assert resp["execution_status"] == "provider_error"
            assert resp["engine"] == "structured_jev"
            assert "JEV blocked" in resp["summary"]


def test_cli_smoke_providers_offline(runner: CliRunner) -> None:
    """smoke-providers without --live stays offline and succeeds."""
    result = runner.invoke(cli, ["smoke-providers"])
    assert result.exit_code == ExitCode.SUCCESS
    assert "OFFLINE MODE" in result.output
    assert "SMOKE-PROVIDERS OFFLINE OK" in result.output


def test_cli_provider_preflight_recognizes_bedrock_mantle(runner: CliRunner) -> None:
    """provider-preflight reports Mantle when bearer/LLM key present (no secret values)."""
    env = {
        "TYPESAFE_API_KEY": "test-typesafe-key",
        "JEV_MODEL_ID": "jev-preview",
        "AWS_BEARER_TOKEN_BEDROCK": "test-bedrock-bearer",
        "LLM_PROVIDER": "bedrock-mantle",
        "LLM_MODEL_ID": "google.gemma-4-31b",
        "LLM_BASE_URL": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
        "BEDROCK_REGION": "us-east-1",
    }
    with patch.dict(os.environ, env, clear=True):
        result = runner.invoke(cli, ["provider-preflight"])
        assert result.exit_code == ExitCode.SUCCESS
        assert "[SYSTEM A: JEV] CREDENTIALS PRESENT" in result.output
        assert "jev-preview" in result.output
        assert "Bedrock Mantle: CONFIGURED" in result.output
        assert "google.gemma-4-31b" in result.output
        assert "PREFLIGHT SUCCESS" in result.output
        # never echo secret material
        assert "test-typesafe-key" not in result.output
        assert "test-bedrock-bearer" not in result.output

