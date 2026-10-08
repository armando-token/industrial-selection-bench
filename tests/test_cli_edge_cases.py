"""Edge cases and hardening tests for CLI commands and exit codes.

Adheres strictly to MEGAPLAN.md §22:
- 0: SUCCESS
- 10: INVALID_CONFIG
- 20: PENDING_DATA
- 30: PROVIDER_BLOCKED
- 40: BUDGET_EXCEEDED
- 50: INTEGRITY_FAILURE
"""

from __future__ import annotations

import json
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


def test_validate_inputs_missing_catalog(runner: CliRunner, tmp_path: Path) -> None:
    """validate-inputs on missing catalog manifest exits with ExitCode.INVALID_CONFIG (10)."""
    missing_cat = tmp_path / "nonexistent_catalog.yaml"
    result = runner.invoke(cli, ["validate-inputs", "--catalog", str(missing_cat)])
    assert result.exit_code == ExitCode.INVALID_CONFIG
    assert result.exit_code == 10
    assert "ERROR: Catalog manifest not found" in result.output


def test_review_export_missing_auto_facts(runner: CliRunner, tmp_path: Path) -> None:
    """review-export on missing auto facts exits with ExitCode.PENDING_DATA (20)."""
    missing_facts = tmp_path / "nonexistent_facts.auto.jsonl"
    result = runner.invoke(cli, ["review-export", "--facts-file", str(missing_facts)])
    assert result.exit_code == ExitCode.PENDING_DATA
    assert result.exit_code == 20
    assert "Error: Auto facts file not found" in result.output


def test_build_knowledge_missing_reviewed_facts_no_force(runner: CliRunner, tmp_path: Path) -> None:
    """build-knowledge without reviewed facts and without --force exits with ExitCode.PENDING_DATA (20)."""
    missing_facts = tmp_path / "nonexistent_facts.reviewed.jsonl"
    result = runner.invoke(cli, ["build-knowledge", "--reviewed-facts", str(missing_facts)])
    assert result.exit_code == ExitCode.PENDING_DATA
    assert result.exit_code == 20
    assert "ERROR: Reviewed facts file" in result.output
    assert "does not exist" in result.output


def test_provider_preflight_require_all_when_system_a_blocked(runner: CliRunner) -> None:
    """provider-preflight with --require-all when System A is blocked exits with ExitCode.PROVIDER_BLOCKED (30)."""
    env = os.environ.copy()
    env.pop("TYPESAFE_API_KEY", None)
    env.pop("JEV_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        result = runner.invoke(cli, ["provider-preflight", "--require-all"])
        assert result.exit_code == ExitCode.PROVIDER_BLOCKED
        assert result.exit_code == 30
        assert "[SYSTEM A: JEV] BLOCKED" in result.output
        assert "PREFLIGHT FAILED" in result.output


def test_benchmark_help_displays_dry_run(runner: CliRunner) -> None:
    """benchmark --help displays the --dry-run option."""
    result = runner.invoke(cli, ["benchmark", "--help"])
    assert result.exit_code == ExitCode.SUCCESS
    assert result.exit_code == 0
    assert "--dry-run" in result.output


def test_benchmark_dry_run_option(runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """benchmark with --dry-run sets LAB_DRY_RUN=1 and logs dry run status."""
    runs_dir = tmp_path / "runs"
    monkeypatch.delenv("LAB_DRY_RUN", raising=False)
    try:
        result = runner.invoke(
            cli,
            [
                "benchmark",
                "--dry-run",
                "--split", "dev",
                "--repetitions", "1",
                "--engines", "structured_rules",
                "--output-dir", str(runs_dir),
            ],
        )
        assert result.exit_code == ExitCode.SUCCESS
        assert os.environ.get("LAB_DRY_RUN") == "1"
        assert "dry run on synthetic fixtures" in result.output.lower() or "dry-run" in result.output.lower()
    finally:
        os.environ.pop("LAB_DRY_RUN", None)



def test_review_import_nonexistent_or_invalid_file(runner: CliRunner, tmp_path: Path) -> None:
    """review-import with non-existent or invalid file exits with code INVALID_CONFIG or PENDING_DATA."""
    # 1. Non-existent file
    missing_file = tmp_path / "nonexistent_approved.json"
    result_missing = runner.invoke(cli, ["review-import", "--file", str(missing_file)])
    assert result_missing.exit_code in (ExitCode.INVALID_CONFIG, ExitCode.PENDING_DATA)
    assert result_missing.exit_code in (10, 20)

    # 2. Invalid file (corrupted json)
    invalid_file = tmp_path / "invalid_review.json"
    invalid_file.write_text("{ corrupt json data ...", encoding="utf-8")
    result_invalid = runner.invoke(cli, ["review-import", "--file", str(invalid_file)])
    assert result_invalid.exit_code in (ExitCode.INVALID_CONFIG, ExitCode.PENDING_DATA)
    assert result_invalid.exit_code == ExitCode.INVALID_CONFIG


@requires_real_facts
def test_freeze_creates_parent_dirs(runner: CliRunner, tmp_path: Path) -> None:
    """freeze handles non-existent output directory by creating parent dirs."""
    nested_manifest = tmp_path / "nested" / "subfolder" / "manifest" / "freeze_test.json"
    assert not nested_manifest.parent.exists()

    result = runner.invoke(cli, ["freeze", "--output", str(nested_manifest)])
    assert result.exit_code == ExitCode.SUCCESS
    assert nested_manifest.exists()
    assert nested_manifest.is_file()

    with open(nested_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "freeze_id" in data
    assert "seal_signature" in data


@requires_real_facts
def test_freeze_rejects_missing_critical_files(runner: CliRunner, tmp_path: Path, monkeypatch) -> None:
    """freeze must fail and not seal when any critical file is missing (§21, Workstream E)."""
    out_manifest = tmp_path / "freeze_manifest.json"
    # Point LAB_DATA_ROOT to an empty directory so data/* are missing
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()
    monkeypatch.setenv("LAB_DATA_ROOT", str(empty_data_dir))

    # Mock resolve_data_path to return a non-existent path
    from unittest.mock import patch
    with patch("industrial_lab.cli.resolve_data_path", return_value=empty_data_dir / "nonexistent.yaml"):
        result = runner.invoke(cli, ["freeze", "--output", str(out_manifest)])
        assert result.exit_code in (ExitCode.INVALID_CONFIG, ExitCode.PENDING_DATA)
        assert result.exit_code == ExitCode.PENDING_DATA
        assert "missing critical file" in result.output.lower()
        assert not out_manifest.exists()


def test_freeze_rejects_empty_critical_files(runner: CliRunner, tmp_path: Path) -> None:
    """freeze must fail and not seal when any critical file is empty (§21, Workstream E)."""
    out_manifest = tmp_path / "freeze_manifest.json"
    empty_file = tmp_path / "empty_catalog.yaml"
    empty_file.touch()

    from unittest.mock import patch
    with patch("industrial_lab.cli.resolve_data_path", return_value=empty_file):
        result = runner.invoke(cli, ["freeze", "--output", str(out_manifest)])
        assert result.exit_code in (ExitCode.INVALID_CONFIG, ExitCode.PENDING_DATA)
        assert "empty critical file" in result.output.lower()
        assert not out_manifest.exists()


def test_freeze_rejects_fake_official_freeze_on_synthetic_catalog(runner: CliRunner, tmp_path: Path) -> None:
    """freeze --official must be blocked if catalog is synthetic_fixture or has null placeholders."""
    out_manifest = tmp_path / "freeze_manifest.json"
    syn_catalog = Path(__file__).resolve().parent.parent / "data" / "manifests" / "catalog.synthetic.yaml"
    assert syn_catalog.exists()
    from unittest.mock import patch

    def _resolve(p):
        ps = str(p)
        if ps == "data/manifests/catalog.yaml" or ps.endswith("/manifests/catalog.yaml"):
            return syn_catalog
        path = Path(p)
        return path

    with patch("industrial_lab.cli.resolve_data_path", side_effect=_resolve):
        result = runner.invoke(cli, ["freeze", "--official", "--output", str(out_manifest)])
    assert result.exit_code == ExitCode.PENDING_DATA
    assert "cannot perform official freeze" in result.output.lower()
    assert not out_manifest.exists()


@requires_real_facts
def test_freeze_verify_machinery(runner: CliRunner, tmp_path: Path) -> None:
    """freeze --verify validates cryptographic seal and catches mismatches/tampering."""
    # 1. Create a valid freeze manifest
    valid_manifest = tmp_path / "valid_freeze.json"
    r_create = runner.invoke(cli, ["freeze", "--output", str(valid_manifest)])
    assert r_create.exit_code == ExitCode.SUCCESS

    # 2. Verify valid manifest
    r_verify = runner.invoke(cli, ["freeze", "--verify", "--manifest", str(valid_manifest)])
    assert r_verify.exit_code == ExitCode.SUCCESS
    assert "cryptographically valid" in r_verify.output.lower()

    # 3. Tamper with seal signature -> CRYPTOGRAPHIC FAILURE
    tampered_manifest = tmp_path / "tampered_freeze.json"
    with open(valid_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["seal_signature"] = "deadbeef" * 8
    with open(tampered_manifest, "w", encoding="utf-8") as f:
        json.dump(data, f)

    r_tampered = runner.invoke(cli, ["freeze", "--verify", "--manifest", str(tampered_manifest)])
    assert r_tampered.exit_code == ExitCode.INTEGRITY_FAILURE
    assert "cryptographic failure" in r_tampered.output.lower()


def test_score_informative_messages(runner: CliRunner, tmp_path: Path) -> None:
    """score provides informative messages if run directory does not exist or lacks responses."""
    # 1. Non-existent run directory -> INVALID_CONFIG (10)
    result_missing = runner.invoke(
        cli, ["score", "--run-id", "missing_run_id", "--runs-dir", str(tmp_path)]
    )
    assert result_missing.exit_code == ExitCode.INVALID_CONFIG
    assert "does not exist" in result_missing.output.lower()

    # 2. Run directory lacks responses file -> PENDING_DATA (20)
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()
    result_no_resp = runner.invoke(
        cli, ["score", "--run-id", "empty_run", "--runs-dir", str(tmp_path)]
    )
    assert result_no_resp.exit_code == ExitCode.PENDING_DATA
    assert "lacks responses" in result_no_resp.output.lower()

    # 3. Run directory has empty responses file -> PENDING_DATA (20)
    (empty_run / "responses_normalized.jsonl").write_text("", encoding="utf-8")
    result_empty_file = runner.invoke(
        cli, ["score", "--run-id", "empty_run", "--runs-dir", str(tmp_path)]
    )
    assert result_empty_file.exit_code == ExitCode.PENDING_DATA
    assert "lacks responses" in result_empty_file.output.lower()


def test_report_informative_messages(runner: CliRunner, tmp_path: Path) -> None:
    """report provides informative messages if run directory does not exist or lacks responses."""
    # 1. Non-existent run directory -> INVALID_CONFIG (10)
    result_missing = runner.invoke(
        cli, ["report", "--run-id", "missing_run_id", "--runs-dir", str(tmp_path)]
    )
    assert result_missing.exit_code == ExitCode.INVALID_CONFIG
    assert "does not exist" in result_missing.output.lower()

    # 2. Run directory lacks responses -> PENDING_DATA (20)
    empty_run = tmp_path / "empty_run_report"
    empty_run.mkdir()
    result_no_resp = runner.invoke(
        cli, ["report", "--run-id", "empty_run_report", "--runs-dir", str(tmp_path)]
    )
    assert result_no_resp.exit_code == ExitCode.PENDING_DATA
    assert "lacks responses" in result_no_resp.output.lower()


def test_replay_informative_messages(runner: CliRunner, tmp_path: Path) -> None:
    """replay provides informative messages if run directory does not exist or lacks responses."""
    # 1. Non-existent run directory -> INVALID_CONFIG (10)
    result_missing = runner.invoke(
        cli, ["replay", "--run-id", "missing_run_id", "--runs-dir", str(tmp_path)]
    )
    assert result_missing.exit_code == ExitCode.INVALID_CONFIG
    assert "does not exist" in result_missing.output.lower()

    # 2. Run directory lacks responses -> PENDING_DATA (20)
    empty_run = tmp_path / "empty_run_replay"
    empty_run.mkdir()
    result_no_resp = runner.invoke(
        cli, ["replay", "--run-id", "empty_run_replay", "--runs-dir", str(tmp_path)]
    )
    assert result_no_resp.exit_code == ExitCode.PENDING_DATA
    assert "lacks responses" in result_no_resp.output.lower()


def test_benchmark_official_system_a_blocked_exits_30(runner: CliRunner, tmp_path: Path) -> None:
    """benchmark in official mode (without --dry-run) with structured_jev when key missing exits with code 30."""
    runs_dir = tmp_path / "runs"
    with patch.dict(os.environ, {}, clear=True):
        if "TYPESAFE_API_KEY" in os.environ:
            del os.environ["TYPESAFE_API_KEY"]
        if "JEV_API_KEY" in os.environ:
            del os.environ["JEV_API_KEY"]
        os.environ.pop("LAB_DRY_RUN", None)

        result = runner.invoke(
            cli,
            [
                "benchmark",
                "--split", "dev",
                "--repetitions", "1",
                "--engines", "structured_jev",
                "--output-dir", str(runs_dir),
            ],
        )
        assert result.exit_code == ExitCode.PROVIDER_BLOCKED
        assert result.exit_code == 30
        assert "Cannot run official benchmark with structured_jev" in result.output
        assert "No llamar éxito a una corrida con A ausente" in result.output


def test_benchmark_budget_exceeded_exits_40(runner: CliRunner, tmp_path: Path) -> None:
    """benchmark exits with ExitCode.BUDGET_EXCEEDED (40) when manifest status is budget_exceeded."""
    from industrial_lab.benchmark.runner import RunManifest
    runs_dir = tmp_path / "runs"

    fake_manifest = RunManifest(
        run_id="fake_budget_run",
        experiment_id="exp_test",
        mode="dry_run",
        seed=42,
        engines=["structured_rules"],
        split="dev",
        repetitions=1,
        started_at_utc="2026-10-02T00:00:00Z",
        status="budget_exceeded",
        total_cost_usd=25.50,
        completed_requests=2,
        total_scheduled_requests=10,
    )

    with patch("industrial_lab.benchmark.runner.BenchmarkRunner.run", return_value=fake_manifest):
        try:
            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--dry-run",
                    "--split", "dev",
                    "--repetitions", "1",
                    "--engines", "structured_rules",
                    "--output-dir", str(runs_dir),
                ],
            )
            assert result.exit_code == ExitCode.BUDGET_EXCEEDED
            assert result.exit_code == 40
            assert "budget exceeded" in result.output.lower()
        finally:
            os.environ.pop("LAB_DRY_RUN", None)


def test_provider_preflight_require_all_when_llm_blocked(runner: CliRunner) -> None:
    """provider-preflight --require-all exits with code 30 when LLM key is missing even if JEV is present."""
    with patch.dict(os.environ, {"TYPESAFE_API_KEY": "fake-jev-key"}, clear=True):
        result = runner.invoke(cli, ["provider-preflight", "--require-all"])
        assert result.exit_code == ExitCode.PROVIDER_BLOCKED
        assert result.exit_code == 30
        assert "PREFLIGHT FAILED" in result.output


def test_benchmark_pre_run_budget_estimate_exceeded_exits_40(runner: CliRunner, tmp_path: Path) -> None:
    """benchmark exits with code 40 when pre-run estimated cost exceeds max_run_usd."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    with patch("industrial_lab.observability.costs.BudgetEstimator.estimate") as mock_est:
        from industrial_lab.observability.costs import BudgetEstimateResult
        mock_est.return_value = BudgetEstimateResult(
            split="test",
            repetitions=100,
            case_count=48,
            scheduled_requests=4800,
            max_cost_usd=95.50,
            per_engine_max_usd={"rag_llm": 95.50},
            budget_limit_usd=20.0,
            exceeds_budget=True,
        )
        try:
            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--dry-run",
                    "--split", "dev",
                    "--repetitions", "100",
                    "--engines", "rag_llm",
                    "--output-dir", str(runs_dir),
                ],
            )
            assert result.exit_code == ExitCode.BUDGET_EXCEEDED
            assert result.exit_code == 40
            assert "exceeds configured budget limit" in result.output
        finally:
            os.environ.pop("LAB_DRY_RUN", None)


def test_benchmark_force_budget_bypasses_pre_run_abort(runner: CliRunner, tmp_path: Path) -> None:
    """benchmark --force-budget bypasses pre-run budget check."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    from industrial_lab.benchmark.runner import RunManifest
    fake_manifest = RunManifest(
        run_id="dryrun_bypassed",
        experiment_id="test-exp",
        mode="dev",
        seed=42,
        engines=["structured_rules"],
        split="dev",
        repetitions=1,
        started_at_utc="2026-10-02T00:00:00Z",
        status="completed",
        total_cost_usd=0.0,
        completed_requests=1,
        total_scheduled_requests=1,
    )

    with patch("industrial_lab.observability.costs.BudgetEstimator.estimate") as mock_est, \
         patch("industrial_lab.benchmark.runner.BenchmarkRunner.run", return_value=fake_manifest):
        from industrial_lab.observability.costs import BudgetEstimateResult
        mock_est.return_value = BudgetEstimateResult(
            split="test",
            repetitions=100,
            case_count=48,
            scheduled_requests=4800,
            max_cost_usd=95.50,
            per_engine_max_usd={"rag_llm": 95.50},
            budget_limit_usd=20.0,
            exceeds_budget=True,
        )
        try:
            result = runner.invoke(
                cli,
                [
                    "benchmark",
                    "--dry-run",
                    "--force-budget",
                    "--split", "dev",
                    "--repetitions", "1",
                    "--engines", "structured_rules",
                    "--output-dir", str(runs_dir),
                ],
            )
            assert result.exit_code == ExitCode.SUCCESS
        finally:
            os.environ.pop("LAB_DRY_RUN", None)


# ==============================================================================
# Edge Cases: Real Profile Separation & PDF Validation (§4.1, §4.3, §12)
# ==============================================================================

def test_validate_real_profile_refuses_synthetic_catalog(tmp_path: Path) -> None:
    """BenchmarkRunner.validate_real_profile() strictly refuses synthetic fixtures in real profile (§4.1)."""
    from industrial_lab.benchmark.runner import BenchmarkRunner

    runner = BenchmarkRunner(output_dir=tmp_path, profile="real")

    # 1. Refuses catalog with synthetic_fixture data_origin
    with patch("industrial_lab.shop.fixtures.load_catalog", return_value={"data_origin": "synthetic_fixture", "products": []}):
        with pytest.raises(ValueError, match="Real profile refuses synthetic or pending catalog"):
            runner.validate_real_profile()

    # 2. Refuses catalog with synthetic P1/P2/P3 products
    fake_cat = {
        "data_origin": "real_user_document",
        "products": [
            {"product_id": "P1", "documents": []},
            {"product_id": "P_X4", "documents": []},
        ],
    }
    with patch("industrial_lab.shop.fixtures.load_catalog", return_value=fake_cat):
        with pytest.raises(ValueError, match="Real profile refuses synthetic product ID: P1"):
            runner.validate_real_profile()

    # 3. Refuses catalog containing known fixture hash
    fixture_hash_cat = {
        "data_origin": "real_user_document",
        "products": [
            {
                "product_id": "P_X4",
                "documents": [
                    {
                        "filename": "some_doc.pdf",
                        "sha256": "ceaf1510e4d8609e2b9c4886156aafea4cb7226878fa5cb2bf7564470ee6e93d",
                    }
                ],
            }
        ],
    }
    with patch("industrial_lab.shop.fixtures.load_catalog", return_value=fixture_hash_cat):
        with pytest.raises(ValueError, match="Real profile refuses synthetic fixture hash"):
            runner.validate_real_profile()

    # 4. Refuses catalog containing fixture filename
    fixture_fn_cat = {
        "data_origin": "real_user_document",
        "products": [
            {
                "product_id": "P_X4",
                "documents": [
                    {
                        "filename": "fixture_p1_manual.txt",
                        "sha256": "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                    }
                ],
            }
        ],
    }
    with patch("industrial_lab.shop.fixtures.load_catalog", return_value=fixture_fn_cat):
        with pytest.raises(ValueError, match="Real profile refuses synthetic fixture file"):
            runner.validate_real_profile()


def test_validate_real_profile_refuses_pending_rules(tmp_path: Path) -> None:
    """BenchmarkRunner.validate_real_profile() strictly refuses pending unreviewed rules (§4.1)."""
    from industrial_lab.benchmark.runner import BenchmarkRunner
    from industrial_lab.rules.engine import ReviewStatus, Rule

    runner = BenchmarkRunner(output_dir=tmp_path, profile="real")

    # Mock real catalog to pass catalog check
    fake_cat = {
        "data_origin": "real_user_document",
        "products": [{"product_id": "P_X4", "documents": []}],
    }
    pending_rule = Rule(rule_id="RULE-PENDING-01", property="supply_voltage", review_status=ReviewStatus.pending)
    def mock_load_rules(self_engine, path):
        self_engine.rules.append(pending_rule)
        return [pending_rule]

    with patch("industrial_lab.shop.fixtures.load_catalog", return_value=fake_cat), \
         patch("industrial_lab.engines.structured_jev.load_all_facts", return_value={"P_X4": []}), \
         patch("industrial_lab.rules.engine.RulesEngine.load_rules_from_file", mock_load_rules):
        with pytest.raises(ValueError, match="Real profile refuses pending unreviewed rule: RULE-PENDING-01"):
            runner.validate_real_profile()


def test_validate_inputs_rejects_corrupt_or_empty_pdf(runner: CliRunner, tmp_path: Path) -> None:
    """validate-inputs detects empty or non-PDF documents and exits with code 10 (§4.1, §4.3)."""
    # Create corrupted catalog and dummy raw dir
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    manifests_dir = data_dir / "manifests"
    raw_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)

    bad_pdf = raw_dir / "bad_document.pdf"
    # Write invalid PDF header (e.g. plain text)
    bad_pdf.write_text("NOT A REAL PDF FILE HEADER", encoding="utf-8")

    cat_yaml = manifests_dir / "catalog.yaml"
    cat_content = {
        "catalog_id": "test-cat",
        "schema_version": "1.0",
        "data_origin": "real_user_document",
        "products": [
            {
                "product_id": "P_TEST",
                "exact_model": "Test Model",
                "sku": "SKU-01",
                "manufacturer": "TestMfg",
                "documents": [
                    {
                        "document_id": "DOC-BAD",
                        "filename": "bad_document.pdf",
                    }
                ],
            }
        ],
    }
    import yaml
    with open(cat_yaml, "w", encoding="utf-8") as f:
        yaml.dump(cat_content, f)

    result = runner.invoke(
        cli,
        ["validate-inputs", "--catalog", str(cat_yaml), "--data-dir", str(data_dir)],
    )
    assert result.exit_code == ExitCode.INVALID_CONFIG
    assert result.exit_code == 10
    assert "lacks '%PDF-' header" in result.output



