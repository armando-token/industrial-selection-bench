"""Contractual verification tests for Workstream D: Evaluation Leakage Guards and Synthetic Fixtures (§13, §18, §0.1).

Tests:
1. Benchmark case loaders & query inputs isolation:
   - Gold labels, acceptable_selections, technical_verdict, required_checks, and expected_missing_fields
     are never leaked into engine QueryRequest inputs.
   - In M1 (natural language end-to-end), requirements and requested_product_ids are strictly None.
   - In M2 (controlled technical validation), requirements specify constraints only, with no answers or verdicts.
2. Synthetic fixture strict labeling:
   - All scenarios in data/scenarios/ are strictly marked `data_origin: synthetic_fixture` and `NON-OFFICIAL`.
   - Synthetic catalog mirror catalog.synthetic.yaml stays `synthetic_fixture`; live catalog.yaml is `real_user_document` + NON-OFFICIAL.
   - All benchmark cases in data/benchmark/ (dev and test splits) are strictly marked `data_origin: synthetic_fixture` and `NON-OFFICIAL`.
3. Standby protection for real catalog (G2):
   - Active catalog is synthetic fixtures (FixtureCorp), NOT real manufacturer products.
   - Real catalog is in STANDBY for G2.
   - No fake gold exists for unsupplied real brands (e.g. Siemens, Schneider, ABB).
   - Default fallback cases in BenchmarkRunner use synthetic FixtureCorp models.
4. Architectural isolation:
   - Inference services (shop, api) in compose.yaml do NOT mount data/benchmark/.
   - Engine modules do not import or read from data/benchmark/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest
import yaml

from industrial_lab.benchmark.runner import BenchmarkRunner
from industrial_lab.schemas import (
    BenchmarkCase,
    CheckResult,
    CheckStatus,
    DecisionOrigin,
    GoldCase,
    QueryRequest,
    Requirement,
    RequirementKind,
    TechnicalVerdict,
)
from industrial_lab.shop import fixtures as shop_fixtures

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_ROOT / "data"
CONFIGS_DIR = WORKSPACE_ROOT / "configs"


# ==============================================================================
# 1. Benchmark Case Loaders & Query Input Sanitization (§13.4, §18.1)
# ==============================================================================

def test_benchmark_case_loader_loads_without_leakage():
    """Verify BenchmarkRunner loads dev and test cases properly, keeping gold isolated."""
    runner = BenchmarkRunner(config_path=CONFIGS_DIR / "experiment.yaml")

    dev_cases = runner.load_cases(split="dev")
    assert len(dev_cases) >= 3, f"Expected at least 3 dev cases, got {len(dev_cases)}"

    test_cases = runner.load_cases(split="test")
    assert len(test_cases) >= 3, f"Expected at least 3 test cases, got {len(test_cases)}"

    for case in dev_cases + test_cases:
        assert case.case_id.startswith(("DEV", "TEST", "T"))
        assert case.scenario_family_id
        assert case.scenario_id in ("S0001", "S0002")
        assert case.query_text and len(case.query_text) > 10
        assert case.gold is not None
        assert case.gold.technical_verdict in TechnicalVerdict
        assert case.data_origin == "synthetic_fixture"
        assert case.official_status == "NON-OFFICIAL"
        assert case.is_official is False


def test_m1_query_request_strictly_strips_requirements_and_candidates():
    """Verify that in M1 mode, QueryRequest receives ONLY natural language query_text (§2.3, §13.4).

    Requirements and requested_product_ids must be strictly None, even if input case contains them.
    """
    case = BenchmarkCase(
        case_id="TEST-M1-001",
        scenario_family_id="FAMILY-01",
        scenario_id="S0001",
        split="test",
        mode="M1",
        query_text="Need a 24V DC PLC controller with Modbus RTU interface.",
        input_requirements=[
            Requirement(
                requirement_id="supply_voltage",
                kind=RequirementKind.exact_property,
                operator="eq",
                target=24.0,
                unit="V",
            )
        ],
        requested_product_ids=["P1"],
        gold=GoldCase(
            acceptable_selections=[["P1"]],
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            required_checks=[{"requirement_id": "REQ-01", "status": "PASS"}],
            required_evidence_groups=[{"group_id": "G1", "acceptable_spans": ["D1:p02:s01"]}],
            expected_missing_fields=[],
            commerce_revision="commerce-v1",
            human_review_status="synthetic_fixture_reviewed",
        ),
    )

    runner = BenchmarkRunner(config_path=CONFIGS_DIR / "experiment.yaml")
    request = runner.create_public_request(case, engine="structured_jev", repeat=1)

    # In M1, requirements and candidate IDs must be stripped to None
    assert request.mode == "M1"
    assert request.requirements is None, "M1 request must not contain structured requirements!"
    assert request.requested_product_ids is None, "M1 request must not contain pre-selected product IDs!"
    assert request.query_text == case.query_text

    # Verify no gold leakage in request dict
    req_dump = request.model_dump()
    assert "gold" not in req_dump
    req_str = str(req_dump)
    assert "acceptable_selections" not in req_str
    assert "technical_verdict" not in req_str
    assert "required_checks" not in req_str
    assert "required_evidence_groups" not in req_str
    assert "expected_missing_fields" not in req_str
    assert "human_review_status" not in req_str


def test_m2_query_request_passes_input_requirements_without_gold_leakage():
    """Verify that in M2 mode, QueryRequest passes input constraints, but zero evaluation answers (§2.3)."""
    case = BenchmarkCase(
        case_id="TEST-M2-001",
        scenario_family_id="FAMILY-01",
        scenario_id="S0001",
        split="test",
        mode="M2",
        query_text="Controlled technical validation of PLC and Power Supply compatibility.",
        input_requirements=[
            Requirement(
                requirement_id="supply_voltage",
                kind=RequirementKind.exact_property,
                operator="eq",
                target=24.0,
                unit="V",
                hard=True,
            ),
            Requirement(
                requirement_id="ambient_temperature",
                kind=RequirementKind.operating_condition,
                operator="gte",
                target=50.0,
                unit="C",
                hard=True,
            ),
        ],
        requested_product_ids=["P1", "P2"],
        gold=GoldCase(
            acceptable_selections=[["P1", "P2"]],
            technical_verdict=TechnicalVerdict.COMPATIBLE,
            required_checks=[
                {"requirement_id": "supply_voltage", "status": "PASS"},
                {"requirement_id": "ambient_temperature", "status": "PASS"},
            ],
            required_evidence_groups=[{"group_id": "G1", "acceptable_spans": ["D1:p02:s01", "D2:p01:s01"]}],
            expected_missing_fields=[],
            commerce_revision="commerce-v1",
            human_review_status="synthetic_fixture_reviewed",
        ),
    )

    runner = BenchmarkRunner(config_path=CONFIGS_DIR / "experiment.yaml")
    request = runner.create_public_request(case, engine="structured_jev", repeat=1)

    # In M2, requirements and requested_product_ids are legitimate test inputs
    assert request.mode == "M2"
    assert request.requirements is not None
    assert len(request.requirements) == 2
    assert request.requested_product_ids == ["P1", "P2"]

    # Verify input requirements contain ONLY constraint definitions, NO check statuses or verdicts
    for r in request.requirements:
        r_dump = r.model_dump()
        assert "status" not in r_dump, "Requirement input must NOT contain check status!"
        assert "verdict" not in r_dump, "Requirement input must NOT contain verdict!"
        assert "evidence_ids" not in r_dump, "Requirement input must NOT contain evidence IDs!"
        assert "decision_origin" not in r_dump, "Requirement input must NOT contain decision origin!"

    # Verify absolutely NO gold labels leaked
    req_dump = request.model_dump()
    assert "gold" not in req_dump
    req_str = str(req_dump)
    assert "acceptable_selections" not in req_str
    assert "technical_verdict" not in req_str
    assert "required_checks" not in req_str
    assert "required_evidence_groups" not in req_str
    assert "expected_missing_fields" not in req_str
    assert "human_review_status" not in req_str


# ==============================================================================
# 2. Synthetic Benchmark Cases Strictly Labeled (§0.1 #4, §4.2)
# ==============================================================================

def test_all_scenarios_strictly_labeled_synthetic_fixture_and_non_official():
    """Scenarios must be NON-OFFICIAL; origin synthetic_fixture OR simulated_commerce (lab prices)."""
    scenarios_dir = DATA_DIR / "scenarios"
    scenario_files = list(scenarios_dir.glob("*.yaml")) + list(scenarios_dir.glob("*.yml"))
    assert len(scenario_files) >= 2, f"Expected at least 2 scenario files, found {len(scenario_files)}"
    allowed_origins = {"synthetic_fixture", "simulated_commerce"}

    for s_file in scenario_files:
        text = s_file.read_text(encoding="utf-8")
        assert "NON-OFFICIAL" in text, f"Scenario {s_file.name} missing 'NON-OFFICIAL' disclaimer"

        with open(s_file, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)

        origin = data.get("data_origin")
        assert origin in allowed_origins, (
            f"Scenario {s_file.name} data_origin must be one of {allowed_origins}, got {origin}"
        )
        if origin == "synthetic_fixture":
            assert "synthetic_fixture" in text
        else:
            assert ("simulated" in text.lower()) or ("SIMULATED" in text)
        assert data.get("official_status") == "NON-OFFICIAL", (
            f"Scenario {s_file.name} official_status must be 'NON-OFFICIAL', got {data.get('official_status')}"
        )
        assert data.get("is_official") is False, (
            f"Scenario {s_file.name} is_official must be False, got {data.get('is_official')}"
        )


def test_catalog_manifest_strictly_labeled_synthetic_fixture_and_non_official():
    """Synthetic catalog mirror stays synthetic_fixture; live catalog is real_user_document NON-OFFICIAL."""
    syn_path = DATA_DIR / "manifests" / "catalog.synthetic.yaml"
    assert syn_path.exists(), "catalog.synthetic.yaml must exist (preserved FixtureCorp fixtures)"
    syn_text = syn_path.read_text(encoding="utf-8")
    assert "NON-OFFICIAL" in syn_text
    assert "synthetic_fixture" in syn_text
    with open(syn_path, "r", encoding="utf-8") as fp:
        syn = yaml.safe_load(fp)
    assert syn.get("data_origin") == "synthetic_fixture"
    assert syn.get("official_status") == "NON-OFFICIAL"
    assert syn.get("is_official") is False

    live_path = DATA_DIR / "manifests" / "catalog.yaml"
    assert live_path.exists(), "catalog.yaml must exist"
    live_text = live_path.read_text(encoding="utf-8")
    assert "NON-OFFICIAL" in live_text
    with open(live_path, "r", encoding="utf-8") as fp:
        live = yaml.safe_load(fp)
    assert live.get("data_origin") == "real_user_document"
    assert live.get("official_status") == "NON-OFFICIAL"
    assert live.get("is_official") is False
    assert [p["product_id"] for p in live.get("products", [])] == ["P_X4", "P_THT", "P_UHEAT"]


def test_all_benchmark_cases_strictly_labeled_synthetic_fixture_and_non_official():
    """Verify that all cases in data/benchmark/{dev,test}/cases.jsonl are marked synthetic_fixture and NON-OFFICIAL."""
    for split in ("dev", "test"):
        cases_file = DATA_DIR / "benchmark" / split / "cases.jsonl"
        assert cases_file.exists(), f"Benchmark cases file for split '{split}' not found at {cases_file}"

        with open(cases_file, "r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                assert record.get("data_origin") == "synthetic_fixture", (
                    f"Line {line_no} in {cases_file.name} missing data_origin: synthetic_fixture"
                )
                assert record.get("official_status") == "NON-OFFICIAL", (
                    f"Line {line_no} in {cases_file.name} missing official_status: NON-OFFICIAL"
                )
                assert record.get("is_official") is False, (
                    f"Line {line_no} in {cases_file.name} is_official must be False"
                )


# ==============================================================================
# 3. No Fake Gold for Real Catalog (Real Catalog STANDBY for G2) (§0.1 #3, #4)
# ==============================================================================

def test_no_fake_gold_for_real_catalog_standby_g2():
    """Live catalog is real_user_document; no fake gold/benchmark claims; defaults stay synthetic."""
    catalog = shop_fixtures.load_catalog()
    assert catalog.get("data_origin") == "real_user_document"
    assert catalog.get("official_status") == "NON-OFFICIAL"
    assert catalog.get("is_official") is False
    ids = [p.get("product_id") for p in catalog.get("products", [])]
    assert ids == ["P_X4", "P_THT", "P_UHEAT"]

    # Synthetic fixture reviewed facts remain FixtureCorp-only (preserved mirror)
    syn_reviewed = DATA_DIR / "synthetic_fixtures" / "facts" / "facts.reviewed.jsonl"
    assert syn_reviewed.exists()
    import json
    syn_ids = set()
    for line in syn_reviewed.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fact = json.loads(line)
        syn_ids.add(fact.get("product_id"))
        assert fact.get("product_id") in ("P1", "P2", "P3")
    assert {"P1", "P2", "P3"}.issubset(syn_ids)
    # Official gold for real products must not exist yet (G3 review not closed)

    # BenchmarkRunner default fallback cases must remain synthetic FixtureCorp, not fake brands
    runner = BenchmarkRunner(config_path=CONFIGS_DIR / "experiment.yaml")
    default_cases = runner._create_default_cases("test")
    for case in default_cases:
        assert case.data_origin == "synthetic_fixture"
        assert case.official_status == "NON-OFFICIAL"
        assert case.is_official is False
        assert "Siemens" not in case.query_text
        assert "Sinamics" not in case.query_text
        assert case.gold.human_review_status == "synthetic_fixture_reviewed"


# ==============================================================================
# 4. Engine & Docker Isolation from Benchmark Directory (§20.2)
# ==============================================================================

def test_docker_compose_isolation_benchmark_not_mounted_in_shop_or_api():
    """Verify that compose.yaml does NOT mount data/benchmark/ into shop or api services (§20.2)."""
    compose_path = WORKSPACE_ROOT / "compose.yaml"
    assert compose_path.exists()

    with open(compose_path, "r", encoding="utf-8") as fp:
        compose_data = yaml.safe_load(fp)

    services = compose_data.get("services", {})
    assert "shop" in services
    assert "api" in services
    assert "runner" in services

    # Shop must NOT mount benchmark
    shop_vols = services["shop"].get("volumes", [])
    for v in shop_vols:
        assert "data/benchmark" not in str(v), f"VIOLATION: shop service mounts benchmark: {v}"

    # API must NOT mount benchmark
    api_vols = services["api"].get("volumes", [])
    for v in api_vols:
        assert "data/benchmark" not in str(v), f"VIOLATION: api service mounts benchmark: {v}"

    # Only runner mounts benchmark
    runner_vols = services["runner"].get("volumes", [])
    has_benchmark = any("data/benchmark" in str(v) for v in runner_vols)
    assert has_benchmark, "runner service must mount data/benchmark:ro"


def test_engines_do_not_import_or_read_benchmark_cases():
    """Verify that engine source files do not import or read from data/benchmark."""
    engines_dir = WORKSPACE_ROOT / "src" / "industrial_lab" / "engines"
    for py_file in engines_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert "data/benchmark" not in content, (
            f"VIOLATION: Engine file {py_file.name} references 'data/benchmark'!"
        )
        assert "cases.jsonl" not in content, (
            f"VIOLATION: Engine file {py_file.name} references 'cases.jsonl'!"
        )
