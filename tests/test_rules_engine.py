"""Tests for Deterministic Rules Engine (§8, §5.5)."""

import json
from pathlib import Path
import pytest
import yaml

from industrial_lab.rules.engine import (
    FactEntry,
    FactIndex,
    MissingPolicy,
    PairEvaluationResult,
    ProductEvaluationResult,
    RangeInterval,
    ReviewStatus,
    Rule,
    RuleCheck,
    RuleCondition,
    RuleScope,
    RulesEngine,
    can_override_check,
    combine_verdicts_with_model,
    eval_eq,
    eval_excludes,
    eval_gte,
    eval_in,
    eval_lte,
    eval_range_contains,
    eval_range_overlap,
    eval_requires,
    eval_role_pair_allowed,
    normalize_property_key,
    normalize_unit,
    parse_numeric_or_range,
)
from industrial_lab.schemas import (
    CheckResult,
    CheckStatus,
    DecisionOrigin,
    Fact,
    Requirement,
    RequirementKind,
    TechnicalVerdict,
)


# ==============================================================================
# 1. Range Interval and Unit Normalization Tests (§8.2)
# ==============================================================================

def test_range_interval_bounds() -> None:
    """Test open and closed boundary checks on RangeInterval (§8.2)."""
    # Closed interval [10, 30]
    closed = RangeInterval(10.0, 30.0, min_inclusive=True, max_inclusive=True)
    assert closed.contains_value(10.0) is True
    assert closed.contains_value(30.0) is True
    assert closed.contains_value(20.0) is True
    assert closed.contains_value(9.99) is False
    assert closed.contains_value(30.01) is False

    # Open interval (10, 30)
    open_int = RangeInterval(10.0, 30.0, min_inclusive=False, max_inclusive=False)
    assert open_int.contains_value(10.0) is False
    assert open_int.contains_value(30.0) is False
    assert open_int.contains_value(20.0) is True

    # Half-open interval [10, 30)
    half_open = RangeInterval(10.0, 30.0, min_inclusive=True, max_inclusive=False)
    assert half_open.contains_value(10.0) is True
    assert half_open.contains_value(30.0) is False


def test_range_contains_range() -> None:
    """Test whether one range contains another range."""
    outer = RangeInterval(10.0, 50.0, min_inclusive=True, max_inclusive=True)
    inner = RangeInterval(20.0, 40.0, min_inclusive=True, max_inclusive=True)
    larger = RangeInterval(5.0, 40.0, min_inclusive=True, max_inclusive=True)

    assert outer.contains_range(inner) is True
    assert outer.contains_range(larger) is False

    # Exact boundary with open/closed sensitivity
    closed_10_50 = RangeInterval(10.0, 50.0, True, True)
    open_10_50 = RangeInterval(10.0, 50.0, False, False)
    assert closed_10_50.contains_range(open_10_50) is True
    assert open_10_50.contains_range(closed_10_50) is False


def test_range_overlap() -> None:
    """Test range overlap with boundary conditions."""
    r1 = RangeInterval(10.0, 20.0, True, True)
    r2 = RangeInterval(20.0, 30.0, True, True)
    r3 = RangeInterval(25.0, 35.0, True, True)

    assert r1.overlaps(r2) is True  # Touch at 20.0
    assert r1.overlaps(r3) is False

    # Non-inclusive boundary touch: [10, 20) and (20, 30] do not overlap
    r_half1 = RangeInterval(10.0, 20.0, True, False)
    r_half2 = RangeInterval(20.0, 30.0, False, True)
    assert r_half1.overlaps(r_half2) is False


def test_unit_normalization() -> None:
    """Test unit conversions to canonical base units (§8.2)."""
    # Voltage: mV -> V
    val, u = normalize_unit(24000.0, "mV")
    assert val == pytest.approx(24.0)
    assert u == "V"

    # Current: mA -> A
    val, u = normalize_unit(500.0, "mA")
    assert val == pytest.approx(0.5)
    assert u == "A"

    # Power: kW -> W
    val, u = normalize_unit(1.5, "kW")
    assert val == pytest.approx(1500.0)
    assert u == "W"

    # Temperature: Fahrenheit -> Celsius
    val, u = normalize_unit(122.0, "degF")
    assert val == pytest.approx(50.0)
    assert u == "C"

    # Temperature: Celsius preservation
    val, u = normalize_unit(45.0, "°C")
    assert val == pytest.approx(45.0)
    assert u == "C"


def test_parse_numeric_or_range() -> None:
    """Test parsing strings and structures into numbers and ranges."""
    res, u = parse_numeric_or_range("24 V")
    assert res == pytest.approx(24.0)
    assert u == "V"

    res, u = parse_numeric_or_range("[-20, 60] °C")
    assert isinstance(res, RangeInterval)
    assert res.min_val == pytest.approx(-20.0)
    assert res.max_val == pytest.approx(60.0)
    assert res.unit == "C"
    assert res.min_inclusive is True
    assert res.max_inclusive is True

    res, u = parse_numeric_or_range("(10 to 30) V")
    assert isinstance(res, RangeInterval)
    assert res.min_inclusive is False
    assert res.max_inclusive is False


# ==============================================================================
# 2. Operator Unit Tests (§8.1)
# ==============================================================================

def test_eval_eq() -> None:
    """Test equality operator with types and unit normalization."""
    # String case-insensitivity
    assert eval_eq("Modbus-RTU", "modbus-rtu") is True
    assert eval_eq("IP67", "ip67") is True
    assert eval_eq("IP67", "IP65") is False

    # Numeric with units
    assert eval_eq("24 V", 24.0, target_unit="V") is True
    assert eval_eq(24000.0, 24.0, fact_unit="mV", target_unit="V") is True
    assert eval_eq(24.0, 12.0, fact_unit="V", target_unit="V") is False

    # Incompatible unit dimensions
    assert eval_eq(24.0, 24.0, fact_unit="V", target_unit="A") is False

    # Boolean
    assert eval_eq(True, "true") is True
    assert eval_eq(False, True) is False


def test_eval_in() -> None:
    """Test membership in set/list."""
    # Scalar in list
    assert eval_in("Modbus-TCP", ["EtherNet/IP", "Modbus-TCP", "PROFINET"]) is True
    assert eval_in("CANopen", ["EtherNet/IP", "Modbus-TCP"]) is False

    # Target in list of facts
    assert eval_in(["Modbus-TCP", "EtherNet/IP"], "EtherNet/IP") is True

    # Case-insensitive
    assert eval_in("modbus-tcp", ["Modbus-TCP", "PROFINET"]) is True


def test_eval_range_contains() -> None:
    """Test range_contains operator."""
    # Range contains scalar
    assert eval_range_contains("[-20, 60] °C", 25.0, target_unit="C") is True
    assert eval_range_contains("[-20, 60] °C", 70.0, target_unit="C") is False

    # Range contains range
    assert eval_range_contains("[10, 50] V", "[20, 30] V") is True
    assert eval_range_contains("[20, 30] V", "[10, 50] V") is False

    # Scalar contained in required target range
    assert eval_range_contains("24 V", "[19.2, 28.8] V") is True
    assert eval_range_contains("48 V", "[19.2, 28.8] V") is False


def test_eval_range_overlap() -> None:
    """Test range_overlap operator."""
    assert eval_range_overlap("[10, 30] V", "[20, 40] V") is True
    assert eval_range_overlap("[10, 20] V", "[25, 35] V") is False


def test_eval_lte_and_gte() -> None:
    """Test lte and gte numeric comparisons with unit normalization."""
    # LTE
    assert eval_lte(12.0, 15.0, "W", "W") is True
    assert eval_lte(15000.0, 15.0, "mW", "W") is True  # 15W <= 15W
    assert eval_lte(18.0, 15.0, "W", "W") is False

    # GTE
    assert eval_gte(60.0, 50.0, "C", "C") is True
    assert eval_gte(122.0, 45.0, "degF", "C") is True  # 50C >= 45C
    assert eval_gte(40.0, 50.0, "C", "C") is False


def test_eval_requires_and_excludes() -> None:
    """Test requires and excludes operators."""
    # Single product with unmet requirement
    facts_with_req = FactIndex("P1", {"requires": ["ACC-MOUNT-01"]})
    st, reason = eval_requires(facts_with_req, "ACC-MOUNT-01")
    assert st == CheckStatus.FAIL
    assert reason == "UNMET_ACCESSORY_REQUIREMENT"

    # Single product without unmet requirement
    facts_clean = FactIndex("P2", {"sku": "P2-MOD"})
    st, reason = eval_requires(facts_clean, None)
    assert st == CheckStatus.PASS

    # Mutual exclusion check (pair)
    facts_a = FactIndex("PA", {"excludes": ["PB"]})
    facts_b = FactIndex("PB", {"sku": "PB"})
    st, reason = eval_excludes(facts_a, facts_b)
    assert st == CheckStatus.FAIL
    assert reason == "MUTUAL_EXCLUSION"


def test_eval_role_pair_allowed() -> None:
    """Test role_pair_allowed for master/slave and PSU source -> PLC sink (§8.1)."""
    # PSU (source 24V) -> PLC (sink [19.2, 28.8]V)
    psu_facts = FactIndex(
        "PSU-01",
        {
            "role": "source",
            "output_voltage": {"value": 24.0, "unit": "V", "evidence_ids": ["E1"]},
        },
    )
    plc_facts = FactIndex(
        "PLC-01",
        {
            "role": "sink",
            "input_voltage_range": {"value": [19.2, 28.8], "unit": "V", "evidence_ids": ["E2"]},
        },
    )

    chk = RuleCheck(operator="role_pair_allowed")
    status, reason, evs = eval_role_pair_allowed(psu_facts, plc_facts, chk)
    assert status == CheckStatus.PASS
    assert reason == "ROLE_PAIR_ALLOWED"
    assert "E1" in evs
    assert "E2" in evs

    # Electrical mismatch (PSU 48V -> PLC 24V range)
    psu_48v_facts = FactIndex(
        "PSU-48V",
        {
            "role": "source",
            "output_voltage": {"value": 48.0, "unit": "V", "evidence_ids": ["E3"]},
        },
    )
    status_mismatch, reason_mismatch, _ = eval_role_pair_allowed(psu_48v_facts, plc_facts, chk)
    assert status_mismatch == CheckStatus.FAIL
    assert "ELECTRICAL_MISMATCH" in reason_mismatch

    # Role conflict: both are sources
    gen_facts = FactIndex("GEN-01", {"role": "source"})
    status_conf, reason_conf, _ = eval_role_pair_allowed(psu_facts, gen_facts, chk)
    assert status_conf == CheckStatus.FAIL
    assert reason_conf == "ROLE_CONFLICT_BOTH_SOURCES"


# ==============================================================================
# 3. Tri-Valued Logic and Aggregation Tests (§5.5)
# ==============================================================================

def test_missing_property_policy() -> None:
    """Never assume zero or true on missing facts (§5.5, §8.1)."""
    engine = RulesEngine()
    empty_facts = FactIndex("P_EMPTY", {})

    check_default = RuleCheck(operator="eq", property="supply_voltage", target=24.0)
    res_default = engine._evaluate_single_check(
        check=check_default,
        facts=empty_facts,
        rule_missing_policy=MissingPolicy.UNKNOWN,
        rule_evidence_ids=[],
        rule_hard=True,
    )
    # Default policy must be UNKNOWN
    assert res_default.status == CheckStatus.UNKNOWN
    assert res_default.reason_code == "MISSING_CRITICAL_PROPERTY"

    # Explicit FAIL policy
    check_fail = RuleCheck(
        operator="eq",
        property="supply_voltage",
        target=24.0,
        missing_policy=MissingPolicy.FAIL,
    )
    res_fail = engine._evaluate_single_check(
        check=check_fail,
        facts=empty_facts,
        rule_missing_policy=MissingPolicy.FAIL,
        rule_evidence_ids=[],
        rule_hard=True,
    )
    assert res_fail.status == CheckStatus.FAIL


def test_verdict_aggregation_tri_valued() -> None:
    """Test three-valued technical verdict aggregation (§5.5):
    1. Any hard check FAIL -> INCOMPATIBLE
    2. No FAIL, but any hard check UNKNOWN -> INSUFFICIENT_EVIDENCE
    3. All hard checks PASS -> COMPATIBLE
    """
    engine = RulesEngine()

    # Case 1: Any hard check FAIL -> INCOMPATIBLE
    facts_fail = {"voltage": 12.0, "ip_rating": "IP20"}
    reqs_fail = [
        Requirement(requirement_id="voltage", kind=RequirementKind.exact_property, operator="eq", target=24.0, hard=True),
        Requirement(requirement_id="ip_rating", kind=RequirementKind.exact_property, operator="eq", target="IP67", hard=True),
    ]
    res_fail = engine.evaluate_product("P1", facts_fail, reqs_fail)
    assert res_fail.verdict == TechnicalVerdict.INCOMPATIBLE

    # Case 2: No FAIL, but hard UNKNOWN -> INSUFFICIENT_EVIDENCE
    facts_unknown = {"voltage": 24.0}  # missing ip_rating
    res_unknown = engine.evaluate_product("P2", facts_unknown, reqs_fail)
    assert res_unknown.verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE
    assert "ip_rating" in res_unknown.missing_evidence

    # Case 3: All hard PASS -> COMPATIBLE
    facts_pass = {"voltage": 24.0, "ip_rating": "IP67"}
    res_pass = engine.evaluate_product("P3", facts_pass, reqs_fail)
    assert res_pass.verdict == TechnicalVerdict.COMPATIBLE


def test_numeric_fail_cannot_be_overridden_by_model() -> None:
    """Test safety invariant (§8.2): Numeric/deterministic FAIL can NEVER be overridden by model confidence."""
    fail_check = CheckResult(
        requirement_id="R_VOLT",
        status=CheckStatus.FAIL,
        evidence_ids=["D1:p12"],
        reason_code="EQUALITY_MISMATCH",
        decision_origin=DecisionOrigin.rule,
    )
    assert can_override_check(fail_check) is False

    # Attempting to combine with a confident model output must not override FAIL
    model_assessments = {
        "R_VOLT": {
            "status": "PASS",
            "model_probabilities": {"PASS": 0.999, "FAIL": 0.001},
            "reason_code": "MODEL_HALLUCINATED_APPROVAL",
        }
    }

    combined = combine_verdicts_with_model([fail_check], model_assessments)
    assert len(combined) == 1
    assert combined[0].status == CheckStatus.FAIL  # MUST REMAIN FAIL!
    assert combined[0].decision_origin == DecisionOrigin.rule

    # Conversely, UNKNOWN check CAN be resolved by model
    unknown_check = CheckResult(
        requirement_id="R_SEMANTIC",
        status=CheckStatus.UNKNOWN,
        evidence_ids=[],
        reason_code="MISSING_CRITICAL_PROPERTY",
        decision_origin=DecisionOrigin.rule,
    )
    assert can_override_check(unknown_check) is True

    model_assessments_sem = {
        "R_SEMANTIC": {
            "status": "PASS",
            "model_probabilities": {"PASS": 0.95, "FAIL": 0.05},
            "reason_code": "SEMANTIC_EVALUATION_POSITIVE",
            "evidence_ids": ["SPAN-01"],
        }
    }
    combined_sem = combine_verdicts_with_model([unknown_check], model_assessments_sem)
    assert combined_sem[0].status == CheckStatus.PASS
    assert combined_sem[0].decision_origin == DecisionOrigin.combined


# ==============================================================================
# 4. RulesEngine DSL Loading & Pair Evaluation Tests (§8.1)
# ==============================================================================

def test_load_rules_from_yaml_file(tmp_path: Path) -> None:
    """Test loading YAML rule definitions following MEGAPLAN §8.1 DSL."""
    yaml_content = """
- rule_id: R-VOLT-COMPAT-001
  scope: pair
  applies_if:
    - property: connection_type
      operator: eq
      value: dc_power
  checks:
    - operator: range_contains
      left_fact: input_voltage_range
      right_fact: output_voltage
      hard: true
  missing_policy: UNKNOWN
  evidence_ids: ["DOC-RULE-01"]
  review_status: approved

- rule_id: R-PENDING-002
  scope: pair
  checks:
    - operator: eq
      left_fact: protocol
      right_fact: protocol
  review_status: pending
"""
    rules_file = tmp_path / "test_rules.yaml"
    rules_file.write_text(yaml_content, encoding="utf-8")

    engine = RulesEngine()
    loaded = engine.load_rules_from_file(rules_file)
    assert len(loaded) == 2
    assert loaded[0].rule_id == "R-VOLT-COMPAT-001"
    assert loaded[0].scope == RuleScope.pair
    assert loaded[0].review_status == ReviewStatus.approved
    assert loaded[1].review_status == ReviewStatus.pending


def test_evaluate_pair_compatible(tmp_path: Path) -> None:
    """Test evaluate_pair with matching voltage and protocol rules."""
    rule_dict = {
        "rule_id": "PAIR-PWR-01",
        "scope": "pair",
        "applies_if": [
            {"property": "role", "operator": "eq", "value": "sink"}
        ],
        "checks": [
            {
                "operator": "range_contains",
                "left_fact": "input_voltage_range",
                "right_fact": "output_voltage",
                "requirement_id": "REQ-PWR-RANGE",
                "hard": True,
            }
        ],
        "review_status": "approved",
    }

    engine = RulesEngine(rules=[rule_dict])

    facts_plc = {
        "role": "sink",
        "input_voltage_range": {"value": [20.4, 28.8], "unit": "V", "evidence_ids": ["E_PLC"]},
    }
    facts_psu = {
        "role": "source",
        "output_voltage": {"value": 24.0, "unit": "V", "evidence_ids": ["E_PSU"]},
    }

    res = engine.evaluate_pair(
        product_a="PLC-100",
        facts_a=facts_plc,
        product_b="PSU-24V",
        facts_b=facts_psu,
    )

    assert res.verdict == TechnicalVerdict.COMPATIBLE
    assert len(res.checks) == 1
    assert res.checks[0].status == CheckStatus.PASS
    assert "E_PLC" in res.checks[0].evidence_ids
    assert "E_PSU" in res.checks[0].evidence_ids

    # Unpack support: verdict, checks = evaluate_pair(...)
    verdict, checks = res
    assert verdict == TechnicalVerdict.COMPATIBLE
    assert len(checks) == 1


def test_evaluate_pair_incompatible() -> None:
    """Test evaluate_pair detecting voltage incompatibility."""
    rule_dict = {
        "rule_id": "PAIR-PWR-01",
        "scope": "pair",
        "checks": [
            {
                "operator": "range_contains",
                "left_fact": "input_voltage_range",
                "right_fact": "output_voltage",
                "requirement_id": "REQ-PWR-RANGE",
                "hard": True,
            }
        ],
        "review_status": "approved",
    }

    engine = RulesEngine(rules=[rule_dict])

    facts_plc = {"input_voltage_range": [20.4, 28.8]}
    facts_psu = {"output_voltage": 48.0}  # Incompatible with 20.4-28.8V

    res = engine.evaluate_pair("PLC-100", facts_plc, "PSU-48V", facts_psu)
    assert res.verdict == TechnicalVerdict.INCOMPATIBLE
    assert res.checks[0].status == CheckStatus.FAIL


def test_pending_rules_skipped_in_official_run() -> None:
    """Test that pending rules are skipped by default (§8.1: 'No ejecutar reglas pending en resultados oficiales')."""
    pending_rule = {
        "rule_id": "R-PENDING-TEST",
        "scope": "single",
        "checks": [{"operator": "eq", "property": "color", "target": "blue"}],
        "review_status": "pending",
    }

    engine = RulesEngine(rules=[pending_rule], allow_pending=False)
    res = engine.evaluate_product("P1", {"color": "red"})

    # The pending rule was skipped, so no checks fired
    assert len(res.checks) == 0
    assert "R-PENDING-TEST" not in res.evaluated_rule_ids

    # When explicitly allowed
    engine_dev = RulesEngine(rules=[pending_rule], allow_pending=True)
    res_dev = engine_dev.evaluate_product("P1", {"color": "red"})
    assert len(res_dev.checks) == 1
    assert res_dev.checks[0].status == CheckStatus.FAIL


def test_load_rules_from_json_and_jsonl(tmp_path: Path) -> None:
    """Test loading rules from JSON and JSONL formats."""
    rule_dict = {
        "rule_id": "R-JSON-01",
        "scope": "single",
        "checks": [{"operator": "gte", "property": "rated_temp", "target": 50.0}],
        "review_status": "approved",
    }

    # JSON array
    json_path = tmp_path / "rules.json"
    json_path.write_text(json.dumps([rule_dict]), encoding="utf-8")
    engine1 = RulesEngine()
    loaded1 = engine1.load_rules_from_file(json_path)
    assert len(loaded1) == 1
    assert loaded1[0].rule_id == "R-JSON-01"

    # JSON object with "rules" key
    json_obj_path = tmp_path / "rules_dict.json"
    json_obj_path.write_text(json.dumps({"rules": [rule_dict]}), encoding="utf-8")
    engine2 = RulesEngine()
    loaded2 = engine2.load_rules_from_file(json_obj_path)
    assert len(loaded2) == 1

    # JSONL format
    jsonl_path = tmp_path / "rules.jsonl"
    jsonl_path.write_text(f"{json.dumps(rule_dict)}\n", encoding="utf-8")
    engine3 = RulesEngine()
    loaded3 = engine3.load_rules_from_file(jsonl_path)
    assert len(loaded3) == 1


def test_evaluate_product_with_pydantic_fact_models() -> None:
    """Test evaluate_product using Pydantic Fact instances directly."""
    facts = [
        Fact(
            fact_id="F1",
            product_id="P_PLC",
            property="supply_voltage",
            value=24.0,
            unit="V",
            evidence_ids=["DOC1:p5"],
        ),
        Fact(
            fact_id="F2",
            product_id="P_PLC",
            property="operating_temperature",
            value=[-20.0, 60.0],
            unit="C",
            evidence_ids=["DOC1:p6"],
        ),
    ]

    reqs = [
        Requirement(
            requirement_id="supply_voltage",
            kind=RequirementKind.exact_property,
            operator="eq",
            target=24.0,
            unit="V",
            hard=True,
        ),
        Requirement(
            requirement_id="operating_temperature",
            kind=RequirementKind.operating_condition,
            operator="range_contains",
            target=25.0,
            unit="C",
            hard=True,
        ),
    ]

    engine = RulesEngine()
    res = engine.evaluate_product("P_PLC", facts, reqs)
    assert res.verdict == TechnicalVerdict.COMPATIBLE
    assert len(res.checks) == 2
    assert all(c.status == CheckStatus.PASS for c in res.checks)
    assert "DOC1:p5" in res.checks[0].evidence_ids
    assert "DOC1:p6" in res.checks[1].evidence_ids


def test_variant_and_condition_filtering() -> None:
    """Test §8.2 variant and condition scoping."""
    entry_standard = FactEntry(
        property_name="voltage",
        value=24.0,
        variant_scope=["V_STANDARD"],
    )
    entry_high_volt = FactEntry(
        property_name="voltage",
        value=230.0,
        variant_scope=["V_HIGH"],
    )

    index = FactIndex("P_VAR", [entry_standard, entry_high_volt])

    # Standard variant lookup
    matched_std = index.get_primary_entry("voltage", variant="V_STANDARD")
    assert matched_std is not None
    assert matched_std.value == 24.0

    # High voltage variant lookup
    matched_high = index.get_primary_entry("voltage", variant="V_HIGH")
    assert matched_high is not None
    assert matched_high.value == 230.0


def test_role_pair_allowed_communication_roles() -> None:
    """Test role_pair_allowed for master/slave, client/server, and conflicts."""
    # Master / Slave pair
    plc_master = FactIndex("PLC_M", {"role": "master"})
    io_slave = FactIndex("IO_S", {"role": "slave"})
    chk = RuleCheck(operator="role_pair_allowed")

    st, reason, _ = eval_role_pair_allowed(plc_master, io_slave, chk)
    assert st == CheckStatus.PASS
    assert reason == "ROLE_PAIR_ALLOWED"

    # Multiple masters conflict
    plc_master2 = FactIndex("PLC_M2", {"role": "master"})
    st_conflict, reason_conflict, _ = eval_role_pair_allowed(plc_master, plc_master2, chk)
    assert st_conflict == CheckStatus.FAIL
    assert reason_conflict == "ROLE_CONFLICT_MULTIPLE_MASTERS"

    # Both sinks conflict
    sink1 = FactIndex("SINK1", {"role": "sink"})
    sink2 = FactIndex("SINK2", {"role": "sink"})
    st_sinks, reason_sinks, _ = eval_role_pair_allowed(sink1, sink2, chk)
    assert st_sinks == CheckStatus.FAIL
    assert reason_sinks == "ROLE_CONFLICT_BOTH_SINKS"

