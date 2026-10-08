"""Contractual tests for Rules Engine and tri-valued logic (§8, §5.5, §24.1).

Tests:
1. Rule operators:
   - range_contains
   - eq
   - in
   - role_pair_allowed
2. Verdict aggregation:
   - FAIL + UNKNOWN -> INCOMPATIBLE (§5.5, §24.1)
   - All PASS -> COMPATIBLE
   - No FAIL, any UNKNOWN -> INSUFFICIENT_EVIDENCE
   - Deterministic FAIL can NEVER be overridden by model confidence (§8.2)
"""

from __future__ import annotations

import pytest

from industrial_lab.rules.engine import (
    FactEntry,
    FactIndex,
    MissingPolicy,
    RangeInterval,
    Rule,
    RuleCheck,
    RuleCondition,
    RuleScope,
    RulesEngine,
    eval_eq,
    eval_in,
    eval_range_contains,
    eval_role_pair_allowed,
)
from industrial_lab.schemas import (
    CheckResult,
    CheckStatus,
    DecisionOrigin,
    Fact,
    Requirement,
    RequirementKind,
    TechnicalVerdict,
    aggregate_verdict,
)


# ==============================================================================
# 1. Test Rule Operators (§8.1)
# ==============================================================================

def test_operator_range_contains() -> None:
    """Test `range_contains` operator with scalars and sub-ranges."""
    # Scalar within [min, max]
    assert eval_range_contains([10, 50], 25) is True
    assert eval_range_contains({"min": 10, "max": 50}, 10) is True
    assert eval_range_contains({"min": 10, "max": 50}, 50) is True
    assert eval_range_contains([10, 50], 5) is False
    assert eval_range_contains([10, 50], 55) is False

    # Range containing sub-range [req_min, req_max]
    assert eval_range_contains([0, 100], [10, 50]) is True
    assert eval_range_contains({"min": -20, "max": 60}, {"min": 0, "max": 50}) is True
    assert eval_range_contains([10, 50], [0, 60]) is False  # target exceeds supported range

    # Unit conversion handling
    assert eval_range_contains([10, 30], 24, range_unit="V", target_unit="VDC") is True


def test_operator_eq() -> None:
    """Test `eq` operator for exact and normalized equality."""
    # Exact numeric and string match
    assert eval_eq(24, 24) is True
    assert eval_eq("Modbus RTU", "Modbus RTU") is True
    assert eval_eq("modbus rtu", "MODBUS RTU") is True  # Case-insensitive
    assert eval_eq(24, 12) is False
    assert eval_eq("Profinet", "Modbus") is False

    # Boolean match
    assert eval_eq(True, True) is True
    assert eval_eq(True, False) is False

    # Normalized electrical units
    assert eval_eq(24.0, 24, fact_unit="V", target_unit="VDC") is True


def test_operator_in() -> None:
    """Test `in` operator for membership in sets or lists."""
    assert eval_in("RS485", ["RS232", "RS485", "Ethernet"]) is True
    assert eval_in("CANopen", ["RS232", "RS485"]) is False
    assert eval_in(24, [12, 24, 48]) is True
    assert eval_in(5, [12, 24, 48]) is False

    # Comma-separated string or case-insensitive string membership
    assert eval_in("modbus", "modbus, profinet, ethernet") is True
    assert eval_in("ethercat", "modbus, profinet") is False


def test_operator_role_pair_allowed() -> None:
    """Test `role_pair_allowed` operator for source/sink and master/slave compatibility."""
    # Power Supply (source 24V) -> PLC (sink 24V) compatible
    psu_facts = FactIndex(
        "P_PSU",
        [
            FactEntry("role", "source"),
            FactEntry("voltage", 24.0, unit="VDC"),
        ],
    )
    plc_facts = FactIndex(
        "P_PLC",
        [
            FactEntry("role", "sink"),
            FactEntry("voltage", 24.0, unit="VDC"),
        ],
    )

    check = RuleCheck(operator="role_pair_allowed")
    status, reason, _ = eval_role_pair_allowed(psu_facts, plc_facts, check)
    assert status == CheckStatus.PASS
    assert "allowed" in reason.lower()

    # Incompatible roles: two sources cannot power each other
    plc_source = FactIndex("P_SRC", [FactEntry("role", "source")])
    status_conflict, reason_conflict, _ = eval_role_pair_allowed(psu_facts, plc_source, check)
    assert status_conflict == CheckStatus.FAIL

    # Incompatible voltages: PSU source 48V -> PLC sink 24V
    psu_48v = FactIndex(
        "P_PSU48",
        [
            FactEntry("role", "source"),
            FactEntry("voltage", 48.0, unit="VDC"),
        ],
    )
    status_volt, reason_volt, _ = eval_role_pair_allowed(psu_48v, plc_facts, check)
    assert status_volt == CheckStatus.FAIL
    assert "mismatch" in reason_volt.lower() or "voltage" in reason_volt.lower()


# ==============================================================================
# 2. Test Tri-Valued Logic Aggregation (§5.5, §24.1)
# ==============================================================================

def test_verdict_fail_plus_unknown_is_incompatible() -> None:
    """MEGAPLAN §5.5 & §24.1: FAIL + UNKNOWN -> INCOMPATIBLE.

    If a hard constraint is in FAIL, the overall verdict is INCOMPATIBLE,
    even if other hard constraints are UNKNOWN.
    """
    reqs = [
        Requirement(requirement_id="R_VOLT", kind=RequirementKind.exact_property, operator="eq", target=24, hard=True),
        Requirement(requirement_id="R_TEMP", kind=RequirementKind.operating_condition, operator="range_contains", target=[0, 40], hard=True),
    ]

    checks = [
        CheckResult(requirement_id="R_VOLT", status=CheckStatus.FAIL, decision_origin=DecisionOrigin.rule, reason_code="VOLTAGE_MISMATCH"),
        CheckResult(requirement_id="R_TEMP", status=CheckStatus.UNKNOWN, decision_origin=DecisionOrigin.rule, reason_code="MISSING_PROPERTY"),
    ]

    verdict = aggregate_verdict(checks, reqs)
    assert verdict == TechnicalVerdict.INCOMPATIBLE, "FAIL + UNKNOWN must aggregate to INCOMPATIBLE"


def test_verdict_pass_plus_unknown_is_insufficient_evidence() -> None:
    """MEGAPLAN §5.5: No FAIL, but any hard constraint UNKNOWN -> INSUFFICIENT_EVIDENCE."""
    reqs = [
        Requirement(requirement_id="R_VOLT", kind=RequirementKind.exact_property, operator="eq", target=24, hard=True),
        Requirement(requirement_id="R_TEMP", kind=RequirementKind.operating_condition, operator="range_contains", target=[0, 40], hard=True),
    ]

    checks = [
        CheckResult(requirement_id="R_VOLT", status=CheckStatus.PASS, decision_origin=DecisionOrigin.rule),
        CheckResult(requirement_id="R_TEMP", status=CheckStatus.UNKNOWN, decision_origin=DecisionOrigin.rule, reason_code="MISSING_PROPERTY"),
    ]

    verdict = aggregate_verdict(checks, reqs)
    assert verdict == TechnicalVerdict.INSUFFICIENT_EVIDENCE


def test_verdict_all_pass_is_compatible() -> None:
    """MEGAPLAN §5.5: All hard constraints PASS -> COMPATIBLE."""
    reqs = [
        Requirement(requirement_id="R_VOLT", kind=RequirementKind.exact_property, operator="eq", target=24, hard=True),
        Requirement(requirement_id="R_COMM", kind=RequirementKind.exact_property, operator="eq", target="Modbus", hard=True),
    ]

    checks = [
        CheckResult(requirement_id="R_VOLT", status=CheckStatus.PASS, decision_origin=DecisionOrigin.rule),
        CheckResult(requirement_id="R_COMM", status=CheckStatus.PASS, decision_origin=DecisionOrigin.rule),
    ]

    verdict = aggregate_verdict(checks, reqs)
    assert verdict == TechnicalVerdict.COMPATIBLE


def test_rules_engine_evaluates_product_with_fail_and_unknown() -> None:
    """Verify that RulesEngine.evaluate_product aggregates FAIL + UNKNOWN into INCOMPATIBLE."""
    engine = RulesEngine()

    facts = [
        Fact(
            fact_id="F1",
            product_id="P1",
            property="voltage",
            value=12,  # Target is 24 -> FAIL
            document_revision="rev1",
        )
        # property "temperature" is missing -> UNKNOWN
    ]

    reqs = [
        Requirement(requirement_id="voltage", kind=RequirementKind.exact_property, operator="eq", target=24, hard=True),
        Requirement(requirement_id="temperature", kind=RequirementKind.operating_condition, operator="range_contains", target=[0, 50], hard=True),
    ]

    res = engine.evaluate_product("P1", facts, reqs)
    assert res.verdict == TechnicalVerdict.INCOMPATIBLE
    statuses = {c.requirement_id: c.status for c in res.checks}
    assert statuses.get("voltage") == CheckStatus.FAIL
    assert statuses.get("temperature") == CheckStatus.UNKNOWN
