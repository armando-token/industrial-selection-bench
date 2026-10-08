"""Deterministic Rules Engine for Industrial Equipment Selection.

Adheres strictly to MEGAPLAN.md §8 and §5.5:
- §8.1: YAML/JSON DSL rule representation and supported operators:
  eq, in, range_contains, range_overlap, lte, gte, requires, excludes, role_pair_allowed.
- §8.2: Execution safety invariants:
  - Unit normalization before numeric comparison, preserving originals
  - Intervals with explicit open/closed boundaries and exact limit testing
  - Strict variant and condition scoping
  - A numeric/deterministic FAIL can NEVER be overridden by model confidence
- §5.5: Tri-valued evaluation logic (PASS, FAIL, UNKNOWN):
  - Missing facts evaluated according to missing_policy (default UNKNOWN, never assume zero or true!)
  - Verdict aggregation:
    1. Any hard check FAIL -> INCOMPATIBLE
    2. No FAIL, but any hard check UNKNOWN -> INSUFFICIENT_EVIDENCE
    3. All hard checks PASS -> COMPATIBLE
"""

from __future__ import annotations

import json
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field

from industrial_lab.schemas import (
    CheckResult,
    CheckStatus,
    DecisionOrigin,
    Fact,
    LabBaseModel,
    Requirement,
    RequirementKind,
    TechnicalVerdict,
    aggregate_verdict,
)


# ==============================================================================
# DSL Enums and Models (§8.1)
# ==============================================================================

class RuleScope(str, Enum):
    """Scope of rule evaluation (§8.1)."""
    single = "single"
    pair = "pair"


class MissingPolicy(str, Enum):
    """Policy applied when a required fact/property is missing (§8.1, §5.5)."""
    UNKNOWN = "UNKNOWN"
    FAIL = "FAIL"
    PASS = "PASS"


class ReviewStatus(str, Enum):
    """Rule review status (§8.1). Rules in 'pending' status are skipped in official runs."""
    pending = "pending"
    reviewed = "reviewed"
    approved = "approved"
    rejected = "rejected"


class RuleCondition(LabBaseModel):
    """Precondition in applies_if determining whether a rule fires (§8.1)."""
    model_config = ConfigDict(extra="ignore")

    property: Optional[str] = None
    operator: str = "eq"
    value: Any = None
    target: Any = None
    left_fact: Optional[str] = None
    right_fact: Optional[str] = None
    product_id: Optional[str] = None


class RuleCheck(LabBaseModel):
    """Atomic check within a rule (§8.1)."""
    model_config = ConfigDict(extra="ignore")

    operator: str
    property: Optional[str] = None
    value: Any = None
    target: Any = None
    left_fact: Optional[str] = None
    right_fact: Optional[str] = None
    requirement_id: Optional[str] = None
    hard: bool = True
    missing_policy: Optional[MissingPolicy] = None
    unit: Optional[str] = None
    evidence_ids: list[str] = Field(default_factory=list)
    reason_code: Optional[str] = None
    description: Optional[str] = None
    tolerance: Optional[float] = None
    role_source: Optional[str] = None
    role_sink: Optional[str] = None
    accessory_id: Optional[str] = None
    excluded_id: Optional[str] = None


class Rule(LabBaseModel):
    """Deterministic validation rule representation from YAML/JSON DSL (§8.1)."""
    model_config = ConfigDict(extra="ignore")

    rule_id: str
    scope: RuleScope = RuleScope.single
    applies_if: list[RuleCondition] = Field(default_factory=list)
    checks: list[RuleCheck] = Field(default_factory=list)
    missing_policy: MissingPolicy = MissingPolicy.UNKNOWN
    evidence_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.pending
    description: Optional[str] = None
    hard: bool = True


class ProductEvaluationResult(LabBaseModel):
    """Evaluation result for a single product (§5.5, §5.6)."""
    model_config = ConfigDict(extra="forbid")

    product_id: str
    verdict: TechnicalVerdict
    checks: list[CheckResult] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    evaluated_rule_ids: list[str] = Field(default_factory=list)

    def __iter__(self):
        """Allow unpacking: verdict, checks = evaluate_product(...)"""
        return iter((self.verdict, self.checks))

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class PairEvaluationResult(LabBaseModel):
    """Evaluation result for a pair of products (§5.5, §8.1)."""
    model_config = ConfigDict(extra="forbid")

    product_a: str
    product_b: str
    verdict: TechnicalVerdict
    checks: list[CheckResult] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    evaluated_rule_ids: list[str] = Field(default_factory=list)

    def __iter__(self):
        """Allow unpacking: verdict, checks = evaluate_pair(...)"""
        return iter((self.verdict, self.checks))

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


# ==============================================================================
# Range and Interval Structures with Boundary Inclusiveness (§8.2)
# ==============================================================================

class RangeInterval:
    """Interval with explicit open/closed boundaries and exact limit testing (§8.2)."""

    def __init__(
        self,
        min_val: float,
        max_val: float,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
        unit: Optional[str] = None,
    ):
        if min_val > max_val:
            raise ValueError(f"Interval min_val ({min_val}) cannot be greater than max_val ({max_val})")
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.min_inclusive = bool(min_inclusive)
        self.max_inclusive = bool(max_inclusive)
        self.unit = unit

    def contains_value(self, val: float, tol: float = 1e-9) -> bool:
        """Test if interval contains scalar value, respecting boundary inclusiveness."""
        if self.min_inclusive:
            if val < self.min_val - tol:
                return False
        else:
            if val <= self.min_val + tol:
                return False

        if self.max_inclusive:
            if val > self.max_val + tol:
                return False
        else:
            if val >= self.max_val - tol:
                return False

        return True

    def contains_range(self, other: RangeInterval, tol: float = 1e-9) -> bool:
        """Test if this interval fully contains another interval."""
        # Lower bound check
        if self.min_val < other.min_val - tol:
            pass
        elif abs(self.min_val - other.min_val) <= tol:
            if not self.min_inclusive and other.min_inclusive:
                return False
        else:
            return False

        # Upper bound check
        if self.max_val > other.max_val + tol:
            pass
        elif abs(self.max_val - other.max_val) <= tol:
            if not self.max_inclusive and other.max_inclusive:
                return False
        else:
            return False

        return True

    def overlaps(self, other: RangeInterval, tol: float = 1e-9) -> bool:
        """Test if two intervals have a non-empty intersection (§8.1)."""
        if self.max_val < other.min_val - tol or other.max_val < self.min_val - tol:
            return False
        if abs(self.max_val - other.min_val) <= tol:
            return self.max_inclusive and other.min_inclusive
        if abs(other.max_val - self.min_val) <= tol:
            return other.max_inclusive and self.min_inclusive
        return True

    def __repr__(self) -> str:
        left = "[" if self.min_inclusive else "("
        right = "]" if self.max_inclusive else ")"
        unit_str = f" {self.unit}" if self.unit else ""
        return f"{left}{self.min_val}, {self.max_val}{right}{unit_str}"


# ==============================================================================
# Unit Normalization (§8.2)
# ==============================================================================

# Canonical base units: V, A, W, Hz, C, Ohm, bar, s
UNIT_CONVERSIONS: dict[str, tuple[float, str]] = {
    # Voltage (base: V)
    "v": (1.0, "V"),
    "volt": (1.0, "V"),
    "volts": (1.0, "V"),
    "vdc": (1.0, "V"),
    "vac": (1.0, "V"),
    "v dc": (1.0, "V"),
    "v ac": (1.0, "V"),
    "mv": (0.001, "V"),
    "millivolt": (0.001, "V"),
    "kv": (1000.0, "V"),
    "kilovolt": (1000.0, "V"),

    # Current (base: A)
    "a": (1.0, "A"),
    "amp": (1.0, "A"),
    "amps": (1.0, "A"),
    "ampere": (1.0, "A"),
    "ma": (0.001, "A"),
    "milliamp": (0.001, "A"),
    "ua": (1e-6, "A"),
    "µa": (1e-6, "A"),
    "microamp": (1e-6, "A"),

    # Power (base: W)
    "w": (1.0, "W"),
    "watt": (1.0, "W"),
    "watts": (1.0, "W"),
    "mw": (0.001, "W"),
    "milliwatt": (0.001, "W"),
    "kw": (1000.0, "W"),
    "kilowatt": (1000.0, "W"),

    # Frequency (base: Hz)
    "hz": (1.0, "Hz"),
    "khz": (1000.0, "Hz"),
    "mhz": (1000000.0, "Hz"),
    "ghz": (1000000000.0, "Hz"),

    # Resistance (base: Ohm)
    "ohm": (1.0, "Ohm"),
    "ohms": (1.0, "Ohm"),
    "ω": (1.0, "Ohm"),
    "kohm": (1000.0, "Ohm"),
    "kω": (1000.0, "Ohm"),
    "mohm": (1000000.0, "Ohm"),
    "mω": (1000000.0, "Ohm"),

    # Pressure (base: bar)
    "bar": (1.0, "bar"),
    "mbar": (0.001, "bar"),
    "psi": (0.0689475729, "bar"),
    "pa": (1e-5, "bar"),
    "kpa": (0.01, "bar"),
    "mpa": (10.0, "bar"),

    # Time (base: s)
    "s": (1.0, "s"),
    "sec": (1.0, "s"),
    "second": (1.0, "s"),
    "seconds": (1.0, "s"),
    "ms": (0.001, "s"),
    "millisecond": (0.001, "s"),
    "min": (60.0, "s"),
    "minute": (60.0, "s"),
    "h": (3600.0, "s"),
    "hr": (3600.0, "s"),
    "hour": (3600.0, "s"),
}


def normalize_unit(val: float, unit_str: Optional[str]) -> tuple[float, Optional[str]]:
    """Normalize numeric value and unit to canonical base unit (§8.2)."""
    if unit_str is None:
        return val, None

    clean_u = unit_str.strip().lower()

    # Temperature special conversions
    if clean_u in ("c", "°c", "degc", "deg c", "celsius"):
        return val, "C"
    if clean_u in ("f", "°f", "degf", "deg f", "fahrenheit"):
        return (val - 32.0) * 5.0 / 9.0, "C"
    if clean_u in ("k", "kelvin"):
        return val - 273.15, "C"

    if clean_u in UNIT_CONVERSIONS:
        factor, canonical = UNIT_CONVERSIONS[clean_u]
        return val * factor, canonical

    return val, unit_str.strip()


def parse_numeric_or_range(
    val: Any,
    unit: Optional[str] = None,
) -> tuple[Union[float, RangeInterval, Any], Optional[str]]:
    """Parse numeric or range values with optional units and interval boundaries."""
    if val is None:
        return None, unit

    if isinstance(val, RangeInterval):
        if unit and not val.unit:
            norm_min, norm_unit = normalize_unit(val.min_val, unit)
            norm_max, _ = normalize_unit(val.max_val, unit)
            return RangeInterval(norm_min, norm_max, val.min_inclusive, val.max_inclusive, norm_unit), norm_unit
        return val, val.unit

    # Dictionary representing range
    if isinstance(val, dict):
        min_v = val.get("min", val.get("min_val", val.get("start")))
        max_v = val.get("max", val.get("max_val", val.get("end")))
        if min_v is not None and max_v is not None:
            min_inc = val.get("min_inclusive", val.get("include_min", True))
            max_inc = val.get("max_inclusive", val.get("include_max", True))
            u = val.get("unit", unit)
            n_min, n_u = normalize_unit(float(min_v), u)
            n_max, _ = normalize_unit(float(max_v), u)
            return RangeInterval(n_min, n_max, min_inc, max_inc, n_u), n_u
        if "value" in val:
            return parse_numeric_or_range(val["value"], val.get("unit", unit))

    # List or tuple representing range [min, max]
    if isinstance(val, (list, tuple)) and len(val) == 2:
        try:
            n_min, n_u = normalize_unit(float(val[0]), unit)
            n_max, _ = normalize_unit(float(val[1]), unit)
            return RangeInterval(n_min, n_max, True, True, n_u), n_u
        except (ValueError, TypeError):
            pass

    # Number
    if isinstance(val, (int, float)):
        norm_v, norm_u = normalize_unit(float(val), unit)
        return norm_v, norm_u

    # String parsing
    if isinstance(val, str):
        val_str = val.strip()

        # Check for range string e.g. "[-20, 60] °C", "[-20, 60 °C]", "(10, 30)", "10-30 V", "10..30", "10 to 30 V"
        range_match = re.match(
            r"^([\[\(])?\s*([-+]?\d*\.?\d+)\s*([a-zA-Z°µΩ/%]+)?\s*(?:to|-|\.\.|,)\s*([-+]?\d*\.?\d+)\s*([a-zA-Z°µΩ/%]+)?\s*([\]\)])?\s*([a-zA-Z°µΩ/%]+)?$",
            val_str,
            re.IGNORECASE,
        )
        if range_match:
            open_left = range_match.group(1) == "("
            min_v = float(range_match.group(2))
            max_v = float(range_match.group(4))
            open_right = range_match.group(6) == ")"
            detected_unit = (
                range_match.group(3)
                or range_match.group(5)
                or range_match.group(7)
                or unit
            )

            n_min, n_u = normalize_unit(min_v, detected_unit)
            n_max, _ = normalize_unit(max_v, detected_unit)
            return (
                RangeInterval(
                    min_val=min(n_min, n_max),
                    max_val=max(n_min, n_max),
                    min_inclusive=not open_left,
                    max_inclusive=not open_right,
                    unit=n_u,
                ),
                n_u,
            )

        # Single number with optional unit e.g. "24 V", "50 deg C", "1500mA"
        num_match = re.match(r"^([-+]?\d*\.?\d+)\s*([a-zA-Z°µΩ/%]+)?$", val_str)
        if num_match:
            v = float(num_match.group(1))
            detected_unit = num_match.group(2) or unit
            return normalize_unit(v, detected_unit)

    return val, unit


# ==============================================================================
# Facts Normalization & Indexing (§5.2, §8.2)
# ==============================================================================

def normalize_property_key(key: str) -> str:
    """Normalize property name for uniform lookup."""
    return key.strip().lower().replace("-", "_").replace(" ", "_")


class FactEntry:
    """Indexed atomic fact with metadata (§5.2)."""

    def __init__(
        self,
        property_name: str,
        value: Any,
        unit: Optional[str] = None,
        evidence_ids: Optional[list[str]] = None,
        conditions: Optional[list[Any]] = None,
        variant_scope: Optional[list[str]] = None,
        port_id: Optional[str] = None,
        extraction_status: str = "reviewed",
        raw: Any = None,
    ):
        self.property = property_name
        self.value = value
        self.unit = unit
        self.evidence_ids = evidence_ids or []
        self.conditions = conditions or []
        self.variant_scope = variant_scope or []
        self.port_id = port_id
        self.extraction_status = extraction_status
        self.raw = raw


class FactIndex:
    """Index of facts for a specific product, supporting variant and condition scoping (§8.2)."""

    def __init__(self, product_id: str, facts: Any):
        self.product_id = product_id
        self.by_property: dict[str, list[FactEntry]] = {}
        self.all_entries: list[FactEntry] = []
        self._index_facts(facts)

    def _index_facts(self, facts: Any) -> None:
        if not facts:
            return

        if isinstance(facts, dict):
            # Dict of facts
            if "facts" in facts and isinstance(facts["facts"], list):
                self._index_facts(facts["facts"])
                return
            for prop_name, val in facts.items():
                if isinstance(val, dict) and "value" in val:
                    entry = FactEntry(
                        property_name=prop_name,
                        value=val.get("value"),
                        unit=val.get("unit"),
                        evidence_ids=val.get("evidence_ids", []),
                        conditions=val.get("conditions", []),
                        variant_scope=val.get("variant_scope", []),
                        port_id=val.get("port_id"),
                        raw=val,
                    )
                else:
                    entry = FactEntry(property_name=prop_name, value=val, raw=val)
                self._add_entry(entry)
            return

        if isinstance(facts, (list, tuple, set)):
            for item in facts:
                if isinstance(item, FactEntry):
                    self._add_entry(item)
                    continue
                if isinstance(item, Fact):
                    entry = FactEntry(
                        property_name=item.property,
                        value=item.value,
                        unit=item.unit,
                        evidence_ids=item.evidence_ids,
                        conditions=item.conditions,
                        variant_scope=item.variant_scope,
                        port_id=item.port_id,
                        extraction_status=item.extraction_status.value if hasattr(item.extraction_status, "value") else str(item.extraction_status),
                        raw=item,
                    )
                elif isinstance(item, dict):
                    prop = item.get("property") or item.get("name") or item.get("key")
                    if prop:
                        entry = FactEntry(
                            property_name=str(prop),
                            value=item.get("value"),
                            unit=item.get("unit"),
                            evidence_ids=item.get("evidence_ids", []),
                            conditions=item.get("conditions", []),
                            variant_scope=item.get("variant_scope", []),
                            port_id=item.get("port_id"),
                            extraction_status=str(item.get("extraction_status", "reviewed")),
                            raw=item,
                        )
                    else:
                        # Could be a single-key dict like {"voltage": 24}
                        for k, v in item.items():
                            entry = FactEntry(property_name=k, value=v, raw=item)
                            self._add_entry(entry)
                        continue
                elif hasattr(item, "property") and hasattr(item, "value"):
                    entry = FactEntry(
                        property_name=item.property,
                        value=item.value,
                        unit=getattr(item, "unit", None),
                        evidence_ids=getattr(item, "evidence_ids", []),
                        conditions=getattr(item, "conditions", []),
                        variant_scope=getattr(item, "variant_scope", []),
                        port_id=getattr(item, "port_id", None),
                        raw=item,
                    )
                else:
                    continue
                self._add_entry(entry)

    def _add_entry(self, entry: FactEntry) -> None:
        norm_key = normalize_property_key(entry.property)
        if norm_key not in self.by_property:
            self.by_property[norm_key] = []
        self.by_property[norm_key].append(entry)
        self.all_entries.append(entry)

    def get_entries(
        self,
        prop_name: str,
        variant: Optional[str] = None,
        condition: Optional[str] = None,
    ) -> list[FactEntry]:
        """Find matching fact entries with optional variant/condition filtering (§8.2)."""
        norm_key = normalize_property_key(prop_name)
        candidates = self.by_property.get(norm_key, [])

        # Alias lookup if not found
        if not candidates:
            alias_map: dict[str, list[str]] = {
                "supply_voltage": ["supply_voltage_nominal_v", "supply_voltage_dc", "operating_voltage", "voltage", "input_voltage"],
                "voltage": ["supply_voltage_nominal_v", "supply_voltage", "supply_voltage_dc", "input_voltage", "volt", "volts"],
                "volt": ["supply_voltage_nominal_v", "voltage", "supply_voltage", "input_voltage"],
                "volts": ["supply_voltage_nominal_v", "voltage", "supply_voltage", "input_voltage"],
                "operating_temperature": ["operating_temp_min_c", "operating_temp_max_c", "ambient_temperature", "operating_temp", "temperature", "temp"],
                "temperature": ["operating_temp_min_c", "operating_temp_max_c", "operating_temperature", "operating_temp", "ambient_temperature", "temp"],
                "temp": ["temperature", "operating_temperature", "ambient_temperature"],
                "ip_rating": ["enclosure_ip_rating", "ingress_protection", "protection_degree"],
                "modbus_protocol": ["communication_protocol", "communication_interface"],
                "protocol": ["communication_protocol", "communication_interface"],
                "communication_protocol": ["communication_protocol", "communication_interface"],
            }
            # Check direct alias
            for alias in alias_map.get(norm_key, []):
                if alias in self.by_property:
                    candidates = self.by_property[alias]
                    break

            # Strip common requirement ID prefixes like r_, req_, chk_
            stripped_key = re.sub(r"^(r_|req_|chk_)", "", norm_key)
            if not candidates:
                if stripped_key in self.by_property:
                    candidates = self.by_property[stripped_key]
                else:
                    for alias in alias_map.get(stripped_key, []):
                        if alias in self.by_property:
                            candidates = self.by_property[alias]
                            break

            # Substring/prefix fallback search across by_property keys
            if not candidates:
                lookup_keys = [norm_key]
                if stripped_key and stripped_key != norm_key:
                    lookup_keys.append(stripped_key)
                for lk in lookup_keys:
                    if len(lk) < 3:
                        continue
                    for bp_key in self.by_property:
                        if lk in bp_key or bp_key in lk:
                            candidates = self.by_property[bp_key]
                            break
                    if candidates:
                        break

        if not candidates:
            return []

        # Filter by variant if specified (§8.2)
        if variant:
            v_matches = [
                e for e in candidates
                if not e.variant_scope or variant in e.variant_scope
            ]
            if v_matches:
                candidates = v_matches

        # Filter by condition if specified (§8.2)
        if condition:
            c_matches = [
                e for e in candidates
                if not e.conditions or any(condition in str(c) for c in e.conditions)
            ]
            if c_matches:
                candidates = c_matches

        return candidates

    def get_primary_entry(
        self,
        prop_name: str,
        variant: Optional[str] = None,
        condition: Optional[str] = None,
    ) -> Optional[FactEntry]:
        """Get the most relevant fact entry for the specified property."""
        entries = self.get_entries(prop_name, variant, condition)
        return entries[0] if entries else None

    def has_property(self, prop_name: str) -> bool:
        """Check if property exists in fact index."""
        return len(self.get_entries(prop_name)) > 0


# ==============================================================================
# Operator Implementations (§8.1)
# ==============================================================================

def eval_eq(
    fact_val: Any,
    target_val: Any,
    fact_unit: Optional[str] = None,
    target_unit: Optional[str] = None,
    tolerance: Optional[float] = None,
) -> bool:
    """Evaluate equality check (e.g. property == value)."""
    norm_fact, u_fact = parse_numeric_or_range(fact_val, fact_unit)
    norm_target, u_target = parse_numeric_or_range(target_val, target_unit)

    # Both numeric
    if isinstance(norm_fact, (int, float)) and isinstance(norm_target, (int, float)):
        # Check unit dimension compatibility
        if u_fact and u_target and u_fact != u_target:
            return False
        tol = tolerance if tolerance is not None else 1e-6
        return math.isclose(float(norm_fact), float(norm_target), abs_tol=tol)

    # Boolean
    if isinstance(fact_val, bool) or isinstance(target_val, bool):
        b_fact = fact_val if isinstance(fact_val, bool) else str(fact_val).lower() in ("true", "1", "yes")
        b_target = target_val if isinstance(target_val, bool) else str(target_val).lower() in ("true", "1", "yes")
        return b_fact == b_target

    # String comparison (case-insensitive, normalized whitespace, hyphens, underscores)
    if isinstance(fact_val, str) and isinstance(target_val, str):
        f_norm = re.sub(r"[\s\-_]+", " ", fact_val.strip().lower())
        t_norm = re.sub(r"[\s\-_]+", " ", target_val.strip().lower())
        return f_norm == t_norm

    return fact_val == target_val


def eval_in(fact_val: Any, target_val: Any) -> bool:
    """Evaluate membership check (value in set/list)."""
    if target_val is None or fact_val is None:
        return False

    # Target is collection: fact_val in target
    if isinstance(target_val, (list, tuple, set)):
        # Normalize strings
        if isinstance(fact_val, str):
            f_str = fact_val.strip().lower()
            return any(
                isinstance(item, str) and item.strip().lower() == f_str
                or item == fact_val
                for item in target_val
            )
        return fact_val in target_val

    # Fact is collection: target_val in fact
    if isinstance(fact_val, (list, tuple, set)):
        if isinstance(target_val, str):
            t_str = target_val.strip().lower()
            return any(
                isinstance(item, str) and item.strip().lower() == t_str
                or item == target_val
                for item in fact_val
            )
        return target_val in fact_val

    # Both strings: check substring
    if isinstance(fact_val, str) and isinstance(target_val, str):
        return fact_val.strip().lower() in target_val.strip().lower()

    return False


def eval_range_contains(
    range_data: Any,
    target_data: Any,
    range_unit: Optional[str] = None,
    target_unit: Optional[str] = None,
    tolerance: Optional[float] = None,
) -> bool:
    """Evaluate range_contains: range [min, max] contains required value or target range [req_min, req_max]."""
    r_val, u_r = parse_numeric_or_range(range_data, range_unit)
    t_val, u_t = parse_numeric_or_range(target_data, target_unit)

    tol = tolerance if tolerance is not None else 1e-9

    # Unit dimension compatibility
    if u_r and u_t and u_r != u_t:
        return False

    # Case 1: Left is RangeInterval, Right is scalar
    if isinstance(r_val, RangeInterval) and isinstance(t_val, (int, float)):
        return r_val.contains_value(float(t_val), tol=tol)

    # Case 2: Left is RangeInterval, Right is RangeInterval
    if isinstance(r_val, RangeInterval) and isinstance(t_val, RangeInterval):
        return r_val.contains_range(t_val, tol=tol)

    # Case 3: Left is scalar, Right is RangeInterval (check if scalar falls within target range)
    if isinstance(r_val, (int, float)) and isinstance(t_val, RangeInterval):
        return t_val.contains_value(float(r_val), tol=tol)

    # Case 4: Both are scalars
    if isinstance(r_val, (int, float)) and isinstance(t_val, (int, float)):
        return math.isclose(float(r_val), float(t_val), abs_tol=tol)

    return False


def eval_range_overlap(
    range_a: Any,
    range_b: Any,
    unit_a: Optional[str] = None,
    unit_b: Optional[str] = None,
    tolerance: Optional[float] = None,
) -> bool:
    """Evaluate range_overlap: two ranges have non-empty intersection (§8.1)."""
    r_a, u_a = parse_numeric_or_range(range_a, unit_a)
    r_b, u_b = parse_numeric_or_range(range_b, unit_b)

    tol = tolerance if tolerance is not None else 1e-9

    if u_a and u_b and u_a != u_b:
        return False

    # Convert scalars to point intervals if necessary
    if isinstance(r_a, (int, float)):
        r_a = RangeInterval(float(r_a), float(r_a), True, True, u_a)
    if isinstance(r_b, (int, float)):
        r_b = RangeInterval(float(r_b), float(r_b), True, True, u_b)

    if isinstance(r_a, RangeInterval) and isinstance(r_b, RangeInterval):
        return r_a.overlaps(r_b, tol=tol)

    return False


def eval_lte(
    fact_val: Any,
    target_val: Any,
    fact_unit: Optional[str] = None,
    target_unit: Optional[str] = None,
    tolerance: Optional[float] = None,
) -> bool:
    """Evaluate numeric less than or equal."""
    norm_fact, u_fact = parse_numeric_or_range(fact_val, fact_unit)
    norm_target, u_target = parse_numeric_or_range(target_val, target_unit)

    if u_fact and u_target and u_fact != u_target:
        return False

    if isinstance(norm_fact, (int, float)) and isinstance(norm_target, (int, float)):
        tol = tolerance if tolerance is not None else 1e-9
        return float(norm_fact) <= float(norm_target) + tol

    return False


def eval_gte(
    fact_val: Any,
    target_val: Any,
    fact_unit: Optional[str] = None,
    target_unit: Optional[str] = None,
    tolerance: Optional[float] = None,
) -> bool:
    """Evaluate numeric greater than or equal."""
    norm_fact, u_fact = parse_numeric_or_range(fact_val, fact_unit)
    norm_target, u_target = parse_numeric_or_range(target_val, target_unit)

    if u_fact and u_target and u_fact != u_target:
        return False

    if isinstance(norm_fact, (int, float)) and isinstance(norm_target, (int, float)):
        tol = tolerance if tolerance is not None else 1e-9
        return float(norm_fact) >= float(norm_target) - tol

    return False


def eval_requires(
    product_facts: FactIndex,
    required_item: Any,
    partner_product_id: Optional[str] = None,
    partner_facts: Optional[FactIndex] = None,
) -> tuple[CheckStatus, str]:
    """Evaluate 'requires': product requires accessory or secondary component (§8.1, §8.2)."""
    # If evaluating a pair
    if partner_product_id is not None:
        target_str = str(required_item).strip().lower() if required_item else ""
        partner_id_norm = partner_product_id.strip().lower()

        # Check if partner product satisfies required ID/category
        if target_str in (partner_id_norm, ""):
            return CheckStatus.PASS, "REQUIRED_COMPONENT_PRESENT"

        # Check if partner facts match required accessory
        if partner_facts:
            sku_entry = partner_facts.get_primary_entry("sku") or partner_facts.get_primary_entry("model")
            if sku_entry and str(sku_entry.value).strip().lower() == target_str:
                return CheckStatus.PASS, "REQUIRED_COMPONENT_PRESENT"

        return CheckStatus.FAIL, "MISSING_REQUIRED_COMPONENT"

    # Single product evaluation
    req_entry = product_facts.get_primary_entry("requires") or product_facts.get_primary_entry("requires_accessory")
    if req_entry is None:
        # Product does not declare an unmet requirement
        return CheckStatus.PASS, "NO_UNMET_REQUIREMENT"

    req_val = req_entry.value
    if not req_val:
        return CheckStatus.PASS, "NO_UNMET_REQUIREMENT"

    return CheckStatus.FAIL, "UNMET_ACCESSORY_REQUIREMENT"


def eval_excludes(
    product_a_facts: FactIndex,
    product_b_facts: Optional[FactIndex] = None,
    target_exclusion: Optional[str] = None,
) -> tuple[CheckStatus, str]:
    """Evaluate 'excludes': mutual exclusion (§8.1)."""
    # Single product scope
    if product_b_facts is None:
        excl_entry = product_a_facts.get_primary_entry("excludes")
        if excl_entry and excl_entry.value:
            if target_exclusion and eval_in(target_exclusion, excl_entry.value):
                return CheckStatus.FAIL, "MUTUAL_EXCLUSION"
        return CheckStatus.PASS, "NO_EXCLUSION"

    # Pair scope: check if A excludes B or B excludes A
    excl_a = product_a_facts.get_primary_entry("excludes")
    if excl_a and excl_a.value:
        if eval_in(product_b_facts.product_id, excl_a.value):
            return CheckStatus.FAIL, "MUTUAL_EXCLUSION"
        b_model = product_b_facts.get_primary_entry("model") or product_b_facts.get_primary_entry("sku")
        if b_model and eval_in(b_model.value, excl_a.value):
            return CheckStatus.FAIL, "MUTUAL_EXCLUSION"

    excl_b = product_b_facts.get_primary_entry("excludes")
    if excl_b and excl_b.value:
        if eval_in(product_a_facts.product_id, excl_b.value):
            return CheckStatus.FAIL, "MUTUAL_EXCLUSION"
        a_model = product_a_facts.get_primary_entry("model") or product_a_facts.get_primary_entry("sku")
        if a_model and eval_in(a_model.value, excl_b.value):
            return CheckStatus.FAIL, "MUTUAL_EXCLUSION"

    return CheckStatus.PASS, "NO_EXCLUSION"


def eval_role_pair_allowed(
    facts_a: FactIndex,
    facts_b: FactIndex,
    check: RuleCheck,
) -> tuple[CheckStatus, str, list[str]]:
    """Validate master/slave, source/sink compatibility (§8.1).

    e.g. PSU source 24V -> PLC sink 24V.
    """
    evidence_ids: list[str] = []

    # 1. Determine roles
    role_a_entry = (
        facts_a.get_primary_entry("role")
        or facts_a.get_primary_entry("power_role")
        or facts_a.get_primary_entry("device_role")
    )
    role_b_entry = (
        facts_b.get_primary_entry("role")
        or facts_b.get_primary_entry("power_role")
        or facts_b.get_primary_entry("device_role")
    )

    role_a = (
        str(role_a_entry.value).strip().lower()
        if role_a_entry and role_a_entry.value
        else (check.role_source or "").lower()
    )
    role_b = (
        str(role_b_entry.value).strip().lower()
        if role_b_entry and role_b_entry.value
        else (check.role_sink or "").lower()
    )

    if role_a_entry:
        evidence_ids.extend(role_a_entry.evidence_ids)
    if role_b_entry:
        evidence_ids.extend(role_b_entry.evidence_ids)

    if not role_a or not role_b:
        return CheckStatus.UNKNOWN, "MISSING_ROLE_SPECIFICATION", evidence_ids

    # 2. Check role compatibility
    compatible_pairs = [
        {"source", "sink"},
        {"power_source", "power_sink"},
        {"master", "slave"},
        {"client", "server"},
        {"transmitter", "receiver"},
        {"peer", "peer"},
        {"bidirectional", "bidirectional"},
    ]

    roles_set = {role_a, role_b}
    is_compatible_role = roles_set in compatible_pairs

    if not is_compatible_role:
        if role_a == "source" and role_b == "source":
            return CheckStatus.FAIL, "ROLE_CONFLICT_BOTH_SOURCES", evidence_ids
        if role_a == "sink" and role_b == "sink":
            return CheckStatus.FAIL, "ROLE_CONFLICT_BOTH_SINKS", evidence_ids
        if role_a == "master" and role_b == "master":
            return CheckStatus.FAIL, "ROLE_CONFLICT_MULTIPLE_MASTERS", evidence_ids
        return CheckStatus.FAIL, "INCOMPATIBLE_ROLES", evidence_ids

    # 3. Check electrical compatibility (e.g. source voltage -> sink voltage range)
    # Identify which is source and which is sink
    source_facts = facts_a if "source" in role_a or "master" in role_a else facts_b
    sink_facts = facts_b if source_facts is facts_a else facts_a

    source_v_entry = (
        source_facts.get_primary_entry("output_voltage")
        or source_facts.get_primary_entry("supply_voltage")
        or source_facts.get_primary_entry("voltage")
    )
    sink_v_entry = (
        sink_facts.get_primary_entry("input_voltage_range")
        or sink_facts.get_primary_entry("supply_voltage_range")
        or sink_facts.get_primary_entry("input_voltage")
        or sink_facts.get_primary_entry("supply_voltage")
    )

    if source_v_entry and sink_v_entry:
        evidence_ids.extend(source_v_entry.evidence_ids)
        evidence_ids.extend(sink_v_entry.evidence_ids)
        v_ok = eval_range_contains(
            sink_v_entry.value,
            source_v_entry.value,
            sink_v_entry.unit,
            source_v_entry.unit,
        )
        if not v_ok:
            return CheckStatus.FAIL, "ELECTRICAL_MISMATCH: VOLTAGE_OUT_OF_RANGE", evidence_ids

    return CheckStatus.PASS, "ROLE_PAIR_ALLOWED", evidence_ids


# ==============================================================================
# Model Confidence / Rule Invariant Enforcement (§8.2)
# ==============================================================================

def can_override_check(check: CheckResult) -> bool:
    """Enforces MEGAPLAN §8.2: A numeric / deterministic FAIL can NEVER be overridden by model confidence."""
    if check.status == CheckStatus.FAIL and check.decision_origin == DecisionOrigin.rule:
        return False
    return True


def combine_verdicts_with_model(
    rule_checks: list[CheckResult],
    model_assessments: Optional[dict[str, Any]] = None,
) -> list[CheckResult]:
    """Combines rule check results with model assessments (§8.2).

    CRITICAL RULE (§8.2): A numeric / deterministic FAIL can NEVER be
    overridden by model confidence. Only UNKNOWN checks may be resolved
    by model predictions.
    """
    if not model_assessments:
        return rule_checks

    combined: list[CheckResult] = []
    for check in rule_checks:
        model_entry = model_assessments.get(check.requirement_id)
        if not model_entry:
            combined.append(check)
            continue

        model_status = model_entry.get("status")
        probs = model_entry.get("model_probabilities")

        # Deterministic FAIL cannot be overridden
        if check.status == CheckStatus.FAIL and check.decision_origin == DecisionOrigin.rule:
            combined.append(check)
            continue

        # If rule was UNKNOWN and model provides high-confidence decision, combine
        if check.status == CheckStatus.UNKNOWN and model_status:
            combined.append(
                CheckResult(
                    requirement_id=check.requirement_id,
                    status=CheckStatus(model_status),
                    evidence_ids=check.evidence_ids + model_entry.get("evidence_ids", []),
                    reason_code=f"RESOLVED_BY_MODEL: {model_entry.get('reason_code', 'MODEL_JUDGMENT')}",
                    decision_origin=DecisionOrigin.combined,
                    model_probabilities=probs,
                )
            )
        else:
            combined.append(check)

    return combined


# ==============================================================================
# RulesEngine Class (§8)
# ==============================================================================

class RulesEngine:
    """Deterministic rules engine supporting DSL evaluation, unit normalization,
    and tri-valued technical verdict aggregation.
    """

    def __init__(
        self,
        rules: Optional[Sequence[Union[Rule, dict[str, Any]]]] = None,
        allow_pending: bool = False,
    ):
        self.rules: list[Rule] = []
        self.allow_pending: bool = allow_pending
        if rules:
            for r in rules:
                self.add_rule(r)

    def add_rule(self, rule: Union[Rule, dict[str, Any]]) -> None:
        """Add a rule to the engine."""
        if isinstance(rule, dict):
            rule = Rule.model_validate(rule)
        self.rules.append(rule)

    def load_rules_from_file(self, path: Union[str, Path]) -> list[Rule]:
        """Load rules from a YAML, JSON, or JSONL file into the engine (§8.1)."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Rules file not found: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        ext = file_path.suffix.lower()

        raw_data: Any = None
        if ext in (".yaml", ".yml"):
            raw_data = yaml.safe_load(content)
        elif ext == ".json":
            raw_data = json.loads(content)
        elif ext == ".jsonl":
            raw_data = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            try:
                raw_data = yaml.safe_load(content)
            except Exception:
                raw_data = json.loads(content)

        loaded_rules: list[Rule] = []
        if isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    loaded_rules.append(Rule.model_validate(item))
        elif isinstance(raw_data, dict):
            if "rules" in raw_data and isinstance(raw_data["rules"], list):
                for item in raw_data["rules"]:
                    if isinstance(item, dict):
                        loaded_rules.append(Rule.model_validate(item))
            else:
                loaded_rules.append(Rule.model_validate(raw_data))

        for r in loaded_rules:
            self.rules.append(r)

        return loaded_rules

    def _condition_matches(
        self,
        cond: RuleCondition,
        facts: FactIndex,
        context: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Check if an applies_if condition matches the candidate facts (§8.1)."""
        if cond.product_id and facts.product_id != cond.product_id:
            return False

        prop = cond.property or cond.left_fact
        if not prop:
            return True

        entries = facts.get_entries(prop)
        if not entries:
            return False

        target_v = cond.value if cond.value is not None else cond.target
        op = (cond.operator or "eq").lower()

        for entry in entries:
            if op == "eq" and eval_eq(entry.value, target_v, entry.unit):
                return True
            if op == "in" and eval_in(entry.value, target_v):
                return True
            if op == "range_contains" and eval_range_contains(entry.value, target_v, entry.unit):
                return True

        return False

    def _rule_applies(
        self,
        rule: Rule,
        facts_a: FactIndex,
        facts_b: Optional[FactIndex] = None,
    ) -> bool:
        """Evaluate all applies_if conditions for a rule (§8.1)."""
        if not rule.applies_if:
            return True

        for cond in rule.applies_if:
            match_a = self._condition_matches(cond, facts_a)
            match_b = self._condition_matches(cond, facts_b) if facts_b else False
            if not (match_a or match_b):
                return False

        return True

    def _evaluate_single_check(
        self,
        check: RuleCheck,
        facts: FactIndex,
        rule_missing_policy: MissingPolicy,
        rule_evidence_ids: list[str],
        rule_hard: bool,
    ) -> CheckResult:
        """Evaluate an atomic check against candidate product facts."""
        req_id = check.requirement_id or f"CHK-{check.operator}-{check.property or 'check'}"
        missing_pol = check.missing_policy or rule_missing_policy or MissingPolicy.UNKNOWN
        op = (check.operator or "eq").lower()
        target_v = check.target if check.target is not None else check.value

        prop = check.property or check.left_fact

        # If operator is not a standard comparison operator but is a property name, infer operator
        standard_ops = ("eq", "in", "range_contains", "range_overlap", "lte", "gte", "requires", "excludes", "role_pair_allowed")
        if op not in standard_ops:
            if facts.has_property(op) or (not prop or not facts.has_property(prop)):
                prop = op
                op = "range_contains" if isinstance(target_v, (list, tuple)) or (isinstance(target_v, str) and ("-" in target_v or "to" in target_v)) else "eq"

        # Missing property handling (§5.5, §8.1)
        if not prop or not facts.has_property(prop):
            if op == "requires":
                # Special evaluation for requires
                st, r_code = eval_requires(facts, target_v)
                return CheckResult(
                    requirement_id=req_id,
                    status=st,
                    evidence_ids=list(set(rule_evidence_ids + check.evidence_ids)),
                    reason_code=r_code,
                    decision_origin=DecisionOrigin.rule,
                    model_probabilities=None,
                )
            if op == "excludes":
                st, r_code = eval_excludes(facts, None, str(target_v) if target_v else None)
                return CheckResult(
                    requirement_id=req_id,
                    status=st,
                    evidence_ids=list(set(rule_evidence_ids + check.evidence_ids)),
                    reason_code=r_code,
                    decision_origin=DecisionOrigin.rule,
                    model_probabilities=None,
                )

            # Property is absent from facts -> apply missing_policy (default UNKNOWN, never assume zero/true!)
            status_map = {
                MissingPolicy.UNKNOWN: CheckStatus.UNKNOWN,
                MissingPolicy.FAIL: CheckStatus.FAIL,
                MissingPolicy.PASS: CheckStatus.PASS,
            }
            res_status = status_map.get(missing_pol, CheckStatus.UNKNOWN)
            return CheckResult(
                requirement_id=req_id,
                status=res_status,
                evidence_ids=list(set(rule_evidence_ids + check.evidence_ids)),
                reason_code="MISSING_CRITICAL_PROPERTY",
                decision_origin=DecisionOrigin.rule,
                model_probabilities=None,
            )

        entry = facts.get_primary_entry(prop)
        fact_val = entry.value if entry else None
        fact_unit = entry.unit if entry else None
        ev_ids = list(set(rule_evidence_ids + check.evidence_ids + (entry.evidence_ids if entry else [])))

        # Evaluate operator
        status = CheckStatus.UNKNOWN
        reason_code = check.reason_code or "EVALUATED"

        if op == "eq":
            passed = eval_eq(fact_val, target_v, fact_unit, check.unit, check.tolerance)
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            reason_code = "REQUIREMENT_SATISFIED" if passed else "EQUALITY_MISMATCH"

        elif op == "in":
            passed = eval_in(fact_val, target_v)
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            reason_code = "REQUIREMENT_SATISFIED" if passed else "VALUE_NOT_IN_SET"

        elif op == "range_contains":
            passed = eval_range_contains(fact_val, target_v, fact_unit, check.unit, check.tolerance)
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            reason_code = "REQUIREMENT_SATISFIED" if passed else "RANGE_DOES_NOT_CONTAIN_TARGET"

        elif op == "range_overlap":
            passed = eval_range_overlap(fact_val, target_v, fact_unit, check.unit, check.tolerance)
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            reason_code = "REQUIREMENT_SATISFIED" if passed else "RANGES_DO_NOT_OVERLAP"

        elif op == "lte":
            passed = eval_lte(fact_val, target_v, fact_unit, check.unit, check.tolerance)
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            reason_code = "REQUIREMENT_SATISFIED" if passed else "VALUE_EXCEEDS_MAXIMUM"

        elif op == "gte":
            passed = eval_gte(fact_val, target_v, fact_unit, check.unit, check.tolerance)
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            reason_code = "REQUIREMENT_SATISFIED" if passed else "VALUE_BELOW_MINIMUM"

        elif op == "exists":
            passed = entry is not None and fact_val is not None
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
            reason_code = "REQUIREMENT_SATISFIED" if passed else "PROPERTY_MISSING"

        elif op == "requires":
            status, reason_code = eval_requires(facts, target_v)

        elif op == "excludes":
            status, reason_code = eval_excludes(facts, None, str(target_v) if target_v else None)

        else:
            status = CheckStatus.UNKNOWN
            reason_code = f"UNSUPPORTED_OPERATOR: {op}"

        return CheckResult(
            requirement_id=req_id,
            status=status,
            evidence_ids=ev_ids,
            reason_code=reason_code,
            decision_origin=DecisionOrigin.rule,
            model_probabilities=None,
        )

    def evaluate_product(
        self,
        product_id: str,
        facts: Any,
        requirements: Optional[Any] = None,
    ) -> ProductEvaluationResult:
        """Evaluate a product against requirements and single-scope deterministic rules (§8, §5.5).

        Parameters
        ----------
        product_id : str
            Candidate product identifier.
        facts : Any
            Product facts (list of Fact/dict or dict of properties).
        requirements : Optional[Any]
            List of Requirement models/dicts or Rule models/dicts.

        Returns
        -------
        ProductEvaluationResult
            Evaluated checks and aggregated TechnicalVerdict.
        """
        fact_index = FactIndex(product_id, facts)
        checks: list[CheckResult] = []
        missing_evidence: list[str] = []
        evaluated_rule_ids: list[str] = []

        req_models: list[Requirement] = []

        # 1. Process explicit requirements if supplied
        if requirements:
            for item in requirements:
                if isinstance(item, Requirement):
                    req_models.append(item)
                    prop_name = getattr(item, "property", None) or getattr(item, "requirement_id", None)
                    check = RuleCheck(
                        operator=item.operator,
                        property=prop_name,
                        target=item.target,
                        unit=item.unit,
                        requirement_id=item.requirement_id,
                        hard=item.hard,
                    )
                    c_res = self._evaluate_single_check(
                        check=check,
                        facts=fact_index,
                        rule_missing_policy=MissingPolicy.UNKNOWN,
                        rule_evidence_ids=[],
                        rule_hard=item.hard,
                    )
                    checks.append(c_res)
                    if c_res.status == CheckStatus.UNKNOWN:
                        missing_evidence.append(item.requirement_id)

                elif isinstance(item, dict) and "operator" in item and "requirement_id" in item:
                    # Dict representing a Requirement
                    req_obj = Requirement.model_validate(item)
                    req_models.append(req_obj)
                    prop_name = item.get("property") or item.get("requirement_id")
                    check = RuleCheck(
                        operator=req_obj.operator,
                        property=prop_name,
                        target=req_obj.target,
                        unit=req_obj.unit,
                        requirement_id=req_obj.requirement_id,
                        hard=req_obj.hard,
                    )
                    c_res = self._evaluate_single_check(
                        check=check,
                        facts=fact_index,
                        rule_missing_policy=MissingPolicy.UNKNOWN,
                        rule_evidence_ids=[],
                        rule_hard=req_obj.hard,
                    )
                    checks.append(c_res)
                    if c_res.status == CheckStatus.UNKNOWN:
                        missing_evidence.append(req_obj.requirement_id)

                elif isinstance(item, (Rule, dict)):
                    rule_obj = item if isinstance(item, Rule) else Rule.model_validate(item)
                    if rule_obj.scope == RuleScope.single:
                        if not self.allow_pending and rule_obj.review_status == ReviewStatus.pending:
                            continue
                        if self._rule_applies(rule_obj, fact_index):
                            evaluated_rule_ids.append(rule_obj.rule_id)
                            for chk in rule_obj.checks:
                                c_res = self._evaluate_single_check(
                                    check=chk,
                                    facts=fact_index,
                                    rule_missing_policy=rule_obj.missing_policy,
                                    rule_evidence_ids=rule_obj.evidence_ids,
                                    rule_hard=rule_obj.hard and chk.hard,
                                )
                                checks.append(c_res)
                                if c_res.status == CheckStatus.UNKNOWN:
                                    missing_evidence.append(c_res.requirement_id)

        # 2. Evaluate loaded single-scope rules
        for rule in self.rules:
            if rule.scope != RuleScope.single:
                continue
            if not self.allow_pending and rule.review_status == ReviewStatus.pending:
                continue
            if rule.rule_id in evaluated_rule_ids:
                continue

            if self._rule_applies(rule, fact_index):
                evaluated_rule_ids.append(rule.rule_id)
                for chk in rule.checks:
                    c_res = self._evaluate_single_check(
                        check=chk,
                        facts=fact_index,
                        rule_missing_policy=rule.missing_policy,
                        rule_evidence_ids=rule.evidence_ids,
                        rule_hard=rule.hard and chk.hard,
                    )
                    checks.append(c_res)
                    if c_res.status == CheckStatus.UNKNOWN:
                        missing_evidence.append(c_res.requirement_id)

        # 3. Aggregate verdict (§5.5)
        verdict = aggregate_verdict(checks, req_models if req_models else None)

        return ProductEvaluationResult(
            product_id=product_id,
            verdict=verdict,
            checks=checks,
            missing_evidence=list(set(missing_evidence)),
            evaluated_rule_ids=evaluated_rule_ids,
        )

    def evaluate_pair(
        self,
        product_a: str,
        facts_a: Any,
        product_b: str,
        facts_b: Any,
        rules: Optional[Sequence[Union[Rule, dict[str, Any]]]] = None,
    ) -> PairEvaluationResult:
        """Evaluate a pair of products against pair compatibility rules (§8.1, §5.5).

        Parameters
        ----------
        product_a : str
            Identifier for component A.
        facts_a : Any
            Facts for component A.
        product_b : str
            Identifier for component B.
        facts_b : Any
            Facts for component B.
        rules : Optional[Sequence[Union[Rule, dict[str, Any]]]]
            Pair rules to evaluate. If omitted, uses loaded pair-scope rules.

        Returns
        -------
        PairEvaluationResult
            Evaluated checks and aggregated TechnicalVerdict.
        """
        fa_index = FactIndex(product_a, facts_a)
        fb_index = FactIndex(product_b, facts_b)

        target_rules: list[Rule] = []
        if rules is not None:
            for r in rules:
                target_rules.append(r if isinstance(r, Rule) else Rule.model_validate(r))
        else:
            target_rules = [r for r in self.rules if r.scope == RuleScope.pair]

        checks: list[CheckResult] = []
        missing_evidence: list[str] = []
        evaluated_rule_ids: list[str] = []

        for rule in target_rules:
            if not self.allow_pending and rule.review_status == ReviewStatus.pending:
                continue

            if not self._rule_applies(rule, fa_index, fb_index):
                continue

            evaluated_rule_ids.append(rule.rule_id)

            for chk in rule.checks:
                req_id = chk.requirement_id or f"{rule.rule_id}-{chk.operator}"
                op = (chk.operator or "").lower()
                missing_pol = chk.missing_policy or rule.missing_policy or MissingPolicy.UNKNOWN
                ev_ids = list(set(rule.evidence_ids + chk.evidence_ids))

                if op == "role_pair_allowed":
                    st, r_code, role_ev = eval_role_pair_allowed(fa_index, fb_index, chk)
                    checks.append(
                        CheckResult(
                            requirement_id=req_id,
                            status=st,
                            evidence_ids=list(set(ev_ids + role_ev)),
                            reason_code=r_code,
                            decision_origin=DecisionOrigin.rule,
                            model_probabilities=None,
                        )
                    )
                    if st == CheckStatus.UNKNOWN:
                        missing_evidence.append(req_id)

                elif op == "excludes":
                    st, r_code = eval_excludes(fa_index, fb_index)
                    checks.append(
                        CheckResult(
                            requirement_id=req_id,
                            status=st,
                            evidence_ids=ev_ids,
                            reason_code=r_code,
                            decision_origin=DecisionOrigin.rule,
                            model_probabilities=None,
                        )
                    )

                elif op == "requires":
                    # Check if A requires B or B requires A
                    st_a, rc_a = eval_requires(fa_index, chk.target or chk.accessory_id, product_b, fb_index)
                    st_b, rc_b = eval_requires(fb_index, chk.target or chk.accessory_id, product_a, fa_index)
                    # If either requires the other and succeeds, pass; if failure, record fail
                    if st_a == CheckStatus.FAIL or st_b == CheckStatus.FAIL:
                        st = CheckStatus.FAIL
                        r_code = rc_a if st_a == CheckStatus.FAIL else rc_b
                    elif st_a == CheckStatus.PASS or st_b == CheckStatus.PASS:
                        st = CheckStatus.PASS
                        r_code = "REQUIRED_COMPONENT_SATISFIED"
                    else:
                        st = CheckStatus.UNKNOWN
                        r_code = "UNKNOWN_REQUIREMENT"
                    checks.append(
                        CheckResult(
                            requirement_id=req_id,
                            status=st,
                            evidence_ids=ev_ids,
                            reason_code=r_code,
                            decision_origin=DecisionOrigin.rule,
                            model_probabilities=None,
                        )
                    )
                    if st == CheckStatus.UNKNOWN:
                        missing_evidence.append(req_id)

                elif op in ("range_contains", "range_overlap", "eq", "in", "lte", "gte"):
                    # Pair fact comparison between A and B
                    left_prop = chk.left_fact or chk.property
                    right_prop = chk.right_fact or chk.property

                    entry_a = fa_index.get_primary_entry(left_prop) if left_prop else None
                    entry_b = fb_index.get_primary_entry(right_prop) if right_prop else None

                    if entry_a:
                        ev_ids.extend(entry_a.evidence_ids)
                    if entry_b:
                        ev_ids.extend(entry_b.evidence_ids)

                    val_a = entry_a.value if entry_a else None
                    unit_a = entry_a.unit if entry_a else None

                    val_b = entry_b.value if entry_b else chk.target or chk.value
                    unit_b = entry_b.unit if entry_b else chk.unit

                    # Check for missing fact (§5.5)
                    if entry_a is None or (val_b is None and entry_b is None):
                        status_map = {
                            MissingPolicy.UNKNOWN: CheckStatus.UNKNOWN,
                            MissingPolicy.FAIL: CheckStatus.FAIL,
                            MissingPolicy.PASS: CheckStatus.PASS,
                        }
                        res_st = status_map.get(missing_pol, CheckStatus.UNKNOWN)
                        checks.append(
                            CheckResult(
                                requirement_id=req_id,
                                status=res_st,
                                evidence_ids=list(set(ev_ids)),
                                reason_code="MISSING_CRITICAL_PROPERTY",
                                decision_origin=DecisionOrigin.rule,
                                model_probabilities=None,
                            )
                        )
                        if res_st == CheckStatus.UNKNOWN:
                            missing_evidence.append(req_id)
                        continue

                    # Evaluate comparison
                    status = CheckStatus.UNKNOWN
                    reason_code = "EVALUATED"

                    if op == "range_contains":
                        passed = eval_range_contains(val_a, val_b, unit_a, unit_b, chk.tolerance)
                        status = CheckStatus.PASS if passed else CheckStatus.FAIL
                        reason_code = "RANGE_CONTAINED" if passed else "RANGE_DOES_NOT_CONTAIN"

                    elif op == "range_overlap":
                        passed = eval_range_overlap(val_a, val_b, unit_a, unit_b, chk.tolerance)
                        status = CheckStatus.PASS if passed else CheckStatus.FAIL
                        reason_code = "RANGES_OVERLAP" if passed else "RANGES_DO_NOT_OVERLAP"

                    elif op == "eq":
                        passed = eval_eq(val_a, val_b, unit_a, unit_b, chk.tolerance)
                        status = CheckStatus.PASS if passed else CheckStatus.FAIL
                        reason_code = "EQUALITY_MATCH" if passed else "EQUALITY_MISMATCH"

                    elif op == "in":
                        passed = eval_in(val_a, val_b)
                        status = CheckStatus.PASS if passed else CheckStatus.FAIL
                        reason_code = "IN_COLLECTION" if passed else "NOT_IN_COLLECTION"

                    elif op == "lte":
                        passed = eval_lte(val_a, val_b, unit_a, unit_b, chk.tolerance)
                        status = CheckStatus.PASS if passed else CheckStatus.FAIL
                        reason_code = "LTE_SATISFIED" if passed else "LTE_VIOLATED"

                    elif op == "gte":
                        passed = eval_gte(val_a, val_b, unit_a, unit_b, chk.tolerance)
                        status = CheckStatus.PASS if passed else CheckStatus.FAIL
                        reason_code = "GTE_SATISFIED" if passed else "GTE_VIOLATED"

                    checks.append(
                        CheckResult(
                            requirement_id=req_id,
                            status=status,
                            evidence_ids=list(set(ev_ids)),
                            reason_code=reason_code,
                            decision_origin=DecisionOrigin.rule,
                            model_probabilities=None,
                        )
                    )
                    if status == CheckStatus.UNKNOWN:
                        missing_evidence.append(req_id)

        verdict = aggregate_verdict(checks)

        return PairEvaluationResult(
            product_a=product_a,
            product_b=product_b,
            verdict=verdict,
            checks=checks,
            missing_evidence=list(set(missing_evidence)),
            evaluated_rule_ids=evaluated_rule_ids,
        )


__all__ = [
    # Enums
    "RuleScope",
    "MissingPolicy",
    "ReviewStatus",
    # DSL Models
    "RuleCondition",
    "RuleCheck",
    "Rule",
    "ProductEvaluationResult",
    "PairEvaluationResult",
    # Range & Units
    "RangeInterval",
    "normalize_unit",
    "parse_numeric_or_range",
    "normalize_property_key",
    # Facts
    "FactEntry",
    "FactIndex",
    # Engine
    "RulesEngine",
    # Operators
    "eval_eq",
    "eval_in",
    "eval_range_contains",
    "eval_range_overlap",
    "eval_lte",
    "eval_gte",
    "eval_requires",
    "eval_excludes",
    "eval_role_pair_allowed",
    # Safety Invariant Functions
    "can_override_check",
    "combine_verdicts_with_model",
]
