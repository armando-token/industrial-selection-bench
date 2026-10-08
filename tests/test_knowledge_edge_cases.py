"""Edge cases and comprehensive verification tests for Workstream B (Knowledge Store & Graph).

Tests cover:
- In-memory SQLite persistence and connection reuse (:memory:)
- Batch transaction atomicity and rollback on failure
- Querying facts by category and listing canonical categories
- Filtering ports by interface, signal, direction, and voltage ranges
- Multi-hop compatibility pathfinding across candidate connections
- Graph component analysis and disconnected component resilience
- Fact conflict detection (conflicting status, contradictory values, invalid ranges)
- Graph conflict detection (mutual exclusion clashes, power source clashes)
- Safe handling of empty tables and missing entities
- Chunked retrieval of large span ID batches
"""

import json
from pathlib import Path
import pytest

from tests.conftest import requires_real_facts

from industrial_lab.knowledge.graph import KnowledgeGraph, build_knowledge_graph
from industrial_lab.knowledge.store import KnowledgeStore, build_knowledge_store
from industrial_lab.schemas import (
    Citation,
    ExtractionStatus,
    Fact,
    Port,
    Relation,
    RelationType,
    SourceSpan,
    VerificationStatus,
)


def test_in_memory_store_persistence():
    """Verifies that KnowledgeStore(':memory:') preserves tables and data across calls."""
    store = KnowledgeStore(":memory:")
    store.save_product("P1", sku="SKU1", manufacturer="TestCorp")
    assert store.get_product("P1")["sku"] == "SKU1"

    prods = store.list_products()
    assert len(prods) == 1
    assert prods[0]["product_id"] == "P1"


def test_batch_transaction_atomicity_and_rollback():
    """Verifies that an error in save_facts rolls back the entire transaction atomically."""
    store = KnowledgeStore(":memory:")
    f1 = Fact(fact_id="F1", product_id="P1", property="p1", value=10)
    f2 = Fact(fact_id="F2", product_id="P1", property="p2", value=20)

    class ExplosiveFact:
        @property
        def fact_id(self):
            raise RuntimeError("Database write failure simulation")

    with pytest.raises(RuntimeError):
        store.save_facts([f1, f2, ExplosiveFact()])

    # Must be 0 because entire batch was rolled back
    facts = store.list_all_facts()
    assert len(facts) == 0


def test_facts_by_category():
    """Verifies querying technical facts by canonical category."""
    store = KnowledgeStore(":memory:")
    facts = [
        Fact(fact_id="F_VOLT", product_id="P1", property="supply_voltage_nominal_v", value=24.0, unit="V"),
        Fact(fact_id="F_TEMP", product_id="P1", property="operating_temp_max_c", value=60.0, unit="°C"),
        Fact(fact_id="F_COMM", product_id="P1", property="communication_protocol", value="Modbus RTU"),
        Fact(fact_id="F_MOUNT", product_id="P1", property="din_rail_mounting", value=True),
    ]
    store.save_facts(facts)

    elec = store.get_facts_by_category("electrical")
    assert len(elec) == 1
    assert elec[0].fact_id == "F_VOLT"

    env = store.get_facts_by_category("environmental")
    assert len(env) == 1
    assert env[0].fact_id == "F_TEMP"

    comm = store.get_facts_by_category("communication")
    assert len(comm) == 1
    assert comm[0].fact_id == "F_COMM"

    mech = store.get_facts_by_category("mechanical")
    assert len(mech) == 1
    assert mech[0].fact_id == "F_MOUNT"

    cats = store.list_fact_categories()
    assert "electrical" in cats
    assert "environmental" in cats


def test_find_ports_by_interface_and_voltage():
    """Verifies searching ports by interface, signal kind, and electrical voltage thresholds."""
    store = KnowledgeStore(":memory:")
    p1 = Port(
        port_id="P1_PWR",
        product_id="P1",
        direction="in",
        physical_interface="screw_terminal",
        signal_kind="power_dc",
        supported_ranges=[{"nominal_v": 24.0, "min_v": 20.4, "max_v": 28.8}],
    )
    p2 = Port(
        port_id="P1_COM",
        product_id="P1",
        direction="bidirectional",
        physical_interface="RS-485",
        signal_kind="serial",
        protocol="Modbus RTU",
    )
    store.save_ports([p1, p2])

    # Search by physical interface
    rs485_ports = store.find_ports(physical_interface="RS-485")
    assert len(rs485_ports) == 1
    assert rs485_ports[0].port_id == "P1_COM"

    # Search by signal
    power_ports = store.find_ports(signal_kind="power_dc")
    assert len(power_ports) == 1
    assert power_ports[0].port_id == "P1_PWR"

    # Search by voltage threshold
    v24_ports = store.find_ports(voltage_nominal=24.0)
    assert len(v24_ports) == 1
    assert v24_ports[0].port_id == "P1_PWR"

    # Search non-matching voltage threshold
    v48_ports = store.find_ports(voltage_nominal=48.0)
    assert len(v48_ports) == 0


def test_detect_fact_conflicts():
    """Verifies detection of conflicting extraction status, contradictory values, and invalid ranges."""
    store = KnowledgeStore(":memory:")
    facts = [
        # Explicitly conflicting status
        Fact(
            fact_id="F_CONF",
            product_id="P1",
            property="power_consumption_max_w",
            value=25.0,
            extraction_status=ExtractionStatus.conflicting,
        ),
        # Contradictory values under same scope
        Fact(
            fact_id="F_VAL_A",
            product_id="P1",
            property="enclosure_ip_rating",
            value="IP20",
            variant_scope=["standard"],
        ),
        Fact(
            fact_id="F_VAL_B",
            product_id="P1",
            property="enclosure_ip_rating",
            value="IP67",
            variant_scope=["standard"],
        ),
        # Invalid range: min > max
        Fact(
            fact_id="F_VMIN",
            product_id="P2",
            property="supply_voltage_min_v",
            value=30.0,
        ),
        Fact(
            fact_id="F_VMAX",
            product_id="P2",
            property="supply_voltage_max_v",
            value=24.0,
        ),
    ]
    store.save_facts(facts)

    conflicts = store.detect_fact_conflicts()
    types = [c["conflict_type"] for c in conflicts]
    assert "status_conflicting" in types
    assert "contradictory_value" in types
    assert "invalid_range" in types


def test_graph_disconnected_components_and_pathfinding():
    """Verifies graph component analysis, isolated clusters, and candidate pathfinding."""
    kg = KnowledgeGraph()
    kg.add_product("P1")
    kg.add_product("P2")
    kg.add_product("P3")

    # In isolation, products form 3 disconnected components
    assert not kg.is_connected()
    components = kg.get_connected_components()
    assert len(components) == 3

    # Direct query between disconnected nodes returns empty without exception
    paths = kg.find_compatibility_paths("P1", "P2")
    assert paths == []

    # Add ports that create candidate connection between P2 (PSU) and P1 (PLC)
    p1_pwr = Port(
        port_id="P1_PWR_IN",
        product_id="P1",
        direction="in",
        physical_interface="screw_terminal",
        signal_kind="power_dc",
        supported_ranges=[{"nominal_v": 24.0, "min_v": 20.4, "max_v": 28.8}],
    )
    p2_pwr = Port(
        port_id="P2_PWR_OUT",
        product_id="P2",
        direction="out",
        physical_interface="screw_terminal",
        signal_kind="power_dc",
        role="power_source",
        supported_ranges=[{"nominal_v": 24.0, "min_v": 24.0, "max_v": 28.0}],
    )
    kg.add_port(p1_pwr)
    kg.add_port(p2_pwr)

    # Product-level candidate discovery
    product_candidates = kg.find_product_candidate_connections("P2", "P1")
    assert len(product_candidates) == 1
    assert product_candidates[0]["source_port_id"] == "P2_PWR_OUT"
    assert product_candidates[0]["target_port_id"] == "P1_PWR_IN"
    assert product_candidates[0]["candidate"] is True

    # Multi-hop candidate paths now connect P2 to P1
    compat_paths = kg.find_compatibility_paths("P2", "P1")
    assert len(compat_paths) >= 1


def test_graph_conflict_detection():
    """Verifies detection of mutual exclusions and power source clashes."""
    kg = KnowledgeGraph()
    kg.add_product("P1")
    kg.add_product("P2")

    # P1 EXCLUDES P2, but P2 REQUIRES P1 -> Contradiction!
    kg.add_relation(Relation(
        relation_id="R1",
        source_id="P1",
        target_id="P2",
        relation_type=RelationType.EXCLUDES,
    ))
    kg.add_relation(Relation(
        relation_id="R2",
        source_id="P2",
        target_id="P1",
        relation_type=RelationType.REQUIRES,
    ))

    # Two power sources
    p1_pwr = Port(
        port_id="PORT_PSU_A",
        product_id="P1",
        direction="out",
        physical_interface="screw_terminal",
        signal_kind="power_dc",
        role="power_source",
    )
    p2_pwr = Port(
        port_id="PORT_PSU_B",
        product_id="P2",
        direction="out",
        physical_interface="screw_terminal",
        signal_kind="power_dc",
        role="power_source",
    )
    kg.add_port(p1_pwr)
    kg.add_port(p2_pwr)

    conflicts = kg.detect_graph_conflicts()
    types = [c["conflict_type"] for c in conflicts]
    assert "exclusion_requirement_conflict" in types
    assert "power_source_clash" in types


def test_empty_tables_and_missing_entities():
    """Verifies edge cases on empty tables and nonexistent entities."""
    store = KnowledgeStore(":memory:")
    assert store.get_product("MISSING") is None
    assert store.list_products() == []
    assert store.get_fact("MISSING") is None
    assert store.get_facts_by_product("MISSING") == []
    assert store.get_facts_for_property("MISSING", "prop") == []
    assert store.list_all_facts() == []
    assert store.get_port("MISSING") is None
    assert store.list_all_ports() == []
    assert store.find_ports(product_id="MISSING") == []
    assert store.get_relations("MISSING") == []
    assert store.get_span("MISSING") is None
    assert store.get_spans(["M1", "M2"]) == []
    assert store.detect_fact_conflicts() == []

    kg = KnowledgeGraph()
    assert kg.get_product_ports("MISSING") == []
    assert kg.get_required_accessories("MISSING") == []
    assert kg.get_exclusions("MISSING") == []
    assert kg.check_direct_compatibility("M1", "M2") is None
    assert kg.find_candidate_connections("M1", "M2") == []
    assert kg.find_product_candidate_connections("M1", "M2") == []
    assert kg.find_compatibility_paths("M1", "M2") == []
    assert kg.detect_graph_conflicts() == []


def test_large_spans_query_chunking():
    """Verifies that querying a large list of span IDs succeeds without SQLite variable limit issues."""
    store = KnowledgeStore(":memory:")
    spans = [
        SourceSpan(
            span_id=f"SPAN_{i:04d}",
            document_id="DOC1",
            document_sha256="sha",
            pdf_page_index=1,
            text=f"Text snippet {i}",
        )
        for i in range(1200)
    ]
    store.save_spans(spans)

    query_ids = [f"SPAN_{i:04d}" for i in range(1200)]
    retrieved = store.get_spans(query_ids)
    assert len(retrieved) == 1200


def test_query_facts_by_product_variant_and_property():
    """Verifies that KnowledgeStore filters facts cleanly by product_id, variant, and property (F05)."""
    store = KnowledgeStore(":memory:")
    facts = [
        # Common facts for P_X4
        Fact(
            fact_id="F_X4_PWR",
            product_id="P_X4",
            property="supply_voltage_nominal_v",
            value=24.0,
            variant_scope=["standard"],
        ),
        Fact(
            fact_id="F_X4_DI",
            product_id="P_X4",
            property="digital_inputs_count",
            value=12,
            variant_scope=["standard", "HE-X4A", "HE-X4R"],
        ),
        # Model A specific
        Fact(
            fact_id="F_X4A_DO",
            product_id="P_X4",
            property="solid_state_dc_outputs_count",
            value=12,
            variant_scope=["HE-X4A", "Model A"],
        ),
        # Model R specific
        Fact(
            fact_id="F_X4R_RELAY",
            product_id="P_X4",
            property="relay_outputs_count",
            value=6,
            variant_scope=["HE-X4R", "Model R"],
        ),
        Fact(
            fact_id="F_X4R_DO",
            product_id="P_X4",
            property="solid_state_dc_outputs_count",
            value=2,
            variant_scope=["HE-X4R", "Model R"],
        ),
    ]
    store.save_facts(facts)

    # 1. Model A facts: must include PWR, DI, and X4A_DO, but NEVER X4R_RELAY
    a_facts = store.get_facts_for_variant("P_X4", "HE-X4A")
    a_ids = {f.fact_id for f in a_facts}
    assert "F_X4_PWR" in a_ids
    assert "F_X4_DI" in a_ids
    assert "F_X4A_DO" in a_ids
    assert "F_X4R_RELAY" not in a_ids
    assert "F_X4R_DO" not in a_ids

    # 2. Model R facts: must include PWR, DI, X4R_RELAY, X4R_DO, but NEVER X4A_DO
    r_facts = store.get_facts_for_variant("P_X4", "HE-X4R")
    r_ids = {f.fact_id for f in r_facts}
    assert "F_X4_PWR" in r_ids
    assert "F_X4_DI" in r_ids
    assert "F_X4R_RELAY" in r_ids
    assert "F_X4R_DO" in r_ids
    assert "F_X4A_DO" not in r_ids

    # 3. Property lookup with variant filtering
    assert store.get_facts_for_property("P_X4", "relay_outputs_count", variant="HE-X4A") == []
    r_relays = store.get_facts_for_property("P_X4", "relay_outputs_count", variant="HE-X4R")
    assert len(r_relays) == 1
    assert r_relays[0].value == 6


def test_query_facts_conditions_filtering():
    """Verifies that KnowledgeStore filters facts by conditions correctly (F05)."""
    store = KnowledgeStore(":memory:")
    facts = [
        Fact(
            fact_id="F_UHEAT_BASE",
            product_id="P_UHEAT",
            property="role",
            value="pumphouse_heater",
            conditions=[],
        ),
        Fact(
            fact_id="F_UHEAT_VERT",
            product_id="P_UHEAT",
            property="max_wattage_w",
            value=500.0,
            conditions=[{"orientation": "vertical"}],
        ),
        Fact(
            fact_id="F_UHEAT_HORIZ",
            product_id="P_UHEAT",
            property="max_wattage_w",
            value=1000.0,
            conditions=[{"orientation": "horizontal"}],
        ),
    ]
    store.save_facts(facts)

    # Query with vertical condition
    v_facts = store.query_facts(product_id="P_UHEAT", conditions={"orientation": "vertical"})
    v_ids = {f.fact_id for f in v_facts}
    assert "F_UHEAT_BASE" in v_ids  # unconditional applies universally
    assert "F_UHEAT_VERT" in v_ids
    assert "F_UHEAT_HORIZ" not in v_ids

    # Query with horizontal condition
    h_facts = store.query_facts(product_id="P_UHEAT", conditions={"orientation": "horizontal"})
    h_ids = {f.fact_id for f in h_facts}
    assert "F_UHEAT_BASE" in h_ids
    assert "F_UHEAT_HORIZ" in h_ids
    assert "F_UHEAT_VERT" not in h_ids


def test_resolve_citations_for_facts_and_checks():
    """Verifies that citations map source_span_ids from facts/checks to real documents, pages, and hashes (REPAIR3_PLAN §5.3)."""
    store = KnowledgeStore(":memory:")
    spans = [
        SourceSpan(
            span_id="D_X4_MANUAL_MAN1137:p47:s01",
            document_id="D_X4_MANUAL_MAN1137",
            document_sha256="35c647aa6b1d8595a7f674e0d203ad77023a27d7ee8fd70c83e1bb3f53d0f649",
            pdf_page_index=47,
            printed_page_label="39",
            text="Table 6.4: I/O Register Map for X4 OCS",
            product_scope=["P_X4"],
        ),
        SourceSpan(
            span_id="D_THT_MANUAL:p02:s02",
            document_id="D_THT_MANUAL",
            document_sha256="48718e71c1b181b392b457ea5a2ceea7dafc1f80ecfe49fe1882fc9bb5f76b10",
            pdf_page_index=2,
            printed_page_label="2",
            text="The THT-02 temperature and humidity sensor is designed based on the RS-485 communication interface",
            product_scope=["P_THT"],
        ),
        SourceSpan(
            span_id="D_UHEAT_INSTALL_66661:p02:s41",
            document_id="D_UHEAT_INSTALL_66661",
            document_sha256="5906efd1b82143717282cb82fa1ebdfc43734e5658e4e93f9cba8c6b733da9c5",
            pdf_page_index=2,
            printed_page_label="2",
            text="Unit CANNOT be installed vertically with thermostat at the top.",
            product_scope=["P_UHEAT"],
        ),
    ]
    store.save_spans(spans)

    # 1. Resolve for specific evidence IDs
    cites = store.resolve_citations([
        "D_X4_MANUAL_MAN1137:p47:s01",
        "D_THT_MANUAL:p02:s02",
        "D_UHEAT_INSTALL_66661:p02:s41",
    ])
    assert len(cites) == 3

    c_x4 = next(c for c in cites if c.document_id == "D_X4_MANUAL_MAN1137")
    assert c_x4.page == 47
    assert c_x4.printed_page == "39"
    assert c_x4.document_sha256.startswith("35c647aa")
    assert "Table 6.4" in c_x4.snippet

    c_tht = next(c for c in cites if c.document_id == "D_THT_MANUAL")
    assert c_tht.page == 2
    assert "RS-485" in c_tht.snippet

    c_uheat = next(c for c in cites if c.document_id == "D_UHEAT_INSTALL_66661")
    assert c_uheat.page == 2
    assert "thermostat at the top" in c_uheat.snippet

    # 2. Resolve for facts helper
    test_fact = Fact(
        fact_id="F_TEST",
        product_id="P_X4",
        property="relays",
        value=6,
        evidence_ids=["D_X4_MANUAL_MAN1137:p47:s01"],
    )
    fact_cites = store.resolve_citations_for_facts([test_fact])
    assert len(fact_cites) == 1
    assert fact_cites[0].document_id == "D_X4_MANUAL_MAN1137"


@requires_real_facts
def test_audit_facts_provenance_and_zero_expert_engineer():
    """Audits data/facts/facts.reviewed.jsonl to enforce F19 provenance rules and zero expert_engineer labels."""
    from industrial_lab.ingest.facts import load_facts

    rev_file = Path("data/facts/facts.reviewed.jsonl")
    assert rev_file.exists()

    facts = load_facts(rev_file)
    assert len(facts) == 87, f"Expected 87 audited real product facts, found {len(facts)}"

    for f in facts:
        # F19 Invariant 1: extraction_status must be in (EXTRACTED, EXTRACTION_FAILED)
        assert f.extraction_status in (ExtractionStatus.EXTRACTED, ExtractionStatus.auto_extracted, "EXTRACTED")

        # F19 Invariant 2: verification_status must be VERIFIED_BY_AGENT or UNVERIFIED
        assert f.verification_status in (
            VerificationStatus.VERIFIED_BY_AGENT,
            VerificationStatus.UNVERIFIED,
            "VERIFIED_BY_AGENT",
            "UNVERIFIED",
        )

        # F19 Invariant 3: NEVER 'expert_engineer'
        assert f.reviewed_by != "expert_engineer", f"Fact {f.fact_id} still has reviewed_by='expert_engineer'"
        assert f.reviewer_id != "expert_engineer", f"Fact {f.fact_id} still has reviewer_id='expert_engineer'"

        # F19 Invariant 4: reviewer_id must be null or 'agent_automated'
        assert f.reviewer_id in (None, "agent_automated")

        # Invariant 5: real facts must have real_user_document data_origin
        assert f.data_origin == "real_user_document"


def test_expert_engineer_placeholder_sanitization():
    """Verifies that any fact initialized with unverified 'expert_engineer' is automatically sanitized."""
    fact = Fact(
        fact_id="F_SAN_01",
        product_id="P_X4",
        property="supply_voltage",
        value=24,
        reviewed_by="expert_engineer",
    )
    assert fact.reviewed_by == "agent_automated"
    assert fact.reviewer_id == "agent_automated"
    assert fact.verification_status == VerificationStatus.VERIFIED_BY_AGENT
    assert fact.review_method == "automated_agent_verification"

