"""Unit tests for fact loading/saving and review export/import edge cases.

Adheres strictly to MEGAPLAN.md §5.2, §7.3, §22:
- Tests load_facts with missing file, empty file, corrupted lines.
- Tests save_facts with missing directories and empty lists.
- Tests export_review with missing input file and empty input file.
- Tests import_review with rejected facts, ensuring rejected facts are excluded.
- Tests import_review with approved and modified facts.
- Tests import_review with invalid decisions, missing required fields, and files with no approved facts.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from industrial_lab.schemas import ExtractionStatus, Fact
from industrial_lab.ingest.facts import load_facts, save_facts
from industrial_lab.ingest.review import export_review, import_review


# ==============================================================================
# 1. Facts loader & saver edge cases
# ==============================================================================

def test_load_facts_missing_file(tmp_path: Path) -> None:
    """load_facts should return empty list when file or directory does not exist."""
    missing_file = tmp_path / "does_not_exist.jsonl"
    assert load_facts(missing_file) == []

    missing_dir_file = tmp_path / "nonexistent_dir" / "facts.jsonl"
    assert load_facts(missing_dir_file) == []

    assert load_facts("") == []
    assert load_facts(None) == []


def test_load_facts_empty_file(tmp_path: Path) -> None:
    """load_facts should return empty list for 0-byte or whitespace-only files."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.touch()
    assert load_facts(empty_file) == []

    whitespace_file = tmp_path / "whitespace.jsonl"
    whitespace_file.write_text("   \n\n  \t\n  \n", encoding="utf-8")
    assert load_facts(whitespace_file) == []


def test_load_facts_corrupted_lines(tmp_path: Path) -> None:
    """load_facts should skip corrupted or invalid lines and parse valid facts."""
    corrupted_file = tmp_path / "corrupted.jsonl"

    valid_fact_1 = Fact(
        fact_id="F_VALID_001",
        product_id="P1",
        property="supply_voltage_nominal_v",
        value=24.0,
        unit="V",
        extraction_status=ExtractionStatus.auto_extracted,
    )
    valid_fact_2 = Fact(
        fact_id="F_VALID_002",
        product_id="P2",
        property="rated_output_current_a",
        value=5.0,
        unit="A",
        extraction_status=ExtractionStatus.auto_extracted,
    )

    lines = [
        valid_fact_1.to_json(),
        "THIS_IS_NOT_JSON",
        '{"fact_id": "F_BROKEN", malformed_json',
        '{"foo": "bar"}',  # Valid JSON but missing required Fact fields
        "",  # Empty line
        '{"fact_id": 12345, "product_id": []}',  # Wrong types
        valid_fact_2.to_json(),
    ]
    corrupted_file.write_text("\n".join(lines), encoding="utf-8")

    loaded = load_facts(corrupted_file)
    assert len(loaded) == 2
    assert loaded[0].fact_id == "F_VALID_001"
    assert loaded[0].property == "supply_voltage_nominal_v"
    assert loaded[0].value == 24.0
    assert loaded[1].fact_id == "F_VALID_002"
    assert loaded[1].property == "rated_output_current_a"
    assert loaded[1].value == 5.0


def test_save_facts_missing_directory_and_empty(tmp_path: Path) -> None:
    """save_facts should create missing parent directories and handle empty lists cleanly."""
    nested_target = tmp_path / "deep" / "nested" / "dir" / "saved_facts.jsonl"

    facts = [
        Fact(
            fact_id="F_TEST_001",
            product_id="P1",
            property="supply_voltage_nominal_v",
            value=24.0,
            unit="V",
        )
    ]

    # Save creates missing parent directories
    save_facts(facts, nested_target)
    assert nested_target.is_file()

    reloaded = load_facts(nested_target)
    assert len(reloaded) == 1
    assert reloaded[0].fact_id == "F_TEST_001"

    # Save empty list
    empty_target = tmp_path / "empty_dir" / "empty_facts.jsonl"
    save_facts([], empty_target)
    assert empty_target.is_file()
    assert load_facts(empty_target) == []

    # Save None
    none_target = tmp_path / "empty_dir" / "none_facts.jsonl"
    save_facts(None, none_target)
    assert none_target.is_file()
    assert load_facts(none_target) == []


# ==============================================================================
# 2. Review Export Edge Cases
# ==============================================================================

def test_export_review_missing_file(tmp_path: Path) -> None:
    """export_review should raise FileNotFoundError when auto facts file is missing."""
    missing_src = tmp_path / "missing_auto_facts.jsonl"
    output_dest = tmp_path / "pending.json"

    with pytest.raises(FileNotFoundError, match="Auto facts file not found"):
        export_review(auto_facts_file=missing_src, output_review_file=output_dest)


def test_export_review_empty_file(tmp_path: Path) -> None:
    """export_review should raise ValueError when auto facts file is empty or has no valid facts."""
    empty_src = tmp_path / "empty_auto_facts.jsonl"
    empty_src.touch()
    output_dest = tmp_path / "pending.json"

    with pytest.raises(ValueError, match="Auto facts file is empty"):
        export_review(auto_facts_file=empty_src, output_review_file=output_dest)

    corrupted_src = tmp_path / "corrupted_auto_facts.jsonl"
    corrupted_src.write_text("NOT_A_FACT\nANOTHER_BAD_LINE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Auto facts file contains no valid facts"):
        export_review(auto_facts_file=corrupted_src, output_review_file=output_dest)


# ==============================================================================
# 3. Review Import Edge Cases
# ==============================================================================

def test_import_review_rejected_facts(tmp_path: Path) -> None:
    """import_review must strictly exclude rejected facts."""
    review_file = tmp_path / "review_package.json"
    output_file = tmp_path / "facts.reviewed.jsonl"

    package = {
        "facts": [
            {
                "fact_id": "F_APPROVED_1",
                "product_id": "P1",
                "property": "supply_voltage_nominal_v",
                "value": 24.0,
                "unit": "V",
                "conditions": [],
                "variant_scope": ["standard"],
                "evidence_ids": ["E1"],
                "extraction_status": "pending_review",
                "review_decision": "approved",
                "reviewed_by": "engineer_a",
            },
            {
                "fact_id": "F_REJECTED_1",
                "product_id": "P1",
                "property": "output_voltage_nominal_v",
                "value": 99.0,
                "unit": "V",
                "conditions": [],
                "variant_scope": ["standard"],
                "evidence_ids": ["E2"],
                "extraction_status": "pending_review",
                "review_decision": "rejected",
                "reviewed_by": "engineer_a",
            },
            {
                "fact_id": "F_REJECTED_2",
                "product_id": "P2",
                "property": "power_consumption_max_w",
                "value": 500.0,
                "unit": "W",
                "conditions": [],
                "variant_scope": ["standard"],
                "evidence_ids": ["E3"],
                "extraction_status": "reviewed",  # Even if extraction_status says reviewed, rejection overrides
                "review_decision": "rejected",
                "reviewed_by": "engineer_b",
            },
        ]
    }
    review_file.write_text(json.dumps(package), encoding="utf-8")

    imported = import_review(review_file=review_file, output_reviewed_file=output_file)

    assert len(imported) == 1
    assert imported[0].fact_id == "F_APPROVED_1"
    assert imported[0].extraction_status == ExtractionStatus.reviewed
    assert imported[0].reviewed_by == "engineer_a"

    # Verify rejected facts are NOT written to the output reviewed facts file
    persisted = load_facts(output_file)
    assert len(persisted) == 1
    persisted_ids = {f.fact_id for f in persisted}
    assert "F_APPROVED_1" in persisted_ids
    assert "F_REJECTED_1" not in persisted_ids
    assert "F_REJECTED_2" not in persisted_ids


def test_import_review_approved_and_modified_facts(tmp_path: Path) -> None:
    """import_review must import approved facts and preserve human modifications."""
    review_file = tmp_path / "review_package.json"
    output_file = tmp_path / "facts.reviewed.jsonl"

    package = {
        "facts": [
            # 1. Unmodified approved fact
            {
                "fact_id": "F_UNMODIFIED_001",
                "product_id": "P1",
                "property": "supply_voltage_nominal_v",
                "value": 24.0,
                "unit": "V",
                "conditions": [],
                "variant_scope": ["standard"],
                "evidence_ids": ["SPAN_001"],
                "extraction_status": "pending_review",
                "review_decision": "approved",
                "reviewed_by": "engineer_1",
            },
            # 2. Approved fact with human modifications (value modified from 24.0 to 48.0, conditions added)
            {
                "fact_id": "F_MODIFIED_001",
                "product_id": "P2",
                "property": "supply_voltage_max_v",
                "value": 48.0,  # Modified value
                "unit": "VDC",  # Modified unit
                "conditions": [{"ambient_temp_max_c": 60}],  # Added condition
                "variant_scope": ["extended_temp"],
                "evidence_ids": ["SPAN_002"],
                "extraction_status": "pending_review",
                "review_decision": "approved",
                "reviewed_by": "engineer_2",
                "review_notes": "Corrected max voltage based on page 4 table 2",
            },
            # 3. Fact with extraction_status == 'reviewed' and modified property value
            {
                "fact_id": "F_MODIFIED_002",
                "product_id": "P3",
                "property": "communication_protocol",
                "value": "Modbus RTU over RS-485",  # Modified string value
                "unit": None,
                "conditions": [],
                "variant_scope": ["standard"],
                "evidence_ids": ["SPAN_003"],
                "extraction_status": "reviewed",
                "reviewed_by": "senior_reviewer",
            },
        ]
    }
    review_file.write_text(json.dumps(package), encoding="utf-8")

    imported = import_review(review_file=review_file, output_reviewed_file=output_file)

    assert len(imported) == 3

    facts_by_id = {f.fact_id: f for f in imported}

    # Verify unmodified approved fact
    f_unmod = facts_by_id["F_UNMODIFIED_001"]
    assert f_unmod.value == 24.0
    assert f_unmod.unit == "V"
    assert f_unmod.extraction_status == ExtractionStatus.reviewed
    assert f_unmod.reviewed_by == "engineer_1"

    # Verify modified approved fact
    f_mod1 = facts_by_id["F_MODIFIED_001"]
    assert f_mod1.value == 48.0
    assert f_mod1.unit == "VDC"
    assert f_mod1.conditions == [{"ambient_temp_max_c": 60}]
    assert f_mod1.variant_scope == ["extended_temp"]
    assert f_mod1.extraction_status == ExtractionStatus.reviewed
    assert f_mod1.reviewed_by == "engineer_2"

    # Verify reviewed status fact
    f_mod2 = facts_by_id["F_MODIFIED_002"]
    assert f_mod2.value == "Modbus RTU over RS-485"
    assert f_mod2.extraction_status == ExtractionStatus.reviewed
    assert f_mod2.reviewed_by == "senior_reviewer"

    # Verify persistence round-trip
    persisted = load_facts(output_file)
    assert len(persisted) == 3
    persisted_by_id = {f.fact_id: f for f in persisted}
    assert persisted_by_id["F_MODIFIED_001"].value == 48.0


def test_import_review_no_approved_facts(tmp_path: Path) -> None:
    """import_review should return an empty list without crashing when no facts are approved."""
    review_file = tmp_path / "all_rejected.json"
    output_file = tmp_path / "facts.reviewed.jsonl"

    package = {
        "facts": [
            {
                "fact_id": "F_REJ_A",
                "product_id": "P1",
                "property": "power_consumption_max_w",
                "review_decision": "rejected",
            },
            {
                "fact_id": "F_REJ_B",
                "product_id": "P2",
                "property": "rated_output_current_a",
                "review_decision": "rejected",
            },
        ]
    }
    review_file.write_text(json.dumps(package), encoding="utf-8")

    imported = import_review(review_file=review_file, output_reviewed_file=output_file)
    assert imported == []
    assert output_file.is_file()
    assert load_facts(output_file) == []


def test_import_review_invalid_decisions_and_missing_fields(tmp_path: Path) -> None:
    """import_review should safely skip entries with invalid decisions or missing required fields."""
    review_file = tmp_path / "invalid_entries.json"
    output_file = tmp_path / "facts.reviewed.jsonl"

    package = {
        "facts": [
            # 1. Valid approved fact
            {
                "fact_id": "F_VALID_KEEP",
                "product_id": "P1",
                "property": "supply_voltage_nominal_v",
                "value": 24.0,
                "unit": "V",
                "review_decision": "approved",
            },
            # 2. Invalid decisions
            {
                "fact_id": "F_PENDING",
                "product_id": "P1",
                "property": "supply_voltage_nominal_v",
                "value": 24.0,
                "review_decision": "pending",
                "extraction_status": "pending_review",
            },
            {
                "fact_id": "F_UNKNOWN_DECISION",
                "product_id": "P1",
                "property": "supply_voltage_nominal_v",
                "value": 24.0,
                "review_decision": "undecided",
                "extraction_status": "pending_review",
            },
            # 3. Missing required fields
            {
                # Missing fact_id
                "product_id": "P1",
                "property": "supply_voltage_nominal_v",
                "value": 24.0,
                "review_decision": "approved",
            },
            {
                # Missing product_id
                "fact_id": "F_NO_PRODUCT",
                "property": "supply_voltage_nominal_v",
                "value": 24.0,
                "review_decision": "approved",
            },
            {
                # Missing property
                "fact_id": "F_NO_PROPERTY",
                "product_id": "P1",
                "value": 24.0,
                "review_decision": "approved",
            },
            # 4. Non-dict entry
            "NOT_A_DICT_ENTRY",
            12345,
        ]
    }
    review_file.write_text(json.dumps(package), encoding="utf-8")

    imported = import_review(review_file=review_file, output_reviewed_file=output_file)
    assert len(imported) == 1
    assert imported[0].fact_id == "F_VALID_KEEP"


def test_import_review_empty_file_and_invalid_json(tmp_path: Path) -> None:
    """import_review should handle empty review files and invalid JSON gracefully."""
    empty_review = tmp_path / "empty_review.json"
    empty_review.touch()
    out1 = tmp_path / "reviewed_1.jsonl"

    assert import_review(review_file=empty_review, output_reviewed_file=out1) == []

    corrupted_review = tmp_path / "corrupted_review.json"
    corrupted_review.write_text("{not_valid_json", encoding="utf-8")
    out2 = tmp_path / "reviewed_2.jsonl"

    assert import_review(review_file=corrupted_review, output_reviewed_file=out2) == []


# ==============================================================================
# 4. Advanced Review Edge Cases: Deduplication, Conflicts, Catalog Validation
# ==============================================================================

def test_import_review_deduplication(tmp_path: Path) -> None:
    """import_review should deduplicate duplicate fact_ids, keeping the latest review entry."""
    review_file = tmp_path / "dedup_review.json"
    output_file = tmp_path / "dedup_reviewed.jsonl"

    package = {
        "facts": [
            {
                "fact_id": "F_DUP_001",
                "product_id": "P1",
                "property": "supply_voltage_nominal_v",
                "value": 12.0,
                "unit": "V",
                "review_decision": "approved",
                "reviewed_by": "engineer_early",
            },
            {
                "fact_id": "F_DUP_001",  # Same fact_id updated
                "product_id": "P1",
                "property": "supply_voltage_nominal_v",
                "value": 24.0,  # Updated value
                "unit": "VDC",
                "review_decision": "approved",
                "reviewed_by": "engineer_latest",
            },
        ]
    }
    review_file.write_text(json.dumps(package), encoding="utf-8")

    imported = import_review(review_file=review_file, output_reviewed_file=output_file, deduplicate=True)
    assert len(imported) == 1
    assert imported[0].fact_id == "F_DUP_001"
    assert imported[0].value == 24.0
    assert imported[0].unit == "VDC"
    assert imported[0].reviewed_by == "engineer_latest"

    persisted = load_facts(output_file)
    assert len(persisted) == 1
    assert persisted[0].value == 24.0


def test_import_review_conflict_handling_and_preservation(tmp_path: Path) -> None:
    """import_review should detect conflicting facts and preserve explicitly marked conflicting facts."""
    review_file = tmp_path / "conflict_review.json"
    output_file = tmp_path / "conflict_reviewed.jsonl"

    package = {
        "facts": [
            # Two approved facts for the same property but conflicting values
            {
                "fact_id": "F_CONF_A",
                "product_id": "P1",
                "property": "supply_voltage_nominal_v",
                "value": 24.0,
                "unit": "V",
                "review_decision": "approved",
            },
            {
                "fact_id": "F_CONF_B",
                "product_id": "P1",
                "property": "supply_voltage_nominal_v",
                "value": 230.0,
                "unit": "V",
                "review_decision": "approved",
            },
            # Explicitly conflicting fact marked by reviewer
            {
                "fact_id": "F_CONF_EXPLICIT",
                "product_id": "P2",
                "property": "rated_current_a",
                "value": None,
                "unit": "A",
                "review_decision": "conflicting",
                "extraction_status": "conflicting",
                "reviewed_by": "lead_engineer",
            },
        ]
    }
    review_file.write_text(json.dumps(package), encoding="utf-8")

    # When preserve_conflicts is True, explicitly conflicting facts are kept with ExtractionStatus.conflicting
    imported = import_review(
        review_file=review_file,
        output_reviewed_file=output_file,
        preserve_conflicts=True,
    )
    assert len(imported) == 3

    by_id = {f.fact_id: f for f in imported}
    assert by_id["F_CONF_A"].extraction_status == ExtractionStatus.reviewed
    assert by_id["F_CONF_B"].extraction_status == ExtractionStatus.reviewed
    assert by_id["F_CONF_EXPLICIT"].extraction_status == ExtractionStatus.conflicting


def test_import_review_unknown_product_ids_filtering(tmp_path: Path) -> None:
    """import_review should filter out facts with unknown product_ids when allowed_product_ids is specified."""
    review_file = tmp_path / "unknown_products.json"
    output_file = tmp_path / "filtered_reviewed.jsonl"

    package = {
        "facts": [
            {
                "fact_id": "F_KNOWN_P1",
                "product_id": "P1",
                "property": "supply_voltage_nominal_v",
                "value": 24.0,
                "review_decision": "approved",
            },
            {
                "fact_id": "F_UNKNOWN_P99",
                "product_id": "P_NONEXISTENT_99",
                "property": "supply_voltage_nominal_v",
                "value": 24.0,
                "review_decision": "approved",
            },
        ]
    }
    review_file.write_text(json.dumps(package), encoding="utf-8")

    # Allowed product IDs specified
    imported = import_review(
        review_file=review_file,
        output_reviewed_file=output_file,
        allowed_product_ids={"P1", "P2", "P3"},
    )
    assert len(imported) == 1
    assert imported[0].fact_id == "F_KNOWN_P1"
    assert imported[0].product_id == "P1"


def test_import_review_jsonl_format_support(tmp_path: Path) -> None:
    """import_review should transparently support JSONL review files as well as JSON files."""
    jsonl_file = tmp_path / "review.jsonl"
    output_file = tmp_path / "from_jsonl.reviewed.jsonl"

    lines = [
        json.dumps({
            "fact_id": "F_JSONL_1",
            "product_id": "P1",
            "property": "supply_voltage_nominal_v",
            "value": 24.0,
            "unit": "V",
            "review_decision": "approved",
        }),
        json.dumps({
            "fact_id": "F_JSONL_2",
            "product_id": "P2",
            "property": "power_consumption_max_w",
            "value": 15.0,
            "unit": "W",
            "review_decision": "approved",
        }),
        json.dumps({
            "fact_id": "F_JSONL_REJ",
            "product_id": "P3",
            "property": "operating_temp_max_c",
            "value": 100.0,
            "review_decision": "rejected",
        }),
    ]
    jsonl_file.write_text("\n".join(lines), encoding="utf-8")

    imported = import_review(review_file=jsonl_file, output_reviewed_file=output_file)
    assert len(imported) == 2
    ids = {f.fact_id for f in imported}
    assert ids == {"F_JSONL_1", "F_JSONL_2"}


def test_import_review_field_cleaning_and_whitespace(tmp_path: Path) -> None:
    """import_review should strip whitespace from IDs, normalize empty port_id to None, and skip invalid conditions."""
    review_file = tmp_path / "whitespace_review.json"
    output_file = tmp_path / "clean_reviewed.jsonl"

    package = {
        "facts": [
            {
                "fact_id": "  F_CLEAN_001  ",
                "product_id": "  P1  ",
                "port_id": "   ",  # Whitespace-only should normalize to None
                "property": "  supply_voltage_nominal_v  ",
                "value": 24.0,
                "unit": "  V  ",
                "conditions": [
                    {"temp": 25},
                    "NOT_A_DICT_CONDITION",  # Malformed item should be skipped
                ],
                "variant_scope": ["  standard  ", 123],
                "evidence_ids": ["  SPAN_01  ", "   "],
                "review_decision": "approved",
            }
        ]
    }
    review_file.write_text(json.dumps(package), encoding="utf-8")

    imported = import_review(review_file=review_file, output_reviewed_file=output_file)
    assert len(imported) == 1
    f = imported[0]
    assert f.fact_id == "F_CLEAN_001"
    assert f.product_id == "P1"
    assert f.port_id is None
    assert f.property == "supply_voltage_nominal_v"
    assert f.unit == "V"
    assert f.conditions == [{"temp": 25}]
    assert f.variant_scope == ["standard", "123"]
    assert f.evidence_ids == ["SPAN_01"]


def test_review_full_lifecycle_roundtrip(tmp_path: Path) -> None:
    """Complete roundtrip test: export pending facts, curate decisions, import reviewed facts."""
    auto_facts_file = tmp_path / "facts.auto.jsonl"
    pending_file = tmp_path / "pending.json"
    reviewed_file = tmp_path / "facts.reviewed.jsonl"

    # 1. Create auto facts
    fact1 = Fact(
        fact_id="F_AUTO_01",
        product_id="P1",
        property="supply_voltage_nominal_v",
        value=24.0,
        unit="V",
        extraction_status=ExtractionStatus.auto_extracted,
    )
    fact2 = Fact(
        fact_id="F_AUTO_02",
        product_id="P2",
        property="output_current_nominal_a",
        value=5.0,
        unit="A",
        extraction_status=ExtractionStatus.auto_extracted,
    )
    save_facts([fact1, fact2], auto_facts_file)

    # 2. Export review package
    export_result = export_review(auto_facts_file=auto_facts_file, output_review_file=pending_file)
    assert export_result["status"] == "exported"
    assert export_result["facts_count"] == 2
    assert pending_file.is_file()

    # 3. Simulate human curation: approve fact1, reject fact2
    with open(pending_file, "r", encoding="utf-8") as f:
        package = json.load(f)

    assert len(package["facts"]) == 2
    package["facts"][0]["review_decision"] = "approved"
    package["facts"][1]["review_decision"] = "rejected"

    with open(pending_file, "w", encoding="utf-8") as f:
        json.dump(package, f)

    # 4. Import reviewed facts
    imported = import_review(review_file=pending_file, output_reviewed_file=reviewed_file)
    assert len(imported) == 1
    assert imported[0].fact_id == "F_AUTO_01"
    assert imported[0].extraction_status == ExtractionStatus.reviewed

    # 5. Verify persisted facts.reviewed.jsonl
    reloaded = load_facts(reviewed_file)
    assert len(reloaded) == 1
    assert reloaded[0].fact_id == "F_AUTO_01"

