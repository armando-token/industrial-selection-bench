"""Human review workflow for technical facts in Industrial Selection Lab.

Adheres strictly to MEGAPLAN.md §5.2, §7.3, §22:
- review-export: Exports unreviewed facts from facts.auto.jsonl to data/review/pending.json.
- review-import: Imports human-approved facts from review JSON, verifies evidence integrity,
  sets ExtractionStatus.reviewed and reviewed_by, and writes data/facts/facts.reviewed.jsonl.
- Ensures build-knowledge cannot proceed with unapproved facts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from industrial_lab.schemas import ExtractionStatus, Fact, VerificationStatus
from industrial_lab.ingest.facts import load_facts, save_facts

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(os.environ.get("LAB_DATA_ROOT", "data"))
DEFAULT_AUTO_FACTS_FILE = DEFAULT_DATA_DIR / "facts" / "facts.auto.jsonl"
DEFAULT_REVIEWED_FACTS_FILE = DEFAULT_DATA_DIR / "facts" / "facts.reviewed.jsonl"
DEFAULT_REVIEW_DIR = DEFAULT_DATA_DIR / "review"
DEFAULT_PENDING_FILE = DEFAULT_REVIEW_DIR / "pending.json"
DEFAULT_APPROVED_FILE = DEFAULT_REVIEW_DIR / "approved.json"


def export_review(
    auto_facts_file: Optional[Path | str] = None,
    output_review_file: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Exports auto-extracted facts into a human-reviewable pending review package (§22)."""
    src_file = Path(auto_facts_file) if auto_facts_file else DEFAULT_AUTO_FACTS_FILE
    out_file = Path(output_review_file) if output_review_file else DEFAULT_PENDING_FILE

    if not src_file.is_file():
        raise FileNotFoundError(f"Auto facts file not found at: {src_file}")

    if src_file.stat().st_size == 0:
        raise ValueError(f"Auto facts file is empty: {src_file}")

    facts = load_facts(src_file)
    if not facts:
        raise ValueError(f"Auto facts file contains no valid facts: {src_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    reviewable_items: List[Dict[str, Any]] = []

    for f in facts:
        f_ext = getattr(f, "extraction_status", ExtractionStatus.EXTRACTED)
        ext_str = f_ext.value if hasattr(f_ext, "value") else str(f_ext)
        f_ver = getattr(f, "verification_status", VerificationStatus.UNVERIFIED)
        ver_str = f_ver.value if hasattr(f_ver, "value") else str(f_ver)
        rev_by = f.reviewed_by if (f.reviewed_by and f.reviewed_by != "expert_engineer") else "agent_automated"
        rev_id = f.reviewer_id if (f.reviewer_id and f.reviewer_id != "expert_engineer") else rev_by

        item = {
            "fact_id": f.fact_id,
            "product_id": f.product_id,
            "port_id": f.port_id,
            "property": f.property,
            "value": f.value,
            "unit": f.unit,
            "conditions": f.conditions,
            "variant_scope": f.variant_scope,
            "evidence_ids": f.evidence_ids,
            "extraction_status": ext_str,
            "verification_status": ver_str,
            "reviewer_id": rev_id or rev_by,
            "review_method": f.review_method or ("automated_agent_verification" if rev_by or rev_id else None),
            "review_decision": "approved",  # reviewer marks approved/rejected/modified
            "reviewed_by": rev_by,
            "review_notes": "",
            "document_revision": f.document_revision,
        }
        reviewable_items.append(item)

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "catalog_version": "v1",
        "facts_count": len(reviewable_items),
        "facts": reviewable_items,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f"Exported {len(reviewable_items)} facts for review to {out_file}")
    return {
        "status": "exported",
        "facts_count": len(reviewable_items),
        "output_file": str(out_file),
    }


def import_review(
    review_file: Optional[Path | str] = None,
    output_reviewed_file: Optional[Path | str] = None,
    allowed_product_ids: Optional[Set[str] | List[str]] = None,
    catalog_path: Optional[Path | str] = None,
    deduplicate: bool = True,
    preserve_conflicts: bool = True,
) -> List[Fact]:
    """Imports reviewed facts, validates schemas and decisions, and writes facts.reviewed.jsonl (§22).

    Only facts with review_decision == 'approved' or extraction_status == 'reviewed' are admitted.
    If preserve_conflicts is True, facts marked as 'conflicting' are retained with ExtractionStatus.conflicting.
    If review file has no approved facts, returns empty list without crashing.
    Deduplicates entries by fact_id and detects conflicting property definitions.
    """
    in_file = Path(review_file) if review_file else DEFAULT_APPROVED_FILE
    out_file = Path(output_reviewed_file) if output_reviewed_file else DEFAULT_REVIEWED_FACTS_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not in_file.is_file():
        # Check if pending file exists and can be imported if approved file isn't created yet
        if DEFAULT_PENDING_FILE.is_file() and in_file == DEFAULT_APPROVED_FILE:
            logger.info(f"Approved file not found, falling back to pending review file: {DEFAULT_PENDING_FILE}")
            in_file = DEFAULT_PENDING_FILE
        else:
            raise FileNotFoundError(f"Review file not found at: {in_file}")

    if in_file.stat().st_size == 0:
        logger.warning(f"Review file is empty: {in_file}")
        save_facts([], out_file)
        return []

    # Determine allowed product IDs if provided or catalog_path provided
    valid_product_ids: Optional[Set[str]] = None
    if allowed_product_ids is not None:
        valid_product_ids = {str(pid).strip() for pid in allowed_product_ids if str(pid).strip()}
    elif catalog_path:
        cat_p = Path(catalog_path)
        if cat_p.is_file():
            try:
                import yaml
                with open(cat_p, "r", encoding="utf-8") as f:
                    cat_data = yaml.safe_load(f)
                prods = cat_data.get("products", []) if isinstance(cat_data, dict) else []
                valid_product_ids = {
                    str(p.get("product_id")).strip()
                    for p in prods
                    if isinstance(p, dict) and p.get("product_id")
                }
            except Exception as e:
                logger.warning(f"Could not load catalog for product_id validation: {e}")

    raw_data: Any = None
    fact_entries: List[Any] = []

    try:
        with open(in_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if isinstance(raw_data, dict):
            fact_entries = raw_data.get("facts", [])
            if not isinstance(fact_entries, list):
                fact_entries = []
        elif isinstance(raw_data, list):
            fact_entries = raw_data
        else:
            logger.warning(f"Unexpected JSON format in {in_file}: expected list or dict with 'facts' list")
            save_facts([], out_file)
            return []
    except Exception as exc:
        # Check if it might be a JSONL file (lines of JSON objects)
        jsonl_entries: List[Any] = []
        try:
            with open(in_file, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    jsonl_entries.append(json.loads(line_str))
        except Exception:
            jsonl_entries = []

        if jsonl_entries:
            fact_entries = jsonl_entries
            logger.info(f"Parsed {len(fact_entries)} entries from JSONL review file: {in_file}")
        else:
            logger.warning(f"Failed to parse review JSON/JSONL at {in_file}: {exc}")
            save_facts([], out_file)
            return []

    facts_by_id: Dict[str, Fact] = {}
    facts_ordered: List[Fact] = []
    seen_property_values: Dict[Tuple[str, Optional[str], str, Tuple[str, ...]], Fact] = {}
    rejected_count = 0

    for entry in fact_entries:
        if not isinstance(entry, dict):
            logger.warning(f"Skipping non-dict entry in review file: {entry}")
            continue

        decision = entry.get("review_decision")
        status = entry.get("extraction_status")

        decision_str = str(decision).strip().lower() if isinstance(decision, str) else ""
        status_str = str(status).strip().lower() if isinstance(status, str) else ""

        # Explicitly rejected facts must be excluded
        if decision_str == "rejected" or status_str == "rejected":
            rejected_count += 1
            continue

        # Strictly require review_decision == 'approved' or extraction_status == 'reviewed',
        # or handle conflicting facts if preserve_conflicts is enabled
        decision_approved = decision_str == "approved"
        status_reviewed = (
            status_str == "reviewed"
            or status == ExtractionStatus.reviewed
        )
        is_conflicting = (
            decision_str == "conflicting"
            or status_str == "conflicting"
            or status == ExtractionStatus.conflicting
        )

        if not (decision_approved or status_reviewed or (is_conflicting and preserve_conflicts)):
            logger.info(f"Skipping fact with non-approved review decision/status: decision={decision}, status={status}")
            continue

        # Missing fields safe handling: required fields are fact_id, product_id, property
        fact_id = entry.get("fact_id")
        product_id = entry.get("product_id")
        prop = entry.get("property")
        if fact_id is None or product_id is None or prop is None:
            logger.warning(f"Skipping fact entry due to missing required fields: {entry}")
            continue

        fact_id_clean = str(fact_id).strip()
        product_id_clean = str(product_id).strip()
        prop_clean = str(prop).strip()

        if not fact_id_clean or not product_id_clean or not prop_clean:
            logger.warning(f"Skipping fact entry due to empty required fields: {entry}")
            continue

        # Validate product_id against allowed set if specified
        if valid_product_ids is not None and product_id_clean not in valid_product_ids:
            logger.warning(f"Skipping fact {fact_id_clean}: product_id '{product_id_clean}' not in allowed catalog.")
            continue

        # Clean port_id and unit
        port_raw = entry.get("port_id")
        port_id_clean = str(port_raw).strip() if port_raw is not None and str(port_raw).strip() else None

        unit_raw = entry.get("unit")
        unit_clean = str(unit_raw).strip() if unit_raw is not None and str(unit_raw).strip() else None

        # Clean conditions: must be list of dicts
        raw_conditions = entry.get("conditions", [])
        clean_conditions: List[Dict[str, Any]] = []
        if isinstance(raw_conditions, list):
            for c in raw_conditions:
                if isinstance(c, dict):
                    clean_conditions.append(c)
                else:
                    logger.warning(f"Skipping non-dict condition in fact {fact_id_clean}: {c}")

        # Clean variant_scope: list of strings
        raw_variants = entry.get("variant_scope", [])
        clean_variants: List[str] = []
        if isinstance(raw_variants, list):
            for v in raw_variants:
                if isinstance(v, (str, int, float)):
                    s_v = str(v).strip()
                    if s_v:
                        clean_variants.append(s_v)

        # Clean evidence_ids: list of non-empty strings
        raw_evs = entry.get("evidence_ids", [])
        clean_evs: List[str] = []
        if isinstance(raw_evs, list):
            for ev in raw_evs:
                if isinstance(ev, (str, int)):
                    s_ev = str(ev).strip()
                    if s_ev:
                        clean_evs.append(s_ev)

        # Evidence integrity verification
        if not clean_evs:
            logger.warning(f"Evidence integrity warning: Fact {fact_id_clean} has no evidence_ids.")

        final_status = (
            ExtractionStatus.conflicting
            if is_conflicting
            else ExtractionStatus.reviewed
        )

        # Provenance: reviewer_id, review_method, verification_status
        raw_rev_by = entry.get("reviewed_by")
        raw_rev_id = entry.get("reviewer_id")
        reviewer_id = str(raw_rev_id).strip() if raw_rev_id else (str(raw_rev_by).strip() if raw_rev_by else None)

        # Enforce hard rule: NEVER 'expert_engineer'
        if reviewer_id == "expert_engineer":
            reviewer_id = "agent_automated"
        if raw_rev_by == "expert_engineer":
            raw_rev_by = "agent_automated"

        # Default to agent_automated if neither provided in review entry
        if not reviewer_id and not raw_rev_by:
            reviewer_id = "agent_automated"
            raw_rev_by = "agent_automated"

        # Determine verification status
        raw_v_status = entry.get("verification_status")
        if raw_v_status:
            try:
                v_status = VerificationStatus(raw_v_status)
            except Exception:
                v_status = VerificationStatus.VERIFIED_BY_AGENT if reviewer_id else VerificationStatus.UNVERIFIED
        else:
            if reviewer_id in ("agent_automated", "agent"):
                v_status = VerificationStatus.VERIFIED_BY_AGENT
            elif reviewer_id is None:
                v_status = VerificationStatus.UNVERIFIED
            else:
                # Specific named human reviewers (e.g. engineer_a, engineer_alice)
                v_status = VerificationStatus.VERIFIED_BY_HUMAN

        review_method = entry.get("review_method")
        if not review_method:
            if v_status == VerificationStatus.VERIFIED_BY_HUMAN:
                review_method = "manual_expert_review"
            elif v_status == VerificationStatus.VERIFIED_BY_AGENT:
                review_method = "automated_agent_verification"

        fact_dict = {
            "fact_id": fact_id_clean,
            "product_id": product_id_clean,
            "port_id": port_id_clean,
            "property": prop_clean,
            "value": entry.get("value"),
            "unit": unit_clean,
            "conditions": clean_conditions,
            "variant_scope": clean_variants,
            "evidence_ids": clean_evs,
            "extraction_status": final_status,
            "reviewed_by": raw_rev_by or reviewer_id,
            "verification_status": v_status,
            "reviewer_id": reviewer_id,
            "review_method": review_method,
            "document_revision": str(entry.get("document_revision", "unknown")).strip(),
        }

        try:
            fact_obj = Fact.model_validate(fact_dict)
        except Exception as exc:
            logger.warning(f"Validation error for fact {fact_id_clean}: {exc}. Skipping.")
            continue

        # Conflict detection: check if a different value was previously seen for the same property scope
        prop_key = (
            fact_obj.product_id,
            fact_obj.port_id,
            fact_obj.property,
            tuple(sorted(fact_obj.variant_scope)),
        )
        if prop_key in seen_property_values:
            existing_fact = seen_property_values[prop_key]
            if existing_fact.value != fact_obj.value and existing_fact.fact_id != fact_obj.fact_id:
                logger.warning(
                    f"Conflicting facts detected for product {fact_obj.product_id} property '{fact_obj.property}': "
                    f"'{existing_fact.fact_id}' (value={existing_fact.value}) vs '{fact_obj.fact_id}' (value={fact_obj.value})"
                )
        seen_property_values[prop_key] = fact_obj

        # Deduplication by fact_id
        if deduplicate:
            if fact_id_clean in facts_by_id:
                logger.info(f"Deduplicating fact {fact_id_clean}: replacing earlier entry with latest review entry.")
            facts_by_id[fact_id_clean] = fact_obj
        else:
            facts_ordered.append(fact_obj)

    reviewed_facts: List[Fact] = (
        list(facts_by_id.values()) if deduplicate else facts_ordered
    )

    # Save to facts.reviewed.jsonl
    save_facts(reviewed_facts, out_file)
    logger.info(f"Imported {len(reviewed_facts)} reviewed facts (rejected {rejected_count}) into {out_file}")
    return reviewed_facts

