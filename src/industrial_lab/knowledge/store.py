"""Knowledge store for Industrial Selection Lab.

Adheres strictly to MEGAPLAN.md §3.2, §5.2, §5.3, §22:
- Dual persistence: Structured SQLite relational database and portable JSON exports.
- Stores catalog products, technical facts, physical/electrical ports, relations, and source spans.
- Indexed fast lookups by product_id, property, port_id, and evidence span_id.
- Full parity with Pydantic schemas (Fact, Port, Relation, SourceSpan, CatalogManifest).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from industrial_lab.schemas import (
    CatalogManifest,
    Citation,
    ExtractionStatus,
    Fact,
    Port,
    ProductManifestItem,
    Relation,
    RelationType,
    SourceSpan,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(os.environ.get("LAB_DATA_ROOT", "data"))
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "knowledge.db"
DEFAULT_JSON_PATH = DEFAULT_DATA_DIR / "knowledge" / "knowledge_store.json"

PROPERTY_CATEGORIES: Dict[str, List[str]] = {
    "electrical": [
        "supply_voltage_nominal_v", "supply_voltage_min_v", "supply_voltage_max_v",
        "power_consumption_max_w", "output_voltage_nominal_v", "rated_output_current_a",
        "rated_output_power_w", "loop_supply_voltage_nominal_v", "loop_supply_voltage_min_v",
        "loop_supply_voltage_max_v", "voltage", "current", "power",
    ],
    "environmental": [
        "operating_temp_min_c", "operating_temp_max_c", "storage_temp_min_c",
        "storage_temp_max_c", "enclosure_ip_rating", "humidity_max_percent", "altitude_max_m",
    ],
    "communication": [
        "communication_interface", "communication_protocol", "baud_rate",
        "parity", "stop_bits", "data_bits",
    ],
    "signal": [
        "sensor_interface", "analog_input", "analog_output",
        "digital_input", "digital_output",
    ],
    "mechanical": [
        "din_rail_mounting", "dimensions", "weight", "mounting_type",
    ],
    "functional": [
        "measurement_range_min_c", "measurement_range_max_c", "accuracy",
        "response_time_ms", "sampling_rate_hz",
    ],
}


class KnowledgeStore:
    """Relational SQLite & JSON store for catalog, facts, ports, relations, and spans."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = str(db_path) if db_path else str(DEFAULT_DB_PATH)
        self._mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes relational tables and indices."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS catalog_products (
                    product_id TEXT PRIMARY KEY,
                    sku TEXT,
                    manufacturer TEXT,
                    exact_model TEXT,
                    variant TEXT,
                    documents_json TEXT
                );

                CREATE TABLE IF NOT EXISTS facts (
                    fact_id TEXT PRIMARY KEY,
                    product_id TEXT,
                    port_id TEXT,
                    property TEXT,
                    value_json TEXT,
                    unit TEXT,
                    conditions_json TEXT,
                    variant_scope_json TEXT,
                    evidence_ids_json TEXT,
                    extraction_status TEXT,
                    reviewed_by TEXT,
                    document_revision TEXT,
                    verification_status TEXT,
                    reviewer_id TEXT,
                    review_method TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_facts_product_id ON facts(product_id);
                CREATE INDEX IF NOT EXISTS idx_facts_property ON facts(product_id, property);
                CREATE INDEX IF NOT EXISTS idx_facts_prop_only ON facts(property);
                CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(extraction_status);
                CREATE INDEX IF NOT EXISTS idx_facts_port_id ON facts(port_id);

                CREATE TABLE IF NOT EXISTS ports (
                    port_id TEXT PRIMARY KEY,
                    product_id TEXT,
                    direction TEXT,
                    physical_interface TEXT,
                    signal_kind TEXT,
                    protocol TEXT,
                    role TEXT,
                    supported_ranges_json TEXT,
                    wiring_conditions_json TEXT,
                    evidence_ids_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_ports_product_id ON ports(product_id);
                CREATE INDEX IF NOT EXISTS idx_ports_interface ON ports(physical_interface);
                CREATE INDEX IF NOT EXISTS idx_ports_signal ON ports(signal_kind);

                CREATE TABLE IF NOT EXISTS relations (
                    relation_id TEXT PRIMARY KEY,
                    source_id TEXT,
                    target_id TEXT,
                    relation_type TEXT,
                    conditions_json TEXT,
                    evidence_ids_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
                CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
                CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);

                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    document_sha256 TEXT,
                    pdf_page_index INTEGER,
                    printed_page_label TEXT,
                    text TEXT,
                    bbox_json TEXT,
                    table_id TEXT,
                    row_header TEXT,
                    column_header TEXT,
                    product_scope_json TEXT,
                    revision TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_spans_doc_page ON spans(document_id, pdf_page_index);
            """)
            cur.execute("PRAGMA table_info(facts)")
            existing_cols = {row["name"] for row in cur.fetchall()}
            if "verification_status" not in existing_cols:
                cur.execute("ALTER TABLE facts ADD COLUMN verification_status TEXT")
            if "reviewer_id" not in existing_cols:
                cur.execute("ALTER TABLE facts ADD COLUMN reviewer_id TEXT")
            if "review_method" not in existing_cols:
                cur.execute("ALTER TABLE facts ADD COLUMN review_method TEXT")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_facts_verification ON facts(verification_status)")
            conn.commit()

    # --------------------------------------------------------------------------
    # Catalog Products
    # --------------------------------------------------------------------------

    def save_product(
        self,
        product_id: str,
        sku: Optional[str] = None,
        manufacturer: Optional[str] = None,
        exact_model: Optional[str] = None,
        variant: Optional[str] = None,
        documents: Optional[List[Any]] = None,
    ) -> None:
        """Upserts a product in the catalog."""
        docs_json = json.dumps(documents or [], ensure_ascii=False)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO catalog_products (product_id, sku, manufacturer, exact_model, variant, documents_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(product_id) DO UPDATE SET
                    sku=excluded.sku,
                    manufacturer=excluded.manufacturer,
                    exact_model=excluded.exact_model,
                    variant=excluded.variant,
                    documents_json=excluded.documents_json
                """,
                (product_id, sku, manufacturer, exact_model, variant, docs_json),
            )
            conn.commit()

    def save_catalog(self, manifest: CatalogManifest | Dict[str, Any]) -> None:
        """Stores all products from a catalog manifest."""
        products = manifest.products if isinstance(manifest, CatalogManifest) else manifest.get("products", [])
        for item in products:
            if isinstance(item, ProductManifestItem):
                self.save_product(
                    product_id=item.product_id,
                    sku=item.sku,
                    manufacturer=item.manufacturer,
                    exact_model=item.exact_model,
                    variant=item.variant,
                    documents=item.documents,
                )
            elif isinstance(item, dict):
                self.save_product(
                    product_id=item.get("product_id"),
                    sku=item.get("sku"),
                    manufacturer=item.get("manufacturer"),
                    exact_model=item.get("exact_model"),
                    variant=item.get("variant"),
                    documents=item.get("documents", []),
                )

    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves single product catalog metadata."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM catalog_products WHERE product_id = ?",
                (product_id,),
            ).fetchone()
            if row:
                return {
                    "product_id": row["product_id"],
                    "sku": row["sku"],
                    "manufacturer": row["manufacturer"],
                    "exact_model": row["exact_model"],
                    "variant": row["variant"],
                    "documents": json.loads(row["documents_json"]),
                }
        return None

    def list_products(self) -> List[Dict[str, Any]]:
        """Lists all products in the catalog."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM catalog_products ORDER BY product_id").fetchall()
            return [
                {
                    "product_id": r["product_id"],
                    "sku": r["sku"],
                    "manufacturer": r["manufacturer"],
                    "exact_model": r["exact_model"],
                    "variant": r["variant"],
                    "documents": json.loads(r["documents_json"]),
                }
                for r in rows
            ]

    # --------------------------------------------------------------------------
    # Facts
    # --------------------------------------------------------------------------

    def save_facts(self, facts: List[Fact]) -> None:
        """Bulk upserts technical facts within a single atomic transaction."""
        if not facts:
            return
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            with conn:
                for fact in facts:
                    conn.execute(
                        """
                        INSERT INTO facts (
                            fact_id, product_id, port_id, property, value_json, unit,
                            conditions_json, variant_scope_json, evidence_ids_json,
                            extraction_status, reviewed_by, document_revision,
                            verification_status, reviewer_id, review_method
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(fact_id) DO UPDATE SET
                            product_id=excluded.product_id,
                            port_id=excluded.port_id,
                            property=excluded.property,
                            value_json=excluded.value_json,
                            unit=excluded.unit,
                            conditions_json=excluded.conditions_json,
                            variant_scope_json=excluded.variant_scope_json,
                            evidence_ids_json=excluded.evidence_ids_json,
                            extraction_status=excluded.extraction_status,
                            reviewed_by=excluded.reviewed_by,
                            document_revision=excluded.document_revision,
                            verification_status=excluded.verification_status,
                            reviewer_id=excluded.reviewer_id,
                            review_method=excluded.review_method
                        """,
                        (
                            fact.fact_id,
                            fact.product_id,
                            fact.port_id,
                            fact.property,
                            json.dumps(fact.value, ensure_ascii=False),
                            fact.unit,
                            json.dumps(fact.conditions, ensure_ascii=False),
                            json.dumps(fact.variant_scope, ensure_ascii=False),
                            json.dumps(fact.evidence_ids, ensure_ascii=False),
                            fact.extraction_status.value if hasattr(fact.extraction_status, "value") else str(fact.extraction_status),
                            fact.reviewed_by,
                            fact.document_revision,
                            fact.verification_status.value if hasattr(fact.verification_status, "value") else str(fact.verification_status),
                            fact.reviewer_id,
                            fact.review_method,
                        ),
                    )
        finally:
            if close_needed:
                conn.close()

    def save_fact(self, fact: Fact) -> None:
        """Upserts a technical fact."""
        self.save_facts([fact])

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        status_str = row["extraction_status"]
        try:
            extraction_status = ExtractionStatus(status_str)
        except (ValueError, KeyError):
            extraction_status = ExtractionStatus.reviewed

        keys = row.keys()
        v_status_str = row["verification_status"] if "verification_status" in keys else None
        try:
            verification_status = VerificationStatus(v_status_str) if v_status_str else VerificationStatus.UNVERIFIED
        except (ValueError, KeyError):
            verification_status = VerificationStatus.UNVERIFIED

        reviewer_id = row["reviewer_id"] if "reviewer_id" in keys else None
        review_method = row["review_method"] if "review_method" in keys else None
        reviewed_by = row["reviewed_by"]
        if reviewed_by == "expert_engineer":
            reviewed_by = "agent_automated"
            reviewer_id = reviewer_id or "agent_automated"

        return Fact(
            fact_id=row["fact_id"],
            product_id=row["product_id"],
            port_id=row["port_id"],
            property=row["property"],
            value=json.loads(row["value_json"]),
            unit=row["unit"],
            conditions=json.loads(row["conditions_json"]),
            variant_scope=json.loads(row["variant_scope_json"]),
            evidence_ids=json.loads(row["evidence_ids_json"]),
            extraction_status=extraction_status,
            reviewed_by=reviewed_by,
            document_revision=row["document_revision"],
            verification_status=verification_status,
            reviewer_id=reviewer_id,
            review_method=review_method,
        )

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            row = conn.execute("SELECT * FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
            if row:
                return self._row_to_fact(row)
            return None
        finally:
            if close_needed:
                conn.close()

    def get_facts_by_product(self, product_id: str, variant: Optional[str] = None) -> List[Fact]:
        """Retrieves all facts for a product, optionally filtered by specific variant (§5.1)."""
        return self.query_facts(product_id=product_id, variant=variant)

    def get_facts_for_variant(self, product_id: str, variant: str) -> List[Fact]:
        """Convenience method to retrieve facts scoped specifically to a product variant (§5.1)."""
        return self.query_facts(product_id=product_id, variant=variant)

    def get_facts_for_property(
        self,
        product_id: str,
        property_name: str,
        variant: Optional[str] = None,
    ) -> List[Fact]:
        """Queries facts for a specific property on a product, with optional variant filtering (§5.1)."""
        return self.query_facts(
            product_id=product_id,
            property_name=property_name,
            variant=variant,
        )

    def query_facts(
        self,
        product_id: Optional[str] = None,
        variant: Optional[str] = None,
        property_name: Optional[str] = None,
        properties: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any] | List[Dict[str, Any]]] = None,
        verification_status: Optional[VerificationStatus | str] = None,
    ) -> List[Fact]:
        """Queries facts with clean filtering by product_id, variant, property, and conditions.

        Adheres strictly to REPAIR3_PLAN §5.1 (F05) and §5.2 (F19).
        Prohibits cross-inheritance between distinct variants without documentary scope.
        """
        query = "SELECT * FROM facts WHERE 1=1"
        params: List[Any] = []
        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)
        if property_name:
            query += " AND property = ?"
            params.append(property_name)
        elif properties:
            placeholders = ",".join("?" for _ in properties)
            query += f" AND property IN ({placeholders})"
            params.extend(properties)

        query += " ORDER BY product_id, fact_id"

        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            rows = conn.execute(query, params).fetchall()
            facts = [self._row_to_fact(r) for r in rows]
        finally:
            if close_needed:
                conn.close()

        # Variant filtering: ensure no illegal cross-inheritance (§5.1)
        if variant:
            v_norm = variant.strip().lower()
            filtered_facts: List[Fact] = []
            for f in facts:
                scope_lower = [s.strip().lower() for s in f.variant_scope]
                # Fact explicitly covers this variant
                if any(v_norm == s or v_norm in s or s in v_norm for s in scope_lower):
                    filtered_facts.append(f)
                # Or fact is general standard baseline without conflicting variant-specific restrictions
                elif "standard" in scope_lower:
                    filtered_facts.append(f)
            facts = filtered_facts

        # Conditions filtering
        if conditions:
            cond_dicts: List[Dict[str, Any]] = (
                conditions if isinstance(conditions, list) else [conditions]
            )
            cond_matched: List[Fact] = []
            for f in facts:
                if not f.conditions:
                    # Unconditional fact applies universally
                    cond_matched.append(f)
                    continue
                # Fact has conditions: check if they match requested conditions
                matches = False
                for req_c in cond_dicts:
                    for f_c in f.conditions:
                        if all(f_c.get(k) == v for k, v in req_c.items() if k in f_c):
                            matches = True
                            break
                    if matches:
                        break
                if matches:
                    cond_matched.append(f)
            facts = cond_matched

        # Verification status filtering
        if verification_status:
            v_target = (
                verification_status.value
                if hasattr(verification_status, "value")
                else str(verification_status)
            ).upper()
            facts = [
                f for f in facts
                if (f.verification_status.value if hasattr(f.verification_status, "value") else str(f.verification_status)).upper() == v_target
            ]

        return facts

    def resolve_citations(
        self,
        evidence_ids: List[str],
        product_id: Optional[str] = None,
    ) -> List[Citation]:
        """Resolves source_span_ids from facts/checks into verifiable Citation objects.

        Adheres strictly to REPAIR3_PLAN §5.3:
        Maps source_span_ids to real documents, document sha256 hashes, PDF page indices,
        printed page labels, and literal snippets.
        Resolves:
          P_X4 -> MAN1138 (datasheet) / MAN1137 (user manual)
          P_THT -> THT-02 user manual
          P_UHEAT -> U Series datasheet / installation sheet 66661
        """
        if not evidence_ids:
            return []

        unique_eids: List[str] = []
        for eid in evidence_ids:
            if eid and eid not in unique_eids:
                unique_eids.append(eid)

        spans = self.get_spans(unique_eids)
        spans_by_id = {s.span_id: s for s in spans}

        citations: List[Citation] = []
        for eid in unique_eids:
            if eid in spans_by_id:
                sp = spans_by_id[eid]
                if product_id and sp.product_scope and product_id not in sp.product_scope:
                    continue
                citations.append(
                    Citation(
                        citation_id=sp.span_id,
                        document_id=sp.document_id,
                        document_sha256=sp.document_sha256,
                        page=sp.pdf_page_index,
                        printed_page=sp.printed_page_label,
                        snippet=sp.text[:400] if sp.text else None,
                        location=f"Page {sp.pdf_page_index}" if sp.pdf_page_index is not None else None,
                    )
                )
            else:
                parts = eid.split(":")
                doc_id = parts[0] if parts else "unknown_doc"
                page_num: Optional[int] = None
                for part in parts[1:]:
                    if part.startswith("p") and part[1:].isdigit():
                        page_num = int(part[1:])
                        break
                citations.append(
                    Citation(
                        citation_id=eid,
                        document_id=doc_id,
                        document_sha256=None,
                        page=page_num,
                        printed_page=str(page_num) if page_num is not None else None,
                        snippet=None,
                        location=f"Page {page_num}" if page_num is not None else None,
                    )
                )

        return citations

    def resolve_citations_for_facts(
        self,
        facts: List[Fact],
        product_id: Optional[str] = None,
    ) -> List[Citation]:
        """Resolves citations for all evidence spans attached to the given facts."""
        evidence_ids: List[str] = []
        for f in facts:
            for eid in f.evidence_ids:
                if eid not in evidence_ids:
                    evidence_ids.append(eid)
        return self.resolve_citations(evidence_ids, product_id=product_id)

    def resolve_citations_for_checks(
        self,
        checks: List[Any],
        product_id: Optional[str] = None,
    ) -> List[Citation]:
        """Resolves citations for all evidence spans attached to the given check results."""
        evidence_ids: List[str] = []
        for c in checks:
            c_ev = getattr(c, "evidence_ids", []) or []
            if isinstance(c_ev, list):
                for eid in c_ev:
                    if eid and eid not in evidence_ids:
                        evidence_ids.append(eid)
        return self.resolve_citations(evidence_ids, product_id=product_id)

    def list_all_facts(self) -> List[Fact]:
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            rows = conn.execute("SELECT * FROM facts ORDER BY product_id, fact_id").fetchall()
            return [self._row_to_fact(r) for r in rows]
        finally:
            if close_needed:
                conn.close()

    def get_facts_by_category(
        self,
        category: str,
        product_id: Optional[str] = None,
    ) -> List[Fact]:
        """Queries facts belonging to a specific technical category (e.g. electrical, environmental)."""
        prefixes_or_props = PROPERTY_CATEGORIES.get(category.lower())
        all_facts = self.get_facts_by_product(product_id) if product_id else self.list_all_facts()
        if not prefixes_or_props:
            cat_lower = category.lower()
            return [f for f in all_facts if cat_lower in f.property.lower()]

        return [
            f for f in all_facts
            if any(p == f.property or f.property.startswith(p) or p in f.property for p in prefixes_or_props)
        ]

    def list_fact_categories(self) -> List[str]:
        """Lists available canonical fact categories."""
        return list(PROPERTY_CATEGORIES.keys())

    def detect_fact_conflicts(self, product_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detects contradictions, conflicting extraction statuses, or physical impossibilities in facts."""
        facts = self.get_facts_by_product(product_id) if product_id else self.list_all_facts()
        conflicts: List[Dict[str, Any]] = []

        # 1. ExtractionStatus.conflicting
        for f in facts:
            if f.extraction_status == ExtractionStatus.conflicting:
                conflicts.append({
                    "conflict_type": "status_conflicting",
                    "product_id": f.product_id,
                    "fact_id": f.fact_id,
                    "property": f.property,
                    "reason": "Fact explicitly marked as conflicting during extraction/review",
                })

        # 2. Contradictory values for the same property under identical scope
        by_prop: Dict[Tuple[str, str], List[Fact]] = {}
        for f in facts:
            key = (f.product_id, f.property)
            by_prop.setdefault(key, []).append(f)

        for (pid, prop), prop_facts in by_prop.items():
            if len(prop_facts) > 1:
                for i in range(len(prop_facts)):
                    for j in range(i + 1, len(prop_facts)):
                        f1, f2 = prop_facts[i], prop_facts[j]
                        if f1.variant_scope == f2.variant_scope and f1.conditions == f2.conditions:
                            if (
                                f1.value != f2.value
                                and f1.extraction_status != ExtractionStatus.rejected
                                and f2.extraction_status != ExtractionStatus.rejected
                            ):
                                conflicts.append({
                                    "conflict_type": "contradictory_value",
                                    "product_id": pid,
                                    "property": prop,
                                    "fact_id_a": f1.fact_id,
                                    "fact_id_b": f2.fact_id,
                                    "value_a": f1.value,
                                    "value_b": f2.value,
                                    "reason": f"Conflicting values ({f1.value} vs {f2.value}) under identical variant/condition scope",
                                })

        # 3. Sanity check: min > max within property groups
        pids = {f.product_id for f in facts}
        for pid in pids:
            p_facts = {f.property: f for f in facts if f.product_id == pid}
            # Supply voltage
            vmin = p_facts.get("supply_voltage_min_v")
            vmax = p_facts.get("supply_voltage_max_v")
            if vmin and vmax and isinstance(vmin.value, (int, float)) and isinstance(vmax.value, (int, float)):
                if vmin.value > vmax.value:
                    conflicts.append({
                        "conflict_type": "invalid_range",
                        "product_id": pid,
                        "property": "supply_voltage",
                        "reason": f"supply_voltage_min_v ({vmin.value}) > supply_voltage_max_v ({vmax.value})",
                    })
            # Operating temperature
            tmin = p_facts.get("operating_temp_min_c")
            tmax = p_facts.get("operating_temp_max_c")
            if tmin and tmax and isinstance(tmin.value, (int, float)) and isinstance(tmax.value, (int, float)):
                if tmin.value > tmax.value:
                    conflicts.append({
                        "conflict_type": "invalid_range",
                        "product_id": pid,
                        "property": "operating_temp",
                        "reason": f"operating_temp_min_c ({tmin.value}) > operating_temp_max_c ({tmax.value})",
                    })

        return conflicts

    # --------------------------------------------------------------------------
    # Ports
    # --------------------------------------------------------------------------

    def save_ports(self, ports: List[Port]) -> None:
        """Bulk upserts physical/electrical ports within a single atomic transaction."""
        if not ports:
            return
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            with conn:
                for port in ports:
                    conn.execute(
                        """
                        INSERT INTO ports (
                            port_id, product_id, direction, physical_interface, signal_kind,
                            protocol, role, supported_ranges_json, wiring_conditions_json, evidence_ids_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(port_id) DO UPDATE SET
                            product_id=excluded.product_id,
                            direction=excluded.direction,
                            physical_interface=excluded.physical_interface,
                            signal_kind=excluded.signal_kind,
                            protocol=excluded.protocol,
                            role=excluded.role,
                            supported_ranges_json=excluded.supported_ranges_json,
                            wiring_conditions_json=excluded.wiring_conditions_json,
                            evidence_ids_json=excluded.evidence_ids_json
                        """,
                        (
                            port.port_id,
                            port.product_id,
                            port.direction,
                            port.physical_interface,
                            port.signal_kind,
                            port.protocol,
                            port.role,
                            json.dumps(port.supported_ranges, ensure_ascii=False),
                            json.dumps(port.wiring_conditions, ensure_ascii=False),
                            json.dumps(port.evidence_ids, ensure_ascii=False),
                        ),
                    )
        finally:
            if close_needed:
                conn.close()

    def save_port(self, port: Port) -> None:
        """Upserts a physical/electrical port."""
        self.save_ports([port])

    def _row_to_port(self, row: sqlite3.Row) -> Port:
        return Port(
            port_id=row["port_id"],
            product_id=row["product_id"],
            direction=row["direction"],
            physical_interface=row["physical_interface"],
            signal_kind=row["signal_kind"],
            protocol=row["protocol"],
            role=row["role"],
            supported_ranges=json.loads(row["supported_ranges_json"]),
            wiring_conditions=json.loads(row["wiring_conditions_json"]),
            evidence_ids=json.loads(row["evidence_ids_json"]),
        )

    def get_port(self, port_id: str) -> Optional[Port]:
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            row = conn.execute("SELECT * FROM ports WHERE port_id = ?", (port_id,)).fetchone()
            if row:
                return self._row_to_port(row)
            return None
        finally:
            if close_needed:
                conn.close()

    def get_ports_by_product(self, product_id: str) -> List[Port]:
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            rows = conn.execute("SELECT * FROM ports WHERE product_id = ? ORDER BY port_id", (product_id,)).fetchall()
            return [self._row_to_port(r) for r in rows]
        finally:
            if close_needed:
                conn.close()

    def list_all_ports(self) -> List[Port]:
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            rows = conn.execute("SELECT * FROM ports ORDER BY product_id, port_id").fetchall()
            return [self._row_to_port(r) for r in rows]
        finally:
            if close_needed:
                conn.close()

    def find_ports(
        self,
        product_id: Optional[str] = None,
        physical_interface: Optional[str] = None,
        signal_kind: Optional[str] = None,
        direction: Optional[str] = None,
        voltage_nominal: Optional[float] = None,
        voltage_min: Optional[float] = None,
        voltage_max: Optional[float] = None,
    ) -> List[Port]:
        """Filters ports by interface, signal kind, direction, and voltage thresholds."""
        query = "SELECT * FROM ports WHERE 1=1"
        params: List[Any] = []
        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)
        if physical_interface:
            query += " AND physical_interface = ?"
            params.append(physical_interface)
        if signal_kind:
            query += " AND signal_kind = ?"
            params.append(signal_kind)
        if direction:
            query += " AND direction = ?"
            params.append(direction)
        query += " ORDER BY product_id, port_id"

        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            rows = conn.execute(query, params).fetchall()
            ports = [self._row_to_port(r) for r in rows]
        finally:
            if close_needed:
                conn.close()

        if voltage_nominal is None and voltage_min is None and voltage_max is None:
            return ports

        matching: List[Port] = []
        for port in ports:
            ranges = port.supported_ranges or []
            matches = False
            for r in ranges:
                if not isinstance(r, dict):
                    continue
                v_nom = r.get("nominal_v") or r.get("loop_nominal_v")
                v_min = r.get("min_v") or r.get("loop_min_v")
                v_max = r.get("max_v") or r.get("loop_max_v")

                v_ok = True
                if voltage_nominal is not None:
                    if v_nom is not None and abs(v_nom - voltage_nominal) > 1e-3:
                        v_ok = False
                if voltage_min is not None:
                    if v_min is not None and v_min > voltage_min:
                        v_ok = False
                if voltage_max is not None:
                    if v_max is not None and v_max < voltage_max:
                        v_ok = False
                if v_ok and (v_nom is not None or v_min is not None or v_max is not None):
                    matches = True
                    break
            if matches:
                matching.append(port)
        return matching

    # --------------------------------------------------------------------------
    # Relations
    # --------------------------------------------------------------------------

    def save_relations(self, relations: List[Relation]) -> None:
        """Bulk upserts relations within a single atomic transaction."""
        if not relations:
            return
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            with conn:
                for relation in relations:
                    conn.execute(
                        """
                        INSERT INTO relations (
                            relation_id, source_id, target_id, relation_type,
                            conditions_json, evidence_ids_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(relation_id) DO UPDATE SET
                            source_id=excluded.source_id,
                            target_id=excluded.target_id,
                            relation_type=excluded.relation_type,
                            conditions_json=excluded.conditions_json,
                            evidence_ids_json=excluded.evidence_ids_json
                        """,
                        (
                            relation.relation_id,
                            relation.source_id,
                            relation.target_id,
                            relation.relation_type.value if hasattr(relation.relation_type, "value") else str(relation.relation_type),
                            json.dumps(relation.conditions, ensure_ascii=False),
                            json.dumps(relation.evidence_ids, ensure_ascii=False),
                        ),
                    )
        finally:
            if close_needed:
                conn.close()

    def save_relation(self, relation: Relation) -> None:
        """Upserts a knowledge graph relation edge."""
        self.save_relations([relation])

    def _row_to_relation(self, row: sqlite3.Row) -> Relation:
        rel_type_str = row["relation_type"]
        try:
            rel_type = RelationType(rel_type_str)
        except (ValueError, KeyError):
            rel_type = RelationType.HAS_PORT
        return Relation(
            relation_id=row["relation_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=rel_type,
            conditions=json.loads(row["conditions_json"]),
            evidence_ids=json.loads(row["evidence_ids_json"]),
        )

    def get_relations(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        rel_type: Optional[RelationType | str] = None,
    ) -> List[Relation]:
        query = "SELECT * FROM relations WHERE 1=1"
        params: List[Any] = []
        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)
        if target_id:
            query += " AND target_id = ?"
            params.append(target_id)
        if rel_type:
            val = rel_type.value if hasattr(rel_type, "value") else str(rel_type)
            query += " AND relation_type = ?"
            params.append(val)

        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_relation(r) for r in rows]
        finally:
            if close_needed:
                conn.close()

    def list_all_relations(self) -> List[Relation]:
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            rows = conn.execute("SELECT * FROM relations ORDER BY relation_id").fetchall()
            return [self._row_to_relation(r) for r in rows]
        finally:
            if close_needed:
                conn.close()

    # --------------------------------------------------------------------------
    # Spans
    # --------------------------------------------------------------------------

    def save_spans(self, spans: List[SourceSpan]) -> None:
        """Bulk upserts source spans within a single atomic transaction."""
        if not spans:
            return
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            with conn:
                for span in spans:
                    conn.execute(
                        """
                        INSERT INTO spans (
                            span_id, document_id, document_sha256, pdf_page_index, printed_page_label,
                            text, bbox_json, table_id, row_header, column_header, product_scope_json, revision
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(span_id) DO UPDATE SET
                            document_id=excluded.document_id,
                            document_sha256=excluded.document_sha256,
                            pdf_page_index=excluded.pdf_page_index,
                            printed_page_label=excluded.printed_page_label,
                            text=excluded.text,
                            bbox_json=excluded.bbox_json,
                            table_id=excluded.table_id,
                            row_header=excluded.row_header,
                            column_header=excluded.column_header,
                            product_scope_json=excluded.product_scope_json,
                            revision=excluded.revision
                        """,
                        (
                            span.span_id,
                            span.document_id,
                            span.document_sha256,
                            span.pdf_page_index,
                            span.printed_page_label,
                            span.text,
                            json.dumps(span.bbox, ensure_ascii=False) if span.bbox is not None else None,
                            span.table_id,
                            span.row_header,
                            span.column_header,
                            json.dumps(span.product_scope, ensure_ascii=False),
                            span.revision,
                        ),
                    )
        finally:
            if close_needed:
                conn.close()

    def save_span(self, span: SourceSpan) -> None:
        """Upserts a source span."""
        self.save_spans([span])

    def _row_to_span(self, row: sqlite3.Row) -> SourceSpan:
        bbox = json.loads(row["bbox_json"]) if row["bbox_json"] is not None else None
        return SourceSpan(
            span_id=row["span_id"],
            document_id=row["document_id"],
            document_sha256=row["document_sha256"],
            pdf_page_index=row["pdf_page_index"],
            printed_page_label=row["printed_page_label"],
            text=row["text"],
            bbox=bbox,
            table_id=row["table_id"],
            row_header=row["row_header"],
            column_header=row["column_header"],
            product_scope=json.loads(row["product_scope_json"]),
            revision=row["revision"],
        )

    def get_span(self, span_id: str) -> Optional[SourceSpan]:
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            row = conn.execute("SELECT * FROM spans WHERE span_id = ?", (span_id,)).fetchone()
            if row:
                return self._row_to_span(row)
            return None
        finally:
            if close_needed:
                conn.close()

    def get_spans(self, span_ids: List[str]) -> List[SourceSpan]:
        if not span_ids:
            return []
        chunk_size = 500
        results: List[SourceSpan] = []
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            for i in range(0, len(span_ids), chunk_size):
                chunk = span_ids[i:i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(f"SELECT * FROM spans WHERE span_id IN ({placeholders})", chunk).fetchall()
                results.extend(self._row_to_span(r) for r in rows)
            return results
        finally:
            if close_needed:
                conn.close()

    def get_document_page_spans(self, document_id: str, page_index: int) -> List[SourceSpan]:
        conn = self._get_connection()
        close_needed = (self._mem_conn is None)
        try:
            rows = conn.execute(
                "SELECT * FROM spans WHERE document_id = ? AND pdf_page_index = ? ORDER BY span_id",
                (document_id, page_index),
            ).fetchall()
            return [self._row_to_span(r) for r in rows]
        finally:
            if close_needed:
                conn.close()

    # --------------------------------------------------------------------------
    # Export / Import JSON
    # --------------------------------------------------------------------------

    def export_json(self, json_path: Optional[Path | str] = None) -> Path:
        target = Path(json_path) if json_path else DEFAULT_JSON_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "products": self.list_products(),
            "facts": [f.to_dict() for f in self.list_all_facts()],
            "ports": [p.to_dict() for p in self.list_all_ports()],
            "relations": [r.to_dict() for r in self.list_all_relations()],
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return target


def build_knowledge_store(
    catalog_path: Optional[Path] = None,
    reviewed_facts_path: Optional[Path] = None,
    spans_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> KnowledgeStore:
    """Builds and populates the KnowledgeStore from catalog, facts, and spans (§22)."""
    cat_file = catalog_path or (DEFAULT_DATA_DIR / "manifests" / "catalog.yaml")
    rev_file = reviewed_facts_path or (DEFAULT_DATA_DIR / "facts" / "facts.reviewed.jsonl")
    spans_file = spans_path or (DEFAULT_DATA_DIR / "pages" / "spans.jsonl")

    # If reviewed facts file doesn't exist, check auto facts
    if not rev_file.is_file():
        auto_file = DEFAULT_DATA_DIR / "facts" / "facts.auto.jsonl"
        if auto_file.is_file():
            logger.info("Using facts.auto.jsonl as fallback for build_knowledge_store")
            rev_file = auto_file

    store = KnowledgeStore(db_path=db_path)

    # 1. Ingest Catalog
    if cat_file.is_file():
        with open(cat_file, "r", encoding="utf-8") as f:
            cat_data = yaml.safe_load(f)
        store.save_catalog(cat_data)

    # 2. Ingest Spans
    if spans_file.is_file():
        with open(spans_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    store.save_span(SourceSpan.from_json(line.strip()))

    # 3. Ingest Facts
    if rev_file.is_file():
        with open(rev_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    store.save_fact(Fact.from_json(line.strip()))

    # 4. Ingest Ports & Relations for 3 products (§5.3)
    # Ports for P1 (PLC)
    p1_com1 = Port(
        port_id="P1_PORT_COM1",
        product_id="P1",
        direction="bidirectional",
        physical_interface="RS-485",
        signal_kind="serial",
        protocol="Modbus RTU",
        role="master_slave",
        supported_ranges=[{"baud_rates": [9600, 19200, 38400, 57600, 115200]}],
        wiring_conditions=["2-wire half-duplex"],
        evidence_ids=["D1_P1_MANUAL:p03:s02"],
    )
    p1_analog_in = Port(
        port_id="P1_PORT_ANALOG_IN",
        product_id="P1",
        direction="in",
        physical_interface="screw_terminal",
        signal_kind="current_loop",
        protocol=None,
        role="receiver",
        supported_ranges=[{"min_ma": 4.0, "max_ma": 20.0, "channels": 2}],
        wiring_conditions=["internal 250 ohm shunt"],
        evidence_ids=["D1_P1_MANUAL:p04:s02"],
    )
    p1_power_in = Port(
        port_id="P1_PORT_PWR_IN",
        product_id="P1",
        direction="in",
        physical_interface="screw_terminal",
        signal_kind="power_dc",
        protocol=None,
        role="power_sink",
        supported_ranges=[{"nominal_v": 24.0, "min_v": 20.4, "max_v": 28.8, "max_w": 15.0}],
        wiring_conditions=["polarity_protected"],
        evidence_ids=["D1_P1_MANUAL:p02:s02"],
    )

    # Ports for P2 (Power Supply)
    p2_power_out = Port(
        port_id="P2_PORT_PWR_OUT",
        product_id="P2",
        direction="out",
        physical_interface="screw_terminal",
        signal_kind="power_dc",
        protocol=None,
        role="power_source",
        supported_ranges=[{"nominal_v": 24.0, "min_v": 24.0, "max_v": 28.0, "max_a": 5.0, "max_w": 120.0}],
        wiring_conditions=["short_circuit_protected"],
        evidence_ids=["D2_P2_DATASHEET:p01:s02"],
    )

    # Ports for P3 (Temperature Sensor)
    p3_analog_loop = Port(
        port_id="P3_PORT_ANALOG_LOOP",
        product_id="P3",
        direction="out",
        physical_interface="screw_terminal",
        signal_kind="current_loop",
        protocol=None,
        role="transmitter",
        supported_ranges=[{"min_ma": 4.0, "max_ma": 20.0, "loop_min_v": 12.0, "loop_max_v": 30.0}],
        wiring_conditions=["2-wire loop powered"],
        evidence_ids=["D3_P3_DATASHEET:p01:s04"],
    )

    ports = [p1_com1, p1_analog_in, p1_power_in, p2_power_out, p3_analog_loop]
    store.save_ports(ports)

    # Standard relations:
    # Product HAS_PORT
    relations = [
        Relation(
            relation_id="REL_P1_HAS_COM1",
            source_id="P1",
            target_id="P1_PORT_COM1",
            relation_type=RelationType.HAS_PORT,
            evidence_ids=["D1_P1_MANUAL:p03:s02"],
        ),
        Relation(
            relation_id="REL_P1_HAS_ANALOG_IN",
            source_id="P1",
            target_id="P1_PORT_ANALOG_IN",
            relation_type=RelationType.HAS_PORT,
            evidence_ids=["D1_P1_MANUAL:p04:s02"],
        ),
        Relation(
            relation_id="REL_P1_HAS_PWR_IN",
            source_id="P1",
            target_id="P1_PORT_PWR_IN",
            relation_type=RelationType.HAS_PORT,
            evidence_ids=["D1_P1_MANUAL:p02:s02"],
        ),
        Relation(
            relation_id="REL_P2_HAS_PWR_OUT",
            source_id="P2",
            target_id="P2_PORT_PWR_OUT",
            relation_type=RelationType.HAS_PORT,
            evidence_ids=["D2_P2_DATASHEET:p01:s02"],
        ),
        Relation(
            relation_id="REL_P3_HAS_ANALOG_LOOP",
            source_id="P3",
            target_id="P3_PORT_ANALOG_LOOP",
            relation_type=RelationType.HAS_PORT,
            evidence_ids=["D3_P3_DATASHEET:p01:s04"],
        ),
        # P1 SUPPORTS_PROTOCOL Modbus RTU
        Relation(
            relation_id="REL_P1_SUPPORTS_MODBUS",
            source_id="P1_PORT_COM1",
            target_id="PROTOCOL_MODBUS_RTU",
            relation_type=RelationType.SUPPORTS_PROTOCOL,
            evidence_ids=["D1_P1_MANUAL:p03:s03"],
        ),
        # P1 and P3 SUPPORTS_SIGNAL current loop
        Relation(
            relation_id="REL_P1_SUPPORTS_420MA",
            source_id="P1_PORT_ANALOG_IN",
            target_id="SIGNAL_4_20MA",
            relation_type=RelationType.SUPPORTS_SIGNAL,
            evidence_ids=["D1_P1_MANUAL:p04:s02"],
        ),
        Relation(
            relation_id="REL_P3_SUPPORTS_420MA",
            source_id="P3_PORT_ANALOG_LOOP",
            target_id="SIGNAL_4_20MA",
            relation_type=RelationType.SUPPORTS_SIGNAL,
            evidence_ids=["D3_P3_DATASHEET:p01:s04"],
        ),
    ]
    store.save_relations(relations)

    logger.info("Knowledge store successfully built and populated.")
    return store
