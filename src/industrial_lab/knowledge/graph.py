"""NetworkX graph representation of industrial products, ports, and relations.

Adheres strictly to MEGAPLAN.md §3.2, §5.3:
- NetworkX DiGraph representation of products, physical/electrical ports, and relations.
- Restricted relation types: HAS_PORT, REQUIRES, EXCLUDES, SUPPORTS_SIGNAL,
  SUPPORTS_PROTOCOL, REQUIRES_ACCESSORY, CONDITION_APPLIES_TO, COMPATIBLE_WITH.
- COMPATIBLE_WITH edges are only present if explicitly stated by authoritative source.
- Pathfinding discovers candidate connections; verification is left to deterministic rule engines.
- Graph export and persistence in portable JSON format.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from industrial_lab.schemas import Port, Relation, RelationType
from industrial_lab.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(os.environ.get("LAB_DATA_ROOT", "data"))
DEFAULT_GRAPH_FILE = DEFAULT_DATA_DIR / "graph" / "knowledge_graph.json"


class KnowledgeGraph:
    """Directed knowledge graph for industrial component compatibility modeling."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._port_objects: Dict[str, Port] = {}

    # --------------------------------------------------------------------------
    # Node & Edge Additions
    # --------------------------------------------------------------------------

    def add_product(self, product_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Adds a product node to the graph."""
        attrs = metadata or {}
        self.graph.add_node(product_id, node_type="product", **attrs)

    def add_port(self, port: Port) -> None:
        """Adds a port node and registers its attributes and Port object."""
        self._port_objects[port.port_id] = port
        self.graph.add_node(
            port.port_id,
            node_type="port",
            product_id=port.product_id,
            direction=port.direction,
            physical_interface=port.physical_interface,
            signal_kind=port.signal_kind,
            protocol=port.protocol,
            role=port.role,
            supported_ranges=port.supported_ranges,
            wiring_conditions=port.wiring_conditions,
            evidence_ids=port.evidence_ids,
        )
        # Ensure parent product node exists
        if not self.graph.has_node(port.product_id):
            self.add_product(port.product_id)
        # Add HAS_PORT relation edge automatically
        self.graph.add_edge(
            port.product_id,
            port.port_id,
            relation_type=RelationType.HAS_PORT.value,
            relation_id=f"HAS_PORT_{port.product_id}_{port.port_id}",
            evidence_ids=port.evidence_ids,
            conditions=[],
        )

    def add_relation(self, relation: Relation) -> None:
        """Adds a typed, directional relation edge adhering strictly to §5.3."""
        rel_type_val = (
            relation.relation_type.value
            if hasattr(relation.relation_type, "value")
            else str(relation.relation_type)
        )
        
        # Ensure nodes exist
        if not self.graph.has_node(relation.source_id):
            self.graph.add_node(relation.source_id, node_type="entity")
        if not self.graph.has_node(relation.target_id):
            self.graph.add_node(relation.target_id, node_type="entity")

        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=rel_type_val,
            relation_id=relation.relation_id,
            conditions=relation.conditions,
            evidence_ids=relation.evidence_ids,
        )

    # --------------------------------------------------------------------------
    # Queries & Traversal
    # --------------------------------------------------------------------------

    def get_port(self, port_id: str) -> Optional[Port]:
        return self._port_objects.get(port_id)

    def get_port_product(self, port_id: str) -> Optional[str]:
        node = self.graph.nodes.get(port_id)
        if node and node.get("node_type") == "port":
            return node.get("product_id")
        return None

    def get_product_ports(self, product_id: str) -> List[Port]:
        """Returns all Port objects associated with a product."""
        ports: List[Port] = []
        if not self.graph.has_node(product_id):
            return ports
        for successor in self.graph.successors(product_id):
            edge_data = self.graph.get_edge_data(product_id, successor)
            if edge_data and edge_data.get("relation_type") == RelationType.HAS_PORT.value:
                port = self._port_objects.get(successor)
                if port:
                    ports.append(port)
        return ports

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
        """Finds port objects in the graph matching interface, signal, direction, and voltage parameters."""
        ports = list(self._port_objects.values())
        if product_id:
            ports = [p for p in ports if p.product_id == product_id]
        if physical_interface:
            ports = [p for p in ports if p.physical_interface == physical_interface]
        if signal_kind:
            ports = [p for p in ports if p.signal_kind == signal_kind]
        if direction:
            ports = [p for p in ports if p.direction == direction]

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

    def find_candidate_connections(
        self,
        source_port_id: str,
        target_port_id: str,
    ) -> List[Dict[str, Any]]:
        """Finds candidate physical/electrical compatibility between two ports (§5.3).
        
        Note: Connectivity indicates candidacy, not guaranteed functional compatibility.
        """
        src = self._port_objects.get(source_port_id)
        tgt = self._port_objects.get(target_port_id)
        if not src or not tgt:
            return []

        candidates = []
        # Check directional pairing: out -> in, or bidirectional
        dir_ok = (
            (src.direction in ("out", "bidirectional") and tgt.direction in ("in", "bidirectional"))
            or (src.direction == "bidirectional" and tgt.direction == "bidirectional")
        )

        if not dir_ok:
            return []

        # Check signal match
        signal_ok = (
            src.signal_kind == tgt.signal_kind
            or "any" in (src.signal_kind, tgt.signal_kind)
        )

        # Check protocol match
        protocol_ok = True
        if src.protocol and tgt.protocol:
            protocol_ok = src.protocol.lower() == tgt.protocol.lower()

        # Check voltage compatibility if both have defined voltage ranges
        voltage_ok = True
        voltage_details = None
        src_v_ranges = [
            r for r in src.supported_ranges
            if isinstance(r, dict) and any(k in r for k in ("nominal_v", "min_v", "loop_min_v", "loop_nominal_v"))
        ]
        tgt_v_ranges = [
            r for r in tgt.supported_ranges
            if isinstance(r, dict) and any(k in r for k in ("nominal_v", "min_v", "loop_min_v", "loop_nominal_v"))
        ]
        if src_v_ranges and tgt_v_ranges:
            s_r = src_v_ranges[0]
            t_r = tgt_v_ranges[0]
            s_nom = s_r.get("nominal_v") or s_r.get("loop_nominal_v")
            t_nom = t_r.get("nominal_v") or t_r.get("loop_nominal_v")
            s_min = s_r.get("min_v") or s_r.get("loop_min_v")
            s_max = s_r.get("max_v") or s_r.get("loop_max_v")
            t_min = t_r.get("min_v") or t_r.get("loop_min_v")
            t_max = t_r.get("max_v") or t_r.get("loop_max_v")

            if s_nom is not None and t_nom is not None and abs(s_nom - t_nom) > 0.1:
                voltage_ok = False
            if s_max is not None and t_min is not None and s_max < t_min:
                voltage_ok = False
            if s_min is not None and t_max is not None and s_min > t_max:
                voltage_ok = False
            voltage_details = {
                "source_nominal_v": s_nom,
                "target_nominal_v": t_nom,
                "voltage_compatible": voltage_ok,
            }

        interface_match = (src.physical_interface == tgt.physical_interface)

        if dir_ok and signal_ok and protocol_ok and voltage_ok:
            cand = {
                "source_port_id": source_port_id,
                "target_port_id": target_port_id,
                "source_product_id": src.product_id,
                "target_product_id": tgt.product_id,
                "signal_kind": src.signal_kind,
                "protocol": src.protocol or tgt.protocol,
                "physical_interface_match": interface_match,
                "candidate": True,
            }
            if voltage_details:
                cand["voltage_details"] = voltage_details
            candidates.append(cand)

        return candidates

    def find_product_candidate_connections(
        self,
        source_product_id: str,
        target_product_id: str,
    ) -> List[Dict[str, Any]]:
        """Discovers all candidate port-to-port connections between two products."""
        src_ports = self.get_product_ports(source_product_id)
        tgt_ports = self.get_product_ports(target_product_id)
        candidates: List[Dict[str, Any]] = []
        for sp in src_ports:
            for tp in tgt_ports:
                conns = self.find_candidate_connections(sp.port_id, tp.port_id)
                candidates.extend(conns)
        return candidates

    def get_connected_components(self) -> List[Set[str]]:
        """Returns weakly connected components of the graph (safe for disconnected graphs)."""
        undir = self.graph.to_undirected()
        return [set(c) for c in nx.connected_components(undir)]

    def is_connected(self) -> bool:
        """Returns True if the entire graph is weakly connected."""
        if len(self.graph) <= 1:
            return True
        return nx.is_weakly_connected(self.graph)

    def find_compatibility_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """Finds candidate multi-hop connection paths between products or entities.
        
        Handles disconnected components gracefully without raising exceptions.
        """
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return []

        aug_graph = self.graph.copy()
        port_items = list(self._port_objects.items())
        for i in range(len(port_items)):
            for j in range(len(port_items)):
                if i != j:
                    p1_id, p1 = port_items[i]
                    p2_id, p2 = port_items[j]
                    if p1.product_id != p2.product_id:
                        conns = self.find_candidate_connections(p1_id, p2_id)
                        if conns:
                            aug_graph.add_edge(p1_id, p2_id, relation_type="CANDIDATE_CONNECTION")

        undir_aug = aug_graph.to_undirected()
        if not nx.has_path(undir_aug, source_id, target_id):
            return []

        paths: List[List[Dict[str, Any]]] = []
        try:
            for raw_path in nx.all_simple_paths(undir_aug, source_id, target_id, cutoff=max_hops):
                path_steps = []
                for idx in range(len(raw_path) - 1):
                    u = raw_path[idx]
                    v = raw_path[idx + 1]
                    edge_data = aug_graph.get_edge_data(u, v) or aug_graph.get_edge_data(v, u) or {}
                    path_steps.append({
                        "from": u,
                        "to": v,
                        "relation_type": edge_data.get("relation_type", "CONNECTED"),
                    })
                paths.append(path_steps)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        return paths

    def detect_graph_conflicts(self) -> List[Dict[str, Any]]:
        """Detects contradictory relations (EXCLUDES vs REQUIRES/COMPATIBLE_WITH) and clashes."""
        conflicts: List[Dict[str, Any]] = []

        # 1. Contradictory relations between nodes
        for u in self.graph.nodes():
            for v in self.graph.successors(u):
                edges_uv = self.graph.get_edge_data(u, v)
                rel_uv = edges_uv.get("relation_type") if edges_uv else None
                if self.graph.has_edge(v, u):
                    edges_vu = self.graph.get_edge_data(v, u)
                    rel_vu = edges_vu.get("relation_type") if edges_vu else None

                    if rel_uv == RelationType.EXCLUDES.value:
                        if rel_vu in (RelationType.REQUIRES.value, RelationType.REQUIRES_ACCESSORY.value):
                            conflicts.append({
                                "conflict_type": "exclusion_requirement_conflict",
                                "source_id": u,
                                "target_id": v,
                                "relation_uv": rel_uv,
                                "relation_vu": rel_vu,
                                "reason": f"{u} EXCLUDES {v}, but {v} REQUIRES {u}",
                            })
                        elif rel_vu == RelationType.COMPATIBLE_WITH.value:
                            conflicts.append({
                                "conflict_type": "exclusion_compatibility_conflict",
                                "source_id": u,
                                "target_id": v,
                                "relation_uv": rel_uv,
                                "relation_vu": rel_vu,
                                "reason": f"{u} EXCLUDES {v}, but {v} states COMPATIBLE_WITH {u}",
                            })

        # 2. Port direction / power role clash
        for pid_a, port_a in self._port_objects.items():
            for pid_b, port_b in self._port_objects.items():
                if pid_a < pid_b and port_a.product_id != port_b.product_id:
                    if port_a.signal_kind == "power_dc" and port_b.signal_kind == "power_dc":
                        if port_a.role == "power_source" and port_b.role == "power_source":
                            conflicts.append({
                                "conflict_type": "power_source_clash",
                                "port_a": pid_a,
                                "port_b": pid_b,
                                "reason": f"Both ports {pid_a} and {pid_b} are power sources (cannot connect source to source)",
                            })

        return conflicts

    def get_required_accessories(self, product_id: str) -> List[str]:
        """Returns list of accessory IDs required by product."""
        reqs = []
        if self.graph.has_node(product_id):
            for succ in self.graph.successors(product_id):
                edge = self.graph.get_edge_data(product_id, succ)
                if edge and edge.get("relation_type") == RelationType.REQUIRES_ACCESSORY.value:
                    reqs.append(succ)
        return reqs

    def get_exclusions(self, product_id: str) -> List[str]:
        """Returns list of products explicitly excluded by this product."""
        excl = []
        if self.graph.has_node(product_id):
            for succ in self.graph.successors(product_id):
                edge = self.graph.get_edge_data(product_id, succ)
                if edge and edge.get("relation_type") == RelationType.EXCLUDES.value:
                    excl.append(succ)
        return excl

    def check_direct_compatibility(self, source_id: str, target_id: str) -> Optional[Relation]:
        """Returns COMPATIBLE_WITH relation if explicitly documented."""
        if self.graph.has_edge(source_id, target_id):
            edge = self.graph.get_edge_data(source_id, target_id)
            if edge and edge.get("relation_type") == RelationType.COMPATIBLE_WITH.value:
                return Relation(
                    relation_id=edge.get("relation_id", f"COMPAT_{source_id}_{target_id}"),
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=RelationType.COMPATIBLE_WITH,
                    conditions=edge.get("conditions", []),
                    evidence_ids=edge.get("evidence_ids", []),
                )
        return None

    # --------------------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the graph structure and port objects to primitive dictionary."""
        nodes = []
        for n, data in self.graph.nodes(data=True):
            node_dict = dict(data)
            node_dict["id"] = n
            nodes.append(node_dict)

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edge_dict = dict(data)
            edge_dict["source"] = u
            edge_dict["target"] = v
            edges.append(edge_dict)

        ports = [p.to_dict() for p in self._port_objects.values()]

        return {
            "nodes": nodes,
            "edges": edges,
            "ports": ports,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Reconstructs KnowledgeGraph from dictionary."""
        kg = cls()
        for p_data in data.get("ports", []):
            port = Port.model_validate(p_data)
            kg.add_port(port)

        for n in data.get("nodes", []):
            nid = n.get("id")
            if not kg.graph.has_node(nid):
                attrs = {k: v for k, v in n.items() if k != "id"}
                kg.graph.add_node(nid, **attrs)

        for e in data.get("edges", []):
            u = e.get("source")
            v = e.get("target")
            attrs = {k: val for k, val in e.items() if k not in ("source", "target")}
            kg.graph.add_edge(u, v, **attrs)

        return kg

    def save_graph(self, file_path: Optional[Path | str] = None) -> Path:
        """Saves knowledge graph JSON to file."""
        p = Path(file_path) if file_path else DEFAULT_GRAPH_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved knowledge graph to {p}")
        return p

    @classmethod
    def load_graph(cls, file_path: Optional[Path | str] = None) -> "KnowledgeGraph":
        """Loads knowledge graph from JSON file."""
        p = Path(file_path) if file_path else DEFAULT_GRAPH_FILE
        if not p.is_file():
            raise FileNotFoundError(f"Knowledge graph file not found at: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def build_knowledge_graph(store: KnowledgeStore) -> KnowledgeGraph:
    """Builds and wires a KnowledgeGraph from an initialized KnowledgeStore."""
    kg = KnowledgeGraph()

    # 1. Add Products
    products = store.list_products()
    for prod in products:
        kg.add_product(
            product_id=prod["product_id"],
            metadata={
                "sku": prod.get("sku"),
                "manufacturer": prod.get("manufacturer"),
                "exact_model": prod.get("exact_model"),
                "variant": prod.get("variant"),
            },
        )

    # 2. Add Ports
    ports = store.list_all_ports()
    for port in ports:
        kg.add_port(port)

    # 3. Add Relations
    relations = store.list_all_relations()
    for rel in relations:
        kg.add_relation(rel)

    # Save to data/graph/knowledge_graph.json
    kg.save_graph()
    return kg
