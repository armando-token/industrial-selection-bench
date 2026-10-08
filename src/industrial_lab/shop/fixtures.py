"""Fixtures and catalog loader for the Shop Simulator.

Follows MEGAPLAN.md §6:
- Loads catalog from data/manifests/catalog.yaml
- Loads commercial data from active scenario (default S0001)
- Manages active scenario state in memory and provides scenario validation
- Generates synthetic PDF documents or serves frozen PDFs from data/raw/
- Separates public discovery catalog (NO curated technical facts) from internal models
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


# Default fallback catalog if YAML is absent or incomplete
DEFAULT_CATALOG: Dict[str, Any] = {
    "catalog_id": "controlnautas-three-products-v1",
    "schema_version": "1.0",
    "data_origin": "real_user_document",
    "official_status": "NON-OFFICIAL",
    "is_official": False,
    "products": [
        {
            "product_id": "P_X4",
            "sku": "HE-X4",
            "manufacturer": "Horner",
            "exact_model": "HE-X4 Micro OCS",
            "variant": "HE-X4A, HE-X4R",
            "name": "Horner HE-X4 Micro OCS",
            "description": "Horner X4 Micro OCS all-in-one controller with integrated I/O and display.",
            "documents": [
                {
                    "document_id": "D_X4_MANUAL_MAN1137",
                    "filename": "horner_x4_user_manual_MAN1137_HE-X4A_HE-X4R.pdf",
                    "title": "Horner X4 User Manual MAN1137",
                    "source_kind": "manual",
                    "mime_type": "application/pdf",
                    "page_count": 145,
                    "sha256": "35c647aa6b1d8595a7f674e0d203ad77023a27d7ee8fd70c83e1bb3f53d0f649",
                },
                {
                    "document_id": "D_X4_DATASHEET_MAN1138",
                    "filename": "horner_x4_micro_ocs_datasheet_MAN1138_R21.pdf",
                    "title": "Horner X4 Micro OCS Datasheet MAN1138",
                    "source_kind": "datasheet",
                    "mime_type": "application/pdf",
                    "page_count": 27,
                    "sha256": "7885574524f1c23cd687a442beb690f45142420913b2684dd464293aa3878f40",
                },
            ],
        },
        {
            "product_id": "P_THT",
            "sku": "THT-02",
            "manufacturer": "TZ",
            "exact_model": "THT-02",
            "variant": "Modbus-RTU SHT30",
            "name": "TZ THT-02 Temperature and Humidity Sensor",
            "description": "TZ THT-02 Modbus-RTU temperature and humidity transmitter with SHT30 element.",
            "documents": [
                {
                    "document_id": "D_THT_MANUAL",
                    "filename": "tz_tht02_temp_humidity_sensor_user_manual.pdf",
                    "title": "TZ THT-02 User Manual",
                    "source_kind": "manual",
                    "mime_type": "application/pdf",
                    "page_count": 9,
                    "sha256": "48718e71596413492f2a871e520c1567e6fe1e8abe05b89822a0f21e091e6a7c",
                },
            ],
        },
        {
            "product_id": "P_UHEAT",
            "sku": "66661",
            "manufacturer": "Pumphouse",
            "exact_model": "U Series Heater",
            "variant": "Item #66661",
            "name": "Pumphouse U Series Heater",
            "description": "Pumphouse U Series enclosure heater with built-in snap-action thermostat.",
            "documents": [
                {
                    "document_id": "D_UHEAT_DATASHEET",
                    "filename": "pumphouse_heater_u_series_datasheet_en.pdf",
                    "title": "Pumphouse Heater U Series Datasheet",
                    "source_kind": "datasheet",
                    "mime_type": "application/pdf",
                    "page_count": 1,
                    "sha256": "7a0517e6f5bc7e6427236436b829412e82da38ac7ac935d39250518e39dc4ce9",
                },
                {
                    "document_id": "D_UHEAT_INSTALL_66661",
                    "filename": "pumphouse_heater_u_series_install_sheet_item66661.pdf",
                    "title": "Pumphouse Heater U Series Install Sheet Item 66661",
                    "source_kind": "manual",
                    "mime_type": "application/pdf",
                    "page_count": 4,
                    "sha256": "192542c508cae7a0fec7ad0fddd89c7a7a75861817b9e85013ed3ad9b1a93059",
                },
            ],
        },
    ],
}

DEFAULT_SCENARIO_S0001: Dict[str, Any] = {
    "scenario_id": "S0001",
    "revision": "rev-2026-10-02-001",
    "catalog_id": "controlnautas-three-products-v1",
    "description": "Standard baseline commercial scenario with full availability",
    "currency": "USD",
    "tax_rate": 0.0,
    "products": {
        "P_X4": {
            "price_minor": 89900,
            "currency": "USD",
            "stock_available": 8,
            "observed_at": "2026-10-02T12:00:00Z",
            "is_simulated": True,
        },
        "P_THT": {
            "price_minor": 4500,
            "currency": "USD",
            "stock_available": 40,
            "observed_at": "2026-10-02T12:00:00Z",
            "is_simulated": True,
        },
        "P_UHEAT": {
            "price_minor": 12900,
            "currency": "USD",
            "stock_available": 15,
            "observed_at": "2026-10-02T12:00:00Z",
            "is_simulated": True,
        },
        # Legacy entries preserved for offline synthetic unit tests
        "P1": {
            "price_minor": 125000,
            "currency": "EUR",
            "stock_available": 15,
            "observed_at": "2026-10-02T12:00:00Z",
            "is_simulated": True,
        },
        "P2": {
            "price_minor": 48000,
            "currency": "EUR",
            "stock_available": 8,
            "observed_at": "2026-10-02T12:00:00Z",
            "is_simulated": True,
        },
        "P3": {
            "price_minor": 31000,
            "currency": "EUR",
            "stock_available": 25,
            "observed_at": "2026-10-02T12:00:00Z",
            "is_simulated": True,
        },
    },
}

# In-memory active scenario state
_ACTIVE_SCENARIO_ID: str = "S0001"


def get_data_root() -> Path:
    """Resolve data root directory using LAB_DATA_ROOT or repository conventions."""
    env_root = os.environ.get("LAB_DATA_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.exists():
            return p

    # Local repo root convention: <repo>/data
    repo_data = Path(__file__).resolve().parents[3] / "data"
    if repo_data.exists():
        return repo_data

    # Docker container root /data
    container_data = Path("/data")
    if container_data.exists():
        return container_data

    cwd_data = Path.cwd() / "data"
    return cwd_data


def get_active_scenario_id() -> str:
    """Return currently active scenario ID."""
    global _ACTIVE_SCENARIO_ID
    return _ACTIVE_SCENARIO_ID


def set_active_scenario_id(scenario_id: str) -> str:
    """Set active scenario ID after validation against allowed scenarios."""
    global _ACTIVE_SCENARIO_ID
    allowed = list_allowed_scenarios()
    if scenario_id not in allowed:
        raise ValueError(
            f"Scenario '{scenario_id}' is not allowed. Available scenarios: {allowed}"
        )
    _ACTIVE_SCENARIO_ID = scenario_id
    return _ACTIVE_SCENARIO_ID


def reset_fixtures_state() -> None:
    """Reset in-memory state to defaults (useful for tests)."""
    global _ACTIVE_SCENARIO_ID
    _ACTIVE_SCENARIO_ID = "S0001"


def list_allowed_scenarios() -> List[str]:
    """List valid scenario IDs found in data/scenarios/ plus defaults."""
    scenarios = set()
    scenarios.add("S0001")

    scenarios_dir = get_data_root() / "scenarios"
    if scenarios_dir.exists() and scenarios_dir.is_dir():
        for f in scenarios_dir.iterdir():
            if f.is_file() and f.suffix in {".yaml", ".yml"}:
                scenarios.add(f.stem)

    return sorted(scenarios)


def load_catalog() -> Dict[str, Any]:
    """Load catalog from data/manifests/catalog.yaml, falling back to default."""
    catalog_path = get_data_root() / "manifests" / "catalog.yaml"
    if catalog_path.exists():
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "products" in data:
                    for p in data["products"]:
                        if not p.get("name"):
                            mfg = p.get("manufacturer") or ""
                            model = p.get("exact_model") or ""
                            p["name"] = f"{mfg} {model}".strip() or p.get("product_id", "")
                        if not p.get("description"):
                            p["description"] = f"{p.get('name')} - Industrial component {p.get('product_id')}"
                    return data
        except Exception:
            pass
    return DEFAULT_CATALOG


def load_scenario(scenario_id: Optional[str] = None) -> Dict[str, Any]:
    """Load commercial scenario by ID, falling back to built-in S0001."""
    target_id = scenario_id or get_active_scenario_id()
    scenario_path = get_data_root() / "scenarios" / f"{target_id}.yaml"
    if not scenario_path.exists():
        scenario_path = get_data_root() / "scenarios" / f"{target_id}.yml"

    if scenario_path.exists():
        try:
            with open(scenario_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    # Ensure scenario_id and revision are set (support both revision and commerce_revision)
                    rev = data.get("revision") or data.get("commerce_revision") or f"rev-{target_id}"
                    data["scenario_id"] = target_id
                    data["revision"] = rev
                    # Normalize products mapping
                    raw_prods = data.get("products", {})
                    norm_prods: Dict[str, Dict[str, Any]] = {}
                    if isinstance(raw_prods, list):
                        for item in raw_prods:
                            pid = item.get("product_id")
                            if pid:
                                norm_prods[pid] = dict(item)
                    elif isinstance(raw_prods, dict):
                        norm_prods = {k: dict(v) for k, v in raw_prods.items()}

                    # Set standard defaults for each product
                    for pid, pdata in norm_prods.items():
                        pdata.setdefault("product_id", pid)
                        pdata.setdefault("currency", data.get("currency", "EUR"))
                        pdata.setdefault("revision", rev)
                        pdata.setdefault("is_simulated", True)
                        pdata.setdefault(
                            "observed_at", datetime.now(timezone.utc).isoformat()
                        )
                    data["products"] = norm_prods
                    return data
        except Exception:
            pass

    # Fallback to default S0001
    return DEFAULT_SCENARIO_S0001


def get_active_revision() -> str:
    """Return revision string of current active scenario."""
    scen = load_scenario()
    return scen.get("revision", "rev-unknown")


def get_product(product_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve product details from catalog."""
    catalog = load_catalog()
    for prod in catalog.get("products", []):
        if prod.get("product_id") == product_id:
            return prod
    return None


def load_synthetic_catalog() -> Dict[str, Any]:
    """Load preserved FixtureCorp synthetic catalog (pipeline unit tests only)."""
    root = get_data_root()
    for candidate in (
        root / "manifests" / "catalog.synthetic.yaml",
        root / "synthetic_fixtures" / "manifests" / "catalog.yaml",
        Path(__file__).resolve().parents[3] / "data" / "manifests" / "catalog.synthetic.yaml",
    ):
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if isinstance(data, dict) and data.get("products"):
                    return data
            except Exception:
                continue
    return {"catalog_id": "synthetic-fallback", "data_origin": "synthetic_fixture", "products": []}


def resolve_product(product_id: str) -> Optional[Dict[str, Any]]:
    """Resolve product from live catalog, else synthetic fixture catalog (tests/offline)."""
    live = get_product(product_id)
    if live is not None:
        return live
    syn = load_synthetic_catalog()
    for prod in syn.get("products", []):
        if prod.get("product_id") == product_id:
            return prod
    return None


def get_commerce(product_id: str, scenario_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve commercial data (price, stock, revision) for product in active scenario."""
    scen = load_scenario(scenario_id)
    prods = scen.get("products", {})
    if product_id in prods:
        item = dict(prods[product_id])
        item.setdefault("product_id", product_id)
        item.setdefault("currency", scen.get("currency", "EUR"))
        item.setdefault("revision", scen.get("revision", "rev-unknown"))
        item.setdefault("is_simulated", True)
        item.setdefault("observed_at", datetime.now(timezone.utc).isoformat())
        return item
    return None


def get_catalog_public() -> Dict[str, Any]:
    """Build public catalog payload for GET /api/catalog.

    MEGAPLAN.md §6.1 / §3.1:
    'Identidad, SKU, nombre y URLs; sin hechos técnicos curados'.
    Explicitly strips internal curated facts, rules, and extraction graphs.
    """
    catalog = load_catalog()
    scen = load_scenario()
    public_products: List[Dict[str, Any]] = []

    for p in catalog.get("products", []):
        pid = p["product_id"]
        docs = []
        for d in p.get("documents", []):
            docs.append(
                {
                    "document_id": d["document_id"],
                    "title": d.get("title", d["document_id"]),
                    "filename": d.get("filename"),
                    "source_kind": d.get("source_kind", "document"),
                    "url": f"/documents/{d['document_id']}.pdf",
                }
            )

        public_products.append(
            {
                "product_id": pid,
                "sku": p.get("sku"),
                "name": p.get("name")
                or f"{p.get('manufacturer', '')} {p.get('exact_model', '')}".strip(),
                "manufacturer": p.get("manufacturer"),
                "exact_model": p.get("exact_model"),
                "variant": p.get("variant"),
                "url": f"/products/{pid}",
                "documents": docs,
            }
        )

    return {
        "catalog_id": catalog.get("catalog_id", "catalog-v1"),
        "scenario_id": scen.get("scenario_id", get_active_scenario_id()),
        "revision": scen.get("revision", "rev-unknown"),
        "products": public_products,
    }


def generate_synthetic_pdf(document_id: str, title: str = "Simulated Industrial Document") -> bytes:
    """Generate minimal valid PDF bytes for synthetic datasheet/manual."""
    safe_title = "".join(c for c in title if 32 <= ord(c) <= 126).replace("(", "[").replace(")", "]")
    safe_id = "".join(c for c in document_id if 32 <= ord(c) <= 126).replace("(", "[").replace(")", "]")

    lines = [
        f"BT /F1 18 Tf 50 720 Td ({safe_title}) Tj ET",
        f"BT /F1 12 Tf 50 680 Td (Document ID: {safe_id}) Tj ET",
        "BT /F1 10 Tf 50 650 Td (DOCUMENTO TECNICO DE SIMULACION INDUSTRIAL - MEGAPLAN S6) Tj ET",
        "BT /F1 10 Tf 50 630 Td (The AI Commerce Gallery Hackathon 2026) Tj ET",
        "BT /F1 10 Tf 50 600 Td (Aviso: Contenido congelado para benchmark y seleccion de equipos.) Tj ET",
    ]
    stream_content = "\n".join(lines) + "\n"
    stream_bytes = stream_content.encode("latin1")
    stream_len = len(stream_bytes)

    header = b"%PDF-1.4\n"
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = (
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    )
    obj4 = (
        f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("latin1")
        + stream_bytes
        + b"endstream\nendobj\n"
    )
    obj5 = b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    offsets = [0]
    curr = len(header)
    offsets.append(curr)
    curr += len(obj1)
    offsets.append(curr)
    curr += len(obj2)
    offsets.append(curr)
    curr += len(obj3)
    offsets.append(curr)
    curr += len(obj4)
    offsets.append(curr)
    curr += len(obj5)

    xref = f"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n"

    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{curr}\n%%EOF\n"

    return header + obj1 + obj2 + obj3 + obj4 + obj5 + xref.encode("latin1") + trailer.encode("latin1")


def get_document_bytes(document_id: str) -> bytes:
    """Retrieve raw document bytes from data/raw/ or generate synthetic PDF."""
    raw_dir = get_data_root() / "raw"
    candidates = [
        raw_dir / f"{document_id}.pdf",
        raw_dir / document_id,
    ]

    # Also search catalog for matching filename
    catalog = load_catalog()
    for p in catalog.get("products", []):
        for doc in p.get("documents", []):
            if doc.get("document_id") == document_id:
                fn = doc.get("filename")
                if fn:
                    candidates.append(raw_dir / fn)
                break

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            try:
                return candidate.read_bytes()
            except Exception:
                pass

    # Fallback to valid synthetic PDF
    title = f"Documento Tecnico {document_id}"
    for p in catalog.get("products", []):
        for doc in p.get("documents", []):
            if doc.get("document_id") == document_id:
                title = doc.get("title", title)
                break

    return generate_synthetic_pdf(document_id, title)
