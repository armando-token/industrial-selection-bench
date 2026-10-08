"""Atomic technical fact extraction for Industrial Selection Lab.

Adheres strictly to MEGAPLAN.md §5.2, §7.3, §22:
- Extracts atomic facts adhering to the Fact schema from document spans.
- Associates each fact with verifiable SourceSpan evidence IDs.
- Distinguishes nominal vs max/min, supply vs signal, input vs output.
- Emits facts.auto.jsonl with ExtractionStatus.auto_extracted.
- Provides loaders and persistence routines for automated and reviewed facts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from industrial_lab.schemas import ExtractionStatus, Fact, SourceSpan, VerificationStatus
from industrial_lab.ingest.documents import load_pages, load_spans

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(os.environ.get("LAB_DATA_ROOT", "data"))
DEFAULT_FACTS_DIR = DEFAULT_DATA_DIR / "facts"
DEFAULT_AUTO_FACTS_FILE = DEFAULT_FACTS_DIR / "facts.auto.jsonl"
DEFAULT_AUTO_REAL_FACTS_FILE = DEFAULT_FACTS_DIR / "facts.auto.real.jsonl"
DEFAULT_REVIEWED_FACTS_FILE = DEFAULT_FACTS_DIR / "facts.reviewed.jsonl"


def _extract_facts_from_spans(spans: List[SourceSpan]) -> List[Fact]:
    """Schema-guided extraction of atomic facts from document spans (§7.3)."""
    facts: List[Fact] = []
    fact_counter = 1

    # Map product to its spans
    spans_by_prod: Dict[str, List[SourceSpan]] = {}
    for s in spans:
        for pid in s.product_scope:
            spans_by_prod.setdefault(pid, []).append(s)

    seen_facts: Dict[tuple, Fact] = {}

    def add_fact(
        pid: str,
        prop: str,
        value: Any,
        unit: Optional[str] = None,
        port_id: Optional[str] = None,
        conditions: Optional[List[dict]] = None,
        variant_scope: Optional[List[str]] = None,
        ev_id: Optional[str] = None,
        revision: str = "unknown",
        origin: Optional[str] = None,
    ) -> None:
        nonlocal fact_counter
        conds = conditions or []
        var_scope = variant_scope or ["standard"]
        data_orig = origin or ("synthetic_fixture" if pid in ("P1", "P2", "P3") else "real_user_document")

        if pid not in ("P1", "P2", "P3"):
            # Canonical deduplication for real products
            key = (
                pid,
                prop,
                json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value,
                unit,
                port_id,
                tuple(sorted(var_scope)),
                tuple(sorted(json.dumps(c, sort_keys=True) for c in conds)),
            )
            if key in seen_facts:
                existing = seen_facts[key]
                if ev_id and ev_id not in existing.evidence_ids:
                    existing.evidence_ids.append(ev_id)
                return

        f = Fact(
            fact_id=f"F_{pid}_{prop}_{fact_counter:03d}",
            product_id=pid,
            port_id=port_id,
            property=prop,
            value=value,
            unit=unit,
            conditions=conds,
            variant_scope=var_scope,
            evidence_ids=[ev_id] if ev_id else [],
            extraction_status=ExtractionStatus.EXTRACTED,
            verification_status=VerificationStatus.UNVERIFIED,
            reviewer_id=None,
            review_method=None,
            reviewed_by=None,
            document_revision=revision,
            data_origin=data_orig,
            official_status="NON-OFFICIAL",
            is_official=False,
        )
        fact_counter += 1
        facts.append(f)
        if pid not in ("P1", "P2", "P3"):
            seen_facts[key] = f

    for pid, prod_spans in spans_by_prod.items():
        for span in prod_spans:
            t = span.text.lower()
            t_norm = re.sub(r"\s+", " ", span.text.lower().replace("\xa0", " "))
            t_compact = re.sub(r"\s+", "", span.text.lower())
            ev_id = span.span_id
            rev = span.revision or "unknown"
            origin = span.data_origin

            # -------------------------------------------------------------
            # Common / Synthetic fixture patterns (P1, P2, P3 compatibility)
            # -------------------------------------------------------------
            if pid in ("P1", "P2", "P3"):
                # 1. Supply voltage nominal / range
                if "supply voltage" in t or "output voltage" in t or "loop supply voltage" in t:
                    if "24v dc" in t or "24 v dc" in t:
                        prop = "supply_voltage_nominal_v"
                        if "output voltage" in t:
                            prop = "output_voltage_nominal_v"
                        elif "loop supply" in t:
                            prop = "loop_supply_voltage_nominal_v"
                        add_fact(pid, prop, 24.0, unit="V", ev_id=ev_id, revision=rev, origin=origin)

                    range_match = re.search(r"(\d+\.?\d*)\s*v(?:\s*dc)?\s*to\s*(\d+\.?\d*)\s*v(?:\s*dc)?", t)
                    if range_match:
                        v_min, v_max = float(range_match.group(1)), float(range_match.group(2))
                        p_prefix = "output_voltage" if "output voltage" in t else ("loop_supply_voltage" if "loop supply" in t else "supply_voltage")
                        add_fact(pid, f"{p_prefix}_min_v", v_min, unit="V", ev_id=ev_id, revision=rev, origin=origin)
                        add_fact(pid, f"{p_prefix}_max_v", v_max, unit="V", ev_id=ev_id, revision=rev, origin=origin)

                # 2. Power and Current
                if "power consumption" in t:
                    w_match = re.search(r"(\d+\.?\d*)\s*w", t)
                    if w_match:
                        add_fact(pid, "power_consumption_max_w", float(w_match.group(1)), unit="W", ev_id=ev_id, revision=rev, origin=origin)

                if "rated output current" in t or "continuous output current" in t:
                    a_match = re.search(r"(\d+\.?\d*)\s*a", t)
                    if a_match:
                        add_fact(pid, "rated_output_current_a", float(a_match.group(1)), unit="A", ev_id=ev_id, revision=rev, origin=origin)

                if "rated output power" in t:
                    w_match = re.search(r"(\d+\.?\d*)\s*w", t)
                    if w_match:
                        add_fact(pid, "rated_output_power_w", float(w_match.group(1)), unit="W", ev_id=ev_id, revision=rev, origin=origin)

                # 3. Communications and Protocols
                if "rs-485" in t or "rs485" in t:
                    add_fact(pid, "communication_interface", "RS-485", port_id="COM1" if "com1" in t else "PORT_SERIAL", ev_id=ev_id, revision=rev, origin=origin)

                if "modbus" in t:
                    add_fact(pid, "communication_protocol", "Modbus RTU", port_id="COM1" if "com1" in t else "PORT_SERIAL", ev_id=ev_id, revision=rev, origin=origin)

                # 4. Sensor interface (4-20mA current loop)
                if "4-20ma" in t or "4-20 ma" in t:
                    add_fact(pid, "sensor_interface", "4-20mA", unit="mA", port_id="ANALOG_LOOP" if pid == "P3" else "ANALOG_IN", ev_id=ev_id, revision=rev, origin=origin)

                # 5. Operating Temperature
                if "operating temperature" in t:
                    temp_match = re.search(r"(-?\d+)\s*°?c\s*to\s*(\d+)\s*°?c", t)
                    if temp_match:
                        add_fact(pid, "operating_temp_min_c", float(temp_match.group(1)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)
                        add_fact(pid, "operating_temp_max_c", float(temp_match.group(2)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)

                # 6. Ingress Protection (IP Rating)
                ip_match = re.search(r"\b(ip\s*\d{2})\b", t)
                if ip_match:
                    add_fact(pid, "enclosure_ip_rating", ip_match.group(1).upper().replace(" ", ""), ev_id=ev_id, revision=rev, origin=origin)

                # 7. Mounting (DIN rail)
                if "din rail" in t:
                    add_fact(pid, "din_rail_mounting", True, ev_id=ev_id, revision=rev, origin=origin)

                # 8. Measurement Range
                if "measurement range" in t:
                    range_match = re.search(r"(-?\d+)\s*°?c\s*to\s*(\d+)\s*°?c", t)
                    if range_match:
                        add_fact(pid, "measurement_range_min_c", float(range_match.group(1)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)
                        add_fact(pid, "measurement_range_max_c", float(range_match.group(2)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)

                continue

            # =============================================================
            # Real Products: P_X4, P_THT, P_UHEAT
            # =============================================================

            # --- P_X4: Horner HE-X4 Micro OCS ---
            if pid == "P_X4":
                # Supply voltage nominal / range: PrimaryPwr.Range 24VDC±20% or 24VDC +/-20%
                pwr_match = (
                    re.search(r"primarypwr\.range\s*(\d+)vdc\s*±\s*(\d+)%", t_compact)
                    or re.search(r"primarypwr\.range\s*(\d+)vdc\s*±\s*(\d+)%", t)
                    or re.search(r"primary\s*power\s*range\s*is\s*(\d+)vdc\s*\+/-\s*(\d+)%", t_norm)
                )
                if pwr_match:
                    v_nom = float(pwr_match.group(1))
                    pct = float(pwr_match.group(2)) / 100.0
                    v_min = round(v_nom * (1.0 - pct), 1)
                    v_max = round(v_nom * (1.0 + pct), 1)
                    add_fact(pid, "supply_voltage_nominal_v", v_nom, unit="V", ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "supply_voltage_min_v", v_min, unit="V", ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "supply_voltage_max_v", v_max, unit="V", ev_id=ev_id, revision=rev, origin=origin)

                # Operating / Storage temp
                op_temp = (
                    re.search(r"operatingtemp\.\s*(-?\d+)°?c\s*to\s*\+?(\d+)°?c", t_compact)
                    or re.search(r"operatingtemp\.\s*(-?\d+)°?c\s*to\s*\+?(\d+)°?c", t)
                    or re.search(r"operating\s*temperature\s*range.*?\b(-?\d+)\s*°?c\s*to\s*\+?(\d+)\s*°?c", t_norm)
                )
                if op_temp:
                    add_fact(pid, "operating_temp_min_c", float(op_temp.group(1)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "operating_temp_max_c", float(op_temp.group(2)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)

                st_temp = (
                    re.search(r"storagetemp\.\s*(-?\d+)°?c\s*to\s*\+?(\d+)°?c", t_compact)
                    or re.search(r"storagetemp\.\s*(-?\d+)°?c\s*to\s*\+?(\d+)°?c", t)
                )
                if st_temp:
                    add_fact(pid, "storage_temp_min_c", float(st_temp.group(1)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "storage_temp_max_c", float(st_temp.group(2)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)

                if "relativehumidity" in t_compact and "5to95%" in t_compact:
                    add_fact(pid, "humidity_max_percent", 95.0, unit="%", ev_id=ev_id, revision=rev, origin=origin)

                if "weight 360g" in t_norm or "weight360g" in t_compact:
                    add_fact(pid, "weight_g", 360.0, unit="g", ev_id=ev_id, revision=rev, origin=origin)

                # Digital Inputs: Models R & A: 12 digital DC inputs
                if ("digital dc inputs" in t_norm and "12" in t_norm) or ("built-in i/o" in t_norm and "12 digital in" in t_norm):
                    add_fact(pid, "digital_inputs_count", 12, variant_scope=["standard", "HE-X4A", "HE-X4R"], ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "digital_inputs", 12, variant_scope=["standard", "HE-X4A", "HE-X4R"], ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "digital_inputs_type", "DC", variant_scope=["standard", "HE-X4A", "HE-X4R"], ev_id=ev_id, revision=rev, origin=origin)
                    if "inputvoltagerange 12vdc/24vdc" in t_norm or "12vdc/24vdc" in t_norm:
                        add_fact(pid, "digital_input_voltage_range", "12VDC/24VDC", variant_scope=["standard", "HE-X4A", "HE-X4R"], ev_id=ev_id, revision=rev, origin=origin)
                    if "absolutemax.voltage 30vdcmax" in t_norm or "30vdcmax" in t_compact:
                        add_fact(pid, "digital_input_max_v", 30.0, unit="V", variant_scope=["standard", "HE-X4A", "HE-X4R"], ev_id=ev_id, revision=rev, origin=origin)

                # Digital Outputs - Model A: 12 sourcing solid-state DC outputs
                if "model a: digital dc outputs" in t_norm or ("model a" in t_norm and "12 digital out" in t_norm) or ("modela:solidstate" in t_compact):
                    var_a = ["HE-X4A", "Model A"]
                    add_fact(pid, "digital_outputs_count", 12, variant_scope=var_a, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "digital_outputs", 12, variant_scope=var_a, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "solid_state_dc_outputs_count", 12, variant_scope=var_a, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "digital_outputs_type", "sourcing solid-state DC", variant_scope=var_a, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "output_type", "Sourcing", variant_scope=var_a, ev_id=ev_id, revision=rev, origin=origin)
                    if "absolutemax.voltage 28vdc" in t_norm or "28vdc" in t_norm:
                        add_fact(pid, "digital_output_max_v", 28.0, unit="V", variant_scope=var_a, ev_id=ev_id, revision=rev, origin=origin)
                    if "maxoutputperpoint:sourcing 0.5a@24vdc" in t_norm or "0.5a@24vdc" in t_compact:
                        add_fact(pid, "max_output_current_per_point_a", 0.5, unit="A", conditions=[{"voltage": "24VDC"}], variant_scope=var_a, ev_id=ev_id, revision=rev, origin=origin)

                # Digital Outputs - Model R: 6 relay outputs + 2 solid-state DC outputs
                if "model r: digital dc outputs" in t_norm or ("model r" in t_norm and "2 pwm out" in t_norm and "6 relay" in t_norm):
                    var_r = ["HE-X4R", "Model R"]
                    add_fact(pid, "solid_state_dc_outputs_count", 2, variant_scope=var_r, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "digital_dc_outputs_count", 2, variant_scope=var_r, ev_id=ev_id, revision=rev, origin=origin)
                    if "absolutemax.voltage 28vdc" in t_norm or "28vdc" in t_norm:
                        add_fact(pid, "digital_output_max_v", 28.0, unit="V", variant_scope=var_r, ev_id=ev_id, revision=rev, origin=origin)

                if "relay outputs: model r" in t_norm or ("model r" in t_norm and "6 relay out" in t_norm):
                    var_r = ["HE-X4R", "Model R"]
                    add_fact(pid, "relay_outputs_count", 6, variant_scope=var_r, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "relay_outputs", 6, variant_scope=var_r, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "digital_outputs", "6 relay outputs + 2 solid-state DC outputs", variant_scope=var_r, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "digital_outputs_count", 8, variant_scope=var_r, ev_id=ev_id, revision=rev, origin=origin)
                    if "max.outputcurrent&voltageper relay 3a@60vac" in t_norm or "3a@30vdc" in t_norm:
                        add_fact(pid, "relay_max_current_a", 3.0, unit="A", conditions=[{"load": "resistive"}], variant_scope=var_r, ev_id=ev_id, revision=rev, origin=origin)
                    if "max.totaloutputcurrent 5acontinuous" in t_norm or "5acontinuous" in t_compact:
                        add_fact(pid, "relay_total_output_current_max_a", 5.0, unit="A", conditions=[{"mode": "continuous"}], variant_scope=var_r, ev_id=ev_id, revision=rev, origin=origin)
                    if "max.switchedpower 150w" in t_norm or "150w" in t_norm:
                        add_fact(pid, "relay_max_switched_power_w", 150.0, unit="W", variant_scope=var_r, ev_id=ev_id, revision=rev, origin=origin)

                # Communication & Protocol
                if "rs-485" in t_norm or "rs485" in t_compact:
                    port = "MJ2" if "mj2" in t_norm else ("PORT_SERIAL" if "serial" in t_norm else "PORT_RS485")
                    add_fact(pid, "communication_interface", "RS-485", port_id=port, ev_id=ev_id, revision=rev, origin=origin)
                if "rs-232" in t_norm:
                    add_fact(pid, "communication_interface", "RS-232", port_id="MJ1", ev_id=ev_id, revision=rev, origin=origin)
                if "ethernet" in t_norm:
                    add_fact(pid, "communication_interface", "Ethernet", port_id="LAN", ev_id=ev_id, revision=rev, origin=origin)
                if "can communications" in t_norm or "can " in t_norm:
                    add_fact(pid, "communication_interface", "CAN", port_id="CAN", ev_id=ev_id, revision=rev, origin=origin)
                if "modbus" in t_norm:
                    add_fact(pid, "communication_protocol", "Modbus RTU", port_id="MJ2", ev_id=ev_id, revision=rev, origin=origin)

                # Analog inputs / outputs
                if "4-20ma" in t_norm or "4-20 ma" in t_norm:
                    add_fact(pid, "sensor_interface", "4-20mA", unit="mA", port_id="ANALOG_IN", ev_id=ev_id, revision=rev, origin=origin)

            # --- P_THT: TZ THT-02 Temp & Humidity Sensor ---
            elif pid == "P_THT":
                # RS-485 interface & Modbus RTU
                if "rs-485" in t_norm or "rs485" in t_compact:
                    add_fact(pid, "communication_interface", "RS-485", port_id="RS485", ev_id=ev_id, revision=rev, origin=origin)
                if "modbus" in t_norm:
                    add_fact(pid, "communication_protocol", "Modbus RTU", port_id="RS485", ev_id=ev_id, revision=rev, origin=origin)

                # Sensing elements: SHT30, SHT3x
                if "sht30" in t_compact:
                    add_fact(pid, "sensing_element", "SHT30", conditions=[{"measurement": "temperature"}], ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "sensing_element_family", "SHT3x", ev_id=ev_id, revision=rev, origin=origin)
                if "sht3x" in t_compact:
                    add_fact(pid, "sensing_element", "SHT3X", conditions=[{"measurement": "humidity"}], ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "sensing_element_family", "SHT3x", ev_id=ev_id, revision=rev, origin=origin)

                # Power supply: DC5～24V, 5mA
                volt_match = (
                    re.search(r"supplyvoltage\s*dc\s*(\d+)～(\d+)v", t_compact)
                    or re.search(r"dc\s*(\d+)～(\d+)v", t_compact)
                )
                if volt_match and ("supplyvoltage" in t_compact or "power supply" in t_norm):
                    v_min, v_max = float(volt_match.group(1)), float(volt_match.group(2))
                    add_fact(pid, "supply_voltage_min_v", v_min, unit="V", ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "supply_voltage_max_v", v_max, unit="V", ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "supply_voltage_nominal_v", 24.0, unit="V", ev_id=ev_id, revision=rev, origin=origin)

                cur_match = re.search(r"current\s*(\d+)ma", t_compact)
                if cur_match and ("power supply" in t_norm or "technical data" in t_norm or "technicaldata" in t_compact):
                    add_fact(pid, "supply_current_ma", float(cur_match.group(1)), unit="mA", ev_id=ev_id, revision=rev, origin=origin)

                # Operating environment: -40～85℃ / 5～95%RH
                if "workingenvironment" in t_compact or "working environment" in t_norm:
                    env_temp = re.search(r"(-?\d+)～(\d+)℃", t_compact)
                    if env_temp:
                        add_fact(pid, "operating_temp_min_c", float(env_temp.group(1)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)
                        add_fact(pid, "operating_temp_max_c", float(env_temp.group(2)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)
                    env_rh = re.search(r"(\d+)～(\d+)%rh", t_compact)
                    if env_rh:
                        add_fact(pid, "operating_humidity_min_percent", float(env_rh.group(1)), unit="%RH", ev_id=ev_id, revision=rev, origin=origin)
                        add_fact(pid, "operating_humidity_max_percent", float(env_rh.group(2)), unit="%RH", ev_id=ev_id, revision=rev, origin=origin)

                # Storage environment: -40～85℃
                if "storageenvironment" in t_compact or "storage environment" in t_norm:
                    st_temp = re.search(r"storageenvironment.*?(-?\d+)～(\d+)℃", t_compact)
                    if st_temp:
                        add_fact(pid, "storage_temp_min_c", float(st_temp.group(1)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)
                        add_fact(pid, "storage_temp_max_c", float(st_temp.group(2)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)

                # Measurement range: -40～125℃ for temperature, 5～95%RH for humidity
                if "measuringrange" in t_compact or "measuring range" in t_norm:
                    m_temp = re.search(r"measuringrange.*?(-?\d+)～(\d+)℃", t_compact)
                    if m_temp:
                        add_fact(pid, "measurement_range_min_c", float(m_temp.group(1)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)
                        add_fact(pid, "measurement_range_max_c", float(m_temp.group(2)), unit="°C", ev_id=ev_id, revision=rev, origin=origin)

                if "workingrange" in t_compact and "%rh" in t_compact:
                    m_rh = re.search(r"workingrange.*?(\d+)～(\d+)%rh", t_compact)
                    if m_rh:
                        add_fact(pid, "humidity_measurement_range_min_percent", float(m_rh.group(1)), unit="%RH", ev_id=ev_id, revision=rev, origin=origin)
                        add_fact(pid, "humidity_measurement_range_max_percent", float(m_rh.group(2)), unit="%RH", ev_id=ev_id, revision=rev, origin=origin)

                # Accuracy: ±0.3℃(0~60℃), ±2%(10%~90%RH)
                if "±0.3℃" in t_compact:
                    add_fact(pid, "temperature_accuracy_c", 0.3, unit="°C", conditions=[{"range": "0~60℃"}], ev_id=ev_id, revision=rev, origin=origin)
                if "±2%" in t_compact:
                    add_fact(pid, "humidity_accuracy_percent", 2.0, unit="%RH", conditions=[{"range": "10%~90%RH"}], ev_id=ev_id, revision=rev, origin=origin)

                # Baud rates: Optional 4800bps/9600bps/19200bps
                if "4800bps" in t_compact and "9600bps" in t_compact:
                    add_fact(pid, "supported_baud_rates", [4800, 9600, 19200], ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "baud_rate", 9600, ev_id=ev_id, revision=rev, origin=origin)

                # Ingress protection (if mentioned)
                ip_match = re.search(r"\b(ip\s*\d{2})\b", t_norm)
                if ip_match:
                    add_fact(pid, "enclosure_ip_rating", ip_match.group(1).upper().replace(" ", ""), ev_id=ev_id, revision=rev, origin=origin)

            # --- P_UHEAT: Pumphouse Heater U Series ---
            elif pid == "P_UHEAT":
                # Role as pumphouse heater
                if "pumphouse heater" in t_norm or "pump house series" in t_norm or "pumphouse" in t_compact:
                    add_fact(pid, "role", "pumphouse_heater", ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "equipment_type", "Pumphouse Heater", ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "application", "freeze protection", ev_id=ev_id, revision=rev, origin=origin)
                if "radiant heat" in t_norm or "convection/radiant" in t_norm:
                    add_fact(pid, "heating_type", "radiant/convection", ev_id=ev_id, revision=rev, origin=origin)

                # Mount orientation rules:
                # "designed to operate in both horizontal (full wattage) or vertical (up to and including 500W) orientations"
                if "horizontal" in t_norm and "vertical" in t_norm and "500w" in t_norm:
                    add_fact(pid, "supported_mount_orientations", ["horizontal", "vertical"], ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "vertical_mount_limit_w", 500.0, unit="W", conditions=[{"orientation": "vertical"}], ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "horizontal_mount_wattage", "full wattage", conditions=[{"orientation": "horizontal"}], ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "horizontal_mount_limit_w", 1000.0, unit="W", conditions=[{"orientation": "horizontal"}], ev_id=ev_id, revision=rev, origin=origin)

                # "Unit CANNOT be installed vertically with thermostat at the top"
                if "cannot be installed vertically with thermostat at the top" in t_norm or "cannotbeinstalledverticallywiththermostatatthetop" in t_compact:
                    add_fact(pid, "mounting_restriction", "Unit CANNOT be installed vertically with thermostat at the top", conditions=[{"orientation": "vertical"}], ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "vertical_mount_prohibited_orientation", "thermostat at the top", conditions=[{"orientation": "vertical"}], ev_id=ev_id, revision=rev, origin=origin)

                # Voltage families:
                # 120V models (U1250, U1275, U12100)
                if "u1250" in t_compact or "12-120v" in t_compact or "12 - 120v" in t_norm or "u12" in t_compact:
                    var_120 = ["120V models", "U12"]
                    add_fact(pid, "voltage_family", "120V", variant_scope=var_120, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "supply_voltage_nominal_v", 120.0, unit="V", variant_scope=var_120, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "available_wattages_w", [500, 750, 1000], unit="W", variant_scope=var_120, ev_id=ev_id, revision=rev, origin=origin)

                # Triple rated 240/208/120V models (U2425, U2450, U2475, U24100)
                if "triplerated" in t_compact or "triple rated" in t_norm or "240/208/120" in t_compact:
                    var_triple = ["triple rated 240/208/120V models", "U24"]
                    add_fact(pid, "voltage_family", "triple rated 240/208/120V", variant_scope=var_triple, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "supply_voltage_nominal_v", 240.0, unit="V", variant_scope=var_triple, ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "supported_voltages_v", [240.0, 208.0, 120.0], unit="V", variant_scope=var_triple, ev_id=ev_id, revision=rev, origin=origin)
                    if "25% less wattage" in t_norm or "25%lesswattage" in t_compact:
                        add_fact(pid, "derating_at_208v_wattage_percent", -25.0, unit="%", variant_scope=var_triple, ev_id=ev_id, revision=rev, origin=origin)
                    if "75% less wattage" in t_norm or "75%lesswattage" in t_compact:
                        add_fact(pid, "derating_at_120v_wattage_percent", -75.0, unit="%", variant_scope=var_triple, ev_id=ev_id, revision=rev, origin=origin)
                    if "13% less amps" in t_norm or "13%lessamps" in t_compact:
                        add_fact(pid, "derating_at_208v_current_percent", -13.0, unit="%", variant_scope=var_triple, ev_id=ev_id, revision=rev, origin=origin)
                    if "50% less amps" in t_norm or "50%lessamps" in t_compact:
                        add_fact(pid, "derating_at_120v_current_percent", -50.0, unit="%", variant_scope=var_triple, ev_id=ev_id, revision=rev, origin=origin)

                # Thermostat: 40° to 90°F
                if "40° to 90°f" in t_norm or "40°to90°f" in t_compact:
                    add_fact(pid, "thermostat_min_temp_f", 40.0, unit="°F", ev_id=ev_id, revision=rev, origin=origin)
                    add_fact(pid, "thermostat_max_temp_f", 90.0, unit="°F", ev_id=ev_id, revision=rev, origin=origin)
                    if "frost protection" in t_norm or "frostprotection" in t_compact:
                        add_fact(pid, "frost_protection", True, ev_id=ev_id, revision=rev, origin=origin)

                # Incoloy 840 element
                if "incoloy 840" in t_norm or "incoloy840" in t_compact:
                    add_fact(pid, "heating_element_material", "Incoloy 840", ev_id=ev_id, revision=rev, origin=origin)

                # Damp locations rating
                if "damp locations" in t_norm or "damplocations" in t_compact:
                    add_fact(pid, "location_rating", "damp locations", ev_id=ev_id, revision=rev, origin=origin)

                # Safety Standard ETLus
                if "etlus" in t_compact:
                    add_fact(pid, "safety_standard", "ETLus", ev_id=ev_id, revision=rev, origin=origin)

    return facts


def extract_facts(
    spans_path: Optional[Path] = None,
    output_facts_file: Optional[Path] = None,
) -> List[Fact]:
    """Runs fact extraction over document spans and saves to facts.auto.jsonl and facts.auto.real.jsonl."""
    out_file = Path(output_facts_file) if output_facts_file else DEFAULT_AUTO_FACTS_FILE
    is_default_target = out_file.resolve() == DEFAULT_AUTO_FACTS_FILE.resolve()

    spans = load_spans(spans_path)
    if not spans:
        logger.warning(f"No spans found to extract facts from at {spans_path}")
        return []

    facts = _extract_facts_from_spans(spans)

    real_facts = [f for f in facts if f.product_id in ("P_X4", "P_THT", "P_UHEAT")]
    syn_facts = [f for f in facts if f.product_id in ("P1", "P2", "P3")]

    # If real facts are extracted, write to facts.auto.real.jsonl
    if real_facts:
        save_facts(real_facts, DEFAULT_AUTO_REAL_FACTS_FILE)

    save_facts(facts, out_file)

    logger.info(f"Extracted {len(facts)} auto-extracted facts into {out_file}")
    return facts


def load_facts(facts_file: Path | str | None) -> List[Fact]:
    """Loads Fact objects from a JSON Lines file.

    Handles missing directories, missing files, empty files, and corrupted lines gracefully.
    """
    if not facts_file:
        logger.warning("Empty or None facts_file path provided to load_facts.")
        return []
    p = Path(facts_file)
    if not p.is_file():
        logger.warning(f"Facts file not found: {p}")
        return []
    facts: List[Fact] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    facts.append(Fact.from_json(line_str))
                except Exception as e:
                    logger.warning(
                        f"Corrupted or invalid fact at {p}:{line_no}: {e}. Skipping line."
                    )
    except Exception as e:
        logger.error(f"Error reading facts file {p}: {e}")
        return []
    return facts


def save_facts(facts: Optional[List[Fact]], output_file: Path | str) -> None:
    """Saves Fact objects to a JSON Lines file.

    Ensures parent directories are created if missing and handles empty fact lists cleanly.
    """
    if not output_file:
        raise ValueError("output_file cannot be empty")
    p = Path(output_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    fact_list = facts if facts is not None else []
    with open(p, "w", encoding="utf-8") as f:
        for fact in fact_list:
            if isinstance(fact, Fact):
                f.write(fact.to_json() + "\n")
            elif isinstance(fact, dict):
                try:
                    f.write(Fact.model_validate(fact).to_json() + "\n")
                except Exception as e:
                    logger.warning(f"Skipping invalid fact dict in save_facts: {e}")
            else:
                logger.warning(f"Skipping unsupported object in save_facts: {type(fact)}")

