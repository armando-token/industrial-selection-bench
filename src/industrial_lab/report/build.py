"""Report generation module adhering to MEGAPLAN.md §25.

Generates:
1. `report.md`: Markdown summary containing the primary benchmark table:
   `motor | casos | éxitos completos | falsas aprobaciones | abstención excesiva | p50 ms | p95 ms | costo total | USD/tarea correcta | preparación`
   and comprehensive analytical insights answering §25.3 questions.
2. `report.html`: Self-contained, responsive HTML dashboard with KPI cards,
   comparison tables, span breakdowns, and amortized cost projections.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from industrial_lab.observability.costs import CostCalculator
from industrial_lab.observability.spans import TelemetryRecord

logger = logging.getLogger(__name__)


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Calculate 95% Wilson score confidence interval for a binomial proportion.

    Well-behaved for small sample sizes and boundary cases (0% or 100%).
    """
    if total <= 0:
        return 0.0, 0.0
    s = max(0, min(int(successes), int(total)))
    p = s / total
    z_val = abs(float(z)) if z else 1.96
    denom = 1.0 + (z_val**2) / total
    center = (p + (z_val**2) / (2.0 * total)) / denom
    variance_term = (p * (1.0 - p) / total) + ((z_val**2) / (4.0 * (total**2)))
    margin = (z_val / denom) * math.sqrt(max(0.0, variance_term))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return round(low * 100.0, 1), round(high * 100.0, 1)


def calculate_percentile(values: list[float], percentile: float) -> float:
    """Calculate percentile from a list of numerical values."""
    if not values:
        return 0.0
    clean_vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean_vals:
        return 0.0
    p = max(0.0, min(100.0, float(percentile)))
    sorted_vals = sorted(clean_vals)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1


@dataclass
class EngineMetrics:
    """Consolidated metrics for an engine or ablation configuration."""

    engine_id: str
    display_name: str
    total_cases: int = 0
    total_requests: int = 0
    successful_tasks: int = 0
    content_fully_correct_tasks: int = 0
    operationally_valid_tasks: int = 0
    false_approvals: int = 0
    incompatible_cases: int = 0
    excessive_abstentions: int = 0
    resolvable_cases: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    total_cost_usd: Optional[float] = 0.0
    cost_status: str = "exact"
    preparation_wall_time_min: float = 0.0
    preparation_compute_cost_usd: float = 0.0
    preparation_human_review_min: float = 0.0
    spans_duration_by_name: dict[str, list[float]] = field(default_factory=dict)
    decisions_by_origin: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate_pct(self) -> float:
        return (self.successful_tasks / self.total_cases * 100.0) if self.total_cases > 0 else 0.0

    @property
    def content_fully_correct_rate_pct(self) -> float:
        return (self.content_fully_correct_tasks / self.total_cases * 100.0) if self.total_cases > 0 else 0.0

    @property
    def operationally_valid_rate_pct(self) -> float:
        return (self.operationally_valid_tasks / self.total_cases * 100.0) if self.total_cases > 0 else 0.0

    @property
    def false_approval_rate_pct(self) -> Optional[float]:
        return (self.false_approvals / self.incompatible_cases * 100.0) if self.incompatible_cases > 0 else None

    @property
    def excessive_abstention_rate_pct(self) -> Optional[float]:
        return (self.excessive_abstentions / self.resolvable_cases * 100.0) if self.resolvable_cases > 0 else None

    @property
    def p50_ms(self) -> float:
        return calculate_percentile(self.latencies_ms, 50.0)

    @property
    def p95_ms(self) -> float:
        return calculate_percentile(self.latencies_ms, 95.0)

    @property
    def mean_latency_ms(self) -> float:
        clean = [v for v in self.latencies_ms if v is not None]
        return statistics.mean(clean) if clean else 0.0

    @property
    def cost_per_correct_task(self) -> Optional[float]:
        return CostCalculator.calculate_cost_per_correct_task(self.total_cost_usd, self.successful_tasks)

    @property
    def total_preparation_cost_usd(self) -> float:
        return float(self.preparation_compute_cost_usd or 0.0)

    @property
    def preparation_summary_str(self) -> str:
        parts = []
        prep_cost = float(self.preparation_compute_cost_usd or 0.0)
        if prep_cost > 0 or self.total_cost_usd is not None:
            parts.append(f"${prep_cost:.2f}")
        prep_human = float(self.preparation_human_review_min or 0.0)
        prep_wall = float(self.preparation_wall_time_min or 0.0)
        if prep_human > 0:
            parts.append(f"{prep_human:.0f} min")
        elif prep_wall > 0:
            parts.append(f"{prep_wall:.0f} min auto")
        else:
            parts.append("0 min")
        return " (".join(parts) + (")" if len(parts) > 1 else "")


@dataclass
class ReportData:
    """Full dataset required to compile the benchmark report."""

    run_id: str
    benchmark_version: str = "v1.0"
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    engines: dict[str, EngineMetrics] = field(default_factory=dict)
    ablations: dict[str, EngineMetrics] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ReportBuilder:
    """Builds Markdown and HTML benchmark reports adhering to §25."""

    def __init__(self, data: ReportData) -> None:
        self.data = data

    @property
    def is_demo(self) -> bool:
        """Return True if this report represents a demo, synthetic fixture, or non-official run."""
        if self.data.metadata.get("is_demo") or self.data.metadata.get("dry_run"):
            return True
        if self.data.metadata.get("data_origin") in ("synthetic_fixture", "simulation_lab", "demo"):
            return True
        if any(term in self.data.run_id.lower() for term in ("demo", "synthetic", "pilot", "freeze", "dryrun", "dry_run")):
            return True
        if os.environ.get("LAB_DRY_RUN", "").lower() in ("1", "true", "yes"):
            return True
        return False

    @classmethod
    def create_demo_data(cls) -> ReportData:
        """Create demonstration dataset adhering strictly to MEGAPLAN.md §0.1 #7.

        IMPORTANT — INTEGRITY POLICY:
        - Evaluates synthetic test fixtures: CATÁLOGO SINTÉTICO DE PRUEBA (3 EQUIPOS FIXTURECORP: P1, P2, P3).
        - System A (structured_jev) is BLOCKED (§0.1 #7) due to empty TYPESAFE_API_KEY.
        - JEV from TypeSafe AI must NEVER be substituted or mocked in official results.
        - ZERO fabricated JEV performance metrics: no success rates, no CI, no invented latencies or costs.
        - Wires to real dry-run pilot artifacts runs/dryrun_20261002_195748 when present.
        """
        # Prefer loading real dry-run artifacts if available
        dryrun_dir = Path("runs/dryrun_20261002_195748")
        if not dryrun_dir.exists():
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            cand = repo_root / "runs" / "dryrun_20261002_195748"
            if cand.exists():
                dryrun_dir = cand

        if dryrun_dir.exists() and (dryrun_dir / "run_manifest.json").exists():
            data = cls.from_run_dir(dryrun_dir)
            data.metadata.update({
                "is_demo": True,
                "dry_run": True,
                "data_origin": "synthetic_fixture",
                "system_a_status": "BLOCKED (§0.1 #7)",
                "system_a_block_reason": "missing TYPESAFE_API_KEY",
                "catalog": "CATÁLOGO SINTÉTICO DE PRUEBA (3 EQUIPOS FIXTURECORP: P1, P2, P3)",
                "disclaimer": "AVISO: DATOS SINTÉTICOS DE DEMOSTRACIÓN — NO CONSTITUYE BENCHMARK OFICIAL NI CORRIDA REAL",
            })
            # Label dry-run engines clearly
            if "structured_rules" in data.engines:
                data.engines["structured_rules"].display_name = "structured_rules [LOCAL - REGLAS DSL]"
            if "rag_llm" in data.engines:
                data.engines["rag_llm"].display_name = "B: rag_llm [DRY-RUN SINTÉTICO - NO OFICIAL]"
                data.engines["rag_llm"].cost_status = "dryrun"
            if "scrape_llm" in data.engines:
                data.engines["scrape_llm"].display_name = "C: scrape_llm [DRY-RUN SINTÉTICO - NO OFICIAL]"
                data.engines["scrape_llm"].cost_status = "dryrun"

            # System A is BLOCKED (§0.1 #7) — zero invented JEV metrics
            data.engines["structured_jev"] = EngineMetrics(
                engine_id="structured_jev",
                display_name="A: structured_jev [BLOQUEADO (§0.1 #7) — Sin API Key]",
                total_cases=0,
                total_requests=0,
                successful_tasks=0,
                false_approvals=0,
                incompatible_cases=0,
                excessive_abstentions=0,
                resolvable_cases=0,
                latencies_ms=[],
                total_cost_usd=None,
                cost_status="BLOCKED (§0.1 #7)",
                preparation_wall_time_min=0.0,
                preparation_compute_cost_usd=0.0,
                preparation_human_review_min=0.0,
                spans_duration_by_name={},
                decisions_by_origin={},
            )
            data.ablations.clear()
            return data

        # Fallback offline demo data without external run directory
        data = ReportData(
            run_id="dryrun_20261002_195748",
            metadata={
                "is_demo": True,
                "dry_run": True,
                "data_origin": "synthetic_fixture",
                "system_a_status": "BLOCKED (§0.1 #7)",
                "system_a_block_reason": "missing TYPESAFE_API_KEY",
                "catalog": "CATÁLOGO SINTÉTICO DE PRUEBA (3 EQUIPOS FIXTURECORP: P1, P2, P3)",
                "disclaimer": "AVISO: DATOS SINTÉTICOS DE DEMOSTRACIÓN — NO CONSTITUYE BENCHMARK OFICIAL NI CORRIDA REAL",
            },
        )

        data.engines["structured_rules"] = EngineMetrics(
            engine_id="structured_rules",
            display_name="structured_rules [LOCAL - REGLAS DSL]",
            total_cases=3,
            total_requests=3,
            successful_tasks=0,
            false_approvals=0,
            incompatible_cases=0,
            excessive_abstentions=0,
            resolvable_cases=3,
            latencies_ms=[16.38, 17.59, 19.89],
            total_cost_usd=0.0,
            cost_status="exact",
            preparation_wall_time_min=0.0,
            preparation_compute_cost_usd=0.0,
            preparation_human_review_min=0.0,
        )

        data.engines["rag_llm"] = EngineMetrics(
            engine_id="rag_llm",
            display_name="B: rag_llm [DRY-RUN SINTÉTICO - NO OFICIAL]",
            total_cases=3,
            total_requests=3,
            successful_tasks=0,
            false_approvals=0,
            incompatible_cases=0,
            excessive_abstentions=0,
            resolvable_cases=3,
            latencies_ms=[14.94, 23.32, 38.37],
            total_cost_usd=0.0,
            cost_status="dryrun",
            preparation_wall_time_min=0.0,
            preparation_compute_cost_usd=0.0,
            preparation_human_review_min=0.0,
        )

        data.engines["scrape_llm"] = EngineMetrics(
            engine_id="scrape_llm",
            display_name="C: scrape_llm [DRY-RUN SINTÉTICO - NO OFICIAL]",
            total_cases=3,
            total_requests=3,
            successful_tasks=0,
            false_approvals=0,
            incompatible_cases=0,
            excessive_abstentions=0,
            resolvable_cases=3,
            latencies_ms=[49.42, 49.92, 50.60],
            total_cost_usd=0.0,
            cost_status="dryrun",
            preparation_wall_time_min=0.0,
            preparation_compute_cost_usd=0.0,
            preparation_human_review_min=0.0,
        )

        data.engines["structured_jev"] = EngineMetrics(
            engine_id="structured_jev",
            display_name="A: structured_jev [BLOQUEADO (§0.1 #7) — Sin API Key]",
            total_cases=0,
            total_requests=0,
            successful_tasks=0,
            false_approvals=0,
            incompatible_cases=0,
            excessive_abstentions=0,
            resolvable_cases=0,
            latencies_ms=[],
            total_cost_usd=None,
            cost_status="BLOCKED (§0.1 #7)",
            preparation_wall_time_min=0.0,
            preparation_compute_cost_usd=0.0,
            preparation_human_review_min=0.0,
            spans_duration_by_name={},
            decisions_by_origin={},
        )

        return data

    @classmethod
    def create_empty_data(cls, run_id: str = "empty_run") -> ReportData:
        """Create empty ReportData for runs with no responses or empty directories."""
        data = ReportData(run_id=run_id)
        for e_id, name in [
            ("structured_jev", "A: structured_jev"),
            ("rag_llm", "B: rag_llm"),
            ("scrape_llm", "C: scrape_llm"),
        ]:
            data.engines[e_id] = EngineMetrics(
                engine_id=e_id,
                display_name=name,
                total_cases=0,
                total_requests=0,
                successful_tasks=0,
                false_approvals=0,
                incompatible_cases=0,
                excessive_abstentions=0,
                resolvable_cases=0,
                latencies_ms=[],
                total_cost_usd=0.0,
                cost_status="exact",
                preparation_compute_cost_usd=0.0,
                preparation_human_review_min=0.0,
            )
        return data

    @classmethod
    def from_telemetry_records(
        cls,
        records: list[TelemetryRecord],
        scoring_data: Optional[dict[str, Any]] = None,
        run_id: str = "run_aggregated",
    ) -> ReportData:
        """Construct ReportData from raw TelemetryRecords and scoring results."""
        data = ReportData(run_id=run_id)
        scoring = scoring_data or {}
        display_names = {
            "structured_jev": "A: structured_jev",
            "rag_llm": "B: rag_llm",
            "scrape_llm": "C: scrape_llm",
            "structured_llm": "Ablation: structured_llm",
            "rag_llm_guarded": "Ablation: rag_llm_guarded",
        }

        engine_had_cost: dict[str, bool] = {}
        engine_case_ids: dict[str, set[str]] = {}

        for rec in records:
            engine_key = rec.engine
            target_map = data.ablations if ("ablation" in engine_key or "guarded" in engine_key) else data.engines

            if engine_key not in target_map:
                target_map[engine_key] = EngineMetrics(
                    engine_id=engine_key,
                    display_name=display_names.get(engine_key, engine_key),
                )

            metric = target_map[engine_key]
            metric.total_requests += 1
            if rec.elapsed_ms is not None:
                metric.latencies_ms.append(rec.elapsed_ms)

            if rec.case_id:
                engine_case_ids.setdefault(engine_key, set()).add(rec.case_id)

            if rec.cost_usd is not None:
                metric.total_cost_usd = (metric.total_cost_usd or 0.0) + rec.cost_usd
                engine_had_cost[engine_key] = True

            if rec.cost_status in ("unknown", "partial"):
                metric.cost_status = rec.cost_status

            # Aggregate spans
            for s in rec.spans:
                if s.duration_ms is not None:
                    metric.spans_duration_by_name.setdefault(s.name, []).append(s.duration_ms)

        # For engines where no record had cost_usd, and cost_status is unknown, set total_cost_usd to None
        for m in list(data.engines.values()) + list(data.ablations.values()):
            if not engine_had_cost.get(m.engine_id, False) and m.cost_status in ("unknown", "partial"):
                m.total_cost_usd = None

        # Merge scoring if provided
        for engine_key, s_info in scoring.get("engines", {}).items():
            target_map = data.ablations if engine_key in data.ablations else data.engines
            if engine_key not in target_map:
                target_map[engine_key] = EngineMetrics(
                    engine_id=engine_key,
                    display_name=display_names.get(engine_key, engine_key),
                )
            m = target_map[engine_key]
            m.total_cases = s_info.get(
                "total_cases",
                len(engine_case_ids.get(engine_key, set()))
                or (len(m.latencies_ms) // 3 or 1),
            )
            m.successful_tasks = s_info.get("successful_tasks", 0)
            m.content_fully_correct_tasks = s_info.get("content_fully_correct_tasks", s_info.get("content_fully_correct", 0))
            m.operationally_valid_tasks = s_info.get("operationally_valid_tasks", s_info.get("operationally_valid", 0))
            m.false_approvals = s_info.get("false_approvals", 0)
            m.incompatible_cases = s_info.get("incompatible_cases", 0)
            m.excessive_abstentions = s_info.get("excessive_abstentions", 0)
            m.resolvable_cases = s_info.get("resolvable_cases", 0)
            m.preparation_compute_cost_usd = s_info.get("preparation_compute_cost_usd", 0.0)
            m.preparation_human_review_min = s_info.get("preparation_human_review_min", 0.0)

        # Fallback case counts if scoring was empty
        for m in list(data.engines.values()) + list(data.ablations.values()):
            if m.total_cases == 0:
                distinct_cases = len(engine_case_ids.get(m.engine_id, set()))
                if distinct_cases > 0:
                    m.total_cases = distinct_cases
                elif m.total_requests > 0:
                    m.total_cases = max(1, m.total_requests // 3 if m.total_requests >= 3 else m.total_requests)

        return data

    @classmethod
    def from_run_dir(cls, run_dir: Path | str) -> ReportData:
        """Scan run directory for telemetry JSONL, normalized logs, and scoring JSON."""
        path = Path(run_dir)
        if not path.exists():
            logger.warning("Run directory %s does not exist; returning empty report data.", path)
            return cls.create_empty_data(run_id=path.name)

        records: list[TelemetryRecord] = []
        scoring_data: dict[str, Any] = {}

        # 1. Primary: telemetry.jsonl
        telemetry_file = path / "telemetry.jsonl"
        if telemetry_file.exists():
            with telemetry_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(TelemetryRecord.model_validate_json(line))
                        except Exception as e:
                            logger.warning("Failed parsing telemetry line: %s", e)

        # 2. Secondary fallback: responses_normalized.jsonl
        if not records:
            norm_file = path / "responses_normalized.jsonl"
            if norm_file.exists():
                with norm_file.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                resp = entry.get("response") or {}
                                status = resp.get("execution_status", "completed")
                                fail_code = status if status != "completed" else None
                                elapsed = float(entry.get("elapsed_ms") or 0.0)
                                rec = TelemetryRecord(
                                    run_id=entry.get("run_id", path.name),
                                    request_id=entry.get("request_id", f"req_{idx}"),
                                    case_id=entry.get("case_id", f"case_{idx}"),
                                    engine=entry.get("engine", "unknown"),
                                    repeat=int(entry.get("repeat", 1)),
                                    start_utc=entry.get("created_at_utc", datetime.now(timezone.utc).isoformat()),
                                    elapsed_ms=elapsed,
                                    failure_code=fail_code,
                                    cost_status="unknown",
                                    cost_usd=None,
                                )
                                records.append(rec)
                            except Exception as e:
                                logger.warning("Failed parsing normalized response line: %s", e)

        # 3. Tertiary fallback: responses_raw.jsonl
        if not records:
            raw_file = path / "responses_raw.jsonl"
            if raw_file.exists():
                with raw_file.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                raw_r = entry.get("raw_response") or {}
                                fail_code = raw_r.get("error") if isinstance(raw_r, dict) else None
                                rec = TelemetryRecord(
                                    run_id=entry.get("run_id", path.name),
                                    request_id=entry.get("request_id", f"req_{idx}"),
                                    case_id=entry.get("case_id", f"case_{idx}"),
                                    engine=entry.get("engine", "unknown"),
                                    repeat=int(entry.get("repeat", 1)),
                                    start_utc=entry.get("created_at_utc", datetime.now(timezone.utc).isoformat()),
                                    elapsed_ms=0.0,
                                    failure_code=str(fail_code) if fail_code else None,
                                    cost_status="unknown",
                                    cost_usd=None,
                                )
                                records.append(rec)
                            except Exception as e:
                                logger.warning("Failed parsing raw response line: %s", e)

        # 4. Scoring JSON
        scoring_file = path / "scoring.json"
        if scoring_file.exists():
            try:
                with scoring_file.open("r", encoding="utf-8") as f:
                    scoring_data = json.load(f)
                    if "engine_summaries" in scoring_data and "engines" not in scoring_data:
                        engines_scoring = {}
                        for eng, s in scoring_data["engine_summaries"].items():
                            ts = s.get("task_success", {})
                            cfc = s.get("content_fully_correct", {})
                            ov = s.get("operationally_valid", {})
                            fa = s.get("false_approval_rate", {})
                            ea = s.get("excessive_abstention_rate", {})
                            engines_scoring[eng] = {
                                "total_cases": s.get("cases", 0),
                                "successful_tasks": ts.get("numerator", 0),
                                "content_fully_correct_tasks": cfc.get("numerator", 0),
                                "operationally_valid_tasks": ov.get("numerator", 0),
                                "false_approvals": fa.get("numerator", 0),
                                "incompatible_cases": fa.get("denominator", 0),
                                "excessive_abstentions": ea.get("numerator", 0),
                                "resolvable_cases": ea.get("denominator", 0),
                                "preparation_compute_cost_usd": 0.0,
                                "preparation_human_review_min": 0.0,
                            }
                        scoring_data["engines"] = engines_scoring
            except Exception as e:
                logger.warning("Failed reading scoring file: %s", e)

        # 5. Scoring JSONL fallback
        if not scoring_data.get("engines"):
            scoring_log = path / "scoring.jsonl"
            if scoring_log.exists():
                engines_scoring: dict[str, dict[str, Any]] = {}
                case_ids_by_engine: dict[str, set[str]] = {}
                try:
                    with scoring_log.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            s_rec = json.loads(line)
                            eng = s_rec.get("engine")
                            if not eng:
                                continue
                            info = engines_scoring.setdefault(eng, {
                                "total_cases": 0,
                                "successful_tasks": 0,
                                "content_fully_correct_tasks": 0,
                                "operationally_valid_tasks": 0,
                                "false_approvals": 0,
                                "incompatible_cases": 0,
                                "excessive_abstentions": 0,
                                "resolvable_cases": 0,
                                "preparation_compute_cost_usd": 0.0,
                                "preparation_human_review_min": 0.0,
                            })
                            c_id = s_rec.get("case_id", "")
                            case_ids_by_engine.setdefault(eng, set()).add(c_id)
                            if s_rec.get("task_success"):
                                info["successful_tasks"] += 1
                            if s_rec.get("content_fully_correct"):
                                info["content_fully_correct_tasks"] += 1
                            if s_rec.get("operationally_valid"):
                                info["operationally_valid_tasks"] += 1
                            if s_rec.get("false_approval"):
                                info["false_approvals"] += 1

                            gold_verdict = s_rec.get("details", {}).get("gold_verdict")
                            # Incompatible case: gold prohibits COMPATIBLE (i.e. != "COMPATIBLE")
                            if gold_verdict != "COMPATIBLE":
                                info["incompatible_cases"] += 1
                            if s_rec.get("excessive_abstention"):
                                info["excessive_abstentions"] += 1
                            # Resolvable case: gold verdict is resoluble (i.e. != "INSUFFICIENT_EVIDENCE")
                            if gold_verdict != "INSUFFICIENT_EVIDENCE":
                                info["resolvable_cases"] += 1

                    for eng, info in engines_scoring.items():
                        info["total_cases"] = len(case_ids_by_engine.get(eng, set())) or 1
                    if engines_scoring:
                        scoring_data = {"engines": engines_scoring}
                except Exception as e:
                    logger.warning("Failed reading scoring log file: %s", e)

        # Check for configured engines from run_manifest.json
        manifest_engines: list[str] = []
        manifest_metadata: dict[str, Any] = {}
        manifest_file = path / "run_manifest.json"
        if manifest_file.exists():
            try:
                with manifest_file.open("r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    manifest_engines = manifest_data.get("engines", [])
                    if isinstance(manifest_data.get("metadata"), dict):
                        manifest_metadata.update(manifest_data["metadata"])
                    if isinstance(manifest_data.get("metadata"), dict):
                        manifest_metadata.update(manifest_data["metadata"])
                    if (
                        manifest_data.get("data_origin") == "synthetic_fixture"
                        or manifest_data.get("mode") == "dry_run"
                        or manifest_metadata.get("dry_run")
                        or manifest_metadata.get("data_origin") == "synthetic_fixture"
                        or "dryrun" in path.name.lower()
                        or os.environ.get("LAB_DRY_RUN", "").lower() in ("1", "true", "yes")
                    ):
                        manifest_metadata["is_demo"] = True
                        manifest_metadata["dry_run"] = True
                        manifest_metadata["data_origin"] = "synthetic_fixture"
            except Exception as e:
                logger.warning("Failed reading manifest file: %s", e)

        if not records:
            logger.info("No telemetry records found in %s; returning empty report data.", path)
            empty_data = cls.create_empty_data(run_id=path.name)
            empty_data.metadata.update(manifest_metadata)
            if manifest_engines:
                empty_data.engines.clear()
                display_names = {
                    "structured_jev": "A: structured_jev",
                    "rag_llm": "B: rag_llm",
                    "scrape_llm": "C: scrape_llm",
                    "structured_rules": "structured_rules (DSL rules)",
                    "structured_llm": "Ablation: structured_llm",
                    "rag_llm_guarded": "Ablation: rag_llm_guarded",
                }
                for e_key in manifest_engines:
                    target_map = empty_data.ablations if ("ablation" in e_key or "guarded" in e_key) else empty_data.engines
                    target_map[e_key] = EngineMetrics(
                        engine_id=e_key,
                        display_name=display_names.get(e_key, e_key),
                        total_cases=0,
                        total_requests=0,
                        successful_tasks=0,
                        latencies_ms=[],
                        total_cost_usd=0.0,
                    )
            return empty_data

        res = cls.from_telemetry_records(records, scoring_data=scoring_data, run_id=path.name)
        res.metadata.update(manifest_metadata)
        return res

    def generate_markdown(self) -> str:
        """Generate GitHub Flavored Markdown report according to MEGAPLAN.md §25."""
        if self.is_demo:
            lines = [
                "# Industrial Selection Lab — Benchmark Report [DEMO / SYNTHETIC / NON-OFFICIAL]",
                "",
                "### `[DEMO / SYNTHETIC / NON-OFFICIAL]` `[DEMO / SYNTHETIC FIXTURE / NON-OFFICIAL]`",
                "",
                "> [!WARNING]",
                "> **AVISO: DATOS SINTÉTICOS DE DEMOSTRACIÓN — NO CONSTITUYE BENCHMARK OFICIAL NI CORRIDA REAL**",
                ">",
                "> **CATÁLOGO SINTÉTICO DE PRUEBA (3 EQUIPOS FIXTURECORP: P1, P2, P3)**",
                ">",
                "> **data_origin:** `synthetic_fixture`",
                ">",
                "> **ESTADO SISTEMA A (`structured_jev`): BLOQUEADO / BLOCKED (§0.1 #7)** debido a `TYPESAFE_API_KEY` vacía / no configurada.",
                "> JEV de TypeSafe AI NUNCA debe ser sustituido por otro modelo ni por mocks en benchmark oficial.",
                "> Sistema A se encuentra completamente bloqueado (0 casos ejecutados, métricas N/A / BLOCKED).",
                "> **Sin reclamos de superioridad técnica sobre productos de producción.**",
                "",
                f"**Run ID:** `{self.data.run_id}` | **Date:** `{self.data.created_utc}` | **Version:** `{self.data.benchmark_version}` | **data_origin:** `synthetic_fixture` | **Modo:** `[DEMO / SYNTHETIC / NON-OFFICIAL]`",
                "",
                "Demostración comparativa de arquitecturas para selección técnica industrial sobre catálogo sintético de prueba (3 equipos FixtureCorp: P1, P2, P3):",
                "- **A — `structured_jev`:** **BLOQUEADO / BLOCKED (§0.1 #7)** por falta de `TYPESAFE_API_KEY`. Arquitectura: Documentación convertida a hechos estructurados, reglas deterministas y evaluador semántico JEV (TypeSafe AI; NUNCA sustituir). En modo demo/dry-run permanece en estado BLOQUEADO / not_run (0 ejecuciones).",
                "- **B — `rag_llm`:** Recuperación híbrida (BM25 + embeddings) con contexto expandido sobre fichas técnicas sintéticas y decisión LLM.",
                "- **C — `scrape_llm`:** Navegación dinámica, extracción de fichas HTML y parsing de PDFs sintéticos con herramientas HTTP y decisión LLM.",
            ]
        else:
            lines = [
                "# Industrial Selection Lab — Benchmark Report",
                "",
                f"**Run ID:** `{self.data.run_id}` | **Date:** `{self.data.created_utc}` | **Version:** `{self.data.benchmark_version}`",
                "",
                "Evaluación comparativa de arquitecturas para selección técnica industrial:",
                "- **A — `structured_jev`:** Hechos técnicos estructurados, reglas deterministas y evaluador semántico JEV (TypeSafe AI — requiere `TYPESAFE_API_KEY` válida; de lo contrario BLOQUEADO §0.1 #7).",
                "- **B — `rag_llm`:** Recuperación híbrida (BM25 + embeddings) con contexto expandido y decisión LLM.",
                "- **C — `scrape_llm`:** Navegación dinámica, extracción de fichas HTML y parsing de PDFs con herramientas HTTP y decisión LLM.",
            ]

        lines.extend([
            "",
            "---",
            "",
            "## 25.1 Tabla Principal de Resultados",
            "",
            "Métrica primaria: **Éxito completo** (6 criterios §15.1: ejecución válida, SKU en gold, veredicto técnico correcto, requisitos críticos verificados, evidencias auditables en span y cotización consistente con stock/precio).",
        ])

        if self.is_demo:
            lines.extend([
                "",
                "> [!NOTE]",
                "> **Integridad de Resultados (Modo Demostración):**",
                "> - Catálogo evaluado: **CATÁLOGO SINTÉTICO DE PRUEBA (3 EQUIPOS FIXTURECORP: P1, P2, P3)**.",
                "> - **Sistema A (`structured_jev`): BLOQUEADO (§0.1 #7)** por ausencia de `TYPESAFE_API_KEY`.",
                "> - JEV de TypeSafe AI NUNCA debe ser sustituido o simulado en reportes oficiales. Cero métricas inventadas para JEV.",
            ])

        lines.extend([
            "",
            "| motor | casos | éxitos completos | falsas aprobaciones | abstención excesiva | p50 ms | p95 ms | costo total | USD/tarea correcta | preparación |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ])

        # Populate main engines
        if not self.data.engines:
            lines.append(
                "| *(Sin datos de ejecución)* | 0 | 0/0 (0.0%) [0.0% - 0.0%] | 0/0 (0.0%) [0.0% - 0.0%] | 0/0 (0.0%) [0.0% - 0.0%] | - | - | $0.000 | N/A | 0 min |"
            )
        else:
            for eng in self.data.engines.values():
                is_blocked = (
                    eng.engine_id == "structured_jev"
                    and (
                        eng.total_cases == 0
                        or "BLOCKED" in eng.cost_status
                        or "BLOQUEADO" in eng.display_name
                        or "BLOCKED" in eng.display_name
                        or self.data.metadata.get("system_a_status") == "BLOCKED (§0.1 #7)"
                    )
                )
                if is_blocked:
                    cases_str = "0 (BLOCKED)"
                    succ_str = "N/A (BLOCKED)"
                    fa_str = "N/A (BLOCKED)"
                    ea_str = "N/A (BLOCKED)"
                    p50_str = "N/A"
                    p95_str = "N/A"
                    cost_str = "N/A (BLOCKED)"
                    cost_task_str = "N/A (BLOCKED)"
                    prep_str = "N/A (BLOCKED)"
                else:
                    cases_str = f"{eng.total_cases}"
                    if eng.total_requests > eng.total_cases:
                        cases_str += f" ({eng.total_requests} reqs)"

                    succ_ci = wilson_score_interval(eng.successful_tasks, eng.total_cases)
                    succ_str = f"{eng.successful_tasks}/{eng.total_cases} ({eng.success_rate_pct:.1f}%) [{succ_ci[0]}% - {succ_ci[1]}%]"

                    if eng.incompatible_cases > 0 and eng.false_approval_rate_pct is not None:
                        fa_ci = wilson_score_interval(eng.false_approvals, eng.incompatible_cases)
                        fa_str = f"{eng.false_approvals}/{eng.incompatible_cases} ({eng.false_approval_rate_pct:.1f}%) [{fa_ci[0]}% - {fa_ci[1]}%]"
                    else:
                        fa_str = "N/A (sin casos no compatibles)"

                    if eng.resolvable_cases > 0 and eng.excessive_abstention_rate_pct is not None:
                        ea_ci = wilson_score_interval(eng.excessive_abstentions, eng.resolvable_cases)
                        ea_str = f"{eng.excessive_abstentions}/{eng.resolvable_cases} ({eng.excessive_abstention_rate_pct:.1f}%) [{ea_ci[0]}% - {ea_ci[1]}%]"
                    else:
                        ea_str = "N/A (sin casos resolubles)"

                    p50_str = f"{eng.p50_ms:,.0f} ms" if (eng.latencies_ms and eng.p50_ms is not None) else "-"
                    p95_str = f"{eng.p95_ms:,.0f} ms" if (eng.latencies_ms and eng.p95_ms is not None) else "-"

                    cost_str = f"${eng.total_cost_usd:.3f}" if eng.total_cost_usd is not None else "Unknown"
                    if eng.cost_status != "exact":
                        cost_str += f" ({eng.cost_status})"

                    cost_task_val = eng.cost_per_correct_task
                    if cost_task_val == float("inf"):
                        cost_task_str = "∞ (0 éxitos)"
                    elif cost_task_val is not None:
                        cost_task_str = f"${cost_task_val:.4f}"
                    else:
                        cost_task_str = "Unknown"

                    prep_str = eng.preparation_summary_str

                lines.append(
                    f"| **{eng.display_name}** | {cases_str} | {succ_str} | {fa_str} | {ea_str} | {p50_str} | {p95_str} | {cost_str} | {cost_task_str} | {prep_str} |"
                )

        # Populate ablations if available
        if self.data.ablations:
            lines.extend([
                "",
                "### Ablaciones Metodológicas",
                "",
                "| motor (ablación) | casos | éxitos completos | falsas aprobaciones | abstención excesiva | p50 ms | p95 ms | costo total | USD/tarea correcta | preparación |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ])
            for abl in self.data.ablations.values():
                is_abl_blocked = (
                    abl.total_cases == 0
                    or "BLOCKED" in abl.display_name
                    or "BLOCKED" in abl.cost_status
                )
                if is_abl_blocked:
                    cases_str = "0 (BLOCKED)"
                    succ_str = "N/A (BLOCKED)"
                    fa_str = "N/A (BLOCKED)"
                    ea_str = "N/A (BLOCKED)"
                    p50_str = "N/A"
                    p95_str = "N/A"
                    cost_str = "N/A (BLOCKED)"
                    cost_task_str = "N/A (BLOCKED)"
                    prep_str = "N/A (BLOCKED)"
                else:
                    cases_str = f"{abl.total_cases}"
                    succ_ci = wilson_score_interval(abl.successful_tasks, abl.total_cases)
                    succ_str = f"{abl.successful_tasks}/{abl.total_cases} ({abl.success_rate_pct:.1f}%) [{succ_ci[0]}% - {succ_ci[1]}%]"

                    if abl.incompatible_cases > 0 and abl.false_approval_rate_pct is not None:
                        fa_ci = wilson_score_interval(abl.false_approvals, abl.incompatible_cases)
                        fa_str = f"{abl.false_approvals}/{abl.incompatible_cases} ({abl.false_approval_rate_pct:.1f}%) [{fa_ci[0]}% - {fa_ci[1]}%]"
                    else:
                        fa_str = "N/A (sin casos no compatibles)"

                    if abl.resolvable_cases > 0 and abl.excessive_abstention_rate_pct is not None:
                        ea_ci = wilson_score_interval(abl.excessive_abstentions, abl.resolvable_cases)
                        ea_str = f"{abl.excessive_abstentions}/{abl.resolvable_cases} ({abl.excessive_abstention_rate_pct:.1f}%) [{ea_ci[0]}% - {ea_ci[1]}%]"
                    else:
                        ea_str = "N/A (sin casos resolubles)"

                    p50_str = f"{abl.p50_ms:,.0f} ms" if (abl.latencies_ms and abl.p50_ms is not None) else "-"
                    p95_str = f"{abl.p95_ms:,.0f} ms" if (abl.latencies_ms and abl.p95_ms is not None) else "-"
                    cost_str = f"${abl.total_cost_usd:.3f}" if abl.total_cost_usd is not None else "Unknown"

                    cost_task_val = abl.cost_per_correct_task
                    if cost_task_val == float("inf"):
                        cost_task_str = "∞ (0 éxitos)"
                    elif cost_task_val is not None:
                        cost_task_str = f"${cost_task_val:.4f}"
                    else:
                        cost_task_str = "Unknown"
                    prep_str = abl.preparation_summary_str

                lines.append(
                    f"| {abl.display_name} | {cases_str} | {succ_str} | {fa_str} | {ea_str} | {p50_str} | {p95_str} | {cost_str} | {cost_task_str} | {prep_str} |"
                )

        # Latency by span breakdown
        lines.extend([
            "",
            "---",
            "",
            "## 25.2 Desglose de Latencia por Etapa (Spans §14.1)",
            "",
            "| Etapa / Span | structured_jev (ms) | rag_llm (ms) | scrape_llm (ms) |",
            "|---|---|---|---|",
        ])

        all_span_names = [
            "query_interpretation",
            "catalog_load",
            "technical_retrieval",
            "html_fetch",
            "pdf_download",
            "pdf_parse",
            "embedding_query",
            "model_request",
            "rule_evaluation",
            "evidence_resolution",
            "commerce_read",
            "quote_generation",
            "render_response",
        ]

        for s_name in all_span_names:
            vals = []
            for e_id in ["structured_jev", "rag_llm", "scrape_llm"]:
                eng = self.data.engines.get(e_id)
                if eng and s_name in eng.spans_duration_by_name:
                    durs = eng.spans_duration_by_name[s_name]
                    vals.append(f"{statistics.mean(durs):.1f} ms" if durs else "-")
                else:
                    vals.append("-")
            lines.append(f"| `{s_name}` | {vals[0]} | {vals[1]} | {vals[2]} |")

        # Proyección de Costo Amortizado vs N
        lines.extend([
            "",
            "---",
            "",
            "## 25.2 Proyección de Costo Amortizado vs Volumen N (§16.3)",
            "",
            "Fórmula: `total_cost_i(N) = preparation_cost_i + N * mean_online_cost_i`  |  `amortized_cost_i(N) = total_cost_i(N) / N`",
            "",
            "| N Consultas | A: structured_jev ($/req) | B: rag_llm ($/req) | C: scrape_llm ($/req) |",
            "|---|---|---|---|",
        ])

        volumes = [1, 10, 100, 1000, 10000]
        a_eng = self.data.engines.get("structured_jev")
        b_eng = self.data.engines.get("rag_llm")
        c_eng = self.data.engines.get("scrape_llm")

        a_blocked = (
            a_eng is None
            or a_eng.total_cases == 0
            or "BLOCKED" in a_eng.cost_status
            or "BLOQUEADO" in a_eng.display_name
            or "BLOCKED" in a_eng.display_name
            or self.data.metadata.get("system_a_status") == "BLOCKED (§0.1 #7)"
        )

        a_prep = float(a_eng.preparation_compute_cost_usd or 0.0) if a_eng else 0.0
        b_prep = float(b_eng.preparation_compute_cost_usd or 0.0) if b_eng else 0.0
        c_prep = float(c_eng.preparation_compute_cost_usd or 0.0) if c_eng else 0.0

        a_on = (a_eng.total_cost_usd / a_eng.total_cases) if (a_eng and a_eng.total_cases > 0 and a_eng.total_cost_usd is not None) else 0.0
        b_on = (b_eng.total_cost_usd / b_eng.total_cases) if (b_eng and b_eng.total_cases > 0 and b_eng.total_cost_usd is not None) else 0.0
        c_on = (c_eng.total_cost_usd / c_eng.total_cases) if (c_eng and c_eng.total_cases > 0 and c_eng.total_cost_usd is not None) else 0.0

        for n in volumes:
            a_str = "N/A (BLOCKED)" if a_blocked else f"${CostCalculator.calculate_amortized_cost(a_prep, a_on, n):.4f}"
            b_am = CostCalculator.calculate_amortized_cost(b_prep, b_on, n)
            c_am = CostCalculator.calculate_amortized_cost(c_prep, c_on, n)
            lines.append(f"| N = {n:,} | {a_str} | ${b_am:.4f} | ${c_am:.4f} |")

        # Break even analysis
        if a_blocked:
            be_note = "System A se encuentra BLOQUEADO (§0.1 #7) por falta de TYPESAFE_API_KEY. No se proyecta punto de equilibrio de A vs C en ausencia de credenciales oficiales."
        else:
            be_ac = CostCalculator.calculate_break_even(a_prep, c_prep, a_on, c_on)
            be_str = f"{be_ac:,.0f} consultas" if be_ac is not None else "No alcanza punto de equilibrio"
            be_note = (
                f"A amortiza su inversión inicial de preparación técnica frente a Scraping (C) a partir de **{be_str}**."
                if be_ac is not None
                else "No alcanza punto de equilibrio entre A y C con los datos observados."
            )
        lines.extend([
            "",
            "> [!NOTE]",
            f"> **Punto de equilibrio A vs C:** {be_note}",
        ])

        # Section 25.3 Mandatory Insights
        lines.extend([
            "",
            "---",
            "",
            "## 25.3 Insights Obligatorios (§25.3)",
            "",
            "### 1. ¿Qué motor ganó cada dimensión y con qué incertidumbre?",
            "- **Integridad y estado de motores:** Evaluación sobre catálogo sintético de prueba (`synthetic_fixture`: FixtureCorp P1, P2, P3).",
            "- **Sistema A (`structured_jev`): BLOQUEADO (§0.1 #7)** por ausencia de `TYPESAFE_API_KEY`. En estricto apego a las reglas absolutas, JEV NUNCA es sustituido ni simulado con métricas fabricadas. No existen tasas de éxito, intervalos de confianza ni latencias inventadas para A (estado: N/A / BLOQUEADO).",
            "- **Motores locales y dry-run no-oficiales:**",
            "  - `structured_rules`: Motor determinista local disponible. Evalúa reglas DSL contra hechos técnicos sin inferencia externa.",
            "  - `rag_llm` (dry-run) y `scrape_llm` (dry-run): Ejecutados en modo dry-run sin credenciales LLM, etiquetados estrictamente como `[DRY-RUN SINTÉTICO - NO OFICIAL]`.",
            "- **Sin reclamos de superioridad:** Al estar System A formalmente bloqueado y los motores B/C en modo dry-run sin LLM keys, **ningún motor ha demostrado superioridad empírica**. La comparación queda en standby hasta desbloquear las credenciales y recibir los 3 productos reales del usuario.",
            "",
            "### 2. ¿Qué parte del ahorro provino de preparar documentos, de JEV o de evitar generación?",
            "- **Estado de System A:** Recordatorio: System A está **BLOQUEADO (§0.1 #7)** por falta de `TYPESAFE_API_KEY`.",
            "- **Evitar generación libre de texto:** El diseño de JEV (`choice`) proyecta **$0.00 en tokens de salida** en evaluación estructurada tipada, eliminando la facturación de texto libre que encarece a B y C.",
            "- **Estado compacto vs contexto RAG:** La extracción previa a hechos sintéticos permite enviar estados compactos (~800 tokens), frente a chunks de texto de RAG que expanden el prompt a ~3,000 tokens en B.",
            "- **Atribución teórica:** La indexación estructurada previa y reglas deterministas eliminan latencias y costos de inferencia en comprobaciones exactas. La validación empírica requerirá las credenciales oficiales de JEV y LLM.",
            "",
            "### 3. ¿Qué costo tuvo convertir manuales a hechos y revisarlos?",
            "- **Datos sintéticos de demostración:** Medición sobre el fixture sintético (3 equipos FixtureCorp: P1, P2, P3):",
            f"- **Costo computacional offline:** ${a_prep:.2f} (extracción sintética en pipeline de desarrollo).",
            "- **Tiempo de auditoría técnica:** Validación de esquemas sintéticos sin costo comercial. Cuando se reciban los productos reales y se desbloquee System A (§0.1 #7), este proceso se ejecutará sobre manuales oficiales de fabricante.",
            "",
            "### 4. ¿Cuántos casos requirieron reglas, JEV, texto recuperado o scraping?",
            "- **Distribución en el piloto dry-run (FixtureCorp P1, P2, P3):**",
            "- **Reglas deterministas (`structured_rules`):** Evaluó los casos de prueba de forma determinista contra hechos locales sin dependencias de red.",
            "- **Evaluador semántico JEV:** Requerido en diseño para juicio semántico; no ejecutado por estar **BLOQUEADO (§0.1 #7, TYPESAFE_API_KEY vacía)**.",
            "- **Scraping online sintético (C):** Flujo de inspección documental y tool-calling simulado sobre endpoints locales del simulador de tienda.",
            "- **RAG sintético (B):** Recuperación híbrida BM25 + embeddings ejecutada sobre documentos de prueba.",
            "",
            "### 5. ¿La precisión de A vino de revisiones humanas que B/C no tenían?",
            "- **Integridad de datos:** System A está **BLOQUEADO (§0.1 #7)** y B/C en dry-run. No se inventan cifras comparativas entre hechos revisados y extracción automática.",
            "- La hipótesis metodológica de que los hechos técnicos auditados previenen alucinaciones frente a RAG/Scraping se evaluará empíricamente en el benchmark oficial una vez desbloqueados los accesos.",
            "",
            "### 6. ¿Qué errores de JEV fueron corregidos por reglas? ¿Qué errores no pudieron corregirse?",
            "- **Arquitectura híbrida determinista (System A BLOQUEADO §0.1 #7):**",
            "- **Corregidos por reglas deterministas:** En el diseño arquitectónico, discrepancias numéricas en umbrales de voltaje y corriente son rechazadas taxativamente por reglas de código antes o después del juicio semántico.",
            "- **No corregibles / abstención obligatoria:** Casos con información omitida intencionalmente (`INSUFFICIENT_EVIDENCE`), donde el sistema se abstiene conforme al protocolo §15.2.",
            "- **Aviso:** Este análisis describe las salvaguardas del diseño de software; no existen ejecuciones de JEV sin API key.",
            "",
            "### 7. ¿Qué pasó con C cuando se habilitó cache documental?",
            "- **Entorno de laboratorio:**",
            "- La política de cache documental entre solicitudes (§11.2) está implementada en el adaptador de Scraping para evitar re-descargas de manuales idénticos.",
            "- La cuantificación experimental de reducción de latencia y llamadas redundantes se registrará formalmente durante la corrida oficial.",
            "",
            "### 8. ¿El catálogo de tres productos era suficiente para poner a prueba selección semántica?",
            "- **CATÁLOGO SINTÉTICO DE PRUEBA (3 EQUIPOS FIXTURECORP: P1, P2, P3):** Los fixtures sintéticos permitieron validar el arnés de pruebas de extremo a extremo sin credenciales externas.",
            "- **Límite metodológico fundamental:** **NO constituye benchmark oficial ni catálogo definitivo.** Se trata de un fixture sintético de desarrollo. Además, **System A está BLOQUEADO (§0.1 #7)** por falta de `TYPESAFE_API_KEY`. Los resultados NUNCA deben comunicarse como superioridad en catálogo de producción ni como validación definitiva de JEV hasta que se complete la corrida oficial con los 3 productos reales del usuario y la API key activa.",
            "",
            "### 9. ¿Cómo cambia la conclusión al mantener el mismo estado técnico y reemplazar JEV por LLM?",
            "- **Ablación metodológica:**",
            "- System A está **BLOQUEADO (§0.1 #7)** y la ablación `structured_llm` se encuentra pendiente de credenciales LLM. Por tanto, **NO se reportan métricas inventadas de comparación entre JEV y LLM**.",
            "- La ablación `structured_rules` disponible confirma que la evaluación determinista local de reglas sobre hechos estructurados opera sin dependencias de red, pero la comparación semántica JEV vs LLM requiere las API keys de ambos proveedores.",
            "",
            "---",
            "",
            "## 25.4 Narrativa para el Hackathon",
            "",
            "> **AVISO DE DEMOSTRACIÓN SINTÉTICA — NO CONSTITUYE BENCHMARK OFICIAL:** En este banco de pruebas controlado sobre catálogo sintético (3 equipos FixtureCorp: P1, P2, P3), se verificó la integridad del arnés de pruebas y la ejecución local de motores. **El Sistema A (`structured_jev`) se encuentra formalmente BLOQUEADO (§0.1 #7)** debido a la ausencia de `TYPESAFE_API_KEY`. En estricto cumplimiento del protocolo, **NO se inventan ni reportan métricas de éxito para JEV**. Los motores B y C se ejecutaron en modo dry-run no-oficial, y `structured_rules` demostró la ejecución determinista local de reglas DSL. No se proclama superioridad técnica ni validación de benchmark hasta la entrega de los 3 productos reales del usuario y las claves de API requeridas.",
        ])

        return "\n".join(lines)

    def generate_html(self) -> str:
        """Generate a self-contained, responsive HTML dashboard with modern CSS."""
        md_content = self.generate_markdown()

        # Engine rows
        engine_rows_html = ""
        for eng in self.data.engines.values():
            is_blocked = (
                eng.engine_id == "structured_jev"
                and (
                    eng.total_cases == 0
                    or "BLOCKED" in eng.cost_status
                    or "BLOQUEADO" in eng.display_name
                    or "BLOCKED" in eng.display_name
                    or self.data.metadata.get("system_a_status") == "BLOCKED (§0.1 #7)"
                )
            )
            if is_blocked:
                engine_rows_html += f"""
            <tr style="background-color: #fff1f2;">
                <td><strong>{eng.display_name}</strong></td>
                <td><span class="badge" style="background:#fee2e2;color:#991b1b;">0 (BLOCKED)</span></td>
                <td><span class="badge" style="background:#fee2e2;color:#991b1b;">N/A (BLOCKED)</span></td>
                <td>N/A (BLOCKED)</td>
                <td>N/A (BLOCKED)</td>
                <td>N/A</td>
                <td>N/A</td>
                <td>N/A (BLOCKED)</td>
                <td>N/A (BLOCKED)</td>
                <td>N/A (BLOCKED)</td>
            </tr>
            """
            else:
                succ_ci = wilson_score_interval(eng.successful_tasks, eng.total_cases)
                fa_ci = wilson_score_interval(eng.false_approvals, eng.incompatible_cases)
                ea_ci = wilson_score_interval(eng.excessive_abstentions, eng.resolvable_cases)
                cost_task_val = eng.cost_per_correct_task
                if cost_task_val == float("inf"):
                    cost_task_str = "∞ (0 éxitos)"
                elif cost_task_val is not None:
                    cost_task_str = f"${cost_task_val:.4f}"
                else:
                    cost_task_str = "Unknown"

                p50_str = f"{eng.p50_ms:,.0f} ms" if (eng.latencies_ms and eng.p50_ms is not None) else "-"
                p95_str = f"{eng.p95_ms:,.0f} ms" if (eng.latencies_ms and eng.p95_ms is not None) else "-"
                cost_str = f"${eng.total_cost_usd:.3f}" if eng.total_cost_usd is not None else "Unknown"

                if eng.incompatible_cases > 0 and eng.false_approval_rate_pct is not None:
                    badge_cls = 'badge-safe' if eng.false_approval_rate_pct < 10 else 'badge-warning'
                    fa_cell = f'<span class="{badge_cls}">{eng.false_approvals}/{eng.incompatible_cases} ({eng.false_approval_rate_pct:.1f}%)</span><br><small class="ci-text">CI: [{fa_ci[0]}% - {fa_ci[1]}%]</small>'
                else:
                    fa_cell = '<span class="badge-neutral">N/A</span><br><small class="ci-text">0 no compatibles</small>'

                if eng.resolvable_cases > 0 and eng.excessive_abstention_rate_pct is not None:
                    ea_cell = f'{eng.excessive_abstentions}/{eng.resolvable_cases} ({eng.excessive_abstention_rate_pct:.1f}%)<br><small class="ci-text">CI: [{ea_ci[0]}% - {ea_ci[1]}%]</small>'
                else:
                    ea_cell = '<span class="badge-neutral">N/A</span><br><small class="ci-text">0 resolubles</small>'

                engine_rows_html += f"""
            <tr>
                <td><strong>{eng.display_name}</strong></td>
                <td>{eng.total_cases} <span class="badge badge-sub">{eng.total_requests} reqs</span></td>
                <td><span class="metric-highlight">{eng.successful_tasks}/{eng.total_cases}</span> ({eng.success_rate_pct:.1f}%)<br><small class="ci-text">CI: [{succ_ci[0]}% - {succ_ci[1]}%]</small></td>
                <td>{fa_cell}</td>
                <td>{ea_cell}</td>
                <td><strong>{p50_str}</strong></td>
                <td>{p95_str}</td>
                <td>{cost_str} <span class="badge badge-neutral">{eng.cost_status}</span></td>
                <td><strong>{cost_task_str}</strong></td>
                <td>{eng.preparation_summary_str}</td>
            </tr>
            """

        if not engine_rows_html:
            engine_rows_html = """
            <tr>
                <td colspan="10" style="text-align: center; color: var(--text-secondary); padding: 18px;">
                    Sin datos de ejecución registrados.
                </td>
            </tr>
            """

        # Ablations rows
        ablation_rows_html = ""
        for abl in self.data.ablations.values():
            is_abl_blocked = (
                abl.total_cases == 0
                or "BLOCKED" in abl.display_name
                or "BLOCKED" in abl.cost_status
            )
            if is_abl_blocked:
                ablation_rows_html += f"""
            <tr style="background-color: #fff1f2;">
                <td><strong>{abl.display_name}</strong></td>
                <td><span class="badge" style="background:#fee2e2;color:#991b1b;">0 (BLOCKED)</span></td>
                <td><span class="badge" style="background:#fee2e2;color:#991b1b;">N/A (BLOCKED)</span></td>
                <td>N/A (BLOCKED)</td>
                <td>N/A (BLOCKED)</td>
                <td>N/A</td>
                <td>N/A</td>
                <td>N/A (BLOCKED)</td>
                <td>N/A (BLOCKED)</td>
                <td>N/A (BLOCKED)</td>
            </tr>
            """
            else:
                succ_ci = wilson_score_interval(abl.successful_tasks, abl.total_cases)
                fa_ci = wilson_score_interval(abl.false_approvals, abl.incompatible_cases)
                ea_ci = wilson_score_interval(abl.excessive_abstentions, abl.resolvable_cases)
                cost_task_val = abl.cost_per_correct_task
                if cost_task_val == float("inf"):
                    cost_task_str = "∞ (0 éxitos)"
                elif cost_task_val is not None:
                    cost_task_str = f"${cost_task_val:.4f}"
                else:
                    cost_task_str = "Unknown"

                p50_str = f"{abl.p50_ms:,.0f} ms" if (abl.latencies_ms and abl.p50_ms is not None) else "-"
                p95_str = f"{abl.p95_ms:,.0f} ms" if (abl.latencies_ms and abl.p95_ms is not None) else "-"
                cost_str = f"${abl.total_cost_usd:.3f}" if abl.total_cost_usd is not None else "Unknown"

                if abl.incompatible_cases > 0 and abl.false_approval_rate_pct is not None:
                    fa_abl_cell = f"{abl.false_approvals}/{abl.incompatible_cases} ({abl.false_approval_rate_pct:.1f}%)"
                else:
                    fa_abl_cell = "N/A"

                if abl.resolvable_cases > 0 and abl.excessive_abstention_rate_pct is not None:
                    ea_abl_cell = f"{abl.excessive_abstentions}/{abl.resolvable_cases} ({abl.excessive_abstention_rate_pct:.1f}%)"
                else:
                    ea_abl_cell = "N/A"

                ablation_rows_html += f"""
            <tr>
                <td><strong>{abl.display_name}</strong></td>
                <td>{abl.total_cases}</td>
                <td>{abl.successful_tasks}/{abl.total_cases} ({abl.success_rate_pct:.1f}%)</td>
                <td>{fa_abl_cell}</td>
                <td>{ea_abl_cell}</td>
                <td>{p50_str}</td>
                <td>{p95_str}</td>
                <td>{cost_str}</td>
                <td>{cost_task_str}</td>
                <td>{abl.preparation_summary_str}</td>
            </tr>
            """

        if not ablation_rows_html:
            ablation_rows_html = """
            <tr>
                <td colspan="10" style="text-align: center; color: var(--text-secondary); padding: 18px;">
                    Sin ablaciones metodológicas registradas para esta ejecución.
                </td>
            </tr>
            """

        # Projections table
        volumes = [1, 10, 100, 1000, 10000]
        a_eng = self.data.engines.get("structured_jev")
        b_eng = self.data.engines.get("rag_llm")
        c_eng = self.data.engines.get("scrape_llm")

        a_blocked = (
            a_eng is None
            or a_eng.total_cases == 0
            or "BLOCKED" in a_eng.cost_status
            or "BLOQUEADO" in a_eng.display_name
            or "BLOCKED" in a_eng.display_name
            or self.data.metadata.get("system_a_status") == "BLOCKED (§0.1 #7)"
        )

        a_prep = float(a_eng.preparation_compute_cost_usd or 0.0) if a_eng else 0.0
        b_prep = float(b_eng.preparation_compute_cost_usd or 0.0) if b_eng else 0.0
        c_prep = float(c_eng.preparation_compute_cost_usd or 0.0) if c_eng else 0.0

        a_on = (a_eng.total_cost_usd / a_eng.total_cases) if (a_eng and a_eng.total_cases > 0 and a_eng.total_cost_usd is not None) else 0.0
        b_on = (b_eng.total_cost_usd / b_eng.total_cases) if (b_eng and b_eng.total_cases > 0 and b_eng.total_cost_usd is not None) else 0.0
        c_on = (c_eng.total_cost_usd / c_eng.total_cases) if (c_eng and c_eng.total_cases > 0 and c_eng.total_cost_usd is not None) else 0.0

        projection_rows_html = ""
        for n in volumes:
            a_cell = "<td>N/A (BLOCKED)</td>" if a_blocked else f"<td>${CostCalculator.calculate_amortized_cost(a_prep, a_on, n):.4f}</td>"
            b_am = CostCalculator.calculate_amortized_cost(b_prep, b_on, n)
            c_am = CostCalculator.calculate_amortized_cost(c_prep, c_on, n)
            projection_rows_html += f"""
            <tr>
                <td><strong>N = {n:,}</strong></td>
                {a_cell}
                <td>${b_am:.4f}</td>
                <td>${c_am:.4f}</td>
            </tr>
            """

        if a_blocked:
            be_html_str = "System A se encuentra BLOQUEADO (§0.1 #7) por falta de TYPESAFE_API_KEY. No se proyecta punto de equilibrio de A vs C en ausencia de credenciales oficiales."
        else:
            be_ac = CostCalculator.calculate_break_even(a_prep, c_prep, a_on, c_on)
            be_html_str = (
                f"A supera a Scraping a partir de ~{be_ac:,.0f} consultas amortizadas."
                if be_ac is not None
                else "No alcanza punto de equilibrio entre A y C con los datos observados."
            )

        # Dynamic KPI Cards
        a_eng = self.data.engines.get("structured_jev")
        a_blocked = (
            a_eng is None
            or a_eng.total_cases == 0
            or "BLOCKED" in a_eng.cost_status
            or "BLOQUEADO" in a_eng.display_name
            or "BLOCKED" in a_eng.display_name
            or self.data.metadata.get("system_a_status") == "BLOCKED (§0.1 #7)"
        )

        if a_blocked:
            kpi_succ_val = "BLOQUEADO"
            kpi_succ_sub = "System A: sin TYPESAFE_API_KEY (§0.1 #7)"
            rules_eng = self.data.engines.get("structured_rules")
            if rules_eng and rules_eng.latencies_ms and rules_eng.p50_ms is not None:
                kpi_p50_val = f"{rules_eng.p50_ms:,.0f} ms"
                kpi_p50_sub = "structured_rules (reglas locales)"
            else:
                kpi_p50_val = "-"
                kpi_p50_sub = "Sin latencias de System A"
            kpi_fa_val = "N/A"
            kpi_fa_sub = "System A bloqueado (§0.1 #7)"
            kpi_cpt_val = "N/A"
            kpi_cpt_sub = "Sin métricas inventadas"
        elif self.data.engines:
            lead_eng = next(iter(self.data.engines.values()))
            if lead_eng.total_cases > 0:
                kpi_succ_val = f"{lead_eng.success_rate_pct:.1f}%"
                kpi_succ_sub = f"{lead_eng.display_name} ({lead_eng.successful_tasks}/{lead_eng.total_cases} casos)"
                kpi_p50_val = f"{lead_eng.p50_ms:,.0f} ms" if (lead_eng.latencies_ms and lead_eng.p50_ms is not None) else "-"
                kpi_p50_sub = f"{lead_eng.display_name} p50"
                if lead_eng.incompatible_cases > 0 and lead_eng.false_approval_rate_pct is not None:
                    kpi_fa_val = f"{lead_eng.false_approval_rate_pct:.1f}%"
                    kpi_fa_sub = f"{lead_eng.false_approvals}/{lead_eng.incompatible_cases} casos incompatibles"
                else:
                    kpi_fa_val = "N/A"
                    kpi_fa_sub = "0 casos no compatibles"
                cpt = lead_eng.cost_per_correct_task
                if cpt == float("inf"):
                    kpi_cpt_val = "∞ (0 éxitos)"
                elif cpt is not None:
                    kpi_cpt_val = f"${cpt:.4f}"
                else:
                    kpi_cpt_val = "Unknown"
                kpi_cpt_sub = "Total online / tareas exitosas"
            else:
                kpi_succ_val = "0.0%"
                kpi_succ_sub = f"{lead_eng.display_name} (0 casos)"
                kpi_p50_val = "-"
                kpi_p50_sub = "Sin latencias"
                kpi_fa_val = "N/A"
                kpi_fa_sub = "0 casos no compatibles"
                kpi_cpt_val = "N/A"
                kpi_cpt_sub = "Sin casos evaluados"
        else:
            kpi_succ_val = "N/A"
            kpi_succ_sub = "Sin motores registrados"
            kpi_p50_val = "-"
            kpi_p50_sub = "Sin latencias"
            kpi_fa_val = "N/A"
            kpi_fa_sub = "Sin datos"
            kpi_cpt_val = "N/A"
            kpi_cpt_sub = "Sin datos"

        html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Industrial Selection Lab — Benchmark Report</title>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --surface-color: #ffffff;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --border-color: #e2e8f0;
            --primary-blue: #0284c7;
            --accent-green: #16a34a;
            --accent-amber: #d97706;
            --accent-red: #dc2626;
            --card-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1300px;
            margin: 0 auto;
        }}
        header {{
            background: var(--surface-color);
            padding: 24px 32px;
            border-radius: 12px;
            box-shadow: var(--card-shadow);
            margin-bottom: 24px;
            border-left: 6px solid var(--primary-blue);
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 1.8rem;
            color: #0369a1;
        }}
        .header-meta {{
            font-size: 0.9rem;
            color: var(--text-secondary);
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--surface-color);
            padding: 20px;
            border-radius: 10px;
            box-shadow: var(--card-shadow);
            border-top: 3px solid var(--primary-blue);
        }}
        .kpi-label {{
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
        }}
        .kpi-subtext {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}
        .section-card {{
            background: var(--surface-color);
            padding: 24px;
            border-radius: 12px;
            box-shadow: var(--card-shadow);
            margin-bottom: 24px;
        }}
        h2 {{
            margin-top: 0;
            font-size: 1.3rem;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 12px;
            color: #0f172a;
        }}
        h3 {{
            font-size: 1.05rem;
            color: #1e293b;
            margin-top: 18px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin-top: 12px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: #f1f5f9;
            font-weight: 600;
            color: #334155;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        tr:hover {{
            background-color: #f8fafc;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-safe {{
            background-color: #dcfce7;
            color: #15803d;
        }}
        .badge-warning {{
            background-color: #fef3c7;
            color: #b45309;
        }}
        .badge-sub {{
            background-color: #e0f2fe;
            color: #0369a1;
        }}
        .badge-neutral {{
            background-color: #f1f5f9;
            color: #475569;
        }}
        .ci-text {{
            color: #64748b;
            font-size: 0.75rem;
        }}
        .metric-highlight {{
            font-weight: 700;
            color: #0284c7;
        }}
        .insight-card {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid var(--primary-blue);
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 14px;
        }}
        .insight-title {{
            font-weight: 700;
            color: #0f172a;
            font-size: 0.95rem;
            margin-bottom: 6px;
        }}
        .insight-body {{
            font-size: 0.9rem;
            color: #334155;
            line-height: 1.5;
        }}
        .narrative-box {{
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border: 1px solid #bae6fd;
            border-radius: 8px;
            padding: 20px;
            font-size: 0.95rem;
            color: #0369a1;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            {f'''<div style="background-color: #fee2e2; border: 2px solid #ef4444; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                <span class="badge" style="background-color: #dc2626; color: #ffffff; font-size: 0.85rem; padding: 4px 10px;">[DEMO / SYNTHETIC / NON-OFFICIAL] [DEMO / SYNTHETIC FIXTURE / NON-OFFICIAL]</span>
                <h3 style="color: #991b1b; margin: 10px 0 6px 0; font-size: 1.05rem; text-transform: uppercase;">
                    AVISO: DATOS SINTÉTICOS DE DEMOSTRACIÓN — NO CONSTITUYE BENCHMARK OFICIAL NI CORRIDA REAL
                </h3>
                <p style="margin: 0 0 6px 0; font-weight: 600; color: #7f1d1d;">
                    CATÁLOGO SINTÉTICO DE PRUEBA (3 EQUIPOS FIXTURECORP: P1, P2, P3)
                </p>
                <p style="margin: 0 0 6px 0; font-size: 0.88rem; color: #7f1d1d;">
                    <strong>data_origin:</strong> <code>synthetic_fixture</code>
                </p>
                <p style="margin: 0; font-size: 0.88rem; color: #991b1b;">
                    <strong>ESTADO SISTEMA A (structured_jev): BLOQUEADO / BLOCKED (§0.1 #7)</strong> debido a <code>TYPESAFE_API_KEY</code> vacía / no configurada. JEV de TypeSafe AI NUNCA debe ser sustituido por otro modelo ni por mocks en benchmark oficial. Sin reclamos de superioridad técnica sobre productos de producción.
                </p>
            </div>''' if self.is_demo else ''}
            <h1>Industrial Selection Lab — Benchmark Report{f' <span class="badge badge-warning" style="font-size: 0.8rem; vertical-align: middle;">[DEMO / SYNTHETIC / NON-OFFICIAL] [DEMO / SYNTHETIC FIXTURE / NON-OFFICIAL]</span>' if self.is_demo else ''}</h1>
            <div class="header-meta">
                <strong>Run ID:</strong> {self.data.run_id} &bull; 
                <strong>Generated:</strong> {self.data.created_utc} &bull; 
                <strong>data_origin:</strong> <code>synthetic_fixture</code> &bull; 
                <strong>Benchmark Spec:</strong> MEGAPLAN.md §25{f' &bull; <span style="color: #dc2626; font-weight: bold;">[DEMO / SYNTHETIC / NON-OFFICIAL] [DEMO / SYNTHETIC FIXTURE / NON-OFFICIAL]</span>' if self.is_demo else ''}
            </div>
        </header>

        <!-- KPI Summary Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Éxito Completo (§15.1){' (DEMO SINTÉTICO)' if self.is_demo else ''}</div>
                <div class="kpi-value" style="color: var(--accent-green);">{kpi_succ_val}</div>
                <div class="kpi-subtext">{kpi_succ_sub}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Latencia Mediana (p50){' (DEMO SINTÉTICO)' if self.is_demo else ''}</div>
                <div class="kpi-value" style="color: var(--primary-blue);">{kpi_p50_val}</div>
                <div class="kpi-subtext">{kpi_p50_sub}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Falsas Aprobaciones{' (DEMO SINTÉTICO)' if self.is_demo else ''}</div>
                <div class="kpi-value" style="color: var(--accent-green);">{kpi_fa_val}</div>
                <div class="kpi-subtext">{kpi_fa_sub}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Costo por Tarea Correcta{' (DEMO)' if self.is_demo else ''}</div>
                <div class="kpi-value">{kpi_cpt_val}</div>
                <div class="kpi-subtext">{kpi_cpt_sub}</div>
            </div>
        </div>

        <!-- Section 25.1 Main Table -->
        <div class="section-card">
            <h2>25.1 Tabla Principal de Resultados</h2>
            {f'''<div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 12px 16px; margin-bottom: 16px; border-radius: 4px; font-size: 0.88rem; color: #991b1b;">
                <strong>Integridad de Resultados (Modo Demostración):</strong> Datos obtenidos sobre <strong>CATÁLOGO SINTÉTICO DE PRUEBA (3 EQUIPOS FIXTURECORP: P1, P2, P3)</strong>.<br>
                <strong>Sistema A (structured_jev) está BLOQUEADO (§0.1 #7)</strong> debido a <code>TYPESAFE_API_KEY</code> vacía. TypeSafe AI JEV NUNCA debe sustituirse por otro modelo ni mock en benchmark oficial. Las métricas de A corresponden a dry-run / reglas de simulación con 0 validez técnica real.
            </div>''' if self.is_demo else ''}
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Motor</th>
                            <th>Casos</th>
                            <th>Éxitos Completos</th>
                            <th>Falsas Aprobaciones</th>
                            <th>Abstención Excesiva</th>
                            <th>p50 ms</th>
                            <th>p95 ms</th>
                            <th>Costo Total</th>
                            <th>USD / Tarea Correcta</th>
                            <th>Preparación</th>
                        </tr>
                    </thead>
                    <tbody>
                        {engine_rows_html}
                    </tbody>
                </table>
            </div>

            <h3 style="margin-top: 28px;">Ablaciones Metodológicas</h3>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Motor (Ablación)</th>
                            <th>Casos</th>
                            <th>Éxitos Completos</th>
                            <th>Falsas Aprobaciones</th>
                            <th>Abstención Excesiva</th>
                            <th>p50 ms</th>
                            <th>p95 ms</th>
                            <th>Costo Total</th>
                            <th>USD / Tarea Correcta</th>
                            <th>Preparación</th>
                        </tr>
                    </thead>
                    <tbody>
                        {ablation_rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Section 25.2 Amortized Cost Projection -->
        <div class="section-card">
            <h2>25.2 Proyección de Costo Amortizado vs Volumen N (§16.3)</h2>
            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                Fórmula: <code>amortized_cost(N) = (preparation_cost + N * mean_online_cost) / N</code>
            </p>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Volumen (N)</th>
                            <th>A: structured_jev</th>
                            <th>B: rag_llm</th>
                            <th>C: scrape_llm</th>
                        </tr>
                    </thead>
                    <tbody>
                        {projection_rows_html}
                    </tbody>
                </table>
            </div>
            <p style="font-size: 0.85rem; color: #64748b; margin-top: 10px;">
                <strong>Punto de equilibrio A vs C:</strong> {be_html_str}
            </p>
        </div>

        <!-- Section 25.3 Mandatory Insights -->
        <div class="section-card">
            <h2>25.3 Insights Obligatorios (§25.3)</h2>

            <div class="insight-card">
                <div class="insight-title">1. ¿Qué motor ganó cada dimensión y con qué incertidumbre?</div>
                <div class="insight-body">
                    <p style="color: #b91c1c; font-weight: 600; margin-top: 0;">
                        [DEMO / SYNTHETIC / NON-OFFICIAL] Sistema A (structured_jev) está BLOQUEADO (§0.1 #7) por falta de TYPESAFE_API_KEY. JEV NUNCA debe ser sustituido ni falseado en benchmark oficial. Cero métricas inventadas para JEV.
                    </p>
                    <strong>Integridad de motores:</strong> Al estar System A formalmente bloqueado y los motores B/C en modo dry-run sin claves LLM, <strong>ningún motor ha demostrado superioridad empírica</strong>.<br>
                    <strong>Motor local:</strong> <code>structured_rules</code> está disponible para evaluación determinista local contra hechos estructurados sin dependencias externas.<br>
                    <strong>Dry-run B/C:</strong> <code>rag_llm</code> y <code>scrape_llm</code> ejecutaron rutas offline de validación etiquetadas como [DRY-RUN SINTÉTICO - NO OFICIAL].
                </div>
            </div>

            <div class="insight-card">
                <div class="insight-title">2. ¿Qué parte del ahorro provino de preparar documentos, de JEV o de evitar generación?</div>
                <div class="insight-body">
                    <p style="color: #b91c1c; font-weight: 600; margin-top: 0;">
                        [ESTADO SYSTEM A: BLOQUEADO §0.1 #7 por TYPESAFE_API_KEY vacía]
                    </p>
                    El ahorro potencial de la arquitectura estructurada se atribuye teóricamente a tres factores de diseño: (1) Ausencia de tokens tarifados de salida gracias a la evaluación tipada Choice proyectada en JEV ($0.00 output); (2) Estado estructurado compacto de ~800 tokens frente a chunks de RAG de ~3,000 tokens en B; (3) Descarte previo determinista mediante reglas numéricas locales. En corridas oficiales, este comportamiento debe verificarse con JEV real sin sustituciones.
                </div>
            </div>

            <div class="insight-card">
                <div class="insight-title">3. ¿Qué costo tuvo convertir manuales a hechos y revisarlos?</div>
                <div class="insight-body">
                    En el banco de pruebas sintético (3 equipos FixtureCorp: P1, P2, P3), la preparación computacional preliminar de prueba se ejecutó en pipeline de desarrollo sin costo comercial. Cuando se reciban los productos reales y se desbloquee System A (§0.1 #7), el proceso se ejecutará sobre manuales oficiales de fabricante.
                </div>
            </div>

            <div class="insight-card">
                <div class="insight-title">4. ¿Cuántos casos requirieron reglas, JEV, texto recuperado o scraping?</div>
                <div class="insight-body">
                    En el piloto dry-run sobre fixtures sintéticos (FixtureCorp P1, P2, P3): <code>structured_rules</code> resolvió reglas locales de forma determinista; <code>rag_llm</code> ejecutó recuperación híbrida sobre documentos de prueba; <code>scrape_llm</code> inspeccionó páginas del simulador local. System A no procesó casos al estar <strong>BLOQUEADO (§0.1 #7)</strong> por ausencia de <code>TYPESAFE_API_KEY</code>.
                </div>
            </div>

            <div class="insight-card">
                <div class="insight-title">5. ¿La precisión de A vino de revisiones humanas que B/C no tenían?</div>
                <div class="insight-body">
                    System A está <strong>BLOQUEADO (§0.1 #7)</strong> y B/C en dry-run. No se inventan cifras comparativas entre hechos revisados y extracción automática. La evaluación empírica de hechos auditados frente a texto libre se realizará en el benchmark oficial.
                </div>
            </div>

            <div class="insight-card">
                <div class="insight-title">6. ¿Qué errores de JEV fueron corregidos por reglas? ¿Qué errores no pudieron corregirse?</div>
                <div class="insight-body">
                    En la integración del pipeline (System A BLOQUEADO §0.1 #7), las reglas deterministas invalidan discrepancias numéricas en umbrales de voltaje y corriente. Los casos con información omitida intencionalmente derivan en abstención obligatoria (<code>INSUFFICIENT_EVIDENCE</code> §15.2). Este análisis describe el diseño del software sin métricas inventadas.
                </div>
            </div>

            <div class="insight-card">
                <div class="insight-title">7. ¿Qué pasó con C cuando se habilitó cache documental?</div>
                <div class="insight-body">
                    La política de cache documental entre solicitudes (§11.2) está implementada en el adaptador de Scraping para evitar re-descargas redundantes. La cuantificación empírica de reducción de latencia y llamadas redundantes se medirá durante la corrida oficial.
                </div>
            </div>

            <div class="insight-card">
                <div class="insight-title">8. ¿El catálogo de tres productos era suficiente para poner a prueba selección semántica?</div>
                <div class="insight-body">
                    <strong>CATÁLOGO SINTÉTICO DE PRUEBA (3 EQUIPOS FIXTURECORP: P1, P2, P3):</strong> Permitió validar el arnés de pruebas de extremo a extremo sin credenciales externas. Se declara expresamente como un banco de pruebas de desarrollo: NO constituye benchmark oficial ni catálogo definitivo. System A está BLOQUEADO (§0.1 #7); nunca debe proclamarse superioridad en catálogo de producción ni como validación definitiva de JEV hasta que se complete la corrida oficial con los 3 productos reales del usuario y la API key activa.
                </div>
            </div>

            <div class="insight-card">
                <div class="insight-title">9. ¿Cómo cambia la conclusión al mantener el mismo estado técnico y reemplazar JEV por LLM?</div>
                <div class="insight-body">
                    System A está <strong>BLOQUEADO (§0.1 #7)</strong> y la ablación <code>structured_llm</code> se encuentra pendiente de credenciales LLM. Por tanto, <strong>NO se reportan métricas inventadas de comparación entre JEV y LLM</strong>. La ablación <code>structured_rules</code> confirma la viabilidad de la evaluación determinista local.
                </div>
            </div>
        </div>

        <!-- Section 25.4 Hackathon Narrative -->
        <div class="section-card">
            <h2>25.4 Narrativa para el Hackathon</h2>
            <div class="narrative-box">
                <strong>AVISO DE DEMOSTRACIÓN SINTÉTICA — NO CONSTITUYE BENCHMARK OFICIAL:</strong> En este banco de pruebas controlado sobre catálogo sintético (3 equipos FixtureCorp: P1, P2, P3), se verificó la integridad del arnés de pruebas y la ejecución local de motores. <strong>El Sistema A (structured_jev) se encuentra formalmente BLOQUEADO (§0.1 #7)</strong> debido a la ausencia de <code>TYPESAFE_API_KEY</code>. En estricto cumplimiento del protocolo, <strong>NO se inventan ni reportan métricas de éxito para JEV</strong>. Los motores B y C se ejecutaron en modo dry-run no-oficial, y <code>structured_rules</code> demostró la ejecución determinista local de reglas DSL. No se proclama superioridad técnica ni validación de benchmark hasta la entrega de los 3 productos reales del usuario y las claves de API requeridas.
            </div>
        </div>
    </div>
</body>
</html>
        """
        return html

    def generate_summary_csv(self) -> str:
        """Generate single-source summary.csv matching evaluation metrics."""
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)
        headers = [
            "engine",
            "cases",
            "completed",
            "task_success_rate",
            "content_fully_correct_rate",
            "operationally_valid_rate",
            "false_approval_rate",
            "excessive_abstention_rate",
            "latency_p50_ms",
            "latency_p95_ms",
            "total_cost_usd",
            "cost_status",
        ]
        writer.writerow(headers)

        for eng in self.data.engines.values():
            fa_val = (
                f"{eng.false_approval_rate_pct / 100.0:.4f}"
                if (eng.incompatible_cases > 0 and eng.false_approval_rate_pct is not None)
                else "N/A"
            )
            ea_val = (
                f"{eng.excessive_abstention_rate_pct / 100.0:.4f}"
                if (eng.resolvable_cases > 0 and eng.excessive_abstention_rate_pct is not None)
                else "N/A"
            )
            ts_val = f"{eng.success_rate_pct / 100.0:.4f}" if eng.total_cases > 0 else "N/A"
            cfc_val = f"{eng.content_fully_correct_rate_pct / 100.0:.4f}" if eng.total_cases > 0 else "N/A"
            ov_val = f"{eng.operationally_valid_rate_pct / 100.0:.4f}" if eng.total_cases > 0 else "N/A"
            p50_val = f"{eng.p50_ms:.2f}" if (eng.latencies_ms and eng.p50_ms is not None) else "N/A"
            p95_val = f"{eng.p95_ms:.2f}" if (eng.latencies_ms and eng.p95_ms is not None) else "N/A"
            cost_val = f"{eng.total_cost_usd:.4f}" if eng.total_cost_usd is not None else "N/A"

            row = [
                eng.engine_id,
                eng.total_cases,
                len([l for l in eng.latencies_ms if l is not None]),
                ts_val,
                cfc_val,
                ov_val,
                fa_val,
                ea_val,
                p50_val,
                p95_val,
                cost_val,
                eng.cost_status,
            ]
            writer.writerow(row)

        return output.getvalue()

    def generate_scorecard_json(self) -> str:
        """Generate single-source scorecard.json matching evaluation metrics."""
        scorecard: dict[str, Any] = {
            "run_id": self.data.run_id,
            "created_utc": self.data.created_utc,
            "benchmark_version": self.data.benchmark_version,
            "metadata": self.data.metadata,
            "engines": {},
            "ablations": {},
        }
        for eng_id, eng in self.data.engines.items():
            scorecard["engines"][eng_id] = {
                "display_name": eng.display_name,
                "total_cases": eng.total_cases,
                "total_requests": eng.total_requests,
                "successful_tasks": eng.successful_tasks,
                "task_success_rate_pct": eng.success_rate_pct,
                "content_fully_correct_tasks": eng.content_fully_correct_tasks,
                "content_fully_correct_rate_pct": eng.content_fully_correct_rate_pct,
                "operationally_valid_tasks": eng.operationally_valid_tasks,
                "operationally_valid_rate_pct": eng.operationally_valid_rate_pct,
                "false_approvals": eng.false_approvals,
                "incompatible_cases": eng.incompatible_cases,
                "false_approval_rate_pct": eng.false_approval_rate_pct,
                "excessive_abstentions": eng.excessive_abstentions,
                "resolvable_cases": eng.resolvable_cases,
                "excessive_abstention_rate_pct": eng.excessive_abstention_rate_pct,
                "p50_ms": eng.p50_ms if eng.latencies_ms else None,
                "p95_ms": eng.p95_ms if eng.latencies_ms else None,
                "total_cost_usd": eng.total_cost_usd,
                "cost_status": eng.cost_status,
                "cost_per_correct_task_usd": eng.cost_per_correct_task,
            }
        for abl_id, abl in self.data.ablations.items():
            scorecard["ablations"][abl_id] = {
                "display_name": abl.display_name,
                "total_cases": abl.total_cases,
                "successful_tasks": abl.successful_tasks,
                "task_success_rate_pct": abl.success_rate_pct,
                "content_fully_correct_tasks": abl.content_fully_correct_tasks,
                "content_fully_correct_rate_pct": abl.content_fully_correct_rate_pct,
                "operationally_valid_tasks": abl.operationally_valid_tasks,
                "operationally_valid_rate_pct": abl.operationally_valid_rate_pct,
                "false_approvals": abl.false_approvals,
                "incompatible_cases": abl.incompatible_cases,
                "false_approval_rate_pct": abl.false_approval_rate_pct,
                "excessive_abstentions": abl.excessive_abstentions,
                "resolvable_cases": abl.resolvable_cases,
                "excessive_abstention_rate_pct": abl.excessive_abstention_rate_pct,
                "p50_ms": abl.p50_ms if abl.latencies_ms else None,
                "total_cost_usd": abl.total_cost_usd,
                "cost_status": abl.cost_status,
            }
        return json.dumps(scorecard, indent=2, ensure_ascii=False)

    def generate_comparison_md(self) -> str:
        """Generate single-source comparison.md matching evaluation metrics."""
        lines = [
            f"# Comparativa de Motores — {self.data.run_id}",
            "",
            f"**Fecha UTC:** {self.data.created_utc}  ",
            f"**Versión Benchmark:** {self.data.benchmark_version}  ",
        ]
        if self.is_demo:
            lines.extend([
                "**Estado:** DEMO / SINTÉTICO / NO OFICIAL  ",
                "",
                "> [!WARNING]",
                "> Datos sintéticos de demostración. No constituye benchmark oficial.",
            ])
        lines.extend([
            "",
            "## 1. Tabla Única de Métricas Consolidadas",
            "",
            "| Motor | Casos | Content Fully Correct | Operationally Valid | Task Success (Primary) | Falsas Aprobaciones | Abstención Excesiva | Latencia p50 (ms) | Costo Total |",
            "|---|---|---|---|---|---|---|---|---|",
        ])
        for eng in self.data.engines.values():
            cfc_str = f"{eng.content_fully_correct_tasks}/{eng.total_cases} ({eng.content_fully_correct_rate_pct:.1f}%)"
            ov_str = f"{eng.operationally_valid_tasks}/{eng.total_cases} ({eng.operationally_valid_rate_pct:.1f}%)"
            ts_str = f"{eng.successful_tasks}/{eng.total_cases} ({eng.success_rate_pct:.1f}%)"
            if eng.incompatible_cases > 0 and eng.false_approval_rate_pct is not None:
                fa_str = f"{eng.false_approvals}/{eng.incompatible_cases} ({eng.false_approval_rate_pct:.1f}%)"
            else:
                fa_str = "N/A (sin casos no compatibles)"
            if eng.resolvable_cases > 0 and eng.excessive_abstention_rate_pct is not None:
                ea_str = f"{eng.excessive_abstentions}/{eng.resolvable_cases} ({eng.excessive_abstention_rate_pct:.1f}%)"
            else:
                ea_str = "N/A (sin casos resolubles)"
            p50_str = f"{eng.p50_ms:,.0f} ms" if (eng.latencies_ms and eng.p50_ms is not None) else "-"
            cost_str = f"${eng.total_cost_usd:.3f}" if eng.total_cost_usd is not None else "Unknown"

            lines.append(
                f"| **{eng.display_name}** | {eng.total_cases} | {cfc_str} | {ov_str} | {ts_str} | {fa_str} | {ea_str} | {p50_str} | {cost_str} |"
            )

        lines.extend([
            "",
            "## 2. Definición Unificada de Métricas (REPAIR3_PLAN §10.3)",
            "",
            "- **`content_fully_correct`**: todos los items exigidos correctos y soportados documentalmente.",
            "- **`operationally_valid`**: ejecución completada exitosamente sin fallos de formato/esquema.",
            "- **`task_success`**: `content_fully_correct AND operationally_valid` (métrica primaria §15.1).",
            "- **`false_approval_rate`**: aprobaciones erróneas entre casos no compatibles (N/A si denominador es 0).",
            "",
        ])
        return "\n".join(lines)

    def generate_informe_final_md(self) -> str:
        """Generate single-source INFORME_FINAL.md matching evaluation metrics."""
        lines = [
            f"# Informe Final de Evaluación — {self.data.run_id}",
            "",
            f"**Fecha UTC:** {self.data.created_utc}  ",
            f"**Versión Benchmark:** {self.data.benchmark_version}  ",
            "",
            "## Resumen Ejecutivo",
            "",
            "Este informe consolida los resultados del benchmark a partir de una **única fuente de evaluación** "
            "(REPAIR3_PLAN §10 y §11), garantizando consistencia absoluta entre `summary.csv`, `scorecard.json`, "
            "`comparison.md` e `INFORME_FINAL.md`.",
            "",
            "## Resultados Principales",
            "",
            "| Motor | Casos | Content Fully Correct | Operationally Valid | Task Success (Primario) | Falsas Aprobaciones | p50 (ms) | Costo ($) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for eng in self.data.engines.values():
            cfc_str = f"{eng.content_fully_correct_tasks}/{eng.total_cases} ({eng.content_fully_correct_rate_pct:.1f}%)"
            ov_str = f"{eng.operationally_valid_tasks}/{eng.total_cases} ({eng.operationally_valid_rate_pct:.1f}%)"
            ts_str = f"{eng.successful_tasks}/{eng.total_cases} ({eng.success_rate_pct:.1f}%)"
            if eng.incompatible_cases > 0 and eng.false_approval_rate_pct is not None:
                fa_str = f"{eng.false_approvals}/{eng.incompatible_cases} ({eng.false_approval_rate_pct:.1f}%)"
            else:
                fa_str = "N/A"
            p50_str = f"{eng.p50_ms:,.0f} ms" if (eng.latencies_ms and eng.p50_ms is not None) else "-"
            cost_str = f"${eng.total_cost_usd:.3f}" if eng.total_cost_usd is not None else "Unknown"

            lines.append(
                f"| **{eng.display_name}** | {eng.total_cases} | {cfc_str} | {ov_str} | {ts_str} | {fa_str} | {p50_str} | {cost_str} |"
            )

        lines.extend([
            "",
            "## Conclusiones Técnicas",
            "",
            "1. **Desacoplamiento de Corrección y Validez Operativa:** Las consultas con respuesta correcta pero "
            "interrumpidas por presupuesto o error de transporte registran `content_fully_correct = 1` y `operationally_valid = 0`, "
            "resultando en `task_success = 0`, eliminando cualquier contradicción cuantitativa.",
            "2. **Seguridad y Falsas Aprobaciones:** Los casos donde el denominador de incompatibilidad es 0 se "
            "declaran explícitamente como `N/A`, evitando falsas afirmaciones de 0.0% de riesgo.",
            "",
        ])
        return "\n".join(lines)

    def build_and_save(self, output_dir: Path | str = "runs/latest_report") -> tuple[Path, Path]:
        """Save single-source report files: report.md, report.html, summary.csv, scorecard.json, comparison.md, and INFORME_FINAL.md."""
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        md_path = target_dir / "report.md"
        html_path = target_dir / "report.html"
        csv_path = target_dir / "summary.csv"
        scorecard_path = target_dir / "scorecard.json"
        comp_path = target_dir / "comparison.md"
        informe_path = target_dir / "INFORME_FINAL.md"

        md_content = self.generate_markdown()
        html_content = self.generate_html()
        csv_content = self.generate_summary_csv()
        scorecard_content = self.generate_scorecard_json()
        comp_content = self.generate_comparison_md()
        informe_content = self.generate_informe_final_md()

        md_path.write_text(md_content, encoding="utf-8")
        html_path.write_text(html_content, encoding="utf-8")
        csv_path.write_text(csv_content, encoding="utf-8")
        scorecard_path.write_text(scorecard_content, encoding="utf-8")
        comp_path.write_text(comp_content, encoding="utf-8")
        informe_path.write_text(informe_content, encoding="utf-8")

        logger.info("Generated single-source benchmark deliverables at %s", target_dir)
        return md_path, html_path


def generate_reports(
    run_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = "runs/latest_report",
    demo: bool = False,
) -> tuple[Path, Path]:
    """High-level function to generate benchmark report.md and report.html."""
    if demo or run_dir is None:
        data = ReportBuilder.create_demo_data()
        data.metadata["is_demo"] = True
    else:
        data = ReportBuilder.from_run_dir(run_dir)
        if demo:
            data.metadata["is_demo"] = True

    builder = ReportBuilder(data)
    return builder.build_and_save(output_dir or "runs/latest_report")


def main() -> None:
    """CLI entry point for building benchmark reports."""
    parser = argparse.ArgumentParser(description="Generate benchmark reports adhering to MEGAPLAN.md §25")
    parser.add_argument("--run-dir", type=str, default=None, help="Directory containing run telemetry and scoring")
    parser.add_argument("--output-dir", type=str, default="runs/latest_report", help="Destination folder for report files")
    parser.add_argument("--demo", action="store_true", help="Generate report from demonstration dataset")

    args = parser.parse_args()
    md_path, html_path = generate_reports(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        demo=args.demo or (args.run_dir is None),
    )
    print(f"Report generated successfully:\n  Markdown: {md_path}\n  HTML: {html_path}")


if __name__ == "__main__":
    main()
