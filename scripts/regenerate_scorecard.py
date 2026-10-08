#!/usr/bin/env python3
"""Offline script to regenerate benchmark scorecard and update statistics.

Adheres strictly to WORKSTREAM 1 requirements:
- Reads results/g4_benchmark/statistics.json + summary.csv (+ evaluation.jsonl / telemetry.jsonl).
- Tests whether recomputing stats with seed 20261002 using StatisticsAnalyzer reproduces
  stored point estimates and CIs exactly.
- Outputs results/g4_benchmark/scorecard.json with pairwise comparisons, gates section,
  regenerated_offline: true, and source_files.
- Updates results/g4_benchmark/statistics.json with English outcome_label,
  outcome_rationale, and outcome_label_original while keeping all numbers identical.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure src/ is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from industrial_lab.benchmark.scoring import CaseScoreRecord
from industrial_lab.benchmark.statistics import (
    PairedComparisonResult,
    StatisticsAnalyzer,
    StatisticsReport,
)
from industrial_lab.observability.spans import TelemetryRecord


def _float_equal(a: float | None, b: float | None, rel_tol: float = 1e-7, abs_tol: float = 1e-9) -> bool:
    """Compare two optional floats for equality within numerical tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def main() -> None:
    g4_dir = REPO_ROOT / "results" / "g4_benchmark"
    stats_path = g4_dir / "statistics.json"
    summary_path = g4_dir / "summary.csv"
    eval_path = g4_dir / "evaluation.jsonl"
    telemetry_path = g4_dir / "telemetry.jsonl"
    scorecard_original_path = g4_dir / "scorecard.original.json"
    scorecard_path = g4_dir / "scorecard.json"

    source_files = [
        "results/g4_benchmark/statistics.json",
        "results/g4_benchmark/summary.csv",
        "results/g4_benchmark/evaluation.jsonl",
        "results/g4_benchmark/telemetry.jsonl",
    ]

    print("=== Step 1: Reading input files ===")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing {eval_path}")
    if not telemetry_path.exists():
        raise FileNotFoundError(f"Missing {telemetry_path}")
    if not scorecard_original_path.exists():
        raise FileNotFoundError(f"Missing {scorecard_original_path}")

    with stats_path.open("r", encoding="utf-8") as f:
        stored_stats: Dict[str, Any] = json.load(f)

    summary_rows: List[Dict[str, str]] = []
    with summary_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary_rows.append(row)
    print(f"Read summary.csv with {len(summary_rows)} engine rows: {[r['engine'] for r in summary_rows]}")

    scores: List[CaseScoreRecord] = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scores.append(CaseScoreRecord.model_validate(json.loads(line)))
    print(f"Read {len(scores)} evaluation score records from {eval_path.name}")

    telemetry: List[TelemetryRecord] = []
    with telemetry_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                telemetry.append(TelemetryRecord.model_validate(json.loads(line)))
    print(f"Read {len(telemetry)} telemetry records from {telemetry_path.name}")

    with scorecard_original_path.open("r", encoding="utf-8") as f:
        scorecard_orig: Dict[str, Any] = json.load(f)

    print("\n=== Step 2: Testing statistical recomputation (seed=20261002) ===")
    analyzer = StatisticsAnalyzer(seed=20261002)
    recomputed_report = analyzer.analyze(scores, telemetry, run_id="g4_benchmark")
    recomputed_dict = json.loads(recomputed_report.model_dump_json())

    # Verify reproduction against stored numbers
    all_matched = True
    metric_keys = [
        "task_success",
        "latency_difference_ms",
        "latency_speedup",
        "cost_difference_usd",
        "cost_reduction_ratio",
    ]

    for comp_id, stored_comp in stored_stats.get("paired_comparisons", {}).items():
        recomp_comp = recomputed_dict.get("paired_comparisons", {}).get(comp_id)
        if not recomp_comp:
            print(f"Mismatch: {comp_id} missing from recomputed report")
            all_matched = False
            continue

        for m_key in metric_keys:
            sm = stored_comp[m_key]
            rm = recomp_comp[m_key]
            pt_eq = _float_equal(sm.get("point_difference"), rm.get("point_difference"))
            lo_eq = _float_equal(sm.get("ci_lower"), rm.get("ci_lower"))
            hi_eq = _float_equal(sm.get("ci_upper"), rm.get("ci_upper"))
            p_eq = _float_equal(sm.get("p_value"), rm.get("p_value"))
            if not (pt_eq and lo_eq and hi_eq and p_eq):
                all_matched = False
                print(f"Mismatch in {comp_id}.{m_key}: stored=(pt={sm.get('point_difference')}, ci=[{sm.get('ci_lower')}, {sm.get('ci_upper')}], p={sm.get('p_value')}) vs recomputed=(pt={rm.get('point_difference')}, ci=[{rm.get('ci_lower')}, {rm.get('ci_upper')}], p={rm.get('p_value')})")

    if all_matched:
        print("Recomputing stats with seed 20261002 reproduces stored point estimates and CIs exactly.")
    else:
        print("Note: Recomputed stats showed differences; reusing stored numeric values and re-deriving label + rationale with fixed function.")

    # Re-derive label + rationale using fixed function on paired comparison results
    paired_objs = {
        cid: PairedComparisonResult.model_validate(cdata)
        for cid, cdata in stored_stats["paired_comparisons"].items()
    }
    english_label, english_rationale = analyzer._determine_outcome_label(
        paired_results=paired_objs,
        num_families=stored_stats.get("num_families", 4),
        total_cases=stored_stats.get("num_cases", 12),
    )
    print(f"\nDerived Outcome Label: {english_label}")
    print(f"Derived Outcome Rationale: {english_rationale}")

    print("\n=== Step 3: Updating results/g4_benchmark/statistics.json ===")
    # Update label and rationale; add outcome_label_original; keep all numbers identical
    original_label = stored_stats.get("outcome_label", "JEV sin beneficio demostrado")
    stored_stats["outcome_label"] = english_label
    stored_stats["outcome_label_original"] = original_label if original_label != english_label else "JEV sin beneficio demostrado"
    stored_stats["outcome_rationale"] = english_rationale

    # Validate schema
    validated_stats_report = StatisticsReport.model_validate(stored_stats)
    with stats_path.open("w", encoding="utf-8") as f:
        f.write(validated_stats_report.model_dump_json(indent=2))
    print(f"Updated {stats_path} successfully.")

    print("\n=== Step 4: Generating results/g4_benchmark/scorecard.json ===")
    scorecard: Dict[str, Any] = {
        "version": scorecard_orig.get("version", "repair3-v1.0"),
        "benchmark_run_id": scorecard_orig.get("benchmark_run_id", "g4_benchmark"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "regenerated_offline": True,
        "source_files": source_files,
        "freeze_signature": scorecard_orig.get("freeze_signature", ""),
        "catalog": scorecard_orig.get("catalog", "controlnautas-three-products-v1"),
        "products": scorecard_orig.get("products", ["P_X4", "P_THT", "P_UHEAT"]),
        "total_requests": scorecard_orig.get("total_requests", 180),
        "cases_evaluated": scorecard_orig.get("cases_evaluated", 12),
        "repetitions": scorecard_orig.get("repetitions", 3),
        "engines": scorecard_orig.get("engines", {}),
        "pairwise_comparisons": {},
        "outcome": {
            "label": english_label,
            "rationale": english_rationale,
            "label_original": stored_stats["outcome_label_original"],
        },
    }

    # Add pairwise comparisons with gates
    for comp_id, comp_data in stored_stats.get("paired_comparisons", {}).items():
        comp_copy = dict(comp_data)
        comp_copy["gates"] = {
            "success_non_inferior": comp_data.get("success_non_inferior"),
            "latency_target_met": comp_data.get("latency_target_met"),
            "cost_target_met": comp_data.get("cost_target_met"),
            "no_new_false_approvals": comp_data.get("no_new_false_approvals"),
            "overall_favorable": comp_data.get("overall_favorable_a", False),
        }
        scorecard["pairwise_comparisons"][comp_id] = comp_copy

    with scorecard_path.open("w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2, ensure_ascii=False)
    print(f"Generated {scorecard_path} successfully.")
    print("Regeneration complete.")


if __name__ == "__main__":
    main()
