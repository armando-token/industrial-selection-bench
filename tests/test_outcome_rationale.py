"""Tests for statistical outcome labels and data-driven rationales.

Verifies:
- Data-driven rationales for PairedComparisonResult objects.
- Correct handling when latency target is met but false approvals gate fails.
- Invariant: Never claim 'no latency advantage' when latency_target_met is True.
- All outcome label branches: Trade-off, Negative result, Advantage supported within pilot,
  Advantage vs. scraping only, No net benefit demonstrated, Inconclusive.
- Integrity and English translation of results/g4_benchmark/scorecard.json and statistics.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest

from industrial_lab.benchmark.statistics import (
    MetricDifferenceCI,
    PairedComparisonResult,
    StatisticsAnalyzer,
    StatisticsReport,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
G4_DIR = REPO_ROOT / "results" / "g4_benchmark"


def _make_metric_ci(
    name: str,
    eng_a: str = "structured_jev",
    eng_b: str = "rag_llm",
    point: float = 0.0,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    p_val: Optional[float] = 1.0,
    confidence_level: float = 0.95,
) -> MetricDifferenceCI:
    """Helper to build a synthetic MetricDifferenceCI."""
    return MetricDifferenceCI(
        metric_name=name,
        engine_a=eng_a,
        engine_b=eng_b,
        point_difference=point,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        p_value=p_val,
        p_value_holm=p_val,
        bootstrap_resamples=10000,
        is_statistically_significant=(p_val < 0.05 if p_val is not None else None),
        details={},
    )


def _make_paired_comp(
    eng_a: str,
    eng_b: str,
    *,
    success_point: float = 0.0833,
    success_ci: Optional[Tuple[float, float]] = (0.0, 0.25),
    success_non_inferior: Optional[bool] = True,
    speedup: float = 14.6,
    speedup_ci: Optional[Tuple[float, float]] = (11.9, 17.7),
    latency_target_met: Optional[bool] = True,
    cost_diff: float = 0.0,
    cost_reduction: float = 0.0,
    cost_target_met: Optional[bool] = False,
    false_approvals_a: int = 3,
    false_approvals_b: int = 0,
    no_new_false_approvals: bool = False,
    overall_favorable_a: bool = False,
    sample_size: int = 36,
    num_families: int = 4,
) -> PairedComparisonResult:
    """Helper to build a synthetic PairedComparisonResult."""
    return PairedComparisonResult(
        comparison_id=f"{eng_a}_vs_{eng_b}",
        engine_a=eng_a,
        engine_b=eng_b,
        sample_size_cases=sample_size,
        num_families=num_families,
        task_success=_make_metric_ci(
            "task_success_diff",
            eng_a,
            eng_b,
            point=success_point,
            ci_lower=success_ci[0] if success_ci else None,
            ci_upper=success_ci[1] if success_ci else None,
            p_val=0.6,
        ),
        latency_difference_ms=_make_metric_ci(
            "latency_diff_ms",
            eng_a,
            eng_b,
            point=-3200.0,
            ci_lower=-4000.0,
            ci_upper=-2500.0,
            p_val=0.0,
        ),
        latency_speedup=_make_metric_ci(
            "latency_speedup",
            eng_a,
            eng_b,
            point=speedup,
            ci_lower=speedup_ci[0] if speedup_ci else None,
            ci_upper=speedup_ci[1] if speedup_ci else None,
            p_val=0.0,
        ),
        cost_difference_usd=_make_metric_ci(
            "cost_diff_usd",
            eng_a,
            eng_b,
            point=cost_diff,
            ci_lower=0.0,
            ci_upper=0.0,
            p_val=1.0,
        ),
        cost_reduction_ratio=_make_metric_ci(
            "cost_reduction_ratio",
            eng_a,
            eng_b,
            point=cost_reduction,
            ci_lower=0.0,
            ci_upper=0.0,
            p_val=1.0,
        ),
        false_approvals_a=false_approvals_a,
        false_approvals_b=false_approvals_b,
        false_approvals_diff=false_approvals_a - false_approvals_b,
        no_new_false_approvals=no_new_false_approvals,
        success_non_inferior=success_non_inferior,
        latency_target_met=latency_target_met,
        cost_target_met=cost_target_met,
        overall_favorable_a=overall_favorable_a,
    )


# ==============================================================================
# 1. Synthetic Tests: Latency Target Met but False Approval Fails
# ==============================================================================

def test_rationale_when_latency_met_but_false_approval_fails() -> None:
    """When latency target is met but false approvals gate fails:

    - Rationale must state latency target is met (with speedup / CI).
    - Rationale must report FAILED no-new-false-approvals gate.
    - Rationale must NEVER claim 'no latency advantage' or Spanish variants.
    """
    analyzer = StatisticsAnalyzer()
    a_vs_b = _make_paired_comp(
        "structured_jev",
        "rag_llm",
        speedup=14.6,
        speedup_ci=(11.9, 17.7),
        latency_target_met=True,
        false_approvals_a=3,
        false_approvals_b=0,
        no_new_false_approvals=False,
        overall_favorable_a=False,
    )
    a_vs_c = _make_paired_comp(
        "structured_jev",
        "scrape_llm",
        speedup=47.5,
        speedup_ci=(46.1, 53.6),
        latency_target_met=True,
        false_approvals_a=3,
        false_approvals_b=0,
        no_new_false_approvals=False,
        overall_favorable_a=False,
    )

    paired: Dict[str, PairedComparisonResult] = {
        "structured_jev_vs_rag_llm": a_vs_b,
        "structured_jev_vs_scrape_llm": a_vs_c,
    }

    label, rationale = analyzer._determine_outcome_label(paired, num_families=4, total_cases=12)

    assert label == "No net benefit demonstrated"
    assert "latency target met (14.6x faster, 95% CI 11.9x-17.7x)" in rationale
    assert "FAILED no-new-false-approvals gate (3 vs 0)" in rationale
    assert "latency target met (47.5x faster, 95% CI 46.1x-53.6x)" in rationale

    # Invariant: Never claim no latency advantage when latency_target_met is True!
    forbidden_phrases = [
        "no latency advantage",
        "sin ventaja",
        "no demostró ventaja",
        "no latency improvement",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in rationale.lower(), f"Forbidden phrase '{phrase}' found in rationale"


# ==============================================================================
# 2. Synthetic Tests: All Outcome Label Branches
# ==============================================================================

def test_outcome_branch_trade_off() -> None:
    """Trade-off branch: task_success inferior despite meeting latency or cost target."""
    analyzer = StatisticsAnalyzer()
    a_vs_b = _make_paired_comp(
        "structured_jev",
        "rag_llm",
        success_non_inferior=False,
        latency_target_met=True,
        overall_favorable_a=False,
    )
    paired = {"structured_jev_vs_rag_llm": a_vs_b}

    label, rationale = analyzer._determine_outcome_label(paired, num_families=4, total_cases=12)
    assert label == "Trade-off"
    assert "structured_jev exhibits lower cost or latency but significant inferiority in task_success" in rationale


def test_outcome_branch_negative_result() -> None:
    """Negative result branch: substantial drop in success and more false approvals."""
    analyzer = StatisticsAnalyzer()
    a_vs_b = _make_paired_comp(
        "structured_jev",
        "rag_llm",
        success_point=-0.25,
        success_non_inferior=None,
        latency_target_met=False,
        cost_target_met=False,
        false_approvals_a=4,
        false_approvals_b=0,
        no_new_false_approvals=False,
        overall_favorable_a=False,
    )
    paired = {"structured_jev_vs_rag_llm": a_vs_b}

    label, rationale = analyzer._determine_outcome_label(paired, num_families=4, total_cases=12)
    assert label == "Negative result"
    assert "structured_jev exhibits lower task success rate and higher false approval rate" in rationale


def test_outcome_branch_advantage_supported_within_pilot() -> None:
    """Advantage supported within pilot branch: favorable vs both B and C with sufficient sample."""
    analyzer = StatisticsAnalyzer()
    a_vs_b = _make_paired_comp(
        "structured_jev",
        "rag_llm",
        overall_favorable_a=True,
        no_new_false_approvals=True,
        false_approvals_a=0,
        false_approvals_b=0,
    )
    a_vs_c = _make_paired_comp(
        "structured_jev",
        "scrape_llm",
        overall_favorable_a=True,
        no_new_false_approvals=True,
        false_approvals_a=0,
        false_approvals_b=0,
    )
    paired = {
        "structured_jev_vs_rag_llm": a_vs_b,
        "structured_jev_vs_scrape_llm": a_vs_c,
    }

    # >= 5 families and >= 20 cases
    label, rationale = analyzer._determine_outcome_label(paired, num_families=5, total_cases=20)
    assert label == "Advantage supported within pilot"
    assert "structured_jev outperforms both rag_llm and scrape_llm" in rationale


def test_outcome_branch_advantage_observed_limited_evidence() -> None:
    """Advantage observed, limited evidence branch: favorable vs both B and C but small sample."""
    analyzer = StatisticsAnalyzer()
    a_vs_b = _make_paired_comp("structured_jev", "rag_llm", overall_favorable_a=True, no_new_false_approvals=True)
    a_vs_c = _make_paired_comp("structured_jev", "scrape_llm", overall_favorable_a=True, no_new_false_approvals=True)
    paired = {
        "structured_jev_vs_rag_llm": a_vs_b,
        "structured_jev_vs_scrape_llm": a_vs_c,
    }

    # < 5 families or < 20 cases
    label, rationale = analyzer._determine_outcome_label(paired, num_families=4, total_cases=12)
    assert label == "Advantage observed, limited evidence"
    assert "limited to support robust significance" in rationale


def test_outcome_branch_advantage_vs_scraping_only() -> None:
    """Advantage vs. scraping only branch: favorable vs scrape_llm, but not vs rag_llm."""
    analyzer = StatisticsAnalyzer()
    a_vs_b = _make_paired_comp(
        "structured_jev",
        "rag_llm",
        overall_favorable_a=False,
        false_approvals_a=2,
        false_approvals_b=0,
        no_new_false_approvals=False,
    )
    a_vs_c = _make_paired_comp(
        "structured_jev",
        "scrape_llm",
        overall_favorable_a=True,
        false_approvals_a=0,
        false_approvals_b=0,
        no_new_false_approvals=True,
    )
    paired = {
        "structured_jev_vs_rag_llm": a_vs_b,
        "structured_jev_vs_scrape_llm": a_vs_c,
    }

    label, rationale = analyzer._determine_outcome_label(paired, num_families=4, total_cases=12)
    assert label == "Advantage vs. scraping only"
    assert "structured_jev vs rag_llm:" in rationale
    assert "structured_jev vs scrape_llm:" in rationale
    assert "FAILED no-new-false-approvals gate (2 vs 0)" in rationale
    assert "no new false approvals (0 vs 0)" in rationale


def test_outcome_branch_no_net_benefit_demonstrated() -> None:
    """No net benefit demonstrated branch: not favorable vs either comparator."""
    analyzer = StatisticsAnalyzer()
    a_vs_b = _make_paired_comp(
        "structured_jev",
        "rag_llm",
        overall_favorable_a=False,
        false_approvals_a=3,
        false_approvals_b=0,
        no_new_false_approvals=False,
    )
    a_vs_c = _make_paired_comp(
        "structured_jev",
        "scrape_llm",
        overall_favorable_a=False,
        false_approvals_a=3,
        false_approvals_b=0,
        no_new_false_approvals=False,
    )
    paired = {
        "structured_jev_vs_rag_llm": a_vs_b,
        "structured_jev_vs_scrape_llm": a_vs_c,
    }

    label, rationale = analyzer._determine_outcome_label(paired, num_families=4, total_cases=12)
    assert label == "No net benefit demonstrated"
    assert "structured_jev vs rag_llm:" in rationale
    assert "structured_jev vs scrape_llm:" in rationale
    assert "latency target met" in rationale


def test_outcome_branch_inconclusive() -> None:
    """Inconclusive branch: empty results, single family, or small sample."""
    analyzer = StatisticsAnalyzer()

    # Empty results
    lbl, rat = analyzer._determine_outcome_label({}, num_families=4, total_cases=12)
    assert lbl == "Inconclusive"
    assert "No paired comparisons executed" in rat

    # Insufficient sample / single family
    a_vs_b = _make_paired_comp("structured_jev", "rag_llm")
    lbl2, rat2 = analyzer._determine_outcome_label(
        {"structured_jev_vs_rag_llm": a_vs_b}, num_families=1, total_cases=3
    )
    assert lbl2 == "Inconclusive"
    assert "Insufficient sample or single-family design" in rat2


# ==============================================================================
# 3. Verification of results/g4_benchmark Output Files
# ==============================================================================

def test_g4_benchmark_files_have_english_label_and_data_driven_rationale() -> None:
    """Verify that scorecard.json and statistics.json in results/g4_benchmark

    have the official English label and data-driven rationales.
    """
    stats_file = G4_DIR / "statistics.json"
    scorecard_file = G4_DIR / "scorecard.json"

    assert stats_file.exists(), f"Missing {stats_file}"
    assert scorecard_file.exists(), f"Missing {scorecard_file}"

    # Verify statistics.json
    with stats_file.open("r", encoding="utf-8") as f:
        stats_data = json.load(f)

    # Validates against Pydantic model
    stats_report = StatisticsReport.model_validate(stats_data)
    assert stats_report.outcome_label == "No net benefit demonstrated"
    assert stats_report.outcome_label_original == "JEV sin beneficio demostrado"
    assert "latency target met (14.6x faster" in stats_report.outcome_rationale
    assert "FAILED no-new-false-approvals gate (3 vs 0)" in stats_report.outcome_rationale
    assert "structured_jev vs scrape_llm" in stats_report.outcome_rationale
    assert "no latency advantage" not in stats_report.outcome_rationale.lower()

    # Verify scorecard.json
    with scorecard_file.open("r", encoding="utf-8") as f:
        scorecard_data = json.load(f)

    assert scorecard_data.get("regenerated_offline") is True
    assert "source_files" in scorecard_data
    assert "results/g4_benchmark/statistics.json" in scorecard_data["source_files"]

    outcome = scorecard_data.get("outcome", {})
    assert outcome.get("label") == "No net benefit demonstrated"
    assert outcome.get("label_original") == "JEV sin beneficio demostrado"
    assert "latency target met" in outcome.get("rationale", "")
    assert "FAILED no-new-false-approvals gate" in outcome.get("rationale", "")

    # Verify gates section per pairwise comparison in scorecard.json
    pairwise = scorecard_data.get("pairwise_comparisons", {})
    assert "structured_jev_vs_rag_llm" in pairwise
    assert "structured_jev_vs_scrape_llm" in pairwise

    jev_rag_gates = pairwise["structured_jev_vs_rag_llm"].get("gates", {})
    assert jev_rag_gates.get("latency_target_met") is True
    assert jev_rag_gates.get("success_non_inferior") is True
    assert jev_rag_gates.get("cost_target_met") is False
    assert jev_rag_gates.get("no_new_false_approvals") is False
    assert jev_rag_gates.get("overall_favorable") is False

    jev_scrape_gates = pairwise["structured_jev_vs_scrape_llm"].get("gates", {})
    assert jev_scrape_gates.get("latency_target_met") is True
    assert jev_scrape_gates.get("success_non_inferior") is True
    assert jev_scrape_gates.get("no_new_false_approvals") is False
    assert jev_scrape_gates.get("overall_favorable") is False
