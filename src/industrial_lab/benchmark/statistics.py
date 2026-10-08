"""Statistical analysis module adhering to MEGAPLAN.md §17.

Implements rigorous experimental analysis:
- Experimental unit: case technical conditions / scenario_family_id (§17.1).
- Paired comparisons across engines (§17.2):
  - Primary comparisons: A vs B (structured_jev vs rag_llm), A vs C (structured_jev vs scrape_llm).
  - Cluster / hierarchical bootstrap by scenario_family_id (10,000 resamples).
  - 95% confidence intervals for paired differences in task_success, latency, and cost.
  - Two-sided p-values with Holm adjustment (Holm-Bonferroni) for multiple hypotheses.
- Interpretation rules & outcome labels (§17.3, §17.4):
  - Non-inferiority margin on task_success (default 5 percentage points: -0.05).
  - Latency reduction target (>= 20%).
  - Cost reduction target (>= 20%).
  - False approval constraint (no new false approvals).
  - Standardized outcome labels from §17.4.
- Persistence:
  - summary.csv: Comprehensive tabular performance overview.
  - statistics.json: Machine-readable paired statistics, CIs, p-values, and labels.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from industrial_lab.benchmark.scoring import BenchmarkSummary, CaseScoreRecord, EngineMetricsSummary
from industrial_lab.observability.spans import TelemetryRecord

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_NONINFERIORITY_MARGIN = 0.05
DEFAULT_LATENCY_REDUCTION_TARGET = 0.20
DEFAULT_COST_REDUCTION_TARGET = 0.20


# ==============================================================================
# Statistical Data Models (§17)
# ==============================================================================

class MetricDifferenceCI(BaseModel):
    """Point estimate and confidence interval for a paired difference (§17.2)."""
    model_config = ConfigDict(extra="forbid")

    metric_name: str
    engine_a: str
    engine_b: str
    point_difference: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    p_value: Optional[float] = None
    p_value_holm: Optional[float] = None
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    is_statistically_significant: Optional[bool] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class PairedComparisonResult(BaseModel):
    """Complete paired comparison between two engines (§17.2, §17.3)."""
    model_config = ConfigDict(extra="forbid")

    comparison_id: str  # e.g. "structured_jev_vs_rag_llm"
    engine_a: str
    engine_b: str
    sample_size_cases: int
    num_families: int

    # Primary paired metrics
    task_success: MetricDifferenceCI
    latency_difference_ms: MetricDifferenceCI
    latency_speedup: MetricDifferenceCI
    cost_difference_usd: MetricDifferenceCI
    cost_reduction_ratio: MetricDifferenceCI

    # Safety: False approvals
    false_approvals_a: int
    false_approvals_b: int
    false_approvals_diff: int
    no_new_false_approvals: bool

    # Target criteria verification (§17.3)
    success_non_inferior: Optional[bool] = None
    latency_target_met: Optional[bool] = None
    cost_target_met: Optional[bool] = None
    overall_favorable_a: bool = False


class StatisticsReport(BaseModel):
    """Final statistical report adhering to MEGAPLAN.md §17.4."""
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at_utc: str
    seed: int
    confidence_level: float
    bootstrap_resamples: int
    num_families: int
    num_cases: int
    num_repeats: int
    engines: List[str]
    paired_comparisons: Dict[str, PairedComparisonResult]
    multiple_testing_adjustments: Dict[str, float]
    outcome_label: str
    outcome_label_original: Optional[str] = None
    outcome_rationale: str
    thresholds: Dict[str, float]


# ==============================================================================
# Holm-Bonferroni Multiple Testing Adjustment (§17.2)
# ==============================================================================

def adjust_p_values_holm(p_values: Dict[str, Optional[float]]) -> Dict[str, float]:
    """Apply Holm-Bonferroni step-down adjustment to multiple hypothesis tests.

    Adheres strictly to MEGAPLAN §17.2:
    Sorts p-values in ascending order: p_(1) <= p_(2) <= ... <= p_(m).
    Adjusted: p_adj_(i) = min(1.0, max_{j <= i} ((m - j + 1) * p_(j))).
    """
    valid_items = [(k, v) for k, v in p_values.items() if v is not None]
    if not valid_items:
        return {}

    # Sort by p-value ascending
    sorted_items = sorted(valid_items, key=lambda x: x[1])
    m = len(sorted_items)

    adjusted_list: List[Tuple[str, float]] = []
    running_max = 0.0

    for i, (name, p_val) in enumerate(sorted_items):
        multiplier = m - i
        val = p_val * multiplier
        running_max = max(running_max, val)
        adj_p = min(1.0, running_max)
        adjusted_list.append((name, adj_p))

    return dict(adjusted_list)


# ==============================================================================
# Statistics Analyzer (§17)
# ==============================================================================

class StatisticsAnalyzer:
    """Analyzes benchmark results with paired cluster bootstrap by scenario_family_id."""

    def __init__(
        self,
        bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        success_noninferiority_margin: float = DEFAULT_NONINFERIORITY_MARGIN,
        latency_reduction_target: float = DEFAULT_LATENCY_REDUCTION_TARGET,
        cost_reduction_target: float = DEFAULT_COST_REDUCTION_TARGET,
        seed: int = 20261002,
    ) -> None:
        self.bootstrap_resamples = bootstrap_resamples
        self.confidence_level = confidence_level
        self.noninferiority_margin = success_noninferiority_margin
        self.latency_reduction_target = latency_reduction_target
        self.cost_reduction_target = cost_reduction_target
        self.seed = seed

    def analyze(
        self,
        score_records: List[CaseScoreRecord],
        telemetry_records: Optional[List[TelemetryRecord]] = None,
        run_id: str = "run_default",
    ) -> StatisticsReport:
        """Run complete paired statistical analysis.

        Parameters
        ----------
        score_records : List[CaseScoreRecord]
            Evaluated per-case records.
        telemetry_records : Optional[List[TelemetryRecord]]
            Telemetry records for latencies and costs.
        run_id : str
            Benchmark run ID.

        Returns
        -------
        StatisticsReport
            Structured report with bootstrap CIs, Holm adjustments, and outcome label.
        """
        # Build lookup table for telemetry
        telemetry_by_req: Dict[str, TelemetryRecord] = {}
        if telemetry_records:
            for t in telemetry_records:
                telemetry_by_req[t.request_id] = t

        # Group observations by (scenario_family_id, case_id, repeat)
        # Structure: family_id -> list of observation dicts per case/repeat
        families: Dict[str, List[Dict[str, Any]]] = {}
        unique_engines: Set[str] = set()
        unique_cases: Set[str] = set()
        repeats_seen: Set[int] = set()

        for r in score_records:
            unique_engines.add(r.engine)
            unique_cases.add(r.case_id)
            repeats_seen.add(r.repeat)

            tel = telemetry_by_req.get(r.request_id)
            lat_ms = tel.elapsed_ms if tel else 0.0
            cost_usd = tel.cost_usd if (tel and tel.cost_usd is not None) else 0.0

            item = {
                "case_id": r.case_id,
                "repeat": r.repeat,
                "engine": r.engine,
                "task_success": 1.0 if r.task_success else 0.0,
                "false_approval": 1 if r.false_approval else 0,
                "completed": r.execution_completed,
                "latency_ms": lat_ms,
                "cost_usd": cost_usd,
            }
            families.setdefault(r.scenario_family_id, []).append(item)

        engine_list = sorted(unique_engines)
        family_ids = sorted(families.keys())

        # Define primary comparisons: A vs B, A vs C, B vs C
        pairs_to_test: List[Tuple[str, str]] = []
        if "structured_jev" in engine_list and "rag_llm" in engine_list:
            pairs_to_test.append(("structured_jev", "rag_llm"))
        if "structured_jev" in engine_list and "scrape_llm" in engine_list:
            pairs_to_test.append(("structured_jev", "scrape_llm"))
        if "rag_llm" in engine_list and "scrape_llm" in engine_list:
            pairs_to_test.append(("rag_llm", "scrape_llm"))

        # Fallback if names differ: test all pairs
        if not pairs_to_test and len(engine_list) >= 2:
            for i in range(len(engine_list)):
                for j in range(i + 1, len(engine_list)):
                    pairs_to_test.append((engine_list[i], engine_list[j]))

        rng = np.random.default_rng(self.seed)

        paired_results: Dict[str, PairedComparisonResult] = {}
        raw_p_values: Dict[str, float] = {}

        for eng_a, eng_b in pairs_to_test:
            comp_id = f"{eng_a}_vs_{eng_b}"
            res = self._compute_paired_comparison(
                eng_a=eng_a,
                eng_b=eng_b,
                families=families,
                family_ids=family_ids,
                rng=rng,
            )
            paired_results[comp_id] = res

            # Register raw p-values for Holm correction (§17.2)
            raw_p_values[f"{comp_id}:task_success"] = res.task_success.p_value
            raw_p_values[f"{comp_id}:latency_speedup"] = res.latency_speedup.p_value
            raw_p_values[f"{comp_id}:cost_reduction"] = res.cost_reduction_ratio.p_value

        # Apply Holm adjustment across primary hypothesis tests
        holm_adjusted = adjust_p_values_holm(raw_p_values)

        # Update adjusted p-values in paired results
        for comp_id, res in paired_results.items():
            if f"{comp_id}:task_success" in holm_adjusted:
                res.task_success.p_value_holm = holm_adjusted[f"{comp_id}:task_success"]
                res.task_success.is_statistically_significant = (res.task_success.p_value_holm < 0.05)
            if f"{comp_id}:latency_speedup" in holm_adjusted:
                res.latency_speedup.p_value_holm = holm_adjusted[f"{comp_id}:latency_speedup"]
                res.latency_speedup.is_statistically_significant = (res.latency_speedup.p_value_holm < 0.05)
            if f"{comp_id}:cost_reduction" in holm_adjusted:
                res.cost_reduction_ratio.p_value_holm = holm_adjusted[f"{comp_id}:cost_reduction"]
                res.cost_reduction_ratio.is_statistically_significant = (res.cost_reduction_ratio.p_value_holm < 0.05)

        # Determine outcome label (§17.4)
        outcome_label, outcome_rationale = self._determine_outcome_label(
            paired_results=paired_results,
            num_families=len(family_ids),
            total_cases=len(unique_cases),
        )

        return StatisticsReport(
            run_id=run_id,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            seed=self.seed,
            confidence_level=self.confidence_level,
            bootstrap_resamples=self.bootstrap_resamples,
            num_families=len(family_ids),
            num_cases=len(unique_cases),
            num_repeats=len(repeats_seen),
            engines=engine_list,
            paired_comparisons=paired_results,
            multiple_testing_adjustments=holm_adjusted,
            outcome_label=outcome_label,
            outcome_rationale=outcome_rationale,
            thresholds={
                "success_noninferiority_margin": self.noninferiority_margin,
                "latency_reduction_target": self.latency_reduction_target,
                "cost_reduction_target": self.cost_reduction_target,
            },
        )

    def _compute_paired_comparison(
        self,
        eng_a: str,
        eng_b: str,
        families: Dict[str, List[Dict[str, Any]]],
        family_ids: List[str],
        rng: np.random.Generator,
    ) -> PairedComparisonResult:
        """Execute paired cluster bootstrap on scenario_family_id."""
        k = len(family_ids)
        b_resamples = self.bootstrap_resamples

        # Collect observations grouped by case and repeat across the dataset
        # To pair observations, key = (case_id, repeat)
        pairs: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
        for fam_id, obs_list in families.items():
            for obs in obs_list:
                key = (obs["case_id"], obs["repeat"])
                pairs.setdefault(key, {})[obs["engine"]] = obs

        # Baseline point estimates on original data
        obs_a = [pair[eng_a] for pair in pairs.values() if eng_a in pair and eng_b in pair]
        obs_b = [pair[eng_b] for pair in pairs.values() if eng_a in pair and eng_b in pair]

        n_paired_cases = len(obs_a)

        success_a = np.array([o["task_success"] for o in obs_a])
        success_b = np.array([o["task_success"] for o in obs_b])
        lat_a = np.array([o["latency_ms"] for o in obs_a])
        lat_b = np.array([o["latency_ms"] for o in obs_b])
        cost_a = np.array([o["cost_usd"] for o in obs_a])
        cost_b = np.array([o["cost_usd"] for o in obs_b])

        # Point estimates
        point_success_diff = float(np.mean(success_a) - np.mean(success_b)) if n_paired_cases > 0 else 0.0
        point_lat_diff = float(np.median(lat_a) - np.median(lat_b)) if n_paired_cases > 0 else 0.0
        med_lat_a = float(np.median(lat_a)) if n_paired_cases > 0 else 1.0
        med_lat_b = float(np.median(lat_b)) if n_paired_cases > 0 else 1.0
        point_speedup = (med_lat_b / med_lat_a) if med_lat_a > 0 else 1.0

        mean_cost_a = float(np.mean(cost_a)) if n_paired_cases > 0 else 0.0
        mean_cost_b = float(np.mean(cost_b)) if n_paired_cases > 0 else 0.0
        point_cost_diff = mean_cost_a - mean_cost_b
        point_cost_reduction = ((mean_cost_b - mean_cost_a) / mean_cost_b) if mean_cost_b > 0 else 0.0

        fa_a = sum(o["false_approval"] for o in obs_a)
        fa_b = sum(o["false_approval"] for o in obs_b)
        fa_diff = fa_a - fa_b
        no_new_fa = (fa_a <= fa_b)

        # Single-family or small-sample guard (§14, §17):
        # A single family (k <= 1) or small sample (n < 5) produces degenerate cluster bootstrap intervals and p=0.
        # Statistical inference must be disabled: CIs, p-values, and statistical significance are set to None.
        inference_disabled = (k <= 1 or n_paired_cases < 5)
        total_successes_a = int(np.sum(success_a)) if n_paired_cases > 0 else 0

        latency_target_met = bool(point_speedup >= (1.0 / (1.0 - self.latency_reduction_target)))
        cost_target_met = bool(point_cost_reduction >= self.cost_reduction_target)

        if inference_disabled:
            reason_msg = (
                f"Single family (k={k}) or small sample (n={n_paired_cases} < 5); "
                "statistical inference disabled to prevent degenerate intervals and p=0."
            )
            ci_success = MetricDifferenceCI(
                metric_name="task_success_diff",
                engine_a=eng_a,
                engine_b=eng_b,
                point_difference=point_success_diff,
                ci_lower=None,
                ci_upper=None,
                confidence_level=self.confidence_level,
                p_value=None,
                p_value_holm=None,
                bootstrap_resamples=b_resamples,
                is_statistically_significant=None,
                details={"inference_disabled": True, "reason": reason_msg},
            )
            ci_lat_diff = MetricDifferenceCI(
                metric_name="latency_diff_ms",
                engine_a=eng_a,
                engine_b=eng_b,
                point_difference=point_lat_diff,
                ci_lower=None,
                ci_upper=None,
                confidence_level=self.confidence_level,
                p_value=None,
                p_value_holm=None,
                bootstrap_resamples=b_resamples,
                is_statistically_significant=None,
                details={"inference_disabled": True, "reason": reason_msg},
            )
            ci_speedup = MetricDifferenceCI(
                metric_name="latency_speedup",
                engine_a=eng_a,
                engine_b=eng_b,
                point_difference=point_speedup,
                ci_lower=None,
                ci_upper=None,
                confidence_level=self.confidence_level,
                p_value=None,
                p_value_holm=None,
                bootstrap_resamples=b_resamples,
                is_statistically_significant=None,
                details={"inference_disabled": True, "reason": reason_msg},
            )
            ci_cost_diff = MetricDifferenceCI(
                metric_name="cost_diff_usd",
                engine_a=eng_a,
                engine_b=eng_b,
                point_difference=point_cost_diff,
                ci_lower=None,
                ci_upper=None,
                confidence_level=self.confidence_level,
                p_value=None,
                p_value_holm=None,
                bootstrap_resamples=b_resamples,
                is_statistically_significant=None,
                details={"inference_disabled": True, "reason": reason_msg},
            )
            ci_cost_reduction = MetricDifferenceCI(
                metric_name="cost_reduction_ratio",
                engine_a=eng_a,
                engine_b=eng_b,
                point_difference=point_cost_reduction,
                ci_lower=None,
                ci_upper=None,
                confidence_level=self.confidence_level,
                p_value=None,
                p_value_holm=None,
                bootstrap_resamples=b_resamples,
                is_statistically_significant=None,
                details={"inference_disabled": True, "reason": reason_msg},
            )
            success_non_inferior = None
            overall_favorable = False
        else:
            # Bootstrap arrays
            boot_success_diffs = np.empty(b_resamples, dtype=float)
            boot_lat_diffs = np.empty(b_resamples, dtype=float)
            boot_speedups = np.empty(b_resamples, dtype=float)
            boot_cost_diffs = np.empty(b_resamples, dtype=float)
            boot_cost_reductions = np.empty(b_resamples, dtype=float)

            # Perform cluster resampling by scenario_family_id
            if k > 0:
                for b in range(b_resamples):
                    # Resample families with replacement
                    sampled_fam_indices = rng.choice(k, size=k, replace=True)
                    sample_obs_a: List[Dict[str, Any]] = []
                    sample_obs_b: List[Dict[str, Any]] = []

                    for idx in sampled_fam_indices:
                        fam_name = family_ids[idx]
                        # Find all paired cases in this family
                        fam_items = families[fam_name]
                        # Map items by (case_id, repeat)
                        fam_pairs: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
                        for item in fam_items:
                            key = (item["case_id"], item["repeat"])
                            fam_pairs.setdefault(key, {})[item["engine"]] = item

                        for key, eng_map in fam_pairs.items():
                            if eng_a in eng_map and eng_b in eng_map:
                                sample_obs_a.append(eng_map[eng_a])
                                sample_obs_b.append(eng_map[eng_b])

                    if sample_obs_a:
                        b_succ_a = [o["task_success"] for o in sample_obs_a]
                        b_succ_b = [o["task_success"] for o in sample_obs_b]
                        b_lat_a = [o["latency_ms"] for o in sample_obs_a]
                        b_lat_b = [o["latency_ms"] for o in sample_obs_b]
                        b_cost_a = [o["cost_usd"] for o in sample_obs_a]
                        b_cost_b = [o["cost_usd"] for o in sample_obs_b]

                        # Success diff
                        boot_success_diffs[b] = np.mean(b_succ_a) - np.mean(b_succ_b)

                        # Latency diff & speedup
                        m_lat_a = np.median(b_lat_a)
                        m_lat_b = np.median(b_lat_b)
                        boot_lat_diffs[b] = m_lat_a - m_lat_b
                        boot_speedups[b] = (m_lat_b / m_lat_a) if m_lat_a > 0 else 1.0

                        # Cost diff & reduction ratio
                        m_cost_a = np.mean(b_cost_a)
                        m_cost_b = np.mean(b_cost_b)
                        boot_cost_diffs[b] = m_cost_a - m_cost_b
                        boot_cost_reductions[b] = ((m_cost_b - m_cost_a) / m_cost_b) if m_cost_b > 0 else 0.0
                    else:
                        boot_success_diffs[b] = point_success_diff
                        boot_lat_diffs[b] = point_lat_diff
                        boot_speedups[b] = point_speedup
                        boot_cost_diffs[b] = point_cost_diff
                        boot_cost_reductions[b] = point_cost_reduction
            else:
                boot_success_diffs.fill(point_success_diff)
                boot_lat_diffs.fill(point_lat_diff)
                boot_speedups.fill(point_speedup)
                boot_cost_diffs.fill(point_cost_diff)
                boot_cost_reductions.fill(point_cost_reduction)

            # Compute 95% Confidence Intervals & two-sided p-values
            alpha = 1.0 - self.confidence_level
            pct_low = 100.0 * (alpha / 2.0)
            pct_high = 100.0 * (1.0 - (alpha / 2.0))

            def make_ci(
                metric_name: str,
                point: float,
                boot_samples: np.ndarray,
                test_direction_positive: bool = True,
            ) -> MetricDifferenceCI:
                ci_low = float(np.percentile(boot_samples, pct_low))
                ci_high = float(np.percentile(boot_samples, pct_high))

                # Two-sided bootstrap p-value
                if point == 0.0 or np.all(boot_samples == 0.0):
                    p_val = 1.0
                else:
                    p_le = np.mean(boot_samples <= 0.0)
                    p_ge = np.mean(boot_samples >= 0.0)
                    p_val = float(min(1.0, 2.0 * min(p_le, p_ge)))

                return MetricDifferenceCI(
                    metric_name=metric_name,
                    engine_a=eng_a,
                    engine_b=eng_b,
                    point_difference=point,
                    ci_lower=ci_low,
                    ci_upper=ci_high,
                    confidence_level=self.confidence_level,
                    p_value=p_val,
                    p_value_holm=p_val,
                    bootstrap_resamples=b_resamples,
                    is_statistically_significant=(p_val < 0.05),
                )

            ci_success = make_ci("task_success_diff", point_success_diff, boot_success_diffs)
            ci_lat_diff = make_ci("latency_diff_ms", point_lat_diff, boot_lat_diffs)
            ci_speedup = make_ci("latency_speedup", point_speedup, boot_speedups - 1.0)  # test against ratio 1.0
            ci_speedup.point_difference = point_speedup
            ci_speedup.ci_lower = float(np.percentile(boot_speedups, pct_low))
            ci_speedup.ci_upper = float(np.percentile(boot_speedups, pct_high))

            ci_cost_diff = make_ci("cost_diff_usd", point_cost_diff, boot_cost_diffs)
            ci_cost_reduction = make_ci("cost_reduction_ratio", point_cost_reduction, boot_cost_reductions)

            # Verification against §17.3 rules:
            # Non-inferiority: lower bound of success difference >= -noninferiority_margin
            success_non_inferior = bool(ci_success.ci_lower >= -self.noninferiority_margin)

            # Invariant (§14, §17): A system with zero successes can NEVER be declared favorable!
            if total_successes_a == 0:
                overall_favorable = False
            else:
                overall_favorable = bool(
                    success_non_inferior
                    and no_new_fa
                    and (latency_target_met or cost_target_met)
                )

        return PairedComparisonResult(
            comparison_id=f"{eng_a}_vs_{eng_b}",
            engine_a=eng_a,
            engine_b=eng_b,
            sample_size_cases=n_paired_cases,
            num_families=k,
            task_success=ci_success,
            latency_difference_ms=ci_lat_diff,
            latency_speedup=ci_speedup,
            cost_difference_usd=ci_cost_diff,
            cost_reduction_ratio=ci_cost_reduction,
            false_approvals_a=fa_a,
            false_approvals_b=fa_b,
            false_approvals_diff=fa_diff,
            no_new_false_approvals=no_new_fa,
            success_non_inferior=success_non_inferior,
            latency_target_met=latency_target_met,
            cost_target_met=cost_target_met,
            overall_favorable_a=overall_favorable,
        )

    @staticmethod
    def _describe_comparison(comp: PairedComparisonResult) -> str:
        """Build data-driven gate summary for a paired comparison."""
        parts: List[str] = []

        # 1. Latency gate
        if comp.latency_target_met:
            sp = comp.latency_speedup.point_difference
            if comp.latency_speedup.ci_lower is not None and comp.latency_speedup.ci_upper is not None:
                ci_pct = int(round(comp.latency_speedup.confidence_level * 100))
                lat_str = (
                    f"latency target met ({sp:.1f}x faster, "
                    f"{ci_pct}% CI {comp.latency_speedup.ci_lower:.1f}x-{comp.latency_speedup.ci_upper:.1f}x)"
                )
            else:
                lat_str = f"latency target met ({sp:.1f}x faster)"
        else:
            sp = comp.latency_speedup.point_difference
            if comp.latency_speedup.ci_lower is not None and comp.latency_speedup.ci_upper is not None:
                ci_pct = int(round(comp.latency_speedup.confidence_level * 100))
                lat_str = (
                    f"latency target not met ({sp:.1f}x speedup, "
                    f"{ci_pct}% CI {comp.latency_speedup.ci_lower:.1f}x-{comp.latency_speedup.ci_upper:.1f}x)"
                )
            else:
                lat_str = f"latency target not met ({sp:.1f}x speedup)"
        parts.append(lat_str)

        # 2. Task success non-inferiority gate
        if comp.success_non_inferior is True:
            parts.append("task success non-inferior")
        elif comp.success_non_inferior is False:
            parts.append("FAILED task-success non-inferiority gate")
        else:
            parts.append("task success non-inferiority not evaluated")

        # 3. Cost gate
        if comp.cost_difference_usd.point_difference == 0.0 and comp.cost_reduction_ratio.point_difference == 0.0:
            parts.append("cost not measured")
        elif comp.cost_target_met:
            cr = comp.cost_reduction_ratio.point_difference * 100.0
            parts.append(f"cost target met ({cr:.1f}% reduction)")
        else:
            parts.append("FAILED cost target gate")

        # 4. False approvals gate
        if comp.no_new_false_approvals:
            parts.append(f"no new false approvals ({comp.false_approvals_a} vs {comp.false_approvals_b})")
        else:
            parts.append(f"FAILED no-new-false-approvals gate ({comp.false_approvals_a} vs {comp.false_approvals_b})")

        gate_summary = "; ".join(parts)
        return f"{comp.engine_a} vs {comp.engine_b}: {gate_summary}."

    def _determine_outcome_label(
        self,
        paired_results: Dict[str, PairedComparisonResult],
        num_families: int,
        total_cases: int,
    ) -> Tuple[str, str]:
        """Determine official outcome label adhering strictly to MEGAPLAN.md §17.4."""
        if not paired_results:
            return (
                "Inconclusive",
                "No paired comparisons executed to determine an outcome.",
            )

        if num_families <= 1 or total_cases < 5:
            return (
                "Inconclusive",
                f"Insufficient sample or single-family design ({num_families} families, {total_cases} cases) prevents conclusive statistical inference.",
            )

        a_vs_b = paired_results.get("structured_jev_vs_rag_llm")
        a_vs_c = paired_results.get("structured_jev_vs_scrape_llm")

        if not a_vs_b and not a_vs_c:
            # Fall back to first available comparison
            first_comp = next(iter(paired_results.values()))
            if first_comp.overall_favorable_a:
                return (
                    "Advantage observed, limited evidence",
                    f"Paired comparison {first_comp.comparison_id} favorable but standard engines not identified.",
                )
            return ("Inconclusive", "Standard comparisons A vs B and A vs C not found.")

        # Check for trade-off (worse quality for better speed/cost)
        if a_vs_b and a_vs_b.success_non_inferior is False and (a_vs_b.latency_target_met or a_vs_b.cost_target_met):
            return (
                "Trade-off",
                "structured_jev exhibits lower cost or latency but significant inferiority in task_success compared to rag_llm.",
            )

        # Check for negative result (loss across metrics)
        if a_vs_b and (a_vs_b.task_success.point_difference < -self.noninferiority_margin) and not a_vs_b.no_new_false_approvals:
            return (
                "Negative result",
                "structured_jev exhibits lower task success rate and higher false approval rate than rag_llm.",
            )

        # Check for advantage backed vs observed with limited evidence
        if a_vs_b and a_vs_c:
            a_beats_b = a_vs_b.overall_favorable_a
            a_beats_c = a_vs_c.overall_favorable_a

            if a_beats_b and a_beats_c:
                if num_families >= 5 and total_cases >= 20:
                    return (
                        "Advantage supported within pilot",
                        "structured_jev outperforms both rag_llm and scrape_llm on latency/cost while maintaining non-inferiority on task_success without additional false approvals, supported by stable CIs.",
                    )
                else:
                    return (
                        "Advantage observed, limited evidence",
                        f"structured_jev shows favorable means against B and C, but the number of families ({num_families}) or cases ({total_cases}) is limited to support robust significance.",
                    )

            if not a_beats_b and a_beats_c:
                desc_b = self._describe_comparison(a_vs_b)
                desc_c = self._describe_comparison(a_vs_c)
                return (
                    "Advantage vs. scraping only",
                    f"{desc_b} {desc_c}",
                )

            if not a_beats_b and not a_beats_c:
                desc_b = self._describe_comparison(a_vs_b)
                desc_c = self._describe_comparison(a_vs_c)
                return (
                    "No net benefit demonstrated",
                    f"{desc_b} {desc_c}",
                )

        if a_vs_b and a_vs_b.overall_favorable_a:
            return (
                "Advantage observed, limited evidence",
                "structured_jev shows practical advantage against rag_llm with observed evidence in the analyzed set.",
            )

        return (
            "Inconclusive",
            "Insufficient sample or mixed results prevent a definitive conclusion.",
        )


# ==============================================================================
# File Generators (§17, §18.4)
# ==============================================================================

def produce_summary_csv(output_path: Path | str, summary: BenchmarkSummary) -> Path:
    """Generate summary.csv adhering to MEGAPLAN.md §18.4 and §25.1.

    Parameters
    ----------
    output_path : Path | str
        Destination path for summary.csv.
    summary : BenchmarkSummary
        Aggregated benchmark summary.

    Returns
    -------
    Path
        Path to generated CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "engine",
        "cases",
        "completed",
        "task_success_rate",
        "verdict_accuracy",
        "selection_accuracy",
        "false_approval_rate",
        "false_rejection_rate",
        "abstention_rate",
        "correct_abstention_rate",
        "excessive_abstention_rate",
        "resolutive_coverage",
        "selective_precision",
        "evidence_validity",
        "evidence_support",
        "evidence_coverage",
        "commercial_accuracy",
        "latency_p50_ms",
        "latency_p90_ms",
        "latency_p95_ms",
        "latency_mean_ms",
        "total_cost_usd",
        "mean_cost_usd",
        "cost_per_correct_task_usd",
        "cost_status",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for engine_name, s in summary.engine_summaries.items():
            row = [
                s.engine,
                s.cases,
                s.completed_cases,
                f"{s.task_success.rate:.4f}" if s.task_success.rate is not None else "N/A",
                f"{s.verdict_accuracy_completed.rate:.4f}" if s.verdict_accuracy_completed.rate is not None else "N/A",
                f"{s.selection_accuracy.rate:.4f}" if s.selection_accuracy.rate is not None else "N/A",
                f"{s.false_approval_rate.rate:.4f}" if s.false_approval_rate.rate is not None else "N/A",
                f"{s.false_rejection_rate.rate:.4f}" if s.false_rejection_rate.rate is not None else "N/A",
                f"{s.abstention_rate.rate:.4f}" if s.abstention_rate.rate is not None else "N/A",
                f"{s.correct_abstention_rate.rate:.4f}" if s.correct_abstention_rate.rate is not None else "N/A",
                f"{s.excessive_abstention_rate.rate:.4f}" if s.excessive_abstention_rate.rate is not None else "N/A",
                f"{s.resolutive_coverage.rate:.4f}" if s.resolutive_coverage.rate is not None else "N/A",
                f"{s.selective_precision.rate:.4f}" if s.selective_precision.rate is not None else "N/A",
                f"{s.evidence_validity.rate:.4f}" if s.evidence_validity.rate is not None else "N/A",
                f"{s.evidence_support.rate:.4f}" if s.evidence_support.rate is not None else "N/A",
                f"{s.evidence_coverage.rate:.4f}" if s.evidence_coverage.rate is not None else "N/A",
                f"{s.commercial_accuracy.rate:.4f}" if s.commercial_accuracy.rate is not None else "N/A",
                f"{s.latency_p50_ms:.2f}" if s.latency_p50_ms is not None else "N/A",
                f"{s.latency_p90_ms:.2f}" if s.latency_p90_ms is not None else "N/A",
                f"{s.latency_p95_ms:.2f}" if s.latency_p95_ms is not None else "N/A",
                f"{s.latency_mean_ms:.2f}" if s.latency_mean_ms is not None else "N/A",
                f"{s.total_cost_usd:.4f}" if s.total_cost_usd is not None else "N/A",
                f"{s.mean_cost_usd:.6f}" if s.mean_cost_usd is not None else "N/A",
                (
                    f"{s.cost_per_correct_task_usd:.4f}"
                    if s.cost_per_correct_task_usd is not None and s.cost_per_correct_task_usd != float("inf")
                    else "N/A"
                ),
                s.cost_status,
            ]
            writer.writerow(row)

    return path


def produce_statistics_json(output_path: Path | str, report: StatisticsReport) -> Path:
    """Generate statistics.json adhering to MEGAPLAN.md §18.4.

    Parameters
    ----------
    output_path : Path | str
        Destination path for statistics.json.
    report : StatisticsReport
        Complete statistics report.

    Returns
    -------
    Path
        Path to generated JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    return path


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_NONINFERIORITY_MARGIN",
    "DEFAULT_LATENCY_REDUCTION_TARGET",
    "DEFAULT_COST_REDUCTION_TARGET",
    "MetricDifferenceCI",
    "PairedComparisonResult",
    "StatisticsReport",
    "adjust_p_values_holm",
    "StatisticsAnalyzer",
    "produce_summary_csv",
    "produce_statistics_json",
]
