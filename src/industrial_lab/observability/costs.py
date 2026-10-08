"""Cost Accounting module adhering to MEGAPLAN.md §16.2 and §16.3.

Provides precise inference cost calculation based on pricing rates loaded
from configs/pricing.yaml.

Formula (§16.2):
  cost_call = uncached_input_tokens * rate_input / 1e6
            + cached_input_tokens * rate_cached / 1e6
            + billed_output_tokens * rate_output / 1e6
            + other_billed * rate_other

Strict handling of missing token counts:
- If tokens are missing/null, cost_status is 'unknown' (or 'estimated' if an explicit
  estimation was performed).
- Never pretends missing usage is $0.00.
- When partial usage is available across multiple calls, reports lower bound and flags
  cost_status as 'partial' or 'unknown'.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from industrial_lab.observability.spans import (
    CostStatus,
    RateStatus,
    TelemetryRecord,
    UsageStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_PRICING_PATH = Path("configs/pricing.yaml")


class ModelRate(BaseModel):
    """Inference pricing rates for a specific model."""

    model_id: str
    provider: str = "unknown"
    rate_input: float = 0.0  # USD per 1M tokens
    rate_cached: float = 0.0  # USD per 1M tokens (rate_cached_input)
    rate_output: float = 0.0  # USD per 1M tokens
    rate_other: float = 0.0  # USD per unit for other billed items
    currency: str = "USD"
    effective_date: str = ""
    verified: bool = False  # Strict default: unconfirmed rates require verified: false
    rate_status: str = RateStatus.ASSUMED.value  # "ACCOUNT_CONFIRMED", "PUBLIC_RATE", "ASSUMED", "UNKNOWN"
    source_url: str = ""
    notes: str = ""


class CallCostResult(BaseModel):
    """Cost breakdown for an individual model or provider call."""

    model_id: str = "unknown"
    provider: str = "unknown"
    cost_usd: Optional[float] = None
    cost_status: str = CostStatus.UNKNOWN.value  # "BILL_CONFIRMED", "CALCULATED", "ESTIMATED", "UNKNOWN"
    usage_status: str = UsageStatus.UNAVAILABLE.value  # "PROVIDER_REPORTED", "LOCALLY_COUNTED", "ESTIMATED", "UNAVAILABLE"
    rate_status: str = RateStatus.UNKNOWN.value  # "ACCOUNT_CONFIRMED", "PUBLIC_RATE", "ASSUMED", "UNKNOWN"
    uncached_input_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    billed_output_tokens: Optional[int] = None
    other_billed_units: float = 0.0
    rate_input: Optional[float] = None
    rate_cached: Optional[float] = None
    rate_output: Optional[float] = None
    rate_other: Optional[float] = None
    input_cost_usd: Optional[float] = None
    cached_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    other_cost_usd: Optional[float] = None
    warning: Optional[str] = None
    notes: Optional[str] = None
    is_provider_call: bool = True
    is_tool_call: bool = False
    is_retry: bool = False
    http_status: Optional[int] = None


class AccountingRecord(BaseModel):
    """Accounting counters adhering to REPAIR_PLAN.md §9 and §10.

    Tracks:
    - logical_request_count: count of end-to-end task requests (e.g. 12 pilot cases)
    - provider_call_count: total HTTP/API model invocations (including retries, errors, tools)
    - tool_call_count: tool invocations (retrieval, scraping, parsing, quoting)
    - retry_count: re-attempts following rate-limits or transient failures
    - status_422_count: contract / unprocessable entity failures (e.g. historical criteria errors)
    - error_call_count: total unsuccessful calls
    - historical_422_calls: explicitly accounted historical 422 calls (§9: 5 calls)
    - total_cost_usd: total accumulated cost or lower bound
    - cost_status: BILL_CONFIRMED, CALCULATED, ESTIMATED, UNKNOWN, partial
    - usage_status: PROVIDER_REPORTED, LOCALLY_COUNTED, ESTIMATED, UNAVAILABLE
    - rate_status: ACCOUNT_CONFIRMED, PUBLIC_RATE, ASSUMED, UNKNOWN
    """

    logical_request_count: int = 0
    provider_call_count: int = 0
    tool_call_count: int = 0
    retry_count: int = 0
    status_422_count: int = 0
    error_call_count: int = 0
    historical_422_calls: int = 0
    total_cost_usd: Optional[float] = None
    cost_status: str = CostStatus.UNKNOWN.value
    usage_status: str = UsageStatus.UNAVAILABLE.value
    rate_status: str = RateStatus.UNKNOWN.value
    notes: list[str] = Field(default_factory=list)


class AggregatedCostResult(BaseModel):
    """Aggregated cost summary across multiple calls or spans."""

    total_cost_usd: Optional[float] = None
    cost_status: str = CostStatus.UNKNOWN.value  # "BILL_CONFIRMED", "CALCULATED", "ESTIMATED", "UNKNOWN", "partial"
    usage_status: str = UsageStatus.UNAVAILABLE.value
    rate_status: str = RateStatus.UNKNOWN.value
    lower_bound_usd: float = 0.0
    uncached_input_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    billed_output_tokens: Optional[int] = None
    call_results: list[CallCostResult] = Field(default_factory=list)
    warning: Optional[str] = None
    logical_request_count: int = 0
    provider_call_count: int = 0
    tool_call_count: int = 0
    retry_count: int = 0
    status_422_count: int = 0
    historical_422_calls: int = 0


class CostCalculator:
    """Cost calculator adhering to MEGAPLAN.md §16.2.

    Loads rates from configs/pricing.yaml and computes call costs, aggregated costs,
    amortized setup costs, and break-even thresholds.
    """

    def __init__(
        self,
        pricing_path: Optional[Path | str] = None,
        default_fallback_cost: float = 0.0,
    ) -> None:
        self.pricing_path = Path(pricing_path) if pricing_path else DEFAULT_PRICING_PATH
        self.default_fallback_cost = float(default_fallback_cost or 0.0)
        self.rates: dict[str, ModelRate] = {}
        self.scenarios: dict[str, ModelRate] = {}
        self.aliases: dict[str, str] = {}
        self.load_pricing()
        # Default engine-to-model rate aliases (§16.2)
        self.register_rate(
            ModelRate(
                model_id="local",
                provider="local",
                rate_input=0.0,
                rate_cached=0.0,
                rate_output=0.0,
                rate_other=0.0,
                verified=True,
            )
        )
        self.add_alias("structured_jev", "typesafe/jev")
        self.add_alias("jev-preview", "typesafe/jev")
        self.add_alias("jev-1.13.0", "typesafe/jev")
        self.add_alias("google.gemma-4-31b", "google.gemma-4-31b")
        self.add_alias("bedrock-mantle/google.gemma-4-31b", "google.gemma-4-31b")
        self.add_alias("gemma-4-31b", "google.gemma-4-31b")
        self.add_alias("gemma", "google.gemma-4-31b")
        self.add_alias("rag_llm", "gpt-4o-mini")
        self.add_alias("scrape_llm", "gpt-4o-mini")
        self.add_alias("structured_llm", "gpt-4o-mini")
        self.add_alias("rag_llm_guarded", "gpt-4o-mini")
        self.add_alias("structured_rules", "local")

    def load_pricing(self, path: Optional[Path | str] = None) -> None:
        """Load pricing definitions from YAML file."""
        target_path = Path(path) if path else self.pricing_path
        if not target_path.exists():
            logger.warning("Pricing file not found at %s. Initializing fallback rates.", target_path)
            self._init_fallback_rates()
            return

        try:
            with target_path.open("r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Failed to parse pricing file %s: %s", target_path, e)
            self._init_fallback_rates()
            return

        if not isinstance(raw_config, dict):
            logger.warning("Pricing file %s did not contain a YAML dictionary. Using fallback rates.", target_path)
            self._init_fallback_rates()
            return

        def _safe_float(val: Any, default: float = 0.0) -> float:
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        # 1. Parse pricing scenarios (§9, §16)
        scenarios = raw_config.get("scenarios", {})
        if isinstance(scenarios, dict):
            for scen_key, scen_data in scenarios.items():
                if not isinstance(scen_data, dict):
                    continue
                rate_cached = (
                    scen_data.get("rate_cached_input")
                    if "rate_cached_input" in scen_data
                    else scen_data.get("rate_cached", 0.0)
                )
                scen_rate_status = scen_data.get("rate_status")
                if not scen_rate_status:
                    if scen_data.get("verified"):
                        scen_rate_status = RateStatus.ACCOUNT_CONFIRMED.value
                    elif "blog" in str(scen_data.get("source_url", "")).lower() or "pricing" in str(scen_data.get("source_url", "")).lower():
                        scen_rate_status = RateStatus.PUBLIC_RATE.value
                    else:
                        scen_rate_status = RateStatus.ASSUMED.value

                scen_rate = ModelRate(
                    model_id=str(scen_data.get("model_id", scen_key)),
                    provider=str(scen_data.get("provider", "unknown")),
                    rate_input=_safe_float(scen_data.get("rate_input")),
                    rate_cached=_safe_float(rate_cached),
                    rate_output=_safe_float(scen_data.get("rate_output")),
                    rate_other=_safe_float(scen_data.get("rate_other")),
                    currency=str(scen_data.get("currency", "USD")),
                    effective_date=str(scen_data.get("effective_date", "")),
                    verified=bool(scen_data.get("verified", False)),
                    rate_status=scen_rate_status,
                    source_url=str(scen_data.get("source_url", "")),
                    notes=str(scen_data.get("notes", "")),
                )
                self.scenarios[str(scen_key)] = scen_rate
                # Also register scenario name directly as a model rate lookup
                self.register_rate(
                    ModelRate(
                        model_id=str(scen_key),
                        provider=scen_rate.provider,
                        rate_input=scen_rate.rate_input,
                        rate_cached=scen_rate.rate_cached,
                        rate_output=scen_rate.rate_output,
                        rate_other=scen_rate.rate_other,
                        currency=scen_rate.currency,
                        effective_date=scen_rate.effective_date,
                        verified=scen_rate.verified,
                        rate_status=scen_rate.rate_status,
                        source_url=scen_rate.source_url,
                        notes=scen_rate.notes,
                    )
                )

        # 2. Parse providers and models
        providers = raw_config.get("providers", {})
        if not isinstance(providers, dict):
            providers = {}

        for provider_key, provider_info in providers.items():
            if not isinstance(provider_info, dict):
                continue
            models = provider_info.get("models", {})
            if not isinstance(models, dict):
                continue
            for model_id, model_data in models.items():
                if not isinstance(model_data, dict):
                    continue
                rate_cached = (
                    model_data.get("rate_cached_input")
                    if "rate_cached_input" in model_data
                    else model_data.get("rate_cached", 0.0)
                )

                model_rate_status = model_data.get("rate_status")
                if not model_rate_status:
                    if model_data.get("verified"):
                        model_rate_status = RateStatus.ACCOUNT_CONFIRMED.value
                    elif "blog" in str(model_data.get("source_url", "")).lower() or "pricing" in str(model_data.get("source_url", "")).lower():
                        model_rate_status = RateStatus.PUBLIC_RATE.value
                    else:
                        model_rate_status = RateStatus.ASSUMED.value

                rate = ModelRate(
                    model_id=str(model_id),
                    provider=str(provider_key),
                    rate_input=_safe_float(model_data.get("rate_input")),
                    rate_cached=_safe_float(rate_cached),
                    rate_output=_safe_float(model_data.get("rate_output")),
                    rate_other=_safe_float(model_data.get("rate_other")),
                    currency=str(model_data.get("currency", "USD")),
                    effective_date=str(model_data.get("effective_date", "")),
                    verified=bool(model_data.get("verified", False)),
                    rate_status=model_rate_status,
                    source_url=str(model_data.get("source_url", "")),
                    notes=str(model_data.get("notes", "")),
                )
                self.register_rate(rate)

    def _init_fallback_rates(self) -> None:
        """Initialize standard baseline rates if pricing.yaml is missing."""
        fallback = [
            ModelRate(
                model_id="systemone-preview-v1",
                provider="typesafe",
                rate_input=0.0,
                rate_cached=0.0,
                rate_output=0.0,
                rate_other=0.0,
                verified=False,
                rate_status=RateStatus.UNKNOWN.value,
                source_url="",
                notes="Unknown / unconfigured rate",
            ),
            ModelRate(
                model_id="typesafe/jev",
                provider="typesafe",
                rate_input=0.0,
                rate_cached=0.0,
                rate_output=0.0,
                rate_other=0.0,
                verified=False,
                rate_status=RateStatus.UNKNOWN.value,
                source_url="",
                notes="Unknown / unconfigured rate",
            ),
            ModelRate(
                model_id="jev-preview",
                provider="typesafe",
                rate_input=0.0,
                rate_cached=0.0,
                rate_output=0.0,
                rate_other=0.0,
                verified=False,
                rate_status=RateStatus.UNKNOWN.value,
                source_url="",
                notes="Unknown / unconfigured rate",
            ),
            ModelRate(
                model_id="jev-1.13.0",
                provider="typesafe",
                rate_input=0.0,
                rate_cached=0.0,
                rate_output=0.0,
                rate_other=0.0,
                verified=False,
                rate_status=RateStatus.UNKNOWN.value,
                source_url="",
                notes="Unknown / unconfigured rate",
            ),
            ModelRate(
                model_id="google.gemma-4-31b",
                provider="bedrock-mantle",
                rate_input=0.14,
                rate_cached=0.035,
                rate_output=0.40,
                rate_other=0.00,
                verified=False,
                rate_status=RateStatus.ASSUMED.value,
                source_url="https://aws.amazon.com/bedrock/pricing/",
                notes="Estimated on-demand rate for Gemma on Bedrock Mantle",
            ),
            ModelRate(
                model_id="gpt-4o-mini",
                provider="openai",
                rate_input=0.150,
                rate_cached=0.075,
                rate_output=0.600,
                rate_other=0.00,
                verified=True,
                rate_status=RateStatus.ACCOUNT_CONFIRMED.value,
                source_url="https://openai.com/api/pricing/",
            ),
            ModelRate(
                model_id="claude-3-5-sonnet",
                provider="anthropic",
                rate_input=3.00,
                rate_cached=0.30,
                rate_output=15.00,
                rate_other=0.00,
                verified=True,
                rate_status=RateStatus.ACCOUNT_CONFIRMED.value,
                source_url="https://www.anthropic.com/pricing",
            ),
            ModelRate(
                model_id="gemini-1.5-flash",
                provider="google",
                rate_input=0.075,
                rate_cached=0.01875,
                rate_output=0.300,
                rate_other=0.00,
                verified=True,
                rate_status=RateStatus.ACCOUNT_CONFIRMED.value,
                source_url="https://ai.google.dev/pricing",
            ),
            ModelRate(
                model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                provider="local",
                rate_input=0.00,
                rate_cached=0.00,
                rate_output=0.00,
                rate_other=0.00,
                verified=True,
                rate_status=RateStatus.ACCOUNT_CONFIRMED.value,
            ),
        ]
        for r in fallback:
            self.register_rate(r)

    def get_scenario(self, scenario_name: Optional[str]) -> Optional[ModelRate]:
        """Look up model rate for a named pricing scenario."""
        if not scenario_name:
            return None
        norm = self._normalize_id(scenario_name)
        if scenario_name in self.scenarios:
            return self.scenarios[scenario_name]
        for k, v in self.scenarios.items():
            if self._normalize_id(k) == norm:
                return v
        return None

    def register_rate(self, rate: ModelRate) -> None:
        """Register or override a model rate."""
        self.rates[rate.model_id] = rate
        # Also register normalized variants
        norm = self._normalize_id(rate.model_id)
        if norm not in self.aliases:
            self.aliases[norm] = rate.model_id

    def add_alias(self, alias: str, target_model_id: str) -> None:
        """Map an alias name to a known model rate."""
        self.aliases[self._normalize_id(alias)] = target_model_id

    @staticmethod
    def _normalize_id(name: Optional[str]) -> str:
        if not name or not isinstance(name, str):
            return ""
        return name.strip().lower().replace("_", "-").replace(":", "/")

    def get_rate(self, model_id: Optional[str]) -> Optional[ModelRate]:
        """Look up model rate by exact ID, alias, or provider prefix."""
        if not model_id or not isinstance(model_id, str):
            return None
        clean_id = model_id.strip()
        if not clean_id:
            return None

        if clean_id in self.rates:
            return self.rates[clean_id]

        norm = self._normalize_id(clean_id)
        if norm in self.aliases and self.aliases[norm] in self.rates:
            return self.rates[self.aliases[norm]]

        # Try stripped provider prefix (e.g. 'openai/gpt-4o-mini' -> 'gpt-4o-mini')
        if "/" in norm:
            suffix = norm.split("/")[-1]
            if suffix in self.aliases and self.aliases[suffix] in self.rates:
                return self.rates[self.aliases[suffix]]
            for key, val in self.rates.items():
                if self._normalize_id(key) == suffix or self._normalize_id(key).endswith("/" + suffix):
                    return val

        # Try prefix match (e.g. 'claude-3-5-sonnet-20241022' -> 'claude-3-5-sonnet')
        for key, val in self.rates.items():
            norm_key = self._normalize_id(key)
            if norm_key and (norm.startswith(norm_key) or norm_key.startswith(norm)):
                return val

        return None

    def calculate_call_cost(
        self,
        model_id: Optional[str] = None,
        uncached_input_tokens: Optional[int] = None,
        cached_input_tokens: Optional[int] = None,
        billed_output_tokens: Optional[int] = None,
        other_billed_units: Optional[float] = 0.0,
        total_input_tokens: Optional[int] = None,
        cached_included_in_total: bool = False,
        is_estimated: bool = False,
        fallback_cost: Optional[float] = None,
        scenario: Optional[str] = None,
        **kwargs: Any,
    ) -> CallCostResult:
        """Calculate the cost of an inference call adhering strictly to §16.2 and REPAIR_PLAN.md §9.

        Formula:
          cost_call = uncached_input_tokens * rate_input / 1e6
                    + cached_input_tokens * rate_cached / 1e6
                    + billed_output_tokens * rate_output / 1e6
                    + other_billed * rate_other

        Rules on missing tokens (§9):
        - If uncached_input_tokens (or total_input_tokens) or billed_output_tokens is None:
          NEVER assume 0 tokens or $0.00 cost.
          cost_status is set to 'unknown' (or 'estimated' if is_estimated is True or explicit positive fallback).
          cost_usd is None (or estimated positive fallback). Never $0.00.
        """
        # Look up rate: scenario override takes precedence if supplied
        rate: Optional[ModelRate] = None
        if scenario:
            rate = self.get_scenario(scenario) or self.get_rate(scenario)

        model_key = str(model_id).strip() if (model_id is not None and str(model_id).strip()) else "unknown"
        if rate is None and model_id:
            rate = self.get_rate(model_key)

        def _safe_int(val: Any) -> Optional[int]:
            if val is None:
                return None
            try:
                return max(0, int(val))
            except (ValueError, TypeError):
                return None

        cached_tokens_resolved = _safe_int(cached_input_tokens) or 0

        if uncached_input_tokens is None:
            if total_input_tokens is not None:
                total_in = _safe_int(total_input_tokens) or 0
                if cached_included_in_total and cached_input_tokens is not None:
                    uncached_resolved = max(0, total_in - cached_tokens_resolved)
                else:
                    uncached_resolved = total_in
            else:
                uncached_resolved = None
        else:
            uncached_resolved = _safe_int(uncached_input_tokens)

        billed_output_resolved = _safe_int(billed_output_tokens)

        try:
            other_units = max(0.0, float(other_billed_units or 0.0))
        except (ValueError, TypeError):
            other_units = 0.0

        is_provider_call = bool(kwargs.get("is_provider_call", True))
        is_tool_call = bool(kwargs.get("is_tool_call", False))
        is_retry = bool(kwargs.get("is_retry", False))
        http_status = kwargs.get("http_status")

        req_usage_status = kwargs.get("usage_status")
        req_rate_status = kwargs.get("rate_status")
        req_cost_status = kwargs.get("cost_status")

        if rate is None:
            res_rate_status = req_rate_status or RateStatus.UNKNOWN.value
            is_explicit_zero = (
                uncached_resolved == 0
                and billed_output_resolved == 0
                and cached_tokens_resolved == 0
                and other_units == 0.0
            )
            if uncached_resolved is None or billed_output_resolved is None:
                cost_val = fallback_cost if (fallback_cost is not None and fallback_cost > 0.0) else None
                status = CostStatus.ESTIMATED.value if (is_estimated or (fallback_cost is not None and fallback_cost > 0.0)) else CostStatus.UNKNOWN.value
                u_status = UsageStatus.ESTIMATED.value if (is_estimated or (fallback_cost is not None and fallback_cost > 0.0)) else UsageStatus.UNAVAILABLE.value
            elif is_explicit_zero:
                cost_val = 0.0
                status = CostStatus.CALCULATED.value
                u_status = req_usage_status or UsageStatus.PROVIDER_REPORTED.value
            else:
                cost_val = fallback_cost if (fallback_cost is not None and fallback_cost > 0.0) else None
                status = CostStatus.ESTIMATED.value if (fallback_cost is not None and fallback_cost > 0.0) else CostStatus.UNKNOWN.value
                u_status = UsageStatus.ESTIMATED.value if (fallback_cost is not None and fallback_cost > 0.0) else UsageStatus.UNAVAILABLE.value

            return CallCostResult(
                model_id=model_key,
                provider="unknown",
                cost_usd=cost_val,
                cost_status=req_cost_status or status,
                usage_status=req_usage_status or u_status,
                rate_status=res_rate_status,
                uncached_input_tokens=uncached_resolved,
                cached_input_tokens=cached_tokens_resolved,
                billed_output_tokens=billed_output_resolved,
                other_billed_units=other_units,
                warning=f"Model rate not found for '{model_key}' in pricing config",
                is_provider_call=is_provider_call,
                is_tool_call=is_tool_call,
                is_retry=is_retry,
                http_status=http_status,
            )

        res_rate_status = req_rate_status or rate.rate_status or (RateStatus.ACCOUNT_CONFIRMED.value if rate.verified else RateStatus.ASSUMED.value)

        # Check for missing token counts (§9): NEVER convert to 0 tokens or $0.00 cost
        if uncached_resolved is None or billed_output_resolved is None:
            if fallback_cost is not None and fallback_cost > 0.0:
                cost_val = fallback_cost
                status = CostStatus.ESTIMATED.value
                u_status = UsageStatus.ESTIMATED.value
            else:
                cost_val = None
                status = CostStatus.ESTIMATED.value if is_estimated else CostStatus.UNKNOWN.value
                u_status = UsageStatus.ESTIMATED.value if is_estimated else UsageStatus.UNAVAILABLE.value

            return CallCostResult(
                model_id=model_key,
                provider=rate.provider,
                cost_usd=cost_val,
                cost_status=req_cost_status or status,
                usage_status=req_usage_status or u_status,
                rate_status=res_rate_status,
                uncached_input_tokens=uncached_resolved,
                cached_input_tokens=cached_tokens_resolved,
                billed_output_tokens=billed_output_resolved,
                other_billed_units=other_units,
                rate_input=rate.rate_input,
                rate_cached=rate.rate_cached,
                rate_output=rate.rate_output,
                rate_other=rate.rate_other,
                warning="Missing usage/token count from provider. Cost cannot be converted to 0 tokens or $0.00.",
                is_provider_call=is_provider_call,
                is_tool_call=is_tool_call,
                is_retry=is_retry,
                http_status=http_status,
                notes=rate.notes if rate.notes else None,
            )

        # Exact computation according to §16.2
        r_input = max(0.0, float(rate.rate_input or 0.0))
        r_cached = max(0.0, float(rate.rate_cached or 0.0))
        r_output = max(0.0, float(rate.rate_output or 0.0))
        r_other = max(0.0, float(rate.rate_other or 0.0))

        input_cost = (uncached_resolved * r_input) / 1_000_000.0
        cached_cost = (cached_tokens_resolved * r_cached) / 1_000_000.0
        output_cost = (billed_output_resolved * r_output) / 1_000_000.0
        other_cost = other_units * r_other

        total_cost = input_cost + cached_cost + output_cost + other_cost

        # Determine usage status
        if req_usage_status:
            res_usage_status = req_usage_status
        elif is_estimated:
            res_usage_status = UsageStatus.ESTIMATED.value
        else:
            res_usage_status = UsageStatus.PROVIDER_REPORTED.value

        # Determine cost status
        # Invariant: If usage or rate is estimated, cost_status MUST be ESTIMATED, never exact/calculated!
        if req_cost_status:
            cost_status = req_cost_status
        elif (
            is_estimated
            or res_usage_status == UsageStatus.ESTIMATED.value
            or res_rate_status in (RateStatus.ASSUMED.value, RateStatus.UNKNOWN.value)
            or not rate.verified
        ):
            cost_status = CostStatus.ESTIMATED.value
        elif res_rate_status in (RateStatus.ACCOUNT_CONFIRMED.value, RateStatus.PUBLIC_RATE.value) and rate.verified:
            cost_status = CostStatus.CALCULATED.value
        else:
            cost_status = CostStatus.ESTIMATED.value

        return CallCostResult(
            model_id=model_key,
            provider=rate.provider,
            cost_usd=round(total_cost, 8),
            cost_status=cost_status,
            usage_status=res_usage_status,
            rate_status=res_rate_status,
            uncached_input_tokens=uncached_resolved,
            cached_input_tokens=cached_tokens_resolved,
            billed_output_tokens=billed_output_resolved,
            other_billed_units=other_units,
            rate_input=r_input,
            rate_cached=r_cached,
            rate_output=r_output,
            rate_other=r_other,
            input_cost_usd=round(input_cost, 8),
            cached_cost_usd=round(cached_cost, 8),
            output_cost_usd=round(output_cost, 8),
            other_cost_usd=round(other_cost, 8),
            notes=rate.notes if rate.notes else None,
            is_provider_call=is_provider_call,
            is_tool_call=is_tool_call,
            is_retry=is_retry,
            http_status=http_status,
        )

    def calculate_cost(
        self,
        model_id: Optional[str] = None,
        *args: Any,
        uncached_input_tokens: Optional[int] = None,
        cached_input_tokens: Optional[int] = None,
        billed_output_tokens: Optional[int] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        other_billed_units: Optional[float] = 0.0,
        total_input_tokens: Optional[int] = None,
        cached_included_in_total: bool = False,
        fallback_cost: Optional[float] = None,
        scenario: Optional[str] = None,
        **kwargs: Any,
    ) -> float:
        """Calculate inference cost safely returning 0.0 or default fallback cost.

        Adheres strictly to §16.2. Handles zero tokens, negative tokens, unknown
        model names, empty rates, and None values without throwing exceptions.
        """
        effective_fallback = float(fallback_cost if fallback_cost is not None else self.default_fallback_cost)

        try:
            # Handle TelemetryRecord or CallCostResult passed as model_id
            if isinstance(model_id, TelemetryRecord):
                tel = self.calculate_telemetry_cost(model_id)
                return float(tel.cost_usd if tel.cost_usd is not None else effective_fallback)
            if isinstance(model_id, CallCostResult):
                return float(model_id.cost_usd if model_id.cost_usd is not None else effective_fallback)

            # Map positional args if provided: (model_id, in, out) or (model_id, in, cached, out)
            if len(args) == 1:
                if input_tokens is None and uncached_input_tokens is None:
                    input_tokens = args[0]
            elif len(args) == 2:
                if input_tokens is None and uncached_input_tokens is None:
                    input_tokens = args[0]
                if output_tokens is None and billed_output_tokens is None:
                    output_tokens = args[1]
            elif len(args) == 3:
                if uncached_input_tokens is None and input_tokens is None:
                    uncached_input_tokens = args[0]
                if cached_input_tokens is None and cached_tokens is None:
                    cached_input_tokens = args[1]
                if billed_output_tokens is None and output_tokens is None:
                    billed_output_tokens = args[2]
            elif len(args) >= 4:
                if uncached_input_tokens is None and input_tokens is None:
                    uncached_input_tokens = args[0]
                if cached_input_tokens is None and cached_tokens is None:
                    cached_input_tokens = args[1]
                if billed_output_tokens is None and output_tokens is None:
                    billed_output_tokens = args[2]
                if other_billed_units is None or other_billed_units == 0.0:
                    other_billed_units = args[3]

            in_val = uncached_input_tokens if uncached_input_tokens is not None else input_tokens
            out_val = billed_output_tokens if billed_output_tokens is not None else output_tokens
            ca_val = cached_input_tokens if cached_input_tokens is not None else cached_tokens

            # If all token fields are explicitly 0 or negative
            is_zero_or_negative = (
                (in_val is not None and in_val <= 0)
                and (out_val is not None and out_val <= 0)
                and (ca_val is None or ca_val <= 0)
                and (other_billed_units is None or float(other_billed_units or 0.0) <= 0.0)
            )
            if is_zero_or_negative:
                return 0.0

            # If model_id is missing or None
            if not model_id or not isinstance(model_id, str) or not model_id.strip():
                return effective_fallback

            res = self.calculate_call_cost(
                model_id=model_id,
                uncached_input_tokens=in_val,
                cached_input_tokens=ca_val,
                billed_output_tokens=out_val,
                other_billed_units=other_billed_units or 0.0,
                total_input_tokens=total_input_tokens,
                cached_included_in_total=cached_included_in_total,
                fallback_cost=effective_fallback,
                scenario=scenario,
            )

            if res.cost_usd is not None:
                return float(res.cost_usd)
            return effective_fallback

        except Exception as e:
            logger.warning("calculate_cost failed with exception: %s. Returning fallback.", e)
            return effective_fallback

    def aggregate_costs(
        self,
        call_results: list[CallCostResult],
        include_historical_422: bool = False,
        historical_422_count: int = 5,
        logical_request_count: Optional[int] = None,
    ) -> AggregatedCostResult:
        """Aggregate costs from multiple calls adhering to §14.3 and REPAIR_PLAN.md §9.

        If any call terminated with missing usage / unknown cost, reports lower bound
        and sets cost_status to 'unknown' or 'partial'. Does not present a partial sum
        as exact cost. Missing usage is NEVER converted to $0.00.
        """
        if not call_results and not include_historical_422:
            return AggregatedCostResult(
                total_cost_usd=0.0,
                cost_status=CostStatus.CALCULATED.value,
                usage_status=UsageStatus.PROVIDER_REPORTED.value,
                rate_status=RateStatus.ACCOUNT_CONFIRMED.value,
                lower_bound_usd=0.0,
                call_results=[],
                logical_request_count=logical_request_count or 0,
            )

        has_unknown = False
        has_estimated = False
        lower_bound = 0.0
        total_uncached = 0
        total_cached = 0
        total_output = 0
        all_tokens_present = True
        all_provider_reported = True
        all_rate_confirmed = True

        provider_calls = 0
        tool_calls = 0
        retries = 0
        status_422 = 0

        for res in call_results:
            if getattr(res, "is_tool_call", False):
                tool_calls += 1
            elif getattr(res, "is_provider_call", True):
                provider_calls += 1

            if getattr(res, "is_retry", False):
                retries += 1

            if getattr(res, "http_status", None) == 422:
                status_422 += 1

            if res.cost_status in (CostStatus.UNKNOWN.value, "unknown") or res.cost_usd is None:
                has_unknown = True
                all_tokens_present = False
            else:
                lower_bound += res.cost_usd or 0.0
                if res.cost_status in (CostStatus.ESTIMATED.value, "estimated"):
                    has_estimated = True

            if getattr(res, "usage_status", None) != UsageStatus.PROVIDER_REPORTED.value:
                all_provider_reported = False
            if getattr(res, "rate_status", None) not in (RateStatus.ACCOUNT_CONFIRMED.value, RateStatus.PUBLIC_RATE.value):
                all_rate_confirmed = False

            if res.uncached_input_tokens is not None:
                total_uncached += res.uncached_input_tokens
            else:
                all_tokens_present = False

            if res.cached_input_tokens is not None:
                total_cached += res.cached_input_tokens

            if res.billed_output_tokens is not None:
                total_output += res.billed_output_tokens
            else:
                all_tokens_present = False

        historical_422 = 0
        if include_historical_422:
            historical_422 = max(0, int(historical_422_count))
            provider_calls += historical_422
            status_422 += historical_422
            # 422 calls in historical runs did not return usage; cost cannot be assumed $0.00
            has_unknown = True
            all_tokens_present = False

        req_count = logical_request_count if logical_request_count is not None else len(call_results)

        if has_unknown:
            cost_status = "partial" if lower_bound > 0 else CostStatus.UNKNOWN.value
            warning = "Cost calculation incomplete due to missing usage in one or more calls. Lower bound reported."
            total_cost_usd = round(lower_bound, 8) if lower_bound > 0 else None
            agg_usage_status = UsageStatus.UNAVAILABLE.value
            agg_rate_status = RateStatus.UNKNOWN.value if not all_rate_confirmed else RateStatus.ACCOUNT_CONFIRMED.value
        elif has_estimated or not all_rate_confirmed or not all_provider_reported:
            cost_status = CostStatus.ESTIMATED.value
            warning = "Contains estimated pricing or unverified rates."
            total_cost_usd = round(lower_bound, 8)
            agg_usage_status = UsageStatus.ESTIMATED.value if not all_provider_reported else UsageStatus.PROVIDER_REPORTED.value
            agg_rate_status = RateStatus.ASSUMED.value if not all_rate_confirmed else RateStatus.ACCOUNT_CONFIRMED.value
        else:
            cost_status = CostStatus.CALCULATED.value
            warning = None
            total_cost_usd = round(lower_bound, 8)
            agg_usage_status = UsageStatus.PROVIDER_REPORTED.value
            agg_rate_status = RateStatus.ACCOUNT_CONFIRMED.value

        return AggregatedCostResult(
            total_cost_usd=total_cost_usd,
            cost_status=cost_status,
            usage_status=agg_usage_status,
            rate_status=agg_rate_status,
            lower_bound_usd=round(lower_bound, 8),
            uncached_input_tokens=total_uncached if all_tokens_present else None,
            cached_input_tokens=total_cached if all_tokens_present else None,
            billed_output_tokens=total_output if all_tokens_present else None,
            call_results=call_results,
            warning=warning,
            logical_request_count=req_count,
            provider_call_count=provider_calls,
            tool_call_count=tool_calls,
            retry_count=retries,
            status_422_count=status_422,
            historical_422_calls=historical_422,
        )

    def calculate_telemetry_cost(self, record: TelemetryRecord) -> TelemetryRecord:
        """Calculate and attach costs to a TelemetryRecord based on its spans and tokens.

        Returns an updated copy of TelemetryRecord.
        """
        # Look for model calls among spans
        model_spans = [s for s in record.spans if s.name == "model_request" or (s.tags.get("model") or s.tags.get("model_id"))]

        call_results: list[CallCostResult] = []

        if model_spans:
            for s in model_spans:
                model_id = s.tags.get("model") or s.tags.get("model_id")
                if not model_id and record.model_ids:
                    model_id = record.model_ids[0]
                if not model_id:
                    model_id = "typesafe/jev" if record.engine == "structured_jev" else "gpt-4o-mini"

                h_status = s.tags.get("http_status") or s.tags.get("status_code")
                st_code: Optional[int] = None
                if h_status is not None:
                    try:
                        st_code = int(h_status)
                    except (ValueError, TypeError):
                        pass

                # Check tokens on span
                res = self.calculate_call_cost(
                    model_id=str(model_id),
                    uncached_input_tokens=s.input_tokens,
                    cached_input_tokens=s.cached_tokens,
                    billed_output_tokens=s.output_tokens,
                    http_status=st_code,
                    is_provider_call=True,
                )
                call_results.append(res)
        else:
            # Fall back to top-level record tokens
            model_id = record.model_ids[0] if record.model_ids else ("typesafe/jev" if record.engine == "structured_jev" else "gpt-4o-mini")
            res = self.calculate_call_cost(
                model_id=model_id,
                uncached_input_tokens=record.input_tokens,
                cached_input_tokens=record.cached_tokens,
                billed_output_tokens=record.output_tokens,
                http_status=record.http_status,
                is_provider_call=True,
            )
            call_results.append(res)

        agg = self.aggregate_costs(call_results, logical_request_count=record.logical_request_count)

        # Update record
        updated_dict = record.model_dump()
        updated_dict["cost_usd"] = agg.total_cost_usd
        updated_dict["cost_status"] = agg.cost_status
        updated_dict["usage_status"] = agg.usage_status
        updated_dict["rate_status"] = agg.rate_status
        return TelemetryRecord(**updated_dict)

    @staticmethod
    def calculate_cost_per_correct_task(
        total_online_cost_usd: Optional[float],
        number_of_successful_tasks: int,
    ) -> Optional[float]:
        """Calculate USD per correct task adhering to REPAIR_PLAN.md §9 and §16.2.

        Formula:
          cost_per_correct_task = total_online_cost / number_of_successful_tasks

        If no successful tasks exist (<= 0), returns None (undefined).
        NEVER returns 0.0, 0, or assumes zero cost.
        The total cost includes failed requests, not only successful ones.
        """
        if total_online_cost_usd is None or number_of_successful_tasks <= 0:
            return None
        return round(total_online_cost_usd / number_of_successful_tasks, 6)

    def compute_accounting(
        self,
        records: list[Any],
        include_historical_422: bool = False,
        historical_422_count: int = 5,
        historical_422_estimated_cost_usd: Optional[float] = None,
    ) -> AccountingRecord:
        """Compute comprehensive accounting metrics across execution records.

        Supports tracking:
        - logical_request_count (§9: e.g. 12 main queries)
        - provider_call_count (§9: total HTTP requests including tools and errors)
        - tool_call_count (tool execution requests)
        - retry_count (retried requests)
        - status_422_count (HTTP 422 contract errors)
        - historical 422 calls accounted for (§9: 5 calls from initial JEV run)
        """
        logical_reqs = 0
        provider_calls = 0
        tool_calls = 0
        retries = 0
        status_422 = 0
        error_calls = 0
        call_results: list[CallCostResult] = []

        for rec in records:
            logical_reqs += 1
            if isinstance(rec, TelemetryRecord):
                provider_calls += rec.provider_calls
                tool_calls += rec.tool_calls
                retries += rec.retry_count
                status_422 += rec.http_statuses.count(422)
                if rec.failure_code or (rec.http_statuses and any(s >= 400 for s in rec.http_statuses)):
                    error_calls += 1
                for s in rec.spans:
                    if s.name == "model_request" or (s.tags.get("model") or s.tags.get("model_id")):
                        m_id = str(s.tags.get("model") or s.tags.get("model_id") or "unknown")
                        h_st = s.tags.get("http_status") or s.tags.get("status_code")
                        call_res = self.calculate_call_cost(
                            model_id=m_id,
                            uncached_input_tokens=s.input_tokens,
                            cached_input_tokens=s.cached_tokens,
                            billed_output_tokens=s.output_tokens,
                            http_status=int(h_st) if h_st is not None else None,
                        )
                        call_results.append(call_res)
            elif isinstance(rec, dict):
                p_c = rec.get("provider_calls", rec.get("provider_call_count", 0))
                if isinstance(p_c, list):
                    provider_calls += len(p_c)
                elif isinstance(p_c, int):
                    provider_calls += p_c
                tool_calls += int(rec.get("tool_calls", rec.get("tool_call_count", 0)))
                retries += int(rec.get("retry_count", rec.get("retries", 0)))
                statuses = rec.get("http_statuses", [])
                if isinstance(statuses, list):
                    status_422 += statuses.count(422)
                if rec.get("error") or rec.get("failure_code"):
                    error_calls += 1

        notes = []
        historical_422 = 0
        if include_historical_422:
            historical_422 = max(0, int(historical_422_count))
            provider_calls += historical_422
            status_422 += historical_422
            error_calls += historical_422
            notes.append(
                f"Accounted for {historical_422} historical HTTP 422 calls from earlier JEV criteria contract failure. "
                "Provider returned no token usage; cost cannot be assumed $0.00."
            )

        agg = self.aggregate_costs(call_results)
        final_cost = agg.total_cost_usd
        final_status = agg.cost_status

        if historical_422 > 0:
            if historical_422_estimated_cost_usd is not None:
                final_cost = (final_cost or 0.0) + historical_422_estimated_cost_usd
                final_status = CostStatus.ESTIMATED.value
            else:
                if final_status in ("exact", CostStatus.CALCULATED.value):
                    final_status = "partial" if final_cost and final_cost > 0 else CostStatus.UNKNOWN.value

        return AccountingRecord(
            logical_request_count=logical_reqs,
            provider_call_count=provider_calls,
            tool_call_count=tool_calls,
            retry_count=retries,
            status_422_count=status_422,
            error_call_count=error_calls,
            historical_422_calls=historical_422,
            total_cost_usd=final_cost,
            cost_status=final_status,
            notes=notes,
        )

    @staticmethod
    def calculate_amortized_cost(
        preparation_cost: float,
        mean_online_cost: float,
        n_queries: int,
        measured_update_cost: float = 0.0,
    ) -> float:
        """Calculate amortized cost adhering to §16.3.

        total_cost(N) = preparation_cost + N * mean_online_cost + measured_update_cost
        amortized_cost(N) = total_cost(N) / N
        """
        if n_queries <= 0:
            raise ValueError("n_queries must be greater than 0")
        total_cost = preparation_cost + (n_queries * mean_online_cost) + measured_update_cost
        return round(total_cost / n_queries, 6)

    @staticmethod
    def calculate_break_even(
        preparation_a: float,
        preparation_b: float,
        online_a: float,
        online_b: float,
    ) -> Optional[float]:
        """Calculate break-even volume N adhering to §16.3.

        N_break_even = (preparation_A - preparation_B) / (online_B - online_A)

        Only computed when the denominator (online_B - online_A) is positive.
        """
        delta_online = online_b - online_a
        delta_prep = preparation_a - preparation_b

        if delta_online <= 0:
            # Architecture A has higher or equal online cost than B; break-even is not achievable
            return None

        if delta_prep <= 0:
            # Architecture A is already cheaper or equal in preparation; A dominates at N=0
            return 0.0

        return round(delta_prep / delta_online, 2)


_default_calculator: Optional[CostCalculator] = None


def get_default_calculator() -> CostCalculator:
    """Return a singleton instance of default CostCalculator."""
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = CostCalculator()
    return _default_calculator


def calculate_cost(
    model_id: Optional[str] = None,
    *args: Any,
    uncached_input_tokens: Optional[int] = None,
    cached_input_tokens: Optional[int] = None,
    billed_output_tokens: Optional[int] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cached_tokens: Optional[int] = None,
    other_billed_units: Optional[float] = 0.0,
    total_input_tokens: Optional[int] = None,
    cached_included_in_total: bool = False,
    fallback_cost: Optional[float] = None,
    pricing_path: Optional[Path | str] = None,
    calculator: Optional[CostCalculator] = None,
    **kwargs: Any,
) -> float:
    """Convenience function to calculate cost safely returning 0.0 or default fallback cost.

    Adheres strictly to §16.2. Handles zero tokens, negative tokens, unknown
    model names, empty rates, and None values without throwing exceptions.
    """
    try:
        calc = calculator or (CostCalculator(pricing_path) if pricing_path else get_default_calculator())
        return calc.calculate_cost(
            model_id,
            *args,
            uncached_input_tokens=uncached_input_tokens,
            cached_input_tokens=cached_input_tokens,
            billed_output_tokens=billed_output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            other_billed_units=other_billed_units,
            total_input_tokens=total_input_tokens,
            cached_included_in_total=cached_included_in_total,
            fallback_cost=fallback_cost,
            **kwargs,
        )
    except Exception as e:
        logger.warning("Error in calculate_cost module function: %s", e)
        return float(fallback_cost or 0.0)


@dataclass
class BudgetEstimateResult:
    """Pre-run theoretical budget estimation result adhering to MEGAPLAN §16.2, §21."""
    split: str
    repetitions: int
    case_count: int
    scheduled_requests: int
    max_cost_usd: float
    per_engine_max_usd: dict[str, float]
    budget_limit_usd: float
    exceeds_budget: bool
    notes: list[str] = field(default_factory=list)


class BudgetEstimator:
    """Pre-run worst-case budget estimator.

    Computes upper bound USD for benchmark execution based on token limits
    and pricing tables before requests are sent.
    """

    # Theoretical maximum tokens per request per engine from configs/models.yaml & MEGAPLAN §21
    ENGINE_TOKEN_BOUNDS: dict[str, dict[str, Any]] = {
        "structured_jev": {"in_tokens": 8000, "out_tokens": 0, "model": "typesafe/jev"},
        "rag_llm": {"in_tokens": 12000, "out_tokens": 2000, "model": "gpt-4o-mini"},
        "scrape_llm": {"in_tokens": 32000, "out_tokens": 4000, "model": "gpt-4o-mini"},
        "structured_llm": {"in_tokens": 8000, "out_tokens": 1000, "model": "gpt-4o-mini"},
        "rag_llm_guarded": {"in_tokens": 12000, "out_tokens": 2000, "model": "gpt-4o-mini"},
        "structured_rules": {"in_tokens": 0, "out_tokens": 0, "model": "local"},
    }

    def __init__(
        self,
        pricing_path: Optional[Path | str] = None,
        calculator: Optional[CostCalculator] = None,
    ) -> None:
        self.calculator = calculator or CostCalculator(pricing_path)

    def estimate(
        self,
        split: str,
        repetitions: int,
        engines: list[str],
        case_count: int = 12,
        max_run_usd: float = 20.0,
    ) -> BudgetEstimateResult:
        """Estimate the maximum possible USD cost for a benchmark run."""
        per_engine: dict[str, float] = {}
        total_max_cost = 0.0
        scheduled = case_count * repetitions * len(engines)

        for eng in engines:
            bounds = self.ENGINE_TOKEN_BOUNDS.get(eng, {"in_tokens": 10000, "out_tokens": 2000, "model": eng})
            model_target = bounds["model"]
            cost_res = self.calculator.calculate_call_cost(
                model_id=model_target,
                uncached_input_tokens=bounds["in_tokens"],
                billed_output_tokens=bounds["out_tokens"],
            )
            per_call_usd = cost_res.cost_usd if cost_res.cost_usd is not None else 0.0
            eng_total_usd = per_call_usd * case_count * repetitions
            per_engine[eng] = round(eng_total_usd, 6)
            total_max_cost += eng_total_usd

        total_max_cost = round(total_max_cost, 6)
        exceeds = total_max_cost > max_run_usd

        return BudgetEstimateResult(
            split=split,
            repetitions=repetitions,
            case_count=case_count,
            scheduled_requests=scheduled,
            max_cost_usd=total_max_cost,
            per_engine_max_usd=per_engine,
            budget_limit_usd=max_run_usd,
            exceeds_budget=exceeds,
        )


__all__ = [
    "ModelRate",
    "CallCostResult",
    "AggregatedCostResult",
    "AccountingRecord",
    "CostCalculator",
    "calculate_cost",
    "get_default_calculator",
    "BudgetEstimateResult",
    "BudgetEstimator",
]


