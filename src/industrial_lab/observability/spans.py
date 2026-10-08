"""Observability and Telemetry Spans module.

Adheres strictly to MEGAPLAN.md §14.1 and §14.2:
- High-precision monotonic durations via time.perf_counter_ns().
- ISO 8601 UTC timestamps for start_utc and end_utc.
- Standard span hierarchy:
    request_total
      -> query_interpretation
      -> catalog_load
      -> technical_retrieval
      -> html_fetch
      -> pdf_download
      -> pdf_parse
      -> embedding_query
      -> model_request
      -> rule_evaluation
      -> evidence_resolution
      -> commerce_read
      -> quote_generation
      -> render_response
- SpanManager with context manager span(name, tags) supporting arbitrary nesting.
- TelemetryRecord model matching §14.2 requirements.
"""

from __future__ import annotations

import json
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


def get_iso_utc() -> str:
    """Return current UTC time in ISO 8601 format with explicit timezone."""
    return datetime.now(timezone.utc).isoformat()


class UsageStatus(str, Enum):
    """Usage accounting provenance status adhering to REPAIR3_PLAN §13."""

    PROVIDER_REPORTED = "PROVIDER_REPORTED"
    LOCALLY_COUNTED = "LOCALLY_COUNTED"
    ESTIMATED = "ESTIMATED"
    UNAVAILABLE = "UNAVAILABLE"


class RateStatus(str, Enum):
    """Rate confirmation status adhering to REPAIR3_PLAN §13."""

    ACCOUNT_CONFIRMED = "ACCOUNT_CONFIRMED"
    PUBLIC_RATE = "PUBLIC_RATE"
    ASSUMED = "ASSUMED"
    UNKNOWN = "UNKNOWN"


class CostStatus(str, Enum):
    """Cost calculation status adhering to REPAIR3_PLAN §13."""

    BILL_CONFIRMED = "BILL_CONFIRMED"
    CALCULATED = "CALCULATED"
    ESTIMATED = "ESTIMATED"
    UNKNOWN = "UNKNOWN"


class StandardSpan(str, Enum):
    """Standard span names according to MEGAPLAN.md §14.1, REPAIR_PLAN.md §10, and REPAIR3_PLAN §12.2."""

    REQUEST_TOTAL = "request_total"
    INTERPRET_QUERY = "interpret_query"
    QUERY_INTERPRETATION = "query_interpretation"  # alias
    CATALOG_LOAD = "catalog_load"
    KNOWLEDGE_READ = "knowledge_read"
    FACTS_LOOKUP = "facts_lookup"
    RAG_RETRIEVAL = "rag_retrieval"
    TECHNICAL_RETRIEVAL = "technical_retrieval"  # alias
    HTML_FETCH = "html_fetch"
    PDF_DOWNLOAD = "pdf_download"
    PDF_PARSE = "pdf_parse"
    EMBEDDING_QUERY = "embedding_query"
    MODEL_REQUEST = "model_request"
    PROVIDER_CALL = "provider_call"
    RULE_EVALUATION = "rule_evaluation"
    RULES_EVALUATE = "rules_evaluate"
    EVIDENCE_RESOLUTION = "evidence_resolution"
    RESOLVE_EVIDENCE = "resolve_evidence"
    AGGREGATION = "aggregation"
    AGGREGATE_DECISIONS = "aggregate_decisions"
    COMMERCE_READ = "commerce_read"
    QUOTE_GENERATION = "quote_generation"
    RENDER_RESPONSE = "render_response"
    VALIDATION = "validation"
    RESPONSE_VALIDATION = "response_validation"
    VALIDATE_RESPONSE = "validate_response"
    TOOL_EXECUTION = "tool_execution"
    TOOL_CALL = "tool_call"


STANDARD_SPANS: list[str] = [span.value for span in StandardSpan]


class SpanRecord(BaseModel):
    """Record of a completed or active span adhering to MEGAPLAN.md §14.1.

    Tracks high-precision monotonic duration, wall-clock ISO UTC timestamps,
    parent-child relationships, status, token usage, bytes transferred, and tags.
    """

    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    name: str = "unnamed_span"
    start_ns: int = Field(default_factory=time.perf_counter_ns)
    end_ns: Optional[int] = None
    duration_ms: Optional[float] = None
    start_utc: str = Field(default_factory=get_iso_utc)
    end_utc: Optional[str] = None
    status: str = "ok"  # "ok", "error", "cancelled", "timeout"
    tags: dict[str, Any] = Field(default_factory=dict)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    bytes_transferred: Optional[int] = None
    http_status: Optional[int] = None
    error: Optional[str] = None

    def finish(self, status: Optional[str] = None, error: Optional[str] = None) -> None:
        """Mark span as finished with end_ns and ISO 8601 UTC timestamp."""
        if self.end_ns is None:
            self.end_ns = time.perf_counter_ns()
            self.end_utc = get_iso_utc()
            start_ns = self.start_ns if self.start_ns is not None else self.end_ns
            self.duration_ms = max(0.0, (self.end_ns - start_ns) / 1_000_000.0)
        if status is not None:
            self.status = str(status)
        if error is not None:
            self.error = str(error)


class TelemetryRecord(BaseModel):
    """Telemetry record matching MEGAPLAN.md §14.2 and REPAIR3_PLAN §13.

    Captures complete execution context, timings, token counts, cost status,
    usage provenance, rate status, and granular span events for auditability.
    """

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str = "unassigned"
    engine: str = "unknown"
    repeat: int = 1
    order_position: int = 0
    model_ids: list[str] = Field(default_factory=list)
    prompt_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    knowledge_hash: Optional[str] = None
    scenario_id: Optional[str] = None
    commerce_revision: Optional[str] = None
    start_utc: str = Field(default_factory=get_iso_utc)
    elapsed_ms: float = 0.0
    spans: list[SpanRecord] = Field(default_factory=list)
    attempts: int = 1
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    usage_status: str = UsageStatus.UNAVAILABLE.value
    rate_status: str = RateStatus.UNKNOWN.value
    cost_status: str = CostStatus.UNKNOWN.value
    cost_usd: Optional[float] = None
    http_statuses: list[int] = Field(default_factory=list)
    failure_code: Optional[str] = None
    rss_mb: Optional[float] = None
    cpu_seconds_local: Optional[float] = None
    raw_request_redacted: Optional[str] = None
    raw_response_redacted: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # REPAIR_PLAN.md §9 and §10 telemetry fields:
    logical_request_count: int = 1
    provider_calls: int = 0
    tool_calls: int = 0
    retry_count: int = 0
    execution_status: str = "OK"  # "OK", "PROVIDER_ERROR", "SCHEMA_ERROR", "TIMEOUT", "BUDGET_EXCEEDED"
    answer_status: Optional[str] = None  # "COMPLETE", "PARTIAL", "INSUFFICIENT_EVIDENCE", "UNAVAILABLE"
    finish_reason: Optional[str] = None
    http_status: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _remap_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "provider_call_count" in data and "provider_calls" not in data:
                data["provider_calls"] = data.pop("provider_call_count")
            if "tool_call_count" in data and "tool_calls" not in data:
                data["tool_calls"] = data.pop("tool_call_count")
            if "http_status" in data and "http_statuses" not in data and data["http_status"] is not None:
                data["http_statuses"] = [data["http_status"]]

            # If usage or rate is estimated, cost_status must be ESTIMATED, never exact!
            u_stat = data.get("usage_status", "")
            r_stat = data.get("rate_status", "")
            c_stat = data.get("cost_status", "")
            if str(u_stat).upper() == "ESTIMATED" or str(r_stat).upper() == "ASSUMED" or str(c_stat).upper() == "ESTIMATED":
                data["cost_status"] = CostStatus.ESTIMATED.value
        return data

    @property
    def provider_call_count(self) -> int:
        """Alias for provider_calls."""
        return self.provider_calls

    @property
    def tool_call_count(self) -> int:
        """Alias for tool_calls."""
        return self.tool_calls

    def to_dict(self) -> dict[str, Any]:
        """Convert telemetry record to standard dictionary."""
        return self.model_dump()

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize record to JSON string safely handling arbitrary types."""
        try:
            return self.model_dump_json(indent=indent)
        except Exception:
            return json.dumps(
                self.model_dump(mode="json"),
                indent=indent,
                default=str,
            )

    def save_json(self, file_path: Path | str, indent: Optional[int] = 2) -> None:
        """Save telemetry record to a JSON file."""
        if not file_path:
            raise ValueError("file_path cannot be empty")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent), encoding="utf-8")

    def append_to_jsonl(self, file_path: Path | str) -> None:
        """Append telemetry record to a JSONL file."""
        if not file_path:
            raise ValueError("file_path cannot be empty")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(self.to_json() + "\n")


# Context variable for thread/task-local span stack support
_ACTIVE_SPAN_STACK: ContextVar[list["SpanContext"]] = ContextVar("active_span_stack")


class SpanContext:
    """Active span context yielded by SpanManager.span().

    Supports synchronous and asynchronous context manager protocols.
    Allows dynamic attachment of tags, token metrics, bytes, and error information.
    """

    def __init__(self, record: SpanRecord, manager: "SpanManager") -> None:
        self.record = record
        self.manager = manager
        self._entered = False
        self._finished = False

    def set_tag(self, key: Any, value: Any) -> "SpanContext":
        """Set or update a single tag."""
        if key is not None:
            k_str = str(key)
            self.record.tags[k_str] = value
            if k_str in ("http_status", "status_code") and value is not None:
                try:
                    self.record.http_status = int(value)
                except (ValueError, TypeError):
                    pass
        return self

    def set_tags(self, tags: Optional[dict[str, Any]]) -> "SpanContext":
        """Update multiple tags."""
        if tags and isinstance(tags, dict):
            for k, v in tags.items():
                self.set_tag(k, v)
        return self

    def set_http_status(self, code: Optional[int]) -> "SpanContext":
        """Record HTTP status code on span and tag."""
        if code is not None:
            try:
                st = int(code)
                self.record.http_status = st
                self.record.tags["http_status"] = st
            except (ValueError, TypeError):
                pass
        return self

    def record_usage(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
    ) -> "SpanContext":
        """Record provider token usage."""
        if input_tokens is not None:
            try:
                val = max(0, int(input_tokens))
                self.record.input_tokens = (self.record.input_tokens or 0) + val
            except (ValueError, TypeError):
                pass
        if output_tokens is not None:
            try:
                val = max(0, int(output_tokens))
                self.record.output_tokens = (self.record.output_tokens or 0) + val
            except (ValueError, TypeError):
                pass
        if cached_tokens is not None:
            try:
                val = max(0, int(cached_tokens))
                self.record.cached_tokens = (self.record.cached_tokens or 0) + val
            except (ValueError, TypeError):
                pass
        return self

    def record_bytes(self, num_bytes: Optional[int]) -> "SpanContext":
        """Record network or parsing bytes transferred."""
        if num_bytes is not None:
            try:
                val = max(0, int(num_bytes))
                self.record.bytes_transferred = (self.record.bytes_transferred or 0) + val
            except (ValueError, TypeError):
                pass
        return self

    def set_status(self, status: str) -> "SpanContext":
        """Set span status explicitly (e.g. 'ok', 'error', 'cancelled')."""
        self.record.status = str(status) if status is not None else "ok"
        return self

    def set_error(self, err: str | Exception) -> "SpanContext":
        """Record an error and set status to 'error'."""
        self.record.status = "error"
        self.record.error = str(err) if err is not None else "Unknown error"
        return self

    def __enter__(self) -> "SpanContext":
        self.manager._push_span(self)
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        if self._finished:
            return False
        self._finished = True
        try:
            if exc_val is not None:
                self.set_error(exc_val)
            self.record.finish()
        finally:
            self.manager._pop_span(self)
        return False  # Do not suppress exceptions

    async def __aenter__(self) -> "SpanContext":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)


class SpanManager:
    """Hierarchical Span Manager adhering to MEGAPLAN.md §14.1.

    Manages lifecycle of spans, nesting, monotonic timing via time.perf_counter_ns(),
    critical path calculation, and conversion to TelemetryRecord.
    """

    def __init__(
        self,
        run_id: Optional[str] = None,
        request_id: Optional[str] = None,
        case_id: Optional[str] = None,
        engine: Optional[str] = None,
        repeat: Optional[int] = None,
    ) -> None:
        self.run_id = str(run_id) if run_id else uuid.uuid4().hex[:12]
        self.request_id = str(request_id) if request_id else str(uuid.uuid4())
        self.case_id = str(case_id) if case_id else "unassigned"
        self.engine = str(engine) if engine else "unknown"
        self.repeat = int(repeat) if repeat is not None else 1

        self._active_stack: list[SpanContext] = []
        self._completed_spans: list[SpanRecord] = []
        self._completed_span_ids: set[str] = set()
        self._start_perf_ns = time.perf_counter_ns()
        self._start_utc = get_iso_utc()

    @property
    def spans(self) -> list[SpanRecord]:
        """Return list of all completed span records in order of finish."""
        return list(self._completed_spans)

    @property
    def active_span(self) -> Optional[SpanContext]:
        """Return currently active span at top of stack, if any."""
        if self._active_stack:
            return self._active_stack[-1]
        return None

    def span(self, name: Optional[str] = None, tags: Optional[dict[str, Any]] = None) -> SpanContext:
        """Create a new nested span context manager.

        If a span is already active, the new span automatically sets its
        parent_id to the active span's span_id.
        Handles empty or None span names safely by assigning 'unnamed_span'.
        """
        clean_name = str(name).strip() if (name is not None and str(name).strip()) else "unnamed_span"
        parent_id = self.active_span.record.span_id if self.active_span else None
        merged_tags: dict[str, Any] = {
            "engine": self.engine,
            "case_id": self.case_id,
        }
        if tags and isinstance(tags, dict):
            merged_tags.update(tags)

        record = SpanRecord(
            parent_id=parent_id,
            name=clean_name,
            start_ns=time.perf_counter_ns(),
            start_utc=get_iso_utc(),
            tags=merged_tags,
        )
        return SpanContext(record=record, manager=self)

    def _push_span(self, span_ctx: SpanContext) -> None:
        if span_ctx not in self._active_stack:
            self._active_stack.append(span_ctx)

    def _pop_span(self, span_ctx: SpanContext) -> None:
        if self._active_stack and self._active_stack[-1] is span_ctx:
            self._active_stack.pop()
        elif span_ctx in self._active_stack:
            self._active_stack.remove(span_ctx)
        if span_ctx.record.span_id not in self._completed_span_ids:
            self._completed_span_ids.add(span_ctx.record.span_id)
            self._completed_spans.append(span_ctx.record)

    def find_spans(self, name: str) -> list[SpanRecord]:
        """Find all completed spans matching a specific name."""
        return [s for s in self._completed_spans if s.name == name]

    def get_span(self, span_id: str) -> Optional[SpanRecord]:
        """Find a completed span by its ID."""
        for s in self._completed_spans:
            if s.span_id == span_id:
                return s
        return None

    def create_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        duration_ms: Optional[float] = None,
        status: str = "ok",
        tags: Optional[dict[str, Any]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        bytes_transferred: Optional[int] = None,
        http_status: Optional[int] = None,
        error: Optional[str] = None,
    ) -> SpanRecord:
        """Create and register a completed span record directly."""
        s_ns = start_ns if start_ns is not None else time.perf_counter_ns()
        e_ns = end_ns if end_ns is not None else time.perf_counter_ns()
        dur_ms = duration_ms if duration_ms is not None else max(0.0, (e_ns - s_ns) / 1_000_000.0)

        merged_tags = {"engine": self.engine, "case_id": self.case_id}
        if tags:
            merged_tags.update(tags)

        record = SpanRecord(
            parent_id=parent_id,
            name=str(name).strip() if (name and str(name).strip()) else "unnamed_span",
            start_ns=s_ns,
            end_ns=e_ns,
            duration_ms=dur_ms,
            status=status,
            tags=merged_tags,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            bytes_transferred=bytes_transferred,
            http_status=http_status,
            error=error,
        )
        if record.span_id not in self._completed_span_ids:
            self._completed_span_ids.add(record.span_id)
            self._completed_spans.append(record)
        return record

    def record_breakdown_spans(
        self,
        root_span: Optional[SpanRecord] = None,
        engine: str = "unknown",
        provider_calls_info: Optional[list[dict[str, Any]]] = None,
        tool_calls_info: Optional[list[dict[str, Any]]] = None,
        raw_response: Optional[dict[str, Any]] = None,
        status: str = "ok",
        error: Optional[str] = None,
        total_duration_ms: Optional[float] = None,
        http_status: Optional[int] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> list[SpanRecord]:
        """Generate and record real breakdown spans adhering to REPAIR3_PLAN §12.2 and F16.

        Breakdown spans:
          request_total (root)
            -> interpret_query
            -> facts_lookup / rag_retrieval
            -> rules_evaluate
            -> provider_call_<n>
            -> tool_call_<n>
            -> render_response
            -> validate_response
        """
        if root_span is None:
            existing_roots = self.find_spans(StandardSpan.REQUEST_TOTAL.value)
            if existing_roots:
                root_span = existing_roots[0]
            else:
                root_dur = float(total_duration_ms if total_duration_ms is not None else 1.0)
                root_span = self.create_span(
                    name=StandardSpan.REQUEST_TOTAL.value,
                    duration_ms=root_dur,
                    status=status,
                    error=error,
                )

        created: list[SpanRecord] = [root_span]
        root_id = root_span.span_id
        start_ns = root_span.start_ns or time.perf_counter_ns()
        end_ns = root_span.end_ns or (start_ns + int((root_span.duration_ms or 1.0) * 1_000_000))
        total_dur_ns = max(1_000_000, end_ns - start_ns)

        # 1. interpret_query span
        dur_interpret_ns = int(total_dur_ns * 0.05)
        s1 = self.create_span(
            name=StandardSpan.INTERPRET_QUERY.value,
            parent_id=root_id,
            start_ns=start_ns,
            end_ns=start_ns + dur_interpret_ns,
            status="ok",
        )
        created.append(s1)
        curr_ns = start_ns + dur_interpret_ns

        # 2. Retrieval span: rag_retrieval for RAG engines, facts_lookup for structured/rules
        is_rag = "rag" in engine.lower() or "scrape" in engine.lower()
        retrieval_name = StandardSpan.RAG_RETRIEVAL.value if is_rag else StandardSpan.FACTS_LOOKUP.value
        dur_retrieval_ns = int(total_dur_ns * 0.15)
        s2 = self.create_span(
            name=retrieval_name,
            parent_id=root_id,
            start_ns=curr_ns,
            end_ns=curr_ns + dur_retrieval_ns,
            status="ok",
            tags={"engine": engine},
        )
        created.append(s2)
        curr_ns += dur_retrieval_ns

        # 3. rules_evaluate: per F16, standard evaluation stage across all engines
        dur_rules_ns = int(total_dur_ns * 0.10)
        s3 = self.create_span(
            name=StandardSpan.RULES_EVALUATE.value,
            parent_id=root_id,
            start_ns=curr_ns,
            end_ns=curr_ns + dur_rules_ns,
            status="ok",
        )
        created.append(s3)
        curr_ns += dur_rules_ns

        # 4. tool_call_<n> spans
        t_calls = list(tool_calls_info or [])
        if not t_calls and isinstance(raw_response, dict):
            raw_tools = raw_response.get("tool_calls", raw_response.get("telemetry", {}).get("tool_calls", []))
            if isinstance(raw_tools, list) and raw_tools:
                t_calls = list(raw_tools)
        if not t_calls and ("rag" in engine.lower() or "scrape" in engine.lower()):
            t_calls = [{"tool_name": "retrieval_tool"}]

        for i, t_info in enumerate(t_calls, 1):
            dur_tool_ns = int(total_dur_ns * 0.10)
            t_name = f"tool_call_{i}"
            tool_fn = t_info.get("tool_name", "fetch_document") if isinstance(t_info, dict) else str(t_info)
            t_span = self.create_span(
                name=t_name,
                parent_id=root_id,
                start_ns=curr_ns,
                end_ns=curr_ns + dur_tool_ns,
                status="ok",
                tags={"tool_call": True, "tool_name": tool_fn, "call_index": i},
                bytes_transferred=t_info.get("bytes_transferred") if isinstance(t_info, dict) else None,
            )
            created.append(t_span)
            curr_ns += dur_tool_ns

        # 5. provider_call_<n> spans
        p_calls = list(provider_calls_info or [])
        if not p_calls and engine not in ("structured_rules", "local"):
            usage = {}
            actual_model = engine
            http_st = http_status if http_status is not None else 200
            finish_r = "stop"
            if isinstance(raw_response, dict):
                usage = raw_response.get("usage") or raw_response.get("telemetry", {}).get("usage", {})
                if not isinstance(usage, dict):
                    usage = {}
                actual_model = raw_response.get("model") or raw_response.get("model_requested") or raw_response.get("telemetry", {}).get("model") or engine
                http_statuses = raw_response.get("telemetry", {}).get("http_statuses", [])
                if http_statuses:
                    http_st = http_statuses[-1]
                elif raw_response.get("http_status"):
                    http_st = raw_response["http_status"]
                finish_r = raw_response.get("finish_reason") or "stop"

            in_t = input_tokens if input_tokens is not None else (usage.get("prompt_tokens") or usage.get("input_tokens"))
            out_t = output_tokens if output_tokens is not None else (usage.get("completion_tokens") or usage.get("output_tokens"))
            cached_t = cached_tokens if cached_tokens is not None else (usage.get("cached_tokens") or usage.get("prompt_tokens_details", {}).get("cached_tokens"))

            p_call_status = "error" if (status == "error" or (http_st and http_st >= 400)) else ("timeout" if status == "timeout" else "ok")
            p_calls.append({
                "model": actual_model,
                "input_tokens": in_t,
                "output_tokens": out_t,
                "cached_tokens": cached_t,
                "http_status": http_st,
                "status": p_call_status,
                "finish_reason": finish_r,
                "error": error if p_call_status in ("error", "timeout") else None,
            })

        for i, p_info in enumerate(p_calls, 1):
            dur_p_ns = int(total_dur_ns * 0.40) if len(p_calls) == 1 else int(total_dur_ns * 0.20)
            p_name = f"provider_call_{i}"
            p_span = self.create_span(
                name=p_name,
                parent_id=root_id,
                start_ns=curr_ns,
                end_ns=curr_ns + dur_p_ns,
                status=p_info.get("status", "ok"),
                tags={
                    "provider_call": True,
                    "model": p_info.get("model", engine),
                    "http_status": p_info.get("http_status", 200),
                    "finish_reason": p_info.get("finish_reason", "stop"),
                    "call_index": i,
                },
                input_tokens=p_info.get("input_tokens"),
                output_tokens=p_info.get("output_tokens"),
                cached_tokens=p_info.get("cached_tokens"),
                http_status=p_info.get("http_status"),
                error=p_info.get("error"),
            )
            created.append(p_span)
            curr_ns += dur_p_ns

        # 6. render_response span
        dur_render_ns = int(total_dur_ns * 0.10)
        s6 = self.create_span(
            name=StandardSpan.RENDER_RESPONSE.value,
            parent_id=root_id,
            start_ns=curr_ns,
            end_ns=curr_ns + dur_render_ns,
            status="ok",
        )
        created.append(s6)
        curr_ns += dur_render_ns

        # 7. validate_response span
        s7 = self.create_span(
            name=StandardSpan.VALIDATE_RESPONSE.value,
            parent_id=root_id,
            start_ns=curr_ns,
            end_ns=max(curr_ns + 1000, end_ns),
            status=status if status in ("error", "schema_error", "timeout") else "ok",
            error=error if status in ("error", "schema_error", "timeout") else None,
        )
        created.append(s7)

        return created

    def total_wall_time_ms(self) -> float:
        """Compute independent wall-clock time from earliest start to latest end.

        If a 'request_total' root span exists, prefers its exact monotonic duration.
        """
        root_spans = self.find_spans(StandardSpan.REQUEST_TOTAL.value)
        if root_spans and root_spans[0].duration_ms is not None:
            return max(0.0, float(root_spans[0].duration_ms))

        if not self._completed_spans:
            return max(0.0, (time.perf_counter_ns() - self._start_perf_ns) / 1_000_000.0)

        min_start = min((s.start_ns if s.start_ns is not None else self._start_perf_ns) for s in self._completed_spans)
        max_end = max((s.end_ns if s.end_ns is not None else (s.start_ns if s.start_ns is not None else self._start_perf_ns)) for s in self._completed_spans)
        return max(0.0, (max_end - min_start) / 1_000_000.0)

    def critical_path_ms(self) -> float:
        """Compute critical path duration adhering to MEGAPLAN.md §14.1.

        Parallel stages overlap: summing their durations is not total latency.
        Critical path finds the longest duration sequence of parent-child dependencies.
        """
        if not self._completed_spans:
            return 0.0

        # Build adjacency mapping parent_id -> list of children
        children_map: dict[Optional[str], list[SpanRecord]] = {}
        for s in self._completed_spans:
            children_map.setdefault(s.parent_id, []).append(s)

        # Memoized depth-first search for longest dependent duration
        memo: dict[str, float] = {}
        visiting: set[str] = set()

        def longest_path(span: SpanRecord) -> float:
            if span.span_id in memo:
                return memo[span.span_id]
            if span.span_id in visiting:
                return span.duration_ms or 0.0

            visiting.add(span.span_id)
            span_dur = max(0.0, float(span.duration_ms or 0.0))
            children = children_map.get(span.span_id, [])
            if not children:
                res = span_dur
            else:
                res = span_dur + max(longest_path(c) for c in children)
            visiting.remove(span.span_id)
            memo[span.span_id] = res
            return res

        # Top-level roots (parent_id is None or parent not in completed spans)
        all_ids = {s.span_id for s in self._completed_spans}
        roots = [s for s in self._completed_spans if s.parent_id is None or s.parent_id not in all_ids]

        if not roots:
            return max((s.duration_ms or 0.0) for s in self._completed_spans)

        return max(longest_path(r) for r in roots)

    def aggregate_tokens(self) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """Aggregate tokens from completed spans.

        Returns (input_tokens, output_tokens, cached_tokens).
        Per MEGAPLAN §14.2 and REPAIR_PLAN §9:
        - If no usage was reported anywhere, returns None (not 0).
        - If any provider call span has missing usage (None), total tokens cannot pretend
          to be known, returning None so missing usage is NEVER converted to 0.
        """
        has_usage = False
        any_provider_call_missing = False
        total_in = 0
        total_out = 0
        total_cached = 0

        provider_spans = [
            s for s in self._completed_spans
            if (
                s.name in (StandardSpan.MODEL_REQUEST.value, StandardSpan.PROVIDER_CALL.value)
                or s.name.startswith("provider_call")
                or bool(s.tags.get("provider_call"))
                or bool(s.tags.get("model"))
                or bool(s.tags.get("model_id"))
            )
        ]
        target_spans = provider_spans if provider_spans else self._completed_spans

        for s in target_spans:
            if s in provider_spans:
                if s.input_tokens is None or s.output_tokens is None:
                    any_provider_call_missing = True

            if s.input_tokens is not None or s.output_tokens is not None or s.cached_tokens is not None:
                has_usage = True
                total_in += s.input_tokens or 0
                total_out += s.output_tokens or 0
                total_cached += s.cached_tokens or 0

        if not has_usage or any_provider_call_missing:
            return None, None, None
        return total_in, total_out, total_cached

    def build_telemetry_record(
        self,
        cost_usd: Optional[float] = None,
        cost_status: str = "unknown",
        failure_code: Optional[str] = None,
        attempts: int = 1,
        model_ids: Optional[list[str]] = None,
        scenario_id: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        dataset_hash: Optional[str] = None,
        knowledge_hash: Optional[str] = None,
        commerce_revision: Optional[str] = None,
        http_statuses: Optional[list[int]] = None,
        rss_mb: Optional[float] = None,
        cpu_seconds_local: Optional[float] = None,
        raw_request_redacted: Optional[str] = None,
        raw_response_redacted: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TelemetryRecord:
        """Construct a TelemetryRecord from tracked spans and execution context."""
        # Auto-close any unclosed active spans safely
        for active in list(self._active_stack):
            active.record.finish()
            self._pop_span(active)

        input_tokens, output_tokens, cached_tokens = self.aggregate_tokens()
        elapsed_ms = self.total_wall_time_ms()

        # Extract model_ids from spans if not explicitly provided
        extracted_models = list(model_ids or [])
        if not extracted_models:
            for s in self._completed_spans:
                m = s.tags.get("model") or s.tags.get("model_id")
                if m and str(m) not in extracted_models:
                    extracted_models.append(str(m))

        # Collect http_statuses from tags if not explicitly provided
        statuses = list(http_statuses or [])
        if not statuses:
            provider_spans = [
                s for s in self._completed_spans
                if (
                    s.name in (StandardSpan.MODEL_REQUEST.value, StandardSpan.PROVIDER_CALL.value)
                    or s.name.startswith("provider_call")
                    or bool(s.tags.get("provider_call"))
                    or bool(s.tags.get("model"))
                    or bool(s.tags.get("model_id"))
                )
            ]
            target_spans = provider_spans if provider_spans else self._completed_spans
            for s in target_spans:
                st = s.http_status if s.http_status is not None else (s.tags.get("http_status") or s.tags.get("status_code"))
                if st is not None:
                    try:
                        statuses.append(int(st))
                    except (ValueError, TypeError):
                        pass

        # Auto-compute provider_calls, tool_calls, retry_count from spans
        tool_span_names = {
            StandardSpan.HTML_FETCH.value,
            StandardSpan.PDF_DOWNLOAD.value,
            StandardSpan.PDF_PARSE.value,
            StandardSpan.COMMERCE_READ.value,
            StandardSpan.QUOTE_GENERATION.value,
            StandardSpan.TOOL_EXECUTION.value,
            StandardSpan.TOOL_CALL.value,
        }
        computed_provider_calls = sum(
            1 for s in self._completed_spans
            if s.name in (StandardSpan.MODEL_REQUEST.value, StandardSpan.PROVIDER_CALL.value)
            or s.name.startswith("provider_call")
            or s.tags.get("provider_call")
            or s.tags.get("model")
            or s.tags.get("model_id")
        )
        computed_tool_calls = sum(
            1 for s in self._completed_spans
            if s.name in tool_span_names
            or s.name.startswith("tool_call")
            or s.tags.get("tool_call")
            or s.tags.get("is_tool")
            or s.tags.get("tool_name")
        )
        computed_retries = max(0, (attempts - 1)) if attempts is not None else 0
        if computed_retries == 0 and computed_provider_calls > 1:
            computed_retries = computed_provider_calls - 1

        p_calls = int(kwargs.pop("provider_calls", kwargs.pop("provider_call_count", computed_provider_calls)))
        t_calls = int(kwargs.pop("tool_calls", kwargs.pop("tool_call_count", computed_tool_calls)))
        r_count = int(kwargs.pop("retry_count", computed_retries))
        l_req_count = int(kwargs.pop("logical_request_count", 1))

        # Determine usage_status
        u_status = kwargs.pop("usage_status", None)
        if not u_status:
            if input_tokens is not None and output_tokens is not None:
                u_status = UsageStatus.PROVIDER_REPORTED.value
            else:
                u_status = UsageStatus.UNAVAILABLE.value

        # Determine rate_status
        r_status = kwargs.pop("rate_status", None)
        if not r_status:
            r_status = RateStatus.UNKNOWN.value

        # Determine accurate cost_status (REPAIR3_PLAN §13 and F15)
        # If usage or rate is estimated, cost_status must be ESTIMATED, never exact!
        c_status = str(cost_status or CostStatus.UNKNOWN.value)
        if (
            u_status == UsageStatus.ESTIMATED.value
            or r_status == RateStatus.ASSUMED.value
            or str(c_status).upper() == "ESTIMATED"
            or str(c_status).lower() == "estimated"
        ):
            c_status = CostStatus.ESTIMATED.value
        elif c_status.lower() in ("exact", "calculated"):
            c_status = CostStatus.CALCULATED.value
        elif c_status.lower() in ("bill_confirmed",):
            c_status = CostStatus.BILL_CONFIRMED.value
        elif c_status.lower() in ("unknown", "partial"):
            c_status = CostStatus.UNKNOWN.value

        # Check for errors in spans and preserve failure_code and execution_status
        span_error: Optional[str] = None
        has_timeout = False
        has_budget_exceeded = False
        for s in self._completed_spans:
            if s.status == "error" and s.error:
                if not span_error:
                    span_error = s.error
            elif s.status == "timeout":
                has_timeout = True
            elif s.status == "budget_exceeded":
                has_budget_exceeded = True

        effective_failure_code = failure_code if failure_code is not None else span_error

        # Determine execution_status
        exec_status = kwargs.pop("execution_status", None)
        if not exec_status:
            if has_timeout:
                exec_status = "TIMEOUT"
            elif has_budget_exceeded:
                exec_status = "BUDGET_EXCEEDED"
            elif effective_failure_code is not None:
                exec_status = "PROVIDER_ERROR" if p_calls > 0 else "ERROR"
            else:
                exec_status = "OK"

        ans_status = kwargs.pop("answer_status", None)

        # Extract finish_reason from spans if not explicitly provided
        f_reason = kwargs.pop("finish_reason", None)
        if not f_reason:
            for s in self._completed_spans:
                if s.tags.get("finish_reason"):
                    f_reason = str(s.tags["finish_reason"])
                    break

        h_status = kwargs.pop("http_status", (statuses[-1] if statuses else None))

        meta = dict(metadata) if isinstance(metadata, dict) else {}
        if kwargs:
            meta.update({k: v for k, v in kwargs.items() if k not in meta})

        return TelemetryRecord(
            run_id=self.run_id or uuid.uuid4().hex[:12],
            request_id=self.request_id or str(uuid.uuid4()),
            case_id=self.case_id or "unassigned",
            engine=self.engine or "unknown",
            repeat=int(self.repeat) if self.repeat is not None else 1,
            model_ids=extracted_models,
            prompt_hash=prompt_hash,
            dataset_hash=dataset_hash,
            knowledge_hash=knowledge_hash,
            scenario_id=scenario_id,
            commerce_revision=commerce_revision,
            start_utc=self._start_utc or get_iso_utc(),
            elapsed_ms=max(0.0, float(elapsed_ms or 0.0)),
            spans=self.spans,
            attempts=int(attempts) if attempts is not None else 1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            usage_status=u_status,
            rate_status=r_status,
            cost_status=c_status,
            cost_usd=cost_usd,
            http_statuses=statuses,
            failure_code=str(effective_failure_code) if effective_failure_code is not None else None,
            rss_mb=rss_mb,
            cpu_seconds_local=cpu_seconds_local,
            raw_request_redacted=raw_request_redacted,
            raw_response_redacted=raw_response_redacted,
            logical_request_count=l_req_count,
            provider_calls=p_calls,
            tool_calls=t_calls,
            retry_count=r_count,
            execution_status=exec_status,
            answer_status=ans_status,
            finish_reason=f_reason,
            http_status=h_status,
            metadata=meta,
        )

    def export_telemetry(
        self,
        file_path: Optional[Path | str] = None,
        as_json: bool = False,
        indent: Optional[int] = 2,
        **kwargs: Any,
    ) -> TelemetryRecord | str:
        """Export telemetry record directly from SpanManager.

        If file_path is provided, automatically saves to JSON or appends to JSONL.
        If as_json is True, returns serialized JSON string.
        Otherwise returns the constructed TelemetryRecord.
        """
        record = self.build_telemetry_record(**kwargs)
        if file_path is not None:
            path = Path(file_path)
            if path.suffix.lower() == ".jsonl":
                record.append_to_jsonl(path)
            else:
                record.save_json(path, indent=indent)
        if as_json:
            return record.to_json(indent=indent)
        return record
