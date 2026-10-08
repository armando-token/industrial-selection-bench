"""Universal OpenAI-compatible LLM client adapter for Systems B and C.

Adheres to MEGAPLAN.md §10, §11, §14.2, §20.3, §24.2.

Bedrock Mantle
--------------
Default production binding uses Amazon Bedrock Mantle (OpenAI-compatible
chat completions) at ``https://bedrock-mantle.us-east-1.api.aws/openai/v1``
with model ``google.gemma-4-31b``. Auth is a bearer token from
``LLM_API_KEY`` or, if empty, ``AWS_BEARER_TOKEN_BEDROCK``. Set
``LLM_BASE_URL`` / ``LLM_MODEL_ID`` / ``LLM_PROVIDER=bedrock-mantle`` via env.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx

from .exceptions import APIRequestError, ProviderBlockedError

logger = logging.getLogger(__name__)


def resolve_llm_api_key(explicit: str | None = None) -> str:
    """Resolve LLM bearer token without logging the value.

    Preference order:
    1. Explicit non-empty ``explicit`` argument
    2. ``LLM_API_KEY``
    3. ``AWS_BEARER_TOKEN_BEDROCK`` (Amazon Bedrock Mantle)
    """
    if explicit is not None and explicit.strip():
        return explicit.strip()
    for env_name in ("LLM_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"):
        raw = os.environ.get(env_name, "")
        if raw and raw.strip():
            return raw.strip()
    return ""




@dataclass
class UsageTelemetry:
    """Token usage telemetry adhering to MEGAPLAN §14.2."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ToolCall:
    """Represents a tool call requested by the model."""

    id: str
    name: str
    arguments: str
    parsed_arguments: dict[str, Any] = field(default_factory=dict)
    type: str = "function"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass
class LLMResponse:
    """Universal response object returned by LLMAdapter."""

    content: str | None
    parsed: dict[str, Any] | list[Any] | None
    tool_calls: list[ToolCall]
    role: str
    finish_reason: str | None
    model: str
    usage: UsageTelemetry
    telemetry: dict[str, Any]
    raw_response: dict[str, Any]
    raw_request: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for backwards compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.raw_response[key]

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.raw_response.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "content": self.content,
            "parsed": self.parsed,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "role": self.role,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "telemetry": self.telemetry,
            "raw_request": self.raw_request,
            "raw_response": self.raw_response,
        }


class LLMAdapter:
    """Universal OpenAI-compatible LLM client for Systems B and C.

    Supports:
    - Chat completions with structured JSON outputs (response_format).
    - Function and tool calling (§11.1).
    - Token telemetry tracking prompt_tokens, completion_tokens, cached_tokens (§14.2).
    - ProviderBlockedError when credentials are not configured.
    - Safe logging with all sensitive authorization tokens redacted.
    - Retries with backoff for 429 and 5xx errors.
    - Amazon Bedrock Mantle via OpenAI-compatible base URL + bearer
      (``LLM_API_KEY`` or ``AWS_BEARER_TOKEN_BEDROCK``).
    """

    DEFAULT_BASE_URL: str = "https://api.openai.com/v1"
    DEFAULT_MODEL: str = "gpt-4o-mini"
    MANTLE_BASE_URL: str = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
    MANTLE_MODEL: str = "google.gemma-4-31b"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_id: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 1,
        http_client: httpx.AsyncClient | None = None,
        require_key: bool = False,
    ) -> None:
        """Initializes LLM client adapter.

        If api_key is missing or empty and require_key is True, raises ProviderBlockedError.
        Otherwise, stores None and defers check until chat() is invoked.
        """
        cleaned_key = resolve_llm_api_key(api_key)

        if require_key and not cleaned_key:
            raise ProviderBlockedError("LLM API key not provided; LLM provider is blocked.")

        self.api_key: str | None = cleaned_key or None
        provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
        default_base = self.MANTLE_BASE_URL if provider in {"bedrock-mantle", "bedrock_mantle", "mantle"} else self.DEFAULT_BASE_URL
        default_model = self.MANTLE_MODEL if provider in {"bedrock-mantle", "bedrock_mantle", "mantle"} else self.DEFAULT_MODEL
        self.base_url: str = (base_url or os.environ.get("LLM_BASE_URL") or default_base).rstrip("/")
        self.model_id: str = (model_id or os.environ.get("LLM_MODEL_ID") or default_model).strip()
        self.timeout: float = timeout
        self.max_retries: int = max(0, min(max_retries, 3))
        self._http_client: httpx.AsyncClient | None = http_client
        self.call_history: list[dict[str, Any]] = []
        self.last_provider_call: dict[str, Any] | None = None

    @property
    def is_configured(self) -> bool:
        """Returns True if an API key is present."""
        return bool(self.api_key)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 2048,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Sends chat completion request to OpenAI-compatible endpoint.

        Args:
            messages: List of message mappings with 'role' and 'content'.
            model: Optional model override. Defaults to self.model_id.
            response_format: Optional structured output format, e.g. {"type": "json_object"}.
            tools: Optional tool definitions for function calling (§11).
            tool_choice: Optional tool choice setting ('auto', 'required', etc.).
            temperature: Sampling temperature (0.0 for deterministic technical tasks).
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.

        Returns:
            LLMResponse containing message content, parsed JSON if applicable,
            tool calls, and usage telemetry.

        Raises:
            ProviderBlockedError: If LLM_API_KEY is not configured.
            APIRequestError: If the remote API call fails.
        """
        if not self.api_key:
            raise ProviderBlockedError("LLM API key not provided; LLM provider is blocked.")

        target_model = (model or self.model_id).strip()
        req_timeout = timeout if timeout is not None else self.timeout

        endpoint = f"{self.base_url}/chat/completions"

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            # Bedrock Mantle's grammar-constrained json_object decoding corrupts token streams for Gemma.
            # Gemma natively follows prompt schema instructions and emits fenced or clean JSON.
            if not ("gemma" in target_model.lower() and "bedrock-mantle" in self.base_url):
                payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        # NEVER log api_key or Authorization header!
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        attempts = 0
        http_statuses: list[int] = []
        call_id = f"call_{uuid.uuid4().hex[:12]}"
        start_utc = datetime.now(timezone.utc).isoformat()
        sanitized_request = {
            "call_id": call_id,
            "endpoint": endpoint,
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "[REDACTED]",
            },
            "payload": payload,
        }
        start_time = time.monotonic()
        last_error: Exception | None = None
        data: dict[str, Any] | None = None

        client = self._http_client
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=req_timeout)

        try:
            while attempts <= self.max_retries:
                attempts += 1
                try:
                    logger.debug(
                        "Sending LLM chat completion to %s (attempt %d, model=%s)",
                        endpoint,
                        attempts,
                        target_model,
                    )
                    resp = await client.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=req_timeout,
                    )
                    http_statuses.append(resp.status_code)

                    # Retry on 429 or 5xx
                    if (resp.status_code == 429 or 500 <= resp.status_code < 600) and attempts <= self.max_retries:
                        retry_after = 1.0
                        retry_header = resp.headers.get("Retry-After")
                        if retry_header:
                            try:
                                retry_after = max(0.1, min(float(retry_header), 5.0))
                            except ValueError:
                                pass
                        logger.warning(
                            "LLM request returned status %d. Retrying after %.2fs backoff (attempt %d/%d)...",
                            resp.status_code,
                            retry_after,
                            attempts,
                            self.max_retries + 1,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.is_error:
                        raise APIRequestError(
                            f"LLM API request failed with status {resp.status_code}: {resp.text}",
                            status_code=resp.status_code,
                            response_text=resp.text,
                        )

                    data = resp.json()
                    break

                except (httpx.RequestError, httpx.TimeoutException) as exc:
                    last_error = exc
                    http_statuses.append(0)
                    if attempts <= self.max_retries:
                        logger.warning("LLM request network error (%s). Retrying (attempt %d)...", exc, attempts)
                        await asyncio.sleep(1.0)
                        continue
                    raise APIRequestError(f"LLM API network request failed after {attempts} attempts: {exc}") from exc

            if data is None:
                raise APIRequestError(
                    f"LLM API request failed after {attempts} attempts. Last error: {last_error}",
                    status_code=http_statuses[-1] if http_statuses else None,
                )

        finally:
            if owns_client and client is not None:
                await client.aclose()

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        # Parse response choices
        choices = data.get("choices", [])
        if not choices:
            raise APIRequestError("LLM response did not contain any choices", response_text=json.dumps(data))

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content")
        role = message.get("role", "assistant")
        finish_reason = first_choice.get("finish_reason")
        returned_model = data.get("model", target_model)

        # Parse tool calls if present
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls: list[ToolCall] = []
        for raw_tc in raw_tool_calls:
            tc_id = raw_tc.get("id", "")
            tc_type = raw_tc.get("type", "function")
            func = raw_tc.get("function", {})
            name = func.get("name", "")
            raw_args = func.get("arguments", "{}")
            parsed_args: dict[str, Any] = {}
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    logger.debug("Failed to parse tool arguments as JSON: %s", raw_args)
                    parsed_args = {}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
                raw_args = json.dumps(raw_args)

            tool_calls.append(
                ToolCall(
                    id=tc_id,
                    name=name,
                    arguments=raw_args,
                    parsed_arguments=parsed_args,
                    type=tc_type,
                )
            )

        # Parse structured JSON output if requested or possible
        parsed_content: dict[str, Any] | list[Any] | None = None
        if content:
            should_parse_json = (
                response_format is not None
                and isinstance(response_format, dict)
                and response_format.get("type") in ("json_object", "json_schema")
            )
            raw_text = content.strip()
            # Allow exactly one syntactic markdown unwrap (```json ... ``` or ``` ... ```)
            if raw_text.startswith("```"):
                fence_match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", raw_text, re.IGNORECASE)
                if fence_match and "```" not in fence_match.group(1):
                    raw_text = fence_match.group(1).strip()

            if should_parse_json or raw_text.startswith(("{", "[")):
                try:
                    parsed_content = json.loads(raw_text)
                except (json.JSONDecodeError, TypeError):
                    if should_parse_json:
                        logger.warning("Requested JSON output format but LLM response was not valid JSON")

        # Telemetry and token usage (§14.2: keep null if missing, never 0)
        raw_usage = data.get("usage")
        if isinstance(raw_usage, Mapping):
            p_tokens = raw_usage.get("prompt_tokens")
            c_tokens = raw_usage.get("completion_tokens")
            tot_tokens = raw_usage.get("total_tokens")

            # Extract cached tokens from prompt_tokens_details or direct key
            cached_tok = raw_usage.get("cached_tokens")
            if cached_tok is None and isinstance(raw_usage.get("prompt_tokens_details"), Mapping):
                cached_tok = raw_usage["prompt_tokens_details"].get("cached_tokens")

            usage = UsageTelemetry(
                prompt_tokens=int(p_tokens) if p_tokens is not None else None,
                completion_tokens=int(c_tokens) if c_tokens is not None else None,
                cached_tokens=int(cached_tok) if cached_tok is not None else None,
                total_tokens=int(tot_tokens) if tot_tokens is not None else None,
            )
        else:
            usage = UsageTelemetry(
                prompt_tokens=None,
                completion_tokens=None,
                cached_tokens=None,
                total_tokens=None,
            )

        telemetry = {
            "elapsed_ms": elapsed_ms,
            "attempts": attempts,
            "http_statuses": http_statuses,
        }

        end_utc = datetime.now(timezone.utc).isoformat()
        call_record = {
            "call_id": call_id,
            "provider": "bedrock-mantle" if "bedrock-mantle" in self.base_url else "openai",
            "endpoint": endpoint,
            "model_requested": target_model,
            "model_effective": returned_model,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "elapsed_ms": elapsed_ms,
            "attempts": attempts,
            "http_status": http_statuses[-1] if http_statuses else 200,
            "finish_reason": finish_reason,
            "usage": usage.to_dict(),
            "raw_request": sanitized_request,
            "raw_response": data,
        }
        self.last_provider_call = call_record
        self.call_history.append(call_record)

        return LLMResponse(
            content=content,
            parsed=parsed_content,
            tool_calls=tool_calls,
            role=role,
            finish_reason=finish_reason,
            model=returned_model,
            usage=usage,
            telemetry=telemetry,
            raw_response=data,
            raw_request=sanitized_request,
            call_id=call_id,
        )

    def write_provider_logs(self, log_dir: Path | str, request_id: str = "", case_id: str = "") -> None:
        """Appends provider_requests.jsonl and provider_responses.jsonl in log_dir (REPAIR3 F20)."""
        target_dir = Path(log_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        req_file = target_dir / "provider_requests.jsonl"
        resp_file = target_dir / "provider_responses.jsonl"

        with open(req_file, "a", encoding="utf-8") as f_req, open(resp_file, "a", encoding="utf-8") as f_resp:
            for call in self.call_history:
                req_entry = {
                    "call_id": call.get("call_id"),
                    "request_id": request_id,
                    "case_id": case_id,
                    "provider": call.get("provider"),
                    "endpoint": call.get("endpoint"),
                    "model_requested": call.get("model_requested"),
                    "method": call.get("raw_request", {}).get("method", "POST"),
                    "headers": call.get("raw_request", {}).get("headers", {}),
                    "payload": call.get("raw_request", {}).get("payload", {}),
                    "timestamp_utc": call.get("start_utc"),
                }
                resp_entry = {
                    "call_id": call.get("call_id"),
                    "request_id": request_id,
                    "case_id": case_id,
                    "provider": call.get("provider"),
                    "endpoint": call.get("endpoint"),
                    "model_effective": call.get("model_effective"),
                    "http_status": call.get("http_status"),
                    "finish_reason": call.get("finish_reason"),
                    "usage": call.get("usage"),
                    "elapsed_ms": call.get("elapsed_ms"),
                    "raw_response": call.get("raw_response"),
                    "timestamp_utc": call.get("end_utc"),
                }
                f_req.write(json.dumps(req_entry, ensure_ascii=False) + "\n")
                f_resp.write(json.dumps(resp_entry, ensure_ascii=False) + "\n")
