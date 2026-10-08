"""TypeSafe JEV Adapter for Industrial Selection Lab (System A).

Adheres to MEGAPLAN.md §0, §9.5, §24.2.
Target endpoint: POST https://api.typesafe.ai/v1/systemone
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from typing import Any, Mapping

import httpx

from .exceptions import (
    APIRequestError,
    ModelValidationError,
    ProviderBlockedError,
    ProviderContractError,
    ResponseValidationError,
)

logger = logging.getLogger(__name__)


def canonical_state_json(state: Any) -> str:
    """Serializes state with canonical JSON formatting.

    Adheres to MEGAPLAN.md §9.5: Ensures deterministic key sorting, compact delimiters,
    and unescaped UTF-8 characters.
    """
    if isinstance(state, str):
        try:
            parsed = json.loads(state)
            return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return state.strip()
    elif isinstance(state, (dict, list)):
        return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    else:
        try:
            return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except TypeError:
            return str(state).strip()


def validate_jev_response(
    data: Mapping[str, Any],
    requested_questions: Mapping[str, Mapping[str, Any]],
    tolerance: float = 0.02,
    model_requested: str | None = None,
    strict_model: bool = False,
    is_mock: bool = False,
) -> dict[str, Any]:
    """Validates JEV response data against contract specifications.

    Adheres to MEGAPLAN.md §9.5, §24.2 and REPAIR_PLAN.md §3:
    - Same validator for real shape and fixture shape.
    - All requested question IDs must be answered.
    - No unexpected question IDs allowed.
    - Choice must be in criteria keys.
    - Probabilities must be finite numbers in [0.0, 1.0] and sum to ~1.0 within tolerance.
    - Captures choice, distribution, confidence, usage tokens, and is_mock flag.
    - Unknown usage tokens are stored as None (never defaulted to 0).
    """
    if not isinstance(data, Mapping):
        raise ResponseValidationError(f"Expected mapping response from JEV API, got {type(data).__name__}")

    # Check for model match
    model_returned = data.get("model")
    if model_requested and model_returned and model_requested != model_returned:
        msg = (
            f"Returned model '{model_returned}' does not match frozen requested model '{model_requested}'. "
            "Adheres to MEGAPLAN.md §24.2: distinct model invalidates run or must be logged as change."
        )
        if strict_model:
            raise ResponseValidationError(msg)
        logger.warning(msg)

    # Locate results/answers mapping
    results_map: Mapping[str, Any] | None = None
    for candidate_key in ("results", "answers", "questions"):
        if candidate_key in data and isinstance(data[candidate_key], Mapping):
            results_map = data[candidate_key]
            break

    if results_map is None:
        # Fallback: check if requested question IDs are top-level keys
        if any(qid in data for qid in requested_questions):
            results_map = data
        else:
            raise ResponseValidationError(
                "Response does not contain a recognized questions/results mapping ('results', 'answers', 'questions')"
            )

    # Verify all requested questions are present (detect missing IDs)
    missing_qids = [qid for qid in requested_questions if qid not in results_map]
    if missing_qids:
        raise ResponseValidationError(
            f"JEV response is missing answers for requested question(s): {missing_qids}"
        )

    # Detect unexpected question IDs (strict ID matching per REPAIR_PLAN §3.2)
    unexpected_qids = [qid for qid in results_map if qid not in requested_questions and qid not in ("model", "usage", "telemetry", "is_mock", "status", "data_origin")]
    if unexpected_qids:
        raise ResponseValidationError(
            f"JEV response contains unexpected question(s) not in sent request: {unexpected_qids}"
        )

    validated_results: dict[str, dict[str, Any]] = {}

    for qid, qspec in requested_questions.items():
        q_answer = results_map[qid]
        if not isinstance(q_answer, Mapping):
            raise ResponseValidationError(
                f"Answer for question '{qid}' must be a mapping, got {type(q_answer).__name__}"
            )

        # Extract distribution / probabilities
        distribution: Mapping[str, Any] | None = None
        for dist_key in ("distribution", "probabilities", "probs"):
            if dist_key in q_answer and isinstance(q_answer[dist_key], Mapping):
                distribution = q_answer[dist_key]
                break

        if distribution is None:
            raise ResponseValidationError(
                f"Question '{qid}' is missing a valid 'distribution' mapping in response"
            )

        validated_dist: dict[str, float] = {}
        total_prob = 0.0

        for outcome, prob in distribution.items():
            if not isinstance(prob, (int, float)) or math.isnan(prob) or math.isinf(prob):
                raise ResponseValidationError(
                    f"Question '{qid}' outcome '{outcome}' has invalid non-finite probability: {prob!r}"
                )
            # Allow minor floating point imprecision beyond [0, 1]
            if prob < -1e-6 or prob > 1.0 + 1e-6:
                raise ResponseValidationError(
                    f"Question '{qid}' outcome '{outcome}' probability {prob} out of bounds [0.0, 1.0]"
                )
            prob_float = float(prob)
            validated_dist[outcome] = prob_float
            total_prob += prob_float

        if abs(total_prob - 1.0) > tolerance:
            raise ResponseValidationError(
                f"Question '{qid}' probabilities do not sum to 1.0 within tolerance {tolerance} "
                f"(sum={total_prob:.5f}, distribution={validated_dist})"
            )

        # Choice validation
        choice = q_answer.get("choice")
        if choice is None:
            # Derive choice from argmax if missing
            choice = max(validated_dist.items(), key=lambda item: item[1])[0]
        else:
            choice = str(choice)
            if validated_dist and choice not in validated_dist:
                raise ResponseValidationError(
                    f"Question '{qid}' reported choice '{choice}' is not present in distribution keys: {list(validated_dist.keys())}"
                )

        # Validate choice against criteria defined on question (REPAIR_PLAN §3.1)
        criteria = qspec.get("criteria")
        if isinstance(criteria, Mapping) and criteria:
            valid_criteria = set(criteria.keys())
            if choice not in valid_criteria:
                raise ResponseValidationError(
                    f"Question '{qid}' reported choice '{choice}' is not one of allowed criteria: {sorted(valid_criteria)}"
                )

        # Confidence validation (optional per spec)
        confidence = q_answer.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)) or math.isnan(confidence) or math.isinf(confidence):
                raise ResponseValidationError(
                    f"Question '{qid}' has invalid non-finite confidence: {confidence!r}"
                )
            confidence = float(confidence)
            if confidence < -1e-6 or confidence > 1.0 + 1e-6:
                raise ResponseValidationError(
                    f"Question '{qid}' confidence {confidence} is out of bounds [0.0, 1.0]"
                )

        validated_results[qid] = {
            "choice": choice,
            "distribution": validated_dist,
            "confidence": confidence,
        }

    # Usage token extraction (adheres to §14.2 & §24.2: unknown usage is null, never 0)
    raw_usage = data.get("usage")
    if isinstance(raw_usage, Mapping):
        in_tokens = raw_usage.get("input_tokens")
        if in_tokens is None:
            in_tokens = raw_usage.get("prompt_tokens")
        out_tokens = raw_usage.get("output_tokens")
        if out_tokens is None:
            out_tokens = raw_usage.get("completion_tokens")
        cached_tok = raw_usage.get("cached_tokens")
        if cached_tok is None and isinstance(raw_usage.get("prompt_tokens_details"), Mapping):
            cached_tok = raw_usage["prompt_tokens_details"].get("cached_tokens")

        usage = {
            "input_tokens": int(in_tokens) if in_tokens is not None else None,
            "output_tokens": int(out_tokens) if out_tokens is not None else None,
            "cached_tokens": int(cached_tok) if cached_tok is not None else None,
        }
    else:
        usage = {
            "input_tokens": None,
            "output_tokens": None,
            "cached_tokens": None,
        }

    # Detect is_mock flag from data or parameter
    effective_is_mock = is_mock or bool(data.get("is_mock", False))

    return {
        "model_requested": model_requested,
        "model_returned": model_returned,
        "results": validated_results,
        "usage": usage,
        "is_mock": effective_is_mock,
    }


class JevAdapter:
    """Adapter for TypeSafe JEV System One API (§0, §9.5, §24.2).

    Target endpoint: POST https://api.typesafe.ai/v1/systemone
    """

    DEFAULT_ENDPOINT: str = "https://api.typesafe.ai/v1/systemone"

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 1,
        http_client: httpx.AsyncClient | None = None,
        tolerance: float = 0.02,
        strict_model: bool = False,
    ) -> None:
        """Initializes the JEV adapter.

        Raises:
            ProviderBlockedError: If TYPESAFE_API_KEY is missing or empty.
            ModelValidationError: If model_id is 'latest' or unversioned.
        """
        raw_key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", "")
        cleaned_key = raw_key.strip()
        if not cleaned_key:
            raise ProviderBlockedError("TypeSafe JEV credentials not provided; system A is blocked.")

        self.api_key: str = cleaned_key
        raw_model = model_id if model_id is not None else os.environ.get("JEV_MODEL_ID", "jev-1.13.0")
        if raw_model == "systemone-preview-v1":
            raw_model = "jev-1.13.0"
        self.model_id: str = self._validate_model_id(raw_model)
        self.endpoint_url: str = (base_url or os.environ.get("TYPESAFE_BASE_URL") or self.DEFAULT_ENDPOINT).strip()
        self.timeout: float = timeout
        self.max_retries: int = max(0, min(max_retries, 1))  # exactly max 1 retry per MEGAPLAN §24.2
        self._http_client: httpx.AsyncClient | None = http_client
        self.tolerance: float = tolerance
        self.strict_model: bool = strict_model

    @staticmethod
    def _validate_model_id(model_id: str) -> str:
        """Validates that model_id is concrete and rejects 'latest'.

        Adheres to MEGAPLAN.md §9.5 and §24.2.
        """
        if not model_id or not model_id.strip():
            raise ModelValidationError("JEV model_id must be provided and non-empty.")

        cleaned = model_id.strip()
        lower = cleaned.lower()
        if lower == "latest" or lower.endswith("-latest") or lower.endswith("/latest") or ":latest" in lower:
            raise ModelValidationError(
                f"JEV model_id must be concrete and frozen; reject 'latest' (got {model_id!r}). "
                "Adheres to MEGAPLAN.md §9.5 and §24.2."
            )
        return cleaned

    @staticmethod
    def _check_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Detects duplicate keys in raw JSON response."""
        res: dict[str, Any] = {}
        for k, v in pairs:
            if k in res:
                raise ResponseValidationError(f"Duplicate key detected in JEV response JSON: '{k}'")
            res[k] = v
        return res

    async def judge_state(
        self,
        state_text: str | Any,
        questions: dict[str, dict[str, Any]],
        is_mock: bool = False,
    ) -> dict[str, Any]:
        """Judges a technical state against a set of atomic questions.

        Adheres strictly to REPAIR_PLAN.md §3:
        - Contract §3.1: criteria is a direct field of choice question, no options, no nested choice.
        - Opaque question IDs mapped locally.
        - Error 422: raises ProviderContractError (PROVIDER_CONTRACT_ERROR), no retry of same body.
        - Duplicate keys detected.
        - is_mock recorded on doubles.

        Args:
            state_text: State description or structured dictionary to be canonically serialized.
            questions: Dictionary mapping question_id to question spec dictionary.
            is_mock: Flag indicating if this call is using recorded test double.

        Returns:
            Dictionary containing validated results, usage, telemetry, raw response, and sanitized request.

        Raises:
            ProviderBlockedError: If credentials are not provided.
            ModelValidationError: If model_id or question spec is invalid.
            ProviderContractError: On HTTP 422 error.
            ResponseValidationError: If response violates distribution/schema invariants.
            APIRequestError: If HTTP request fails after retries.
        """
        if not questions:
            raise ValueError("Questions dictionary must not be empty.")

        # Re-validate credentials and model
        if not self.api_key:
            raise ProviderBlockedError("TypeSafe JEV credentials not provided; system A is blocked.")
        self._validate_model_id(self.model_id)

        # Validate question specifications adhere to Section 3.1
        for qid, qspec in questions.items():
            if not isinstance(qspec, Mapping):
                raise ModelValidationError(f"Question '{qid}' spec must be a dictionary, got {type(qspec).__name__}")
            if qspec.get("type") != "choice":
                raise ModelValidationError(f"Question '{qid}' type must be 'choice', got {qspec.get('type')!r}")
            if "options" in qspec:
                raise ModelValidationError(
                    f"Question '{qid}' contains forbidden 'options' field (caused 422 in candidate run; see REPAIR_PLAN §3.1)"
                )
            if "choice" in qspec:
                raise ModelValidationError(
                    f"Question '{qid}' contains nested 'choice' object; 'criteria' must be a direct field (see REPAIR_PLAN §3.1)"
                )
            criteria = qspec.get("criteria")
            if not isinstance(criteria, Mapping) or not criteria:
                raise ModelValidationError(
                    f"Question '{qid}' must have a direct non-empty 'criteria' mapping (see REPAIR_PLAN §3.1)"
                )

        # Canonical JSON serialization of state (§9.5)
        canonical_state = canonical_state_json(state_text)

        payload = {
            "model": self.model_id,
            "state": canonical_state,
            "questions": questions,
        }

        # NEVER log api_key or Authorization header!
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        attempts = 0
        http_statuses: list[int] = []
        start_time = time.monotonic()
        last_error: Exception | None = None
        data: dict[str, Any] | None = None

        # Client lifecycle
        client = self._http_client
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self.timeout)

        try:
            while attempts <= self.max_retries:
                attempts += 1
                try:
                    logger.debug(
                        "Sending JEV judge_state request to %s (attempt %d, model=%s)",
                        self.endpoint_url,
                        attempts,
                        self.model_id,
                    )
                    resp = await client.post(
                        self.endpoint_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                    http_statuses.append(resp.status_code)

                    # 422 is PROVIDER_CONTRACT_ERROR: DO NOT RETRY per REPAIR_PLAN §3.3
                    if resp.status_code == 422:
                        raise ProviderContractError(
                            f"JEV API contract error 422 (PROVIDER_CONTRACT_ERROR): {resp.text}",
                            status_code=422,
                            response_text=resp.text,
                        )

                    # Retry on 429 or 5xx (max 1 retry)
                    if (resp.status_code == 429 or 500 <= resp.status_code < 600) and attempts <= self.max_retries:
                        retry_after = 1.0
                        retry_header = resp.headers.get("Retry-After")
                        if retry_header:
                            try:
                                retry_after = max(0.1, min(float(retry_header), 5.0))
                            except ValueError:
                                pass
                        logger.warning(
                            "JEV request returned status %d. Retrying after %.2fs backoff (attempt %d/%d)...",
                            resp.status_code,
                            retry_after,
                            attempts,
                            self.max_retries + 1,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.is_error:
                        raise APIRequestError(
                            f"JEV API request failed with status {resp.status_code}: {resp.text}",
                            status_code=resp.status_code,
                            response_text=resp.text,
                        )

                    try:
                        data = json.loads(resp.text, object_pairs_hook=self._check_duplicate_keys)
                    except json.JSONDecodeError as exc:
                        raise ResponseValidationError(f"Invalid JSON returned from JEV API: {exc}") from exc
                    break

                except ProviderContractError:
                    # Do not retry contract error
                    raise

                except (httpx.RequestError, httpx.TimeoutException) as exc:
                    last_error = exc
                    http_statuses.append(0)
                    if attempts <= self.max_retries:
                        logger.warning("JEV request network error (%s). Retrying (attempt %d)...", exc, attempts)
                        await asyncio.sleep(1.0)
                        continue
                    raise APIRequestError(f"JEV API network request failed after {attempts} attempts: {exc}") from exc

            if data is None:
                raise APIRequestError(
                    f"JEV API request failed after {attempts} attempts. Last error: {last_error}",
                    status_code=http_statuses[-1] if http_statuses else None,
                )

        finally:
            if owns_client and client is not None:
                await client.aclose()

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        # Validate response adhering to §9.5, §24.2 and REPAIR_PLAN §3
        validation_result = validate_jev_response(
            data=data,
            requested_questions=questions,
            tolerance=self.tolerance,
            model_requested=self.model_id,
            strict_model=self.strict_model,
            is_mock=is_mock,
        )

        telemetry = {
            "elapsed_ms": elapsed_ms,
            "attempts": attempts,
            "http_statuses": http_statuses,
        }

        # Sanitize payload for persistence (never contain credentials)
        sanitized_payload = {
            "model": self.model_id,
            "state": canonical_state,
            "questions": questions,
        }

        return {
            "model_requested": self.model_id,
            "model_returned": validation_result.get("model_returned"),
            "results": validation_result["results"],
            "usage": validation_result["usage"],
            "telemetry": telemetry,
            "raw_response": data,
            "is_mock": validation_result.get("is_mock", is_mock),
            "sanitized_request": sanitized_payload,
        }
