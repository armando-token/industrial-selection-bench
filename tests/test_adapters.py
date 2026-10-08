"""Comprehensive tests for Model & API Adapters.

Tests:
1. TypeSafe JEV Adapter (MEGAPLAN §0, §9.5, §24.2)
2. Universal LLM Adapter (MEGAPLAN §10, §11, §14.2, §20.3)
3. Embedding Adapter & Sliding Window Chunking (MEGAPLAN §10.1)
"""

from __future__ import annotations

import json
import math
import os
from typing import Any
from unittest.mock import patch

import httpx
import numpy as np
import pytest

from industrial_lab.adapters import (
    EMBEDDING_DEFAULT_MODEL_ID,
    MAX_WINDOW_TOKENS,
    MODEL_MAX_SEQ_LENGTH,
    WINDOW_OVERLAP_TOKENS,
    APIRequestError,
    EmbeddingAdapter,
    FallbackTokenizer,
    JevAdapter,
    LLMAdapter,
    ModelValidationError,
    ParentChunk,
    ProviderBlockedError,
    ResponseValidationError,
    aggregate_window_scores_to_parents,
    canonical_state_json,
    chunk_document_into_parents_and_windows,
    chunk_text_with_sliding_window,
    validate_jev_response,
)


# ============================================================================
# 1. JEV Adapter Tests (§0, §9.5, §24.2)
# ============================================================================


def test_jev_credentials_missing_raises_provider_blocked() -> None:
    """If TYPESAFE_API_KEY is missing or empty, raise ProviderBlockedError."""
    with patch.dict(os.environ, {}, clear=True):
        if "TYPESAFE_API_KEY" in os.environ:
            del os.environ["TYPESAFE_API_KEY"]

        with pytest.raises(ProviderBlockedError) as exc_info:
            JevAdapter(api_key=None)
        assert "TypeSafe JEV credentials not provided; system A is blocked." in str(exc_info.value)

        with pytest.raises(ProviderBlockedError) as exc_info:
            JevAdapter(api_key="   ")
        assert "TypeSafe JEV credentials not provided; system A is blocked." in str(exc_info.value)


def test_jev_model_rejects_latest() -> None:
    """Validates model_id is concrete and rejects 'latest'."""
    with pytest.raises(ModelValidationError):
        JevAdapter(api_key="test-key", model_id="latest")

    with pytest.raises(ModelValidationError):
        JevAdapter(api_key="test-key", model_id="jev-latest")

    with pytest.raises(ModelValidationError):
        JevAdapter(api_key="test-key", model_id="")


def test_jev_canonical_state_json() -> None:
    """Serializes state with canonical JSON formatting (§9.5)."""
    # Key sorting and compact separators
    unordered_dict = {"z": 1, "a": 2, "m": {"c": 3, "b": 4}}
    canonical = canonical_state_json(unordered_dict)
    assert canonical == '{"a":2,"m":{"b":4,"c":3},"z":1}'

    # JSON string input parsed and sorted canonically
    json_str = '{"x": 10,  "b": "test" }'
    canonical_str = canonical_state_json(json_str)
    assert canonical_str == '{"b":"test","x":10}'

    # Non-JSON string trimmed
    plain_text = "   State description in natural text   "
    assert canonical_state_json(plain_text) == "State description in natural text"


def test_jev_response_validation_success() -> None:
    """Validates full response fixture: choices, finite probabilities summing to 1.0, confidence, usage."""
    requested_questions = {
        "p1_requirement_r1": {
            "type": "choice",
            "criteria": {
                "supported": "Evidence supports R1",
                "contradicted": "Evidence contradicts R1",
                "insufficient": "Insufficient evidence",
            },
        }
    }

    valid_response = {
        "model": "jev-1.0",
        "results": {
            "p1_requirement_r1": {
                "choice": "supported",
                "distribution": {
                    "supported": 0.85,
                    "contradicted": 0.10,
                    "insufficient": 0.05,
                },
                "confidence": 0.88,
            }
        },
        "usage": {
            "input_tokens": 150,
            "output_tokens": 20,
            "cached_tokens": 10,
        },
    }

    validated = validate_jev_response(
        data=valid_response,
        requested_questions=requested_questions,
        tolerance=0.02,
        model_requested="jev-1.0",
    )

    assert validated["results"]["p1_requirement_r1"]["choice"] == "supported"
    assert validated["results"]["p1_requirement_r1"]["confidence"] == 0.88
    assert validated["usage"]["input_tokens"] == 150
    assert validated["usage"]["output_tokens"] == 20
    assert validated["usage"]["cached_tokens"] == 10


def test_jev_response_validation_failures() -> None:
    """Verifies validation failures on missing questions, invalid sums, NaNs, and bounds."""
    requested_questions = {
        "q1": {"type": "choice"},
        "q2": {"type": "choice"},
    }

    # Missing question q2
    missing_data = {
        "results": {
            "q1": {"choice": "supported", "distribution": {"supported": 1.0}}
        }
    }
    with pytest.raises(ResponseValidationError, match="missing answers for requested question"):
        validate_jev_response(missing_data, requested_questions)

    # Probabilities don't sum to 1.0
    bad_sum_data = {
        "results": {
            "q1": {"choice": "supported", "distribution": {"supported": 0.5, "contradicted": 0.1}},
            "q2": {"choice": "supported", "distribution": {"supported": 1.0}},
        }
    }
    with pytest.raises(ResponseValidationError, match="do not sum to 1.0"):
        validate_jev_response(bad_sum_data, requested_questions)

    # NaN probability
    nan_data = {
        "results": {
            "q1": {"choice": "supported", "distribution": {"supported": float("nan"), "insufficient": 0.5}},
            "q2": {"choice": "supported", "distribution": {"supported": 1.0}},
        }
    }
    with pytest.raises(ResponseValidationError, match="invalid non-finite probability"):
        validate_jev_response(nan_data, requested_questions)

    # Negative probability
    neg_data = {
        "results": {
            "q1": {"choice": "supported", "distribution": {"supported": 1.1, "insufficient": -0.1}},
            "q2": {"choice": "supported", "distribution": {"supported": 1.0}},
        }
    }
    with pytest.raises(ResponseValidationError, match="out of bounds"):
        validate_jev_response(neg_data, requested_questions)


def test_jev_unknown_usage_tokens_stored_as_none() -> None:
    """MEGAPLAN §14.2 & §24.2: Usage desconocido no vale cero; tokens se guardan como None."""
    requested_questions = {"q1": {"type": "choice"}}
    response_no_usage = {
        "results": {
            "q1": {"choice": "supported", "distribution": {"supported": 1.0}}
        }
    }
    validated = validate_jev_response(response_no_usage, requested_questions)
    assert validated["usage"]["input_tokens"] is None
    assert validated["usage"]["output_tokens"] is None
    assert validated["usage"]["cached_tokens"] is None


@pytest.mark.asyncio
async def test_jev_judge_state_success_and_headers() -> None:
    """Tests successful judge_state execution with Bearer header and telemetry."""
    requested_questions = {
        "q1": {
            "type": "choice",
            "instructions": "Evaluate R1",
            "criteria": {"supported": "ok", "contradicted": "no", "insufficient": "lack"},
        }
    }

    mock_response_data = {
        "model": "jev-1.0",
        "results": {
            "q1": {
                "choice": "supported",
                "distribution": {"supported": 0.95, "contradicted": 0.03, "insufficient": 0.02},
                "confidence": 0.95,
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer secret-jev-key"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "jev-1.0"
        assert "q1" in payload["questions"]
        return httpx.Response(200, json=mock_response_data)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = JevAdapter(
            api_key="secret-jev-key",
            model_id="jev-1.0",
            http_client=client,
        )
        res = await adapter.judge_state(state_text={"key": "val"}, questions=requested_questions)

        assert res["model_requested"] == "jev-1.0"
        assert res["results"]["q1"]["choice"] == "supported"
        assert res["usage"]["input_tokens"] == 100
        assert res["usage"]["cached_tokens"] is None
        assert res["telemetry"]["attempts"] == 1
        assert res["telemetry"]["http_statuses"] == [200]
        assert res["telemetry"]["elapsed_ms"] >= 0.0


@pytest.mark.asyncio
async def test_jev_retries_on_429() -> None:
    requested_questions = {
        "q1": {
            "type": "choice",
            "criteria": {
                "supported": "Supports requirement",
                "contradicted": "Contradicts requirement",
                "insufficient": "Insufficient evidence",
            },
        }
    }
    calls = 0

    mock_response_data = {
        "model": "jev-1.0",
        "results": {
            "q1": {"choice": "supported", "distribution": {"supported": 1.0}}
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0.01"}, json={"error": "Rate limit"})
        return httpx.Response(200, json=mock_response_data)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = JevAdapter(
            api_key="secret-key",
            model_id="jev-1.0",
            http_client=client,
            max_retries=1,
        )
        res = await adapter.judge_state(state_text="state", questions=requested_questions)
        assert res["telemetry"]["attempts"] == 2
        assert res["telemetry"]["http_statuses"] == [429, 200]
        assert calls == 2


# ============================================================================
# 2. Universal LLM Adapter Tests (§10, §11, §14.2, §20.3)
# ============================================================================


def test_llm_credentials_missing_raises_provider_blocked() -> None:
    """If LLM_API_KEY is not provided, raises ProviderBlockedError."""
    with patch.dict(os.environ, {}, clear=True):
        if "LLM_API_KEY" in os.environ:
            del os.environ["LLM_API_KEY"]

        adapter = LLMAdapter(api_key=None)
        assert not adapter.is_configured

        with pytest.raises(ProviderBlockedError) as exc_info:
            adapter_strict = LLMAdapter(api_key=None, require_key=True)
        assert "LLM API key not provided; LLM provider is blocked." in str(exc_info.value)


@pytest.mark.asyncio
async def test_llm_chat_missing_credentials_raises_on_call() -> None:
    """Calling chat() without API key raises ProviderBlockedError."""
    with patch.dict(os.environ, {}, clear=True):
        adapter = LLMAdapter(api_key=None)
        with pytest.raises(ProviderBlockedError):
            await adapter.chat(messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_llm_chat_structured_json_and_usage() -> None:
    """Tests chat completion with structured JSON output and token telemetry (§14.2)."""
    mock_payload = {
        "id": "chatcmpl-123",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"selection": "SKU-100", "status": "PASS", "confidence": 0.9}',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 250,
            "completion_tokens": 40,
            "total_tokens": 290,
            "prompt_tokens_details": {
                "cached_tokens": 50,
            },
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer test-llm-key"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "gpt-4o-mini"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json=mock_payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = LLMAdapter(
            api_key="test-llm-key",
            model_id="gpt-4o-mini",
            http_client=client,
        )

        resp = await adapter.chat(
            messages=[{"role": "user", "content": "Select PLC"}],
            response_format={"type": "json_object"},
        )

        assert resp.content is not None
        assert resp.parsed == {"selection": "SKU-100", "status": "PASS", "confidence": 0.9}
        assert resp.usage.prompt_tokens == 250
        assert resp.usage.completion_tokens == 40
        assert resp.usage.cached_tokens == 50
        assert resp.usage.total_tokens == 290
        assert resp.telemetry["attempts"] == 1
        assert resp.telemetry["http_statuses"] == [200]


@pytest.mark.asyncio
async def test_llm_chat_function_tool_calling() -> None:
    """Tests tool/function calling support (§11.1)."""
    mock_payload = {
        "id": "chatcmpl-456",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "open_document",
                                "arguments": '{"document_id": "doc_manual_s7"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 15},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert "tools" in body
        assert body["tools"][0]["function"]["name"] == "open_document"
        return httpx.Response(200, json=mock_payload)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        adapter = LLMAdapter(
            api_key="test-llm-key",
            http_client=client,
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "open_document",
                    "description": "Opens manual PDF",
                    "parameters": {"type": "object", "properties": {"document_id": {"type": "string"}}},
                },
            }
        ]

        resp = await adapter.chat(
            messages=[{"role": "user", "content": "Find manual"}],
            tools=tools,
        )

        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "call_abc123"
        assert tc.name == "open_document"
        assert tc.parsed_arguments == {"document_id": "doc_manual_s7"}


# ============================================================================
# 3. Embedding Adapter & Sliding Window Tests (§10.1)
# ============================================================================


def test_embedding_constants_and_limits() -> None:
    """Verifies candidate model and sequence limits specified in §10.1."""
    assert EMBEDDING_DEFAULT_MODEL_ID == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert MODEL_MAX_SEQ_LENGTH == 128
    assert MAX_WINDOW_TOKENS == 96
    assert WINDOW_OVERLAP_TOKENS == 16


def test_sliding_window_chunking_exact_stride() -> None:
    """Verifies sliding window chunking: max 96 tokens, 16 overlap, stride 80."""
    tokenizer = FallbackTokenizer()
    # Create parent text with ~200 tokens
    words = [f"word{i}" for i in range(200)]
    parent_text = " ".join(words)
    parent_id = "chunk_p0"

    windows = chunk_text_with_sliding_window(
        parent_text=parent_text,
        parent_id=parent_id,
        max_window_tokens=96,
        overlap_tokens=16,
        tokenizer=tokenizer,
    )

    # 200 tokens with stride 80:
    # w0: 0..96
    # w1: 80..176
    # w2: 160..200
    assert len(windows) == 3
    assert windows[0].token_start == 0
    assert windows[0].token_end == 96
    assert windows[0].parent_id == parent_id
    assert windows[0].parent_text == parent_text

    assert windows[1].token_start == 80
    assert windows[1].token_end == 176
    assert windows[1].parent_id == parent_id

    assert windows[2].token_start == 160
    assert windows[2].token_end == 200
    assert windows[2].parent_id == parent_id

    # Verify every window has <= 96 tokens
    for w in windows:
        assert w.token_count <= 96


def test_chunk_document_into_parents_and_windows() -> None:
    """Verifies full hierarchy: 500-token parents with 80-token overlap, subdividing into 96-token windows."""
    words = [f"token{i}" for i in range(1200)]
    doc_text = " ".join(words)

    parents, windows = chunk_document_into_parents_and_windows(
        text=doc_text,
        document_id="doc_plc_spec",
        product_id="PROD-01",
        variant_id="VAR-A",
        parent_chunk_tokens=500,
        parent_overlap_tokens=80,
        max_window_tokens=96,
        window_overlap_tokens=16,
    )

    assert len(parents) > 1
    assert len(windows) > len(parents)

    # Every window points to a valid parent_id
    parent_ids = {p.chunk_id for p in parents}
    for w in windows:
        assert w.parent_id in parent_ids
        assert w.metadata.get("product_id") == "PROD-01"
        assert w.token_count <= 96


def test_aggregate_window_scores_to_parents() -> None:
    """Tests max score aggregation to deduplicate parents (§10.1)."""
    tokenizer = FallbackTokenizer()
    w1 = chunk_text_with_sliding_window("text 1", parent_id="p1", tokenizer=tokenizer)[0]
    w2 = chunk_text_with_sliding_window("text 2", parent_id="p1", tokenizer=tokenizer)[0]
    w3 = chunk_text_with_sliding_window("text 3", parent_id="p2", tokenizer=tokenizer)[0]

    scores = [(w1, 0.45), (w2, 0.88), (w3, 0.65)]
    aggregated = aggregate_window_scores_to_parents(scores)

    # p1 should take max(0.45, 0.88) = 0.88
    # p2 should take 0.65
    assert aggregated[0] == ("p1", 0.88)
    assert aggregated[1] == ("p2", 0.65)


def test_deterministic_fallback_embeddings() -> None:
    """Verifies fallback embedder: dim 384, normalized unit vectors, deterministic, semantic alignment."""
    adapter = EmbeddingAdapter(force_fallback=True)
    assert adapter.is_fallback
    assert adapter.dimension == 384

    t1 = "Siemens S7-1200 CPU 1214C DC/DC/DC 24V supply"
    t2 = "Siemens S7-1200 compact CPU 24VDC power"
    t3 = "Completely unrelated hydraulic piston pump high pressure"

    e1 = adapter.embed_query(t1)
    e2 = adapter.embed_query(t2)
    e3 = adapter.embed_query(t3)

    # Dimension checks
    assert e1.shape == (384,)
    assert e2.shape == (384,)
    assert e3.shape == (384,)

    # Unit norm checks
    assert math.isclose(float(np.linalg.norm(e1)), 1.0, rel_tol=1e-5)
    assert math.isclose(float(np.linalg.norm(e2)), 1.0, rel_tol=1e-5)
    assert math.isclose(float(np.linalg.norm(e3)), 1.0, rel_tol=1e-5)

    # Determinism check
    e1_repeat = adapter.embed_query(t1)
    assert np.allclose(e1, e1_repeat)

    # Semantic similarity: S7-1200 CPU 24V should be much closer to e2 than to hydraulic pump e3
    sim_12 = adapter.cosine_similarity(e1, e2)
    sim_13 = adapter.cosine_similarity(e1, e3)
    assert sim_12 > sim_13, f"Expected {sim_12} > {sim_13}"

    # Batch embedding check
    matrix = adapter.embed_texts([t1, t2, t3])
    assert matrix.shape == (3, 384)
    sims = adapter.cosine_similarities(e1, matrix)
    assert math.isclose(float(sims[0]), 1.0, rel_tol=1e-5)
    assert sims[1] > sims[2]
