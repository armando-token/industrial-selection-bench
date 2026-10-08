"""Model and API adapters for Industrial Selection Lab.

Adheres to MEGAPLAN.md §0, §9.5, §10.1, §11, §14.2, §20.3, §24.2.
"""

from .embeddings import (
    DEFAULT_MODEL_ID as EMBEDDING_DEFAULT_MODEL_ID,
    MAX_WINDOW_TOKENS,
    MODEL_MAX_SEQ_LENGTH,
    WINDOW_OVERLAP_TOKENS,
    DeterministicFallbackEmbedder,
    EmbeddingAdapter,
    FallbackTokenizer,
    ParentChunk,
    SlidingWindow,
    aggregate_window_scores_to_parents,
    chunk_document_into_parents_and_windows,
    chunk_text_with_sliding_window,
)
from .exceptions import (
    AdapterError,
    APIRequestError,
    ModelValidationError,
    ProviderBlockedError,
    ProviderContractError,
    ResponseValidationError,
)
from .jev import (
    JevAdapter,
    canonical_state_json,
    validate_jev_response,
)
from .llm import (
    LLMAdapter,
    LLMResponse,
    ToolCall,
    UsageTelemetry,
)

__all__ = [
    # Exceptions
    "AdapterError",
    "ProviderBlockedError",
    "ModelValidationError",
    "ResponseValidationError",
    "APIRequestError",
    "ProviderContractError",
    # JEV
    "JevAdapter",
    "validate_jev_response",
    "canonical_state_json",
    # LLM
    "LLMAdapter",
    "LLMResponse",
    "ToolCall",
    "UsageTelemetry",
    # Embeddings
    "EmbeddingAdapter",
    "ParentChunk",
    "SlidingWindow",
    "FallbackTokenizer",
    "DeterministicFallbackEmbedder",
    "chunk_text_with_sliding_window",
    "chunk_document_into_parents_and_windows",
    "aggregate_window_scores_to_parents",
    "EMBEDDING_DEFAULT_MODEL_ID",
    "MODEL_MAX_SEQ_LENGTH",
    "MAX_WINDOW_TOKENS",
    "WINDOW_OVERLAP_TOKENS",
]
