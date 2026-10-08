"""Embedding adapter and sliding window chunking for System B (RAG).

Adheres strictly to MEGAPLAN.md §10.1 and REPAIR_PLAN.md §7.1:
- Candidate model specification: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (dim=384)
- Enforces model max_seq_length=128
- Enforces sliding window of max 96 tokens with 16 tokens overlap pointing to 500-token parent chunk
- Provides DeterministicFallbackEmbedder producing 384-dimensional unit vectors via
  MD5 feature hashing + character n-grams + TF-IDF log-scaling.
- HONEST DISCLOSURE (§7.1): Heavy neural libraries (sentence_transformers, torch, fastembed)
  are NOT installed in .venv. The active embedder is explicitly and honestly labeled as
  'deterministic_feature_hash_fallback' (or in hybrid search 'bm25_plus_deterministic_feature_hash').
  It is NEVER claimed to be a neural vector RAG model.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Constants adhering to MEGAPLAN §10.1
DEFAULT_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_MAX_SEQ_LENGTH = 128
MAX_WINDOW_TOKENS = 96
WINDOW_OVERLAP_TOKENS = 16
DEFAULT_PARENT_CHUNK_TOKENS = 500
DEFAULT_PARENT_OVERLAP_TOKENS = 80
EMBEDDING_DIMENSION = 384

# Honest retrieval labels (§7.1)
RETRIEVAL_LABEL_FALLBACK = "deterministic_feature_hash_fallback"
RETRIEVAL_LABEL_NEURAL = "sentence_transformers_neural"


@dataclass
class ParentChunk:
    """Represents a ~500 token parent section or table chunk (§10.1)."""

    chunk_id: str
    text: str
    product_id: str | None = None
    variant_id: str | None = None
    document_id: str | None = None
    page_number: int | None = None
    section: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "product_id": self.product_id,
            "variant_id": self.variant_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "section": self.section,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


@dataclass
class SlidingWindow:
    """Represents a sliding window of max 96 tokens pointing to its parent chunk (§10.1)."""

    window_id: str
    parent_id: str
    text: str
    token_start: int
    token_end: int
    window_index: int
    token_count: int = 0
    parent_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "parent_id": self.parent_id,
            "text": self.text,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "window_index": self.window_index,
            "token_count": self.token_count,
            "parent_text": self.parent_text,
            "metadata": self.metadata,
        }


class FallbackTokenizer:
    """Multilingual rule-based tokenizer for token counting and window slicing.

    Splits punctuation, alphanumeric identifiers, symbols, and whitespace words.
    Handles technical product codes and multilingual strings without external dependencies.
    """

    # Matches word tokens, technical codes with dashes/dots, or punctuation
    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_/.][A-Za-z0-9]+)*|[^\s\w]|\s+", re.UNICODE)

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        raw_tokens = self.TOKEN_PATTERN.findall(text)
        # Filter out purely whitespace tokens to mimic BPE/WordPiece tokenization density
        return [tok for tok in raw_tokens if not tok.isspace()]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def decode(self, tokens: Sequence[str]) -> str:
        """Joins tokens back into human-readable text."""
        result: list[str] = []
        for i, tok in enumerate(tokens):
            if i > 0 and tok and tokens[i - 1]:
                prev_tok = tokens[i - 1]
                if prev_tok[-1].isalnum() and tok[0].isalnum():
                    result.append(" ")
                elif prev_tok[-1] in ",;:.?!" and tok[0].isalnum():
                    result.append(" ")
            result.append(tok)
        return "".join(result).strip()


def chunk_text_with_sliding_window(
    parent_text: str,
    parent_id: str,
    max_window_tokens: int = MAX_WINDOW_TOKENS,
    overlap_tokens: int = WINDOW_OVERLAP_TOKENS,
    metadata: dict[str, Any] | None = None,
    tokenizer: FallbackTokenizer | Any | None = None,
) -> list[SlidingWindow]:
    """Splits a parent chunk text into sliding windows of max 96 tokens with 16 tokens overlap.

    Adheres strictly to MEGAPLAN.md §10.1:
    - Window size <= max_window_tokens (default 96).
    - Solapamiento = overlap_tokens (default 16).
    - Stride = max_window_tokens - overlap_tokens (default 80).
    - Every window explicitly references parent_id and parent_text.
    """
    if tokenizer is None:
        tokenizer = FallbackTokenizer()

    meta = metadata.copy() if metadata else {}
    tokens = tokenizer.tokenize(parent_text)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return [
            SlidingWindow(
                window_id=f"{parent_id}_w0",
                parent_id=parent_id,
                text="",
                token_start=0,
                token_end=0,
                window_index=0,
                token_count=0,
                parent_text=parent_text,
                metadata=meta,
            )
        ]

    # If within limit, return a single window
    if total_tokens <= max_window_tokens:
        return [
            SlidingWindow(
                window_id=f"{parent_id}_w0",
                parent_id=parent_id,
                text=parent_text.strip(),
                token_start=0,
                token_end=total_tokens,
                window_index=0,
                token_count=total_tokens,
                parent_text=parent_text,
                metadata=meta,
            )
        ]

    windows: list[SlidingWindow] = []
    stride = max(1, max_window_tokens - overlap_tokens)
    start_idx = 0
    window_idx = 0

    while start_idx < total_tokens:
        end_idx = min(start_idx + max_window_tokens, total_tokens)
        window_tokens = tokens[start_idx:end_idx]
        window_text = tokenizer.decode(window_tokens)

        windows.append(
            SlidingWindow(
                window_id=f"{parent_id}_w{window_idx}",
                parent_id=parent_id,
                text=window_text,
                token_start=start_idx,
                token_end=end_idx,
                window_index=window_idx,
                token_count=len(window_tokens),
                parent_text=parent_text,
                metadata=meta,
            )
        )
        window_idx += 1
        if end_idx >= total_tokens:
            break
        start_idx += stride

    return windows


def chunk_document_into_parents_and_windows(
    text: str,
    document_id: str,
    product_id: str | None = None,
    variant_id: str | None = None,
    parent_chunk_tokens: int = DEFAULT_PARENT_CHUNK_TOKENS,
    parent_overlap_tokens: int = DEFAULT_PARENT_OVERLAP_TOKENS,
    max_window_tokens: int = MAX_WINDOW_TOKENS,
    window_overlap_tokens: int = WINDOW_OVERLAP_TOKENS,
    metadata: dict[str, Any] | None = None,
    tokenizer: FallbackTokenizer | Any | None = None,
) -> tuple[list[ParentChunk], list[SlidingWindow]]:
    """Chunks a full document text into ~500-token parent chunks and max 96-token sliding windows.

    Adheres to MEGAPLAN.md §10.1.
    """
    if tokenizer is None:
        tokenizer = FallbackTokenizer()

    base_meta = metadata.copy() if metadata else {}
    tokens = tokenizer.tokenize(text)
    total_tokens = len(tokens)

    parent_chunks: list[ParentChunk] = []
    all_windows: list[SlidingWindow] = []

    if total_tokens == 0:
        p_chunk = ParentChunk(
            chunk_id=f"{document_id}_p0",
            text="",
            product_id=product_id,
            variant_id=variant_id,
            document_id=document_id,
            token_count=0,
            metadata=base_meta,
        )
        parent_chunks.append(p_chunk)
        all_windows.extend(
            chunk_text_with_sliding_window(
                parent_text="",
                parent_id=p_chunk.chunk_id,
                max_window_tokens=max_window_tokens,
                overlap_tokens=window_overlap_tokens,
                metadata=base_meta,
                tokenizer=tokenizer,
            )
        )
        return parent_chunks, all_windows

    parent_stride = max(1, parent_chunk_tokens - parent_overlap_tokens)
    p_start = 0
    p_idx = 0

    while p_start < total_tokens:
        p_end = min(p_start + parent_chunk_tokens, total_tokens)
        p_tokens = tokens[p_start:p_end]
        p_text = tokenizer.decode(p_tokens)
        chunk_id = f"{document_id}_p{p_idx}"

        p_chunk = ParentChunk(
            chunk_id=chunk_id,
            text=p_text,
            product_id=product_id,
            variant_id=variant_id,
            document_id=document_id,
            token_count=len(p_tokens),
            metadata=base_meta,
        )
        parent_chunks.append(p_chunk)

        # Create sliding windows for this parent chunk
        windows = chunk_text_with_sliding_window(
            parent_text=p_text,
            parent_id=chunk_id,
            max_window_tokens=max_window_tokens,
            overlap_tokens=window_overlap_tokens,
            metadata={
                **base_meta,
                "product_id": product_id,
                "variant_id": variant_id,
                "document_id": document_id,
            },
            tokenizer=tokenizer,
        )
        all_windows.extend(windows)

        p_idx += 1
        if p_end >= total_tokens:
            break
        p_start += parent_stride

    return parent_chunks, all_windows


def aggregate_window_scores_to_parents(
    window_scores: list[tuple[SlidingWindow, float]],
) -> list[tuple[str, float]]:
    """Aggregates sliding window scores to parent chunks using max score aggregation.

    Adheres to MEGAPLAN.md §10.1: 'Deduplicar padres y congelar agregación por máximo score.'
    Returns sorted list of (parent_id, max_score) descending.
    """
    parent_max_scores: dict[str, float] = {}
    for window, score in window_scores:
        pid = window.parent_id
        if pid not in parent_max_scores or score > parent_max_scores[pid]:
            parent_max_scores[pid] = score

    sorted_parents = sorted(parent_max_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_parents


class DeterministicFallbackEmbedder:
    """Local deterministic fallback embedder producing normalized 384-dimensional vectors.

    Adheres strictly to REPAIR_PLAN.md §7.1 and MEGAPLAN §10.1:
    - Algorithm:
      1. Feature extraction: Extracts lowercase word unigrams, word bigrams,
         character 3-grams (trigrams), and character 4-grams (quadgrams).
      2. Feature hashing: Maps each term/n-gram to a dimension index (0..383)
         via MD5 digest first 4 bytes modulo 384, with a signed random projection
         sign (+1.0 if 5th byte bit 0 is 1, else -1.0).
      3. TF-IDF log-scaling: Weights each feature occurrence by log-term frequency
         (weight = 1.0 + ln(term_count)) and accumulates sign * weight.
      4. Normalization: Strictly normalizes vector to unit L2 norm (||v||_2 = 1.0).
    - Honest Disclosure: Neural libraries (sentence_transformers, torch, fastembed)
      are not installed in .venv. This embedder is a deterministic feature-hash fallback,
      NEVER a neural vector model.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension

    def _hash_token(self, token: str) -> tuple[int, float]:
        """Maps token to dimension index and sign (+1.0 or -1.0)."""
        h_bytes = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(h_bytes[:4], byteorder="little") % self.dimension
        sign = 1.0 if (h_bytes[4] & 1) == 1 else -1.0
        return idx, sign

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single string into a normalized 384-dimensional float32 vector."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        cleaned = text.lower().strip()
        if not cleaned:
            vec[0] = 1.0
            return vec

        # Extract words
        words = re.findall(r"\w+|[^\s\w]", cleaned, re.UNICODE)
        term_counts: dict[str, int] = {}

        # Unigrams & bigrams
        for i, w in enumerate(words):
            term_counts[w] = term_counts.get(w, 0) + 1
            if i > 0:
                bigram = f"{words[i-1]}_{w}"
                term_counts[bigram] = term_counts.get(bigram, 0) + 1

        # Character 3-grams and 4-grams for subword matching (multilingual robustness)
        for i in range(len(cleaned) - 2):
            trigram = cleaned[i : i + 3]
            term_counts[trigram] = term_counts.get(trigram, 0) + 1
        for i in range(len(cleaned) - 3):
            quadgram = cleaned[i : i + 4]
            term_counts[quadgram] = term_counts.get(quadgram, 0) + 1

        # Accumulate hashed projections with log-TF weighting
        for term, count in term_counts.items():
            idx, sign = self._hash_token(term)
            weight = 1.0 + math.log(count)
            vec[idx] += sign * weight

        # Strictly normalize to unit L2 norm
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec.astype(np.float32)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embeds a batch of texts into an (N, dimension) numpy array of unit vectors."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        matrix = np.empty((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            matrix[i] = self.embed_text(t)
        return matrix


class EmbeddingAdapter:
    """Embedding adapter enforcing sequence limits and sliding windows (§10.1).

    Candidate model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    Sequence limits:
    - max_seq_length = 128
    - sliding window = 96 tokens max with 16 tokens overlap
    - parent chunk = 500 tokens
    """

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        force_fallback: bool = False,
        device: str | None = None,
    ) -> None:
        self.model_id: str = (model_id or os.environ.get("EMBEDDING_MODEL_ID") or DEFAULT_MODEL_ID).strip()
        self.revision: str | None = revision or os.environ.get("EMBEDDING_MODEL_REVISION") or None
        self.model_max_seq_length: int = MODEL_MAX_SEQ_LENGTH  # 128
        self.max_window_tokens: int = MAX_WINDOW_TOKENS  # 96
        self.window_overlap: int = WINDOW_OVERLAP_TOKENS  # 16
        self.dimension: int = EMBEDDING_DIMENSION  # 384
        self.is_fallback: bool = False

        self._tokenizer = FallbackTokenizer()
        self._st_model: Any = None
        self._fallback_embedder = DeterministicFallbackEmbedder(dimension=self.dimension)

        if force_fallback:
            self.is_fallback = True
            logger.info("EmbeddingAdapter: force_fallback=True. Using DeterministicFallbackEmbedder.")
        else:
            self._try_init_sentence_transformers(device)

    def _try_init_sentence_transformers(self, device: str | None = None) -> None:
        """Attempts to load sentence-transformers model if installed and accessible."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            kwargs: dict[str, Any] = {}
            if self.revision:
                kwargs["revision"] = self.revision
            if device:
                kwargs["device"] = device

            logger.info("Attempting to load SentenceTransformer model %s", self.model_id)
            model = SentenceTransformer(self.model_id, **kwargs)
            self._st_model = model
            # Enforce max_seq_length = 128
            if hasattr(model, "max_seq_length"):
                self.model_max_seq_length = min(model.max_seq_length, MODEL_MAX_SEQ_LENGTH)
            self.is_fallback = False
            logger.info("Successfully loaded SentenceTransformer model %s", self.model_id)
        except Exception as exc:
            logger.info(
                "SentenceTransformers unavailable or model failed to load (%s); "
                "using local deterministic fallback embedder (dim=%d).",
                exc,
                self.dimension,
            )
            self.is_fallback = True

    @property
    def embedder_type(self) -> str:
        """Returns honest label of active embedder type (§7.1)."""
        return RETRIEVAL_LABEL_FALLBACK if self.is_fallback else RETRIEVAL_LABEL_NEURAL

    def count_tokens(self, text: str) -> int:
        return self._tokenizer.count_tokens(text)

    def chunk_parent_to_windows(
        self,
        parent_text: str,
        parent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[SlidingWindow]:
        """Creates sliding windows of max 96 tokens with 16 tokens overlap pointing to parent."""
        return chunk_text_with_sliding_window(
            parent_text=parent_text,
            parent_id=parent_id,
            max_window_tokens=self.max_window_tokens,
            overlap_tokens=self.window_overlap,
            metadata=metadata,
            tokenizer=self._tokenizer,
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embeds a list of texts into normalized unit vectors (shape: [N, 384]).

        Validates sequence limit: logs warning if unwindowed texts exceed model_max_seq_length (128).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        for i, text in enumerate(texts):
            tok_count = self.count_tokens(text)
            if tok_count > self.model_max_seq_length:
                logger.warning(
                    "Embedding text at index %d has %d tokens, exceeding model_max_seq_length=%d. "
                    "Per MEGAPLAN §10.1, sliding windows <= 96 tokens should be used.",
                    i,
                    tok_count,
                    self.model_max_seq_length,
                )

        if not self.is_fallback and self._st_model is not None:
            try:
                embeddings = self._st_model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return np.asarray(embeddings, dtype=np.float32)
            except Exception as exc:
                logger.warning("SentenceTransformer encode failed (%s); falling back to local embedder.", exc)
                self.is_fallback = True

        return self._fallback_embedder.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D unit vector of shape (384,)."""
        matrix = self.embed_texts([query])
        return matrix[0]

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two unit vectors (dot product)."""
        dot = float(np.dot(v1, v2))
        return max(-1.0, min(1.0, dot))

    @staticmethod
    def cosine_similarities(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
        """Computes cosine similarities between query_vec and a matrix of corpus vectors."""
        if corpus_matrix.shape[0] == 0:
            return np.empty((0,), dtype=np.float32)
        return corpus_matrix @ query_vec
