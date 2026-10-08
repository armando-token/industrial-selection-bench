"""Hybrid Retrieval module adhering to MEGAPLAN.md §10.1 and REPAIR_PLAN.md §7.1.

Honest Retrieval System Labeling (§7.1):
- Lexical retrieval: Okapi BM25 over document parent chunks.
- Dense retrieval: Local DeterministicFallbackEmbedder (dimension 384, MD5 feature hashing
  with character n-grams and TF-IDF log-scaling, L2 normalized).
- Heavy neural libraries (sentence_transformers, torch, fastembed) are NOT installed in .venv.
- This retrieval pipeline is honestly labeled as 'bm25_plus_deterministic_feature_hash'
  and is NEVER claimed to be a neural vector RAG model.
- Fuses lexical and feature-hash dense rankings using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


ENGLISH_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "which", "who",
    "whom", "this", "that", "these", "those", "it", "its", "as", "if",
}


def clean_text_boundaries(text: str) -> str:
    """Expands text boundaries between letters, digits, and camelCase for robust tokenization.

    Handles PDF table text where whitespace between cells or columns was lost
    (e.g., '12including4configurable' -> '12 including 4 configurable',
    'InputsperModule' -> 'Inputs per Module').
    """
    if not text:
        return ""
    # Separate digit-letter and letter-digit transitions
    t = re.sub(r"(\d+)([a-zA-Z]+)", r"\1 \2", text)
    t = re.sub(r"([a-zA-Z]+)(\d+)", r"\1 \2", t)
    # Separate camelCase transitions (e.g. InputsperModule -> Inputs per Module)
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    return t


def tokenize_tokens(text: str) -> list[re.Match]:
    """Tokenize text into subword/punctuation-aware tokens with regex matches (for character spans)."""
    return list(re.finditer(r"\w+|[^\w\s]", text, re.UNICODE))


def count_tokens(text: str) -> int:
    """Approximate token count of text using subword/punctuation regex matching."""
    return len(tokenize_tokens(text))


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 lexical search.

    Extracts lowercase alphanumeric words and expands hyphenated/dotted identifiers
    (e.g., 's7-1200' -> ['s7-1200', 's7', '1200', 's71200']) to ensure robust matching
    for industrial part numbers, voltages, and serial protocols.
    Also separates letter/digit transitions to preserve table cell contents (e.g. '12 DI').
    """
    raw_tokens = re.findall(r"[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*", text.lower())
    tokens = []
    for tok in raw_tokens:
        tokens.append(tok)
        if "-" in tok or "_" in tok or "." in tok:
            parts = re.split(r"[-_.]+", tok)
            tokens.extend(p for p in parts if p)
            tokens.append("".join(parts))
        if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
            if re.fullmatch(r"\d+[a-zA-Z]{1,2}", tok):
                continue
            subparts = re.findall(r"[a-zA-Z]{2,}|\d+", tok)
            if len(subparts) > 1:
                tokens.extend(subparts)
    camel_words = re.findall(r"[A-Z][a-z]{2,}", text)
    for cw in camel_words:
        cw_lower = cw.lower()
        if cw_lower not in tokens:
            tokens.append(cw_lower)
    return tokens


class BM25Index:
    """Inverted index implementing Okapi BM25 with non-negative IDF."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self.num_docs: int = 0
        self.inverted_index: dict[str, dict[int, int]] = {}
        self.idfs: dict[str, float] = {}

    def fit(self, corpus: list[str]) -> None:
        """Build BM25 inverted index and document statistics over a corpus of texts."""
        self.num_docs = len(corpus)
        self.doc_lengths = []
        self.inverted_index = collections.defaultdict(dict)

        for doc_idx, text in enumerate(corpus):
            terms = bm25_tokenize(text)
            self.doc_lengths.append(len(terms))
            tf = collections.Counter(terms)
            for term, count in tf.items():
                self.inverted_index[term][doc_idx] = count

        if self.num_docs > 0:
            self.avgdl = sum(self.doc_lengths) / self.num_docs
        else:
            self.avgdl = 0.0

        self.idfs = {}
        for term, postings in self.inverted_index.items():
            n_t = len(postings)
            # Okapi BM25 with + 1.0 inside log ensures IDF is always positive
            self.idfs[term] = math.log((self.num_docs - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def score(self, query: str) -> np.ndarray:
        """Compute BM25 scores for all documents in the corpus for the given query."""
        if self.num_docs == 0:
            return np.zeros(0, dtype=np.float64)

        raw_query_terms = bm25_tokenize(query)
        # Filter English stop words from query to prevent short docs from being inflated by 'at', 'is', 'the'
        query_terms = [t for t in raw_query_terms if t not in ENGLISH_STOP_WORDS]
        if not query_terms:
            query_terms = raw_query_terms

        scores = np.zeros(self.num_docs, dtype=np.float64)

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            idf = self.idfs[term]
            postings = self.inverted_index[term]
            for doc_idx, count in postings.items():
                dl = self.doc_lengths[doc_idx]
                denominator = count + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                if denominator > 0:
                    scores[doc_idx] += idf * (count * (self.k1 + 1.0) / denominator)

        return scores

    def to_dict(self) -> dict:
        """Serialize BM25 index state to JSON-serializable dictionary."""
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "num_docs": self.num_docs,
            "idfs": self.idfs,
            "inverted_index": {
                term: {str(doc_idx): count for doc_idx, count in postings.items()}
                for term, postings in self.inverted_index.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> BM25Index:
        """Deserialize BM25 index state from dictionary."""
        index = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        index.doc_lengths = data.get("doc_lengths", [])
        index.avgdl = float(data.get("avgdl", 0.0))
        index.num_docs = int(data.get("num_docs", 0))
        index.idfs = {k: float(v) for k, v in data.get("idfs", {}).items()}
        raw_inv = data.get("inverted_index", {})
        index.inverted_index = {
            term: {int(doc_idx): int(count) for doc_idx, count in postings.items()}
            for term, postings in raw_inv.items()
        }
        return index


def deterministic_embed(texts: list[str], dim: int = 384) -> np.ndarray:
    """Deterministic hashing-based dense embedder (dim=384, L2 normalized).

    Used as zero-dependency embedder when neural model dependencies (e.g. torch,
    sentence-transformers) are not loaded or in offline/test mode. Uses token and
    subword n-gram MD5 hashing with signed random projection to produce 384-dimensional
    unit-norm embeddings.
    """
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        words = bm25_tokenize(text)
        if not words:
            continue
        for word in words:
            if word in ENGLISH_STOP_WORDS:
                continue
            features = [word]
            if len(word) >= 3:
                for n in range(3, min(6, len(word) + 1)):
                    for s in range(len(word) - n + 1):
                        features.append(word[s : s + n])
            for feat in features:
                h = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if ((h >> 16) & 1) else -1.0
                vectors[i, idx] += sign

        norm = float(np.linalg.norm(vectors[i]))
        if norm > 0:
            vectors[i] /= norm
        else:
            vectors[i, 0] = 1.0
    return vectors


def create_embedding_windows(
    text: str,
    window_tokens: int = 96,
    overlap_tokens: int = 16,
) -> list[str]:
    """Split parent chunk text into embedding windows of window_tokens with overlap_tokens."""
    matches = tokenize_tokens(text)
    if not matches:
        return [text] if text else []
    if len(matches) <= window_tokens:
        return [text]

    step = max(1, window_tokens - overlap_tokens)
    windows: list[str] = []
    start = 0
    total = len(matches)
    while start < total:
        end = min(start + window_tokens, total)
        start_char = matches[start].start()
        end_char = matches[end - 1].end()
        window_slice = text[start_char:end_char]
        windows.append(window_slice)
        if end >= total:
            break
        start += step
    return windows


def chunk_document(
    doc: dict,
    target_tokens: int = 500,
    overlap_tokens: int = 80,
) -> list[dict]:
    """Break a document into ~500-token parent chunks preserving headers and footnotes.

    Preserves document_id, page_index, span_ids, and metadata across all emitted chunks.
    If headers or footnotes are present in the doc (or detected in markdown tables),
    they are attached to each generated parent chunk.
    """
    text = doc.get("text") or doc.get("content") or doc.get("body") or ""
    doc_id = str(doc.get("document_id") or doc.get("doc_id") or doc.get("id", "doc"))

    page_index = doc.get("page_index")
    if page_index is None:
        page_index = doc.get("pdf_page_index", doc.get("page", 0))
    page_index = int(page_index) if page_index is not None else 0

    span_ids = doc.get("span_ids")
    if span_ids is None and "span_id" in doc:
        span_ids = [doc["span_id"]]
    elif isinstance(span_ids, str):
        span_ids = [span_ids]
    elif span_ids is None:
        span_ids = [f"{doc_id}:p{page_index}:s01"]

    headers = str(doc.get("headers") or doc.get("header") or doc.get("table_header") or "")
    footnotes = str(doc.get("footnotes") or doc.get("footnote") or "")

    metadata = {
        k: v
        for k, v in doc.items()
        if k not in {"text", "content", "body", "document_id", "doc_id", "page_index", "pdf_page_index", "page", "span_ids", "span_id"}
    }

    matches = tokenize_tokens(text)
    total_tokens = len(matches)

    # If the text fits in target_tokens, keep as single parent chunk
    if total_tokens <= target_tokens:
        full_text = text
        if headers and not full_text.startswith(headers):
            full_text = f"{headers}\n{full_text}"
        if footnotes and not full_text.endswith(footnotes):
            full_text = f"{full_text}\n{footnotes}"

        return [
            {
                "chunk_id": f"{doc_id}:p{page_index}:c0",
                "document_id": doc_id,
                "page_index": page_index,
                "span_ids": list(span_ids),
                "text": full_text,
                "token_count": count_tokens(full_text),
                "metadata": metadata,
            }
        ]

    # Split into target_tokens chunks with overlap_tokens
    step = max(1, target_tokens - overlap_tokens)
    parent_chunks: list[dict] = []
    start = 0
    chunk_idx = 0

    while start < total_tokens:
        end = min(start + target_tokens, total_tokens)
        start_char = matches[start].start()
        end_char = matches[end - 1].end()
        chunk_text = text[start_char:end_char]

        if headers and not chunk_text.startswith(headers):
            chunk_text = f"{headers}\n{chunk_text}"
        if footnotes and not chunk_text.endswith(footnotes):
            chunk_text = f"{chunk_text}\n{footnotes}"

        parent_chunks.append({
            "chunk_id": f"{doc_id}:p{page_index}:c{chunk_idx}",
            "document_id": doc_id,
            "page_index": page_index,
            "span_ids": list(span_ids),
            "text": chunk_text,
            "token_count": count_tokens(chunk_text),
            "metadata": metadata,
        })
        chunk_idx += 1
        if end >= total_tokens:
            break
        start += step

    return parent_chunks


class HybridRetriever:
    """Hybrid Retriever combining Okapi BM25 and dense vector search via Reciprocal Rank Fusion.

    Adheres to MEGAPLAN.md §10.1:
    - Lexical retrieval: BM25 score calculation over document chunks.
    - Dense/Vector retrieval: Cosine similarity over normalized embedding vectors (using NumPy
      dot product, no external vector DB needed for 3 products!).
    - Reciprocal Rank Fusion (RRF):
        score = sum(1.0 / (60.0 + rank)) where rank starts at 1 for both lexical and vector ranks.
    - Chunk hierarchy:
      - 500-token parent chunks (sections, tables with headers and footnotes preserved).
      - 96-token embedding windows with 16 tokens overlap.
      - When an embedding window matches, expand to its 500-token parent chunk. Deduplicate parent
        chunks taking the maximum score.
    - Top-K retrieval:
      - Retrieve lexical top_k (default 12), vector top_k (default 12), fuse with RRF, return
        context top_k (default 8).
    """

    def __init__(
        self,
        lexical_top_k: int = 24,
        vector_top_k: int = 24,
        context_top_k: int = 8,
        parent_chunk_target_tokens: int = 500,
        embedding_window_tokens: int = 96,
        embedding_window_overlap: int = 16,
        parent_chunk_overlap: int = 80,
        rrf_constant: float = 60.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None,
    ):
        self.lexical_top_k = int(lexical_top_k)
        self.vector_top_k = int(vector_top_k)
        self.context_top_k = int(context_top_k)
        self.parent_chunk_target_tokens = int(parent_chunk_target_tokens)
        self.embedding_window_tokens = int(embedding_window_tokens)
        self.embedding_window_overlap = int(embedding_window_overlap)
        self.parent_chunk_overlap = int(parent_chunk_overlap)
        self.rrf_constant = float(rrf_constant)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)

        self._custom_embed_fn = embed_fn

        # Internal storage
        self.parent_chunks: list[dict] = []
        self.embedding_windows: list[dict] = []
        self.normalized_embeddings: Optional[np.ndarray] = None
        self.bm25_index: BM25Index = BM25Index(k1=self.bm25_k1, b=self.bm25_b)

    @property
    def embed_fn(self) -> Callable[[list[str]], np.ndarray]:
        """Return active embedding function, falling back to deterministic embedder if unset."""
        if self._custom_embed_fn is not None:
            return self._custom_embed_fn
        try:
            from industrial_lab.adapters.embeddings import EmbeddingAdapter
            if not hasattr(self, "_default_adapter") or self._default_adapter is None:
                self._default_adapter = EmbeddingAdapter()
            return self._default_adapter.embed_texts
        except Exception:
            return deterministic_embed

    @embed_fn.setter
    def embed_fn(self, fn: Callable[[list[str]], np.ndarray]) -> None:
        self._custom_embed_fn = fn

    @property
    def retrieval_label(self) -> str:
        """Returns honest label of the retrieval method (§7.1)."""
        adapter = getattr(self, "_default_adapter", None)
        if adapter is not None and not getattr(adapter, "is_fallback", True):
            return "hybrid_bm25_dense_neural"
        return "bm25_plus_deterministic_feature_hash"

    @classmethod
    def from_config(cls, config: dict, embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None) -> HybridRetriever:
        """Create a HybridRetriever configured from a dictionary (e.g. experiment.yaml 'rag' section)."""
        rag_cfg = config.get("rag", config)
        return cls(
            lexical_top_k=rag_cfg.get("lexical_top_k", 24),
            vector_top_k=rag_cfg.get("vector_top_k", 24),
            context_top_k=rag_cfg.get("context_top_k", 8),
            parent_chunk_target_tokens=rag_cfg.get("parent_chunk_target_tokens", 500),
            embedding_window_tokens=rag_cfg.get("embedding_window_tokens", 96),
            embedding_window_overlap=rag_cfg.get("embedding_window_overlap", 16),
            parent_chunk_overlap=rag_cfg.get("parent_chunk_overlap", 80),
            rrf_constant=rag_cfg.get("rrf_constant", 60.0),
            bm25_k1=rag_cfg.get("bm25_k1", 1.5),
            bm25_b=rag_cfg.get("bm25_b", 0.75),
            embed_fn=embed_fn,
        )

    def index_documents(self, documents: list[dict]) -> None:
        """Build inverted index for BM25 and store normalized embeddings matrix.

        Processes raw documents or sections into 500-token parent chunks and generates
        96-token embedding windows with 16 tokens overlap. Embeds all windows and normalizes
        the resulting embedding matrix for cosine similarity search.
        """
        all_parent_chunks: list[dict] = []
        for doc in documents:
            chunks = chunk_document(
                doc=doc,
                target_tokens=self.parent_chunk_target_tokens,
                overlap_tokens=self.parent_chunk_overlap,
            )
            all_parent_chunks.extend(chunks)

        self.parent_chunks = all_parent_chunks

        # Build BM25 index over parent chunks
        parent_texts = [p["text"] for p in self.parent_chunks]
        self.bm25_index = BM25Index(k1=self.bm25_k1, b=self.bm25_b)
        self.bm25_index.fit(parent_texts)

        # Generate embedding windows (96 tokens, 16 overlap) referencing parent chunk indices
        self.embedding_windows = []
        window_texts: list[str] = []
        for p_idx, p_chunk in enumerate(self.parent_chunks):
            windows = create_embedding_windows(
                text=p_chunk["text"],
                window_tokens=self.embedding_window_tokens,
                overlap_tokens=self.embedding_window_overlap,
            )
            for w_idx, w_text in enumerate(windows):
                self.embedding_windows.append({
                    "window_id": f"{p_chunk['chunk_id']}_w{w_idx}",
                    "parent_chunk_idx": p_idx,
                    "parent_chunk_id": p_chunk["chunk_id"],
                    "text": w_text,
                    "token_count": count_tokens(w_text),
                })
                window_texts.append(w_text)

        # Embed all windows and compute normalized embedding matrix
        if window_texts:
            raw_embeddings = self.embed_fn(window_texts)
            if not isinstance(raw_embeddings, np.ndarray):
                raw_embeddings = np.array(raw_embeddings, dtype=np.float32)
            else:
                raw_embeddings = raw_embeddings.astype(np.float32)

            norms = np.linalg.norm(raw_embeddings, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self.normalized_embeddings = raw_embeddings / norms
        else:
            self.normalized_embeddings = np.zeros((0, 384), dtype=np.float32)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Compute normalized dense embedding vector for search query."""
        raw_emb = self.embed_fn([query])
        if not isinstance(raw_emb, np.ndarray):
            raw_emb = np.array(raw_emb, dtype=np.float32)
        else:
            raw_emb = raw_emb.astype(np.float32)

        q_vec = raw_emb[0]
        norm = float(np.linalg.norm(q_vec))
        if norm > 0:
            return q_vec / norm
        return q_vec

    def lexical_search(self, query: str, top_k: Optional[int] = None) -> list[tuple[int, float]]:
        """Perform BM25 lexical search over parent chunks, returning list of (parent_idx, score)."""
        if not self.parent_chunks or not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.lexical_top_k
        scores = self.bm25_index.score(query)

        # Only consider chunks with strictly positive relevance
        positive_indices = [idx for idx, s in enumerate(scores) if s > 0.0]
        positive_indices.sort(key=lambda idx: scores[idx], reverse=True)

        return [(idx, float(scores[idx])) for idx in positive_indices[:k]]

    def vector_search(self, query: str, top_k: Optional[int] = None) -> list[tuple[int, float]]:
        """Perform dense vector search over embedding windows, expanding to parent chunks with max score."""
        if not self.parent_chunks or self.normalized_embeddings is None or len(self.normalized_embeddings) == 0 or not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.vector_top_k
        q_emb = self._get_query_embedding(query)

        # Cosine similarity via NumPy dot product over normalized vectors
        sims = self.normalized_embeddings @ q_emb
        sorted_window_indices = np.argsort(-sims)

        # Deduplicate to parent chunks keeping the maximum score
        seen_parents: dict[int, float] = {}
        results: list[tuple[int, float]] = []

        for w_idx in sorted_window_indices:
            w = self.embedding_windows[w_idx]
            p_idx = w["parent_chunk_idx"]
            sim = float(sims[w_idx])
            if p_idx not in seen_parents:
                seen_parents[p_idx] = sim
                results.append((p_idx, sim))
                if len(results) >= k:
                    break

        return results

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        """Perform hybrid retrieval over indexed documents.

        1. Retrieves lexical top_k (BM25) and vector top_k (cosine similarity).
        2. Fuses rankings using Reciprocal Rank Fusion:
           score = sum(1.0 / (60.0 + rank))
           where rank starts at 1 for both lexical and vector ranks.
        3. Returns top_k ranked chunk dictionaries with span_ids, document_id, page_index, and text.
        """
        if not self.parent_chunks or not query or not query.strip():
            return []

        # 1. Retrieve lexical candidates
        lexical_candidates = self.lexical_search(query, top_k=self.lexical_top_k)

        # 2. Retrieve vector candidates
        vector_candidates = self.vector_search(query, top_k=self.vector_top_k)

        # 3. Reciprocal Rank Fusion
        rrf_scores: dict[int, float] = collections.defaultdict(float)
        lexical_ranks: dict[int, int] = {}
        vector_ranks: dict[int, int] = {}
        bm25_scores: dict[int, float] = {}
        vector_scores: dict[int, float] = {}

        for rank, (p_idx, score) in enumerate(lexical_candidates, start=1):
            lexical_ranks[p_idx] = rank
            bm25_scores[p_idx] = score
            rrf_scores[p_idx] += 1.0 / (self.rrf_constant + rank)

        for rank, (p_idx, score) in enumerate(vector_candidates, start=1):
            vector_ranks[p_idx] = rank
            vector_scores[p_idx] = score
            rrf_scores[p_idx] += 1.0 / (self.rrf_constant + rank)

        if not rrf_scores:
            return []

        # Sort candidate parent chunks by RRF score descending, breaking ties with best individual score
        all_candidates = list(rrf_scores.keys())
        all_candidates.sort(
            key=lambda idx: (
                rrf_scores[idx],
                max(bm25_scores.get(idx, 0.0), vector_scores.get(idx, 0.0)),
                -idx,
            ),
            reverse=True,
        )

        selected_indices = all_candidates[:top_k]

        # Format output dictionaries adhering to system contract
        results = []
        for p_idx in selected_indices:
            parent = self.parent_chunks[p_idx]
            rrf_val = float(rrf_scores[p_idx])
            chunk_dict = {
                "chunk_id": parent["chunk_id"],
                "document_id": parent["document_id"],
                "page_index": parent["page_index"],
                "span_ids": list(parent["span_ids"]),
                "text": parent["text"],
                "score": rrf_val,
                "rrf_score": rrf_val,
                "lexical_rank": lexical_ranks.get(p_idx),
                "vector_rank": vector_ranks.get(p_idx),
                "bm25_score": bm25_scores.get(p_idx, 0.0),
                "vector_score": vector_scores.get(p_idx, 0.0),
                "metadata": parent.get("metadata", {}),
            }
            results.append(chunk_dict)

        return results

    def _serialize_metadata(self) -> dict:
        """Serialize configuration, parent chunks, windows, and BM25 index to dict."""
        return {
            "retrieval_label": self.retrieval_label,
            "config": {
                "lexical_top_k": self.lexical_top_k,
                "vector_top_k": self.vector_top_k,
                "context_top_k": self.context_top_k,
                "parent_chunk_target_tokens": self.parent_chunk_target_tokens,
                "embedding_window_tokens": self.embedding_window_tokens,
                "embedding_window_overlap": self.embedding_window_overlap,
                "parent_chunk_overlap": self.parent_chunk_overlap,
                "rrf_constant": self.rrf_constant,
                "bm25_k1": self.bm25_k1,
                "bm25_b": self.bm25_b,
            },
            "parent_chunks": self.parent_chunks,
            "embedding_windows": self.embedding_windows,
            "bm25_index": self.bm25_index.to_dict(),
        }

    def _deserialize_metadata(self, metadata: dict) -> None:
        """Restore configuration, parent chunks, windows, and BM25 index from dict."""
        cfg = metadata.get("config", {})
        self.lexical_top_k = int(cfg.get("lexical_top_k", self.lexical_top_k))
        self.vector_top_k = int(cfg.get("vector_top_k", self.vector_top_k))
        self.context_top_k = int(cfg.get("context_top_k", self.context_top_k))
        self.parent_chunk_target_tokens = int(cfg.get("parent_chunk_target_tokens", self.parent_chunk_target_tokens))
        self.embedding_window_tokens = int(cfg.get("embedding_window_tokens", self.embedding_window_tokens))
        self.embedding_window_overlap = int(cfg.get("embedding_window_overlap", self.embedding_window_overlap))
        self.parent_chunk_overlap = int(cfg.get("parent_chunk_overlap", self.parent_chunk_overlap))
        self.rrf_constant = float(cfg.get("rrf_constant", self.rrf_constant))
        self.bm25_k1 = float(cfg.get("bm25_k1", self.bm25_k1))
        self.bm25_b = float(cfg.get("bm25_b", self.bm25_b))

        self.parent_chunks = metadata.get("parent_chunks", [])
        self.embedding_windows = metadata.get("embedding_windows", [])

        if "bm25_index" in metadata:
            self.bm25_index = BM25Index.from_dict(metadata["bm25_index"])
        else:
            self.bm25_index = BM25Index(k1=self.bm25_k1, b=self.bm25_b)
            self.bm25_index.fit([p["text"] for p in self.parent_chunks])

    def save_index(self, path: Union[str, Path]) -> None:
        """Save index state, chunks, BM25 index, and normalized embedding matrix to disk.

        Supports saving to:
        - Directory path: saves 'index_metadata.json' and 'embeddings.npy'
        - .npz archive: compressed bundle with embeddings and metadata
        - .json file: JSON metadata and adjacent .npy embeddings file
        """
        p = Path(path)
        metadata = self._serialize_metadata()

        emb_matrix = (
            self.normalized_embeddings
            if self.normalized_embeddings is not None
            else np.zeros((0, 384), dtype=np.float32)
        )

        if p.suffix == ".npz":
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, embeddings=emb_matrix, metadata=json.dumps(metadata))
        elif p.suffix == ".json":
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            npy_path = p.with_suffix(".npy")
            np.save(npy_path, emb_matrix)
        else:
            p.mkdir(parents=True, exist_ok=True)
            meta_path = p / "index_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            npy_path = p / "embeddings.npy"
            np.save(npy_path, emb_matrix)

    def load_index(self, path: Union[str, Path]) -> None:
        """Load index state, chunks, BM25 index, and normalized embedding matrix from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Index path does not exist: {p}")

        if p.is_file() and p.suffix == ".npz":
            data = np.load(p, allow_pickle=False)
            self.normalized_embeddings = data["embeddings"].astype(np.float32)
            metadata = json.loads(str(data["metadata"]))
        elif p.is_file() and p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            npy_path = p.with_suffix(".npy")
            if npy_path.exists():
                self.normalized_embeddings = np.load(npy_path).astype(np.float32)
            else:
                self.normalized_embeddings = np.zeros((len(metadata.get("embedding_windows", [])), 384), dtype=np.float32)
        elif p.is_dir():
            meta_file = p / "index_metadata.json"
            if not meta_file.exists():
                meta_file = p / "hybrid_index.json"
            if not meta_file.exists():
                json_files = list(p.glob("*.json"))
                if json_files:
                    meta_file = json_files[0]
                else:
                    raise FileNotFoundError(f"No metadata JSON found in directory {p}")

            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            emb_file = p / "embeddings.npy"
            if emb_file.exists():
                self.normalized_embeddings = np.load(emb_file).astype(np.float32)
            else:
                self.normalized_embeddings = np.zeros((len(metadata.get("embedding_windows", [])), 384), dtype=np.float32)
        else:
            raise ValueError(f"Unrecognized index path: {p}")

        self._deserialize_metadata(metadata)

    @classmethod
    def from_saved(cls, path: Union[str, Path], embed_fn: Optional[Callable[[list[str]], np.ndarray]] = None) -> HybridRetriever:
        """Load a saved HybridRetriever index from disk."""
        retriever = cls(embed_fn=embed_fn)
        retriever.load_index(path)
        return retriever
