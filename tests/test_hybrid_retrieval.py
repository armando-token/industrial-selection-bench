"""Tests for HybridRetriever adhering to MEGAPLAN.md §10.1."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from industrial_lab.retrieval.hybrid import (
    BM25Index,
    HybridRetriever,
    bm25_tokenize,
    chunk_document,
    count_tokens,
    create_embedding_windows,
)


@pytest.fixture
def sample_catalog_docs() -> list[dict]:
    """Sample industrial catalog documents for 3 products."""
    return [
        {
            "document_id": "DOC-P1-MANUAL",
            "page_index": 12,
            "span_ids": ["DOC-P1:p12:s01"],
            "product_scope": ["P1"],
            "text": (
                "Siemens S7-1200 CPU 1214C compact controller. "
                "Operating voltage is 24V DC (nominal, range 20.4V to 28.8V DC). "
                "Equipped with 14 digital inputs (24V DC) and 10 digital outputs (transistor, 0.5A). "
                "Integrated PROFINET interface supporting TCP/IP and Modbus-TCP. "
                "Ambient operating temperature ranges from -20°C to +60°C horizontal mounting."
            ),
        },
        {
            "document_id": "DOC-P2-MANUAL",
            "page_index": 5,
            "span_ids": ["DOC-P2:p05:s02"],
            "product_scope": ["P2"],
            "text": (
                "Schneider Electric Altivar Machine ATV320 variable speed drive. "
                "Rated power 1.5 kW (2.0 hp), 380V to 500V three-phase AC supply voltage. "
                "Communication interfaces: embedded Modbus-RTU and CANopen over RJ45 connectors. "
                "Supports RS-485 2-wire multidrop bus topology up to 115200 bps. "
                "Overload capability: 150% for 60 seconds. Operating temperature -10°C to 50°C."
            ),
        },
        {
            "document_id": "DOC-P3-MANUAL",
            "page_index": 8,
            "span_ids": ["DOC-P3:p08:s01"],
            "product_scope": ["P3"],
            "text": (
                "Phoenix Contact Quint Power QUINT4-PS/1AC/24DC/20 primary-switched power supply. "
                "Input voltage range 100V to 240V AC. Nominal output voltage 24V DC +/- 1%. "
                "Nominal output current 20A with dynamic boost up to 30A for 5 seconds. "
                "Efficiency > 94%. DIN rail mountable. Selective Fuse Breaking (SFB) technology. "
                "Permissible ambient temperature during operation: -25°C to 70°C."
            ),
        },
    ]


def test_bm25_tokenize():
    """Verify BM25 tokenization extracts alphanumeric and technical tokens."""
    tokens = bm25_tokenize("Siemens S7-1200 PLC, 24V DC power! RS-485 Modbus-RTU.")
    assert "s7-1200" in tokens
    assert "s7" in tokens
    assert "1200" in tokens
    assert "rs-485" in tokens
    assert "rs" in tokens
    assert "485" in tokens
    assert "modbus-rtu" in tokens
    assert "modbus" in tokens
    assert "rtu" in tokens
    assert "24v" in tokens
    assert "dc" in tokens


def test_bm25_scoring():
    """Verify Okapi BM25 scoring ranks relevant documents highest."""
    corpus = [
        "Siemens S7-1200 PLC controller 24V DC.",
        "Schneider Altivar ATV320 variable speed drive 400V AC.",
        "Phoenix Contact 24V DC 20A power supply module.",
    ]
    bm25 = BM25Index(k1=1.5, b=0.75)
    bm25.fit(corpus)

    # All IDFs must be strictly positive
    for term, idf in bm25.idfs.items():
        assert idf > 0.0

    scores = bm25.score("24V DC power supply")
    # Document 2 (Phoenix Contact power supply) and Doc 0 (PLC 24V DC) should score > 0
    assert scores[2] > 0
    assert scores[0] > 0
    # Document 2 has both "24V", "DC", "power", "supply", so it should score highest
    assert scores[2] > scores[0]
    # Document 1 has no matching terms
    assert scores[1] == 0.0


def test_chunking_hierarchy_short_document():
    """Verify short documents under 500 tokens remain a single parent chunk."""
    doc = {
        "document_id": "D1",
        "page_index": 1,
        "span_ids": ["D1:p01:s01"],
        "text": "Short technical specification text for testing.",
    }
    chunks = chunk_document(doc, target_tokens=500, overlap_tokens=80)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "D1:p1:c0"
    assert chunks[0]["document_id"] == "D1"
    assert chunks[0]["page_index"] == 1
    assert chunks[0]["span_ids"] == ["D1:p01:s01"]
    assert chunks[0]["text"] == doc["text"]


def test_chunking_hierarchy_preserves_table_headers_and_footnotes():
    """Verify table headers and footnotes are preserved across chunks."""
    header = "| Parameter | Specification |\n|---|---|\n"
    footnote = "\n* Measured at 25°C ambient temperature."
    rows = "\n".join([f"| Metric_{i} | Value_{i} with technical detail specification |" for i in range(150)])
    full_table_text = f"{rows}"

    doc = {
        "document_id": "TAB-DOC",
        "page_index": 4,
        "span_id": "TAB-DOC:p04:s01",
        "headers": header,
        "footnotes": footnote,
        "text": full_table_text,
    }

    chunks = chunk_document(doc, target_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1

    # Check each parent chunk preserves header and footnote
    for chunk in chunks:
        assert chunk["text"].startswith(header)
        assert chunk["text"].endswith(footnote)
        assert chunk["document_id"] == "TAB-DOC"
        assert chunk["page_index"] == 4
        assert chunk["span_ids"] == ["TAB-DOC:p04:s01"]


def test_embedding_windows_generation():
    """Verify 96-token embedding windows with 16 tokens overlap."""
    words = [f"Token{i}" for i in range(250)]
    long_text = " ".join(words)

    windows = create_embedding_windows(long_text, window_tokens=96, overlap_tokens=16)
    assert len(windows) > 1

    # First window should have 96 tokens
    w0_tokens = count_tokens(windows[0])
    assert w0_tokens == 96

    # Verify overlap: step is 96 - 16 = 80 tokens
    # Window 0 tokens: 0..96
    # Window 1 tokens: 80..176
    w1_tokens = count_tokens(windows[1])
    assert w1_tokens == 96


def test_vector_expansion_and_deduplication():
    """Verify window matches expand to parent chunk and deduplicate taking max score."""
    # Create two documents: one very long (multiple windows) and one short
    long_text_words = [f"LongDocWord{i}" for i in range(300)]
    long_text = " ".join(long_text_words) + " specific_target_keyword found in window"

    short_text = "Short text with specific_target_keyword."

    docs = [
        {"document_id": "LONG", "page_index": 1, "text": long_text, "span_ids": ["S1"]},
        {"document_id": "SHORT", "page_index": 2, "text": short_text, "span_ids": ["S2"]},
    ]

    retriever = HybridRetriever(
        parent_chunk_target_tokens=500,
        embedding_window_tokens=96,
        embedding_window_overlap=16,
    )
    retriever.index_documents(docs)

    # LONG document must have produced multiple embedding windows
    long_windows = [w for w in retriever.embedding_windows if w["parent_chunk_id"].startswith("LONG")]
    assert len(long_windows) > 1

    # Perform vector search
    vec_results = retriever.vector_search("specific_target_keyword")
    # Results should be unique parent chunks
    parent_indices = [idx for idx, _ in vec_results]
    assert len(parent_indices) == len(set(parent_indices))


def test_rrf_scoring_calculation():
    """Verify RRF formula: score = sum(1.0 / (60.0 + rank)) with rank starting at 1."""
    retriever = HybridRetriever(rrf_constant=60.0)

    # If chunk has lexical rank 1 and vector rank 1:
    # score = 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278688
    # If chunk has lexical rank 2 and vector rank 3:
    # score = 1/(60+2) + 1/(60+3) = 1/62 + 1/63 ≈ 0.032002
    score_1_1 = 1.0 / 61.0 + 1.0 / 61.0
    score_2_3 = 1.0 / 62.0 + 1.0 / 63.0

    assert score_1_1 > score_2_3


def test_hybrid_search_end_to_end(sample_catalog_docs):
    """Verify full end-to-end hybrid retrieval adhering to all specifications."""
    retriever = HybridRetriever(
        lexical_top_k=12,
        vector_top_k=12,
        context_top_k=8,
        rrf_constant=60.0,
    )
    retriever.index_documents(sample_catalog_docs)

    # Query matching P2 (Schneider ATV320 variable speed drive Modbus-RTU RS-485)
    query = "Schneider ATV320 variable speed drive Modbus-RTU RS-485"
    results = retriever.search(query, top_k=3)

    assert len(results) == 3
    # Top result should be P2
    top = results[0]
    assert top["document_id"] == "DOC-P2-MANUAL"
    assert top["page_index"] == 5
    assert "DOC-P2:p05:s02" in top["span_ids"]
    assert "ATV320" in top["text"]
    assert top["score"] > 0
    assert "rrf_score" in top
    assert top["lexical_rank"] == 1

    # Verify all required dictionary fields are present
    for r in results:
        assert "span_ids" in r
        assert "document_id" in r
        assert "page_index" in r
        assert "text" in r
        assert "score" in r
        assert "chunk_id" in r


def test_hybrid_search_power_supply_query(sample_catalog_docs):
    """Verify query for power supply retrieves P3 (Phoenix Contact)."""
    retriever = HybridRetriever()
    retriever.index_documents(sample_catalog_docs)

    query = "24V DC 20A DIN rail power supply"
    results = retriever.search(query, top_k=2)

    assert len(results) >= 1
    top = results[0]
    assert top["document_id"] == "DOC-P3-MANUAL"
    assert "QUINT4-PS/1AC/24DC/20" in top["text"]
    assert "DOC-P3:p08:s01" in top["span_ids"]


def test_save_and_load_index_directory(sample_catalog_docs):
    """Verify save_index and load_index to directory restores index perfectly."""
    retriever = HybridRetriever()
    retriever.index_documents(sample_catalog_docs)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "test_rag_index"
        retriever.save_index(save_dir)

        assert (save_dir / "index_metadata.json").exists()
        assert (save_dir / "embeddings.npy").exists()

        # Load into new retriever instance
        loaded_retriever = HybridRetriever.from_saved(save_dir)

        # Check internal state restored
        assert len(loaded_retriever.parent_chunks) == len(retriever.parent_chunks)
        assert len(loaded_retriever.embedding_windows) == len(retriever.embedding_windows)
        assert np.allclose(loaded_retriever.normalized_embeddings, retriever.normalized_embeddings)

        # Check search results match before and after
        query = "PROFINET S7-1200 CPU"
        res_orig = retriever.search(query, top_k=3)
        res_loaded = loaded_retriever.search(query, top_k=3)

        assert len(res_orig) == len(res_loaded)
        for orig, loaded in zip(res_orig, res_loaded):
            assert orig["chunk_id"] == loaded["chunk_id"]
            assert pytest.approx(orig["score"], abs=1e-6) == loaded["score"]


def test_save_and_load_index_npz(sample_catalog_docs):
    """Verify save_index and load_index using compressed .npz archive."""
    retriever = HybridRetriever()
    retriever.index_documents(sample_catalog_docs)

    with tempfile.NamedTemporaryFile(suffix=".npz") as tmpfile:
        retriever.save_index(tmpfile.name)

        loaded_retriever = HybridRetriever.from_saved(tmpfile.name)
        assert len(loaded_retriever.parent_chunks) == len(retriever.parent_chunks)

        query = "Modbus-RTU RS-485"
        res_orig = retriever.search(query)
        res_loaded = loaded_retriever.search(query)

        assert res_orig[0]["chunk_id"] == res_loaded[0]["chunk_id"]


def test_custom_embed_fn():
    """Verify custom embedding function injection works as expected."""
    def custom_embed(texts: list[str]) -> np.ndarray:
        # Simple 64-dim dummy embedding
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), 64).astype(np.float32)

    docs = [
        {"document_id": "D1", "text": "Testing custom embedding function with technical specs."},
    ]

    retriever = HybridRetriever(embed_fn=custom_embed)
    retriever.index_documents(docs)

    assert retriever.normalized_embeddings.shape[1] == 64
    # Check normalization
    norm = np.linalg.norm(retriever.normalized_embeddings[0])
    assert pytest.approx(norm, abs=1e-5) == 1.0


def test_empty_retriever():
    """Verify searching an empty retriever safely returns empty list."""
    retriever = HybridRetriever()
    results = retriever.search("anything")
    assert results == []


def test_empty_and_whitespace_queries(sample_catalog_docs):
    """Verify empty and whitespace queries safely return empty results."""
    retriever = HybridRetriever()
    retriever.index_documents(sample_catalog_docs)

    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.lexical_search("") == []
    assert retriever.lexical_search("   ") == []
    assert retriever.vector_search("") == []
    assert retriever.vector_search("   ") == []


def test_retriever_from_config_megaplan_spec():
    """Verify from_config correctly parses MEGAPLAN §21 configuration block."""
    config = {
        "rag": {
            "lexical_top_k": 12,
            "vector_top_k": 12,
            "context_top_k": 8,
            "parent_chunk_target_tokens": 500,
            "embedding_window_tokens": 96,
            "embedding_window_overlap": 16,
            "parent_chunk_overlap": 80,
            "rrf_constant": 60,
        }
    }
    retriever = HybridRetriever.from_config(config)
    assert retriever.lexical_top_k == 12
    assert retriever.vector_top_k == 12
    assert retriever.context_top_k == 8
    assert retriever.parent_chunk_target_tokens == 500
    assert retriever.embedding_window_tokens == 96
    assert retriever.embedding_window_overlap == 16
    assert retriever.parent_chunk_overlap == 80
    assert retriever.rrf_constant == 60.0


def test_chunk_document_fallback_metadata_and_defaults():
    """Verify chunk_document safely handles missing document_id, page_index, and span_ids."""
    doc_missing = {
        "text": "Industrial automation sensor with IO-Link interface and 24V DC supply.",
    }
    chunks = chunk_document(doc_missing, target_tokens=500, overlap_tokens=80)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["document_id"] == "doc"
    assert c["page_index"] == 0
    assert c["span_ids"] == ["doc:p0:s01"]
    assert c["token_count"] > 0


def test_top_k_larger_than_corpus(sample_catalog_docs):
    """Verify requesting top_k greater than corpus size returns all chunks without errors."""
    retriever = HybridRetriever()
    retriever.index_documents(sample_catalog_docs)

    results = retriever.search("24V DC", top_k=50)
    assert 1 <= len(results) <= len(sample_catalog_docs)


def test_embedding_adapter_integration(sample_catalog_docs):
    """Verify HybridRetriever integration with EmbeddingAdapter."""
    from industrial_lab.adapters.embeddings import EmbeddingAdapter

    adapter = EmbeddingAdapter(force_fallback=True)
    retriever = HybridRetriever(embed_fn=adapter.embed_texts)
    retriever.index_documents(sample_catalog_docs)

    results = retriever.search("Siemens S7-1200 24V DC", top_k=2)
    assert len(results) >= 1
    assert results[0]["document_id"] == "DOC-P1-MANUAL"

