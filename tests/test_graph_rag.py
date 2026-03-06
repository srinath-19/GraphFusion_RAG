"""Tests for graph_rag module (graph traversal and entity matching logic)."""

import json
import tempfile
from pathlib import Path

import pro_implementation.graph_rag as gr_module
from pro_implementation.graph_rag import find_related_chunk_ids


SAMPLE_GRAPH = {
    "nodes": {
        "jessica liu": {"name": "Jessica Liu", "type": "person", "chunk_ids": ["0", "1"]},
        "rellm": {"name": "Rellm", "type": "product", "chunk_ids": ["2", "3"]},
        "insurellm": {"name": "Insurellm", "type": "company", "chunk_ids": ["4", "5"]},
        "sarah chen": {"name": "Sarah Chen", "type": "person", "chunk_ids": ["6"]},
        "healthllm": {"name": "Healthllm", "type": "product", "chunk_ids": ["7"]},
        "bizllm": {"name": "Bizllm", "type": "product", "chunk_ids": ["8"]},
    },
    "edges": [
        {"source": "jessica liu", "target": "rellm", "relation": "works_on", "chunk_id": "1"},
        {"source": "rellm", "target": "insurellm", "relation": "product_of", "chunk_id": "3"},
        {"source": "sarah chen", "target": "healthllm", "relation": "signed_contract", "chunk_id": "6"},
        {"source": "healthllm", "target": "insurellm", "relation": "product_of", "chunk_id": "7"},
        {"source": "insurellm", "target": "bizllm", "relation": "offers", "chunk_id": "5"},
    ],
}


def test_find_direct_entity_match():
    """Querying for an entity name directly should return its chunks."""
    result = find_related_chunk_ids("Tell me about Jessica Liu", SAMPLE_GRAPH)
    assert "0" in result
    assert "1" in result


def test_find_entity_by_partial_match():
    """Query words that match entity keys should still find them."""
    result = find_related_chunk_ids("What is Rellm?", SAMPLE_GRAPH)
    assert "2" in result or "3" in result


def test_graph_traversal_finds_connected_entities():
    """Traversal should find entities connected via edges."""
    result = find_related_chunk_ids("Jessica Liu", SAMPLE_GRAPH, max_hops=1)
    # Jessica -> Rellm via works_on edge
    assert any(cid in result for cid in ["2", "3"]), "Should find Rellm chunks via graph traversal"


def test_multi_hop_traversal():
    """Multi-hop traversal should reach entities 2 hops away."""
    result = find_related_chunk_ids("Jessica Liu", SAMPLE_GRAPH, max_hops=2, max_chunks=20)
    # Jessica -> Rellm -> Insurellm (2 hops)
    assert any(cid in result for cid in ["4", "5"]), "Should find Insurellm chunks via 2-hop traversal"


def test_empty_graph_returns_empty():
    """Empty or None graph should return empty list."""
    assert find_related_chunk_ids("anything", None) == []
    assert find_related_chunk_ids("anything", {}) == []
    assert find_related_chunk_ids("anything", {"nodes": {}}) == []


def test_no_matching_entities_returns_empty():
    """Query with no matching entities should return empty list."""
    result = find_related_chunk_ids("quantum computing research", SAMPLE_GRAPH)
    assert result == []


def test_max_chunks_limit():
    """Result should be limited by max_chunks parameter."""
    result = find_related_chunk_ids("Insurellm", SAMPLE_GRAPH, max_hops=2, max_chunks=3)
    assert len(result) <= 3


def test_direct_matches_prioritized():
    """Direct entity matches should come before connected entity chunks."""
    result = find_related_chunk_ids("Jessica Liu", SAMPLE_GRAPH, max_hops=2, max_chunks=20)
    # Jessica's direct chunks (0, 1) should come before connected chunks
    direct_indices = [result.index(cid) for cid in ["0", "1"] if cid in result]
    if direct_indices:
        min_direct = min(direct_indices)
        assert min_direct == 0, "Direct match chunks should be first in results"


def test_no_duplicate_chunk_ids():
    """Result should not contain duplicate chunk IDs."""
    result = find_related_chunk_ids("Insurellm products", SAMPLE_GRAPH, max_hops=2, max_chunks=20)
    assert len(result) == len(set(result)), "Result should have no duplicate chunk IDs"


def test_load_graph_returns_none_when_no_file():
    """load_graph should return None when the graph file doesn't exist."""
    original_path = gr_module.GRAPH_PATH
    gr_module.GRAPH_PATH = "/tmp/nonexistent_graph_test_12345.json"
    try:
        result = gr_module.load_graph()
        assert result is None
    finally:
        gr_module.GRAPH_PATH = original_path


def test_load_graph_reads_valid_json():
    """load_graph should correctly read a valid graph JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_GRAPH, f)
        tmp_path = f.name

    original_path = gr_module.GRAPH_PATH
    gr_module.GRAPH_PATH = tmp_path
    try:
        result = gr_module.load_graph()
        assert result is not None
        assert "nodes" in result
        assert "edges" in result
        assert "jessica liu" in result["nodes"]
    finally:
        gr_module.GRAPH_PATH = original_path
        Path(tmp_path).unlink(missing_ok=True)