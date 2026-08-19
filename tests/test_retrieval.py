import pytest
from backend.app.services.vector_store import vector_store
from backend.app.services.embedding_service import embedding_service
from backend.app.rag.chunker import HierarchicalChunker

def test_vector_search_execution():
    query = "Who is the chief architect of the Constitution of India?"
    q_vec = embedding_service.embed_query(query)
    assert len(q_vec) == 384

    hits = vector_store.search_children(q_vec, top_k=3, score_threshold=0.3)
    assert isinstance(hits, list)

def test_hierarchical_context_resolution():
    mock_child_hits = [
        {
            "score": 0.85,
            "child_id": "doc1_p0_c0",
            "parent_id": "doc1_p0",
            "doc_id": "doc1",
            "child_text": "Ambedkar was chairman of drafting committee.",
            "parent_text": "The Constitution of India is the supreme law. B. R. Ambedkar is chief architect.",
            "language": "en",
            "metadata": {}
        },
        {
            "score": 0.78,
            "child_id": "doc1_p0_c1",
            "parent_id": "doc1_p0",
            "doc_id": "doc1",
            "child_text": "It was adopted in 1949.",
            "parent_text": "The Constitution of India is the supreme law. B. R. Ambedkar is chief architect.",
            "language": "en",
            "metadata": {}
        }
    ]

    parents = vector_store.resolve_hierarchical_context(mock_child_hits, max_parents=2)
    # Should deduplicate multiple children from the same parent into 1 parent entry
    assert len(parents) == 1
    assert parents[0]["parent_id"] == "doc1_p0"
    assert parents[0]["max_child_score"] == 0.85
    assert len(parents[0]["matched_children"]) == 2
