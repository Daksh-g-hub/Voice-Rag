import pytest
from backend.app.rag.chunker import HierarchicalChunker, ParentChunk, ChildChunk

def test_hierarchical_chunker_basic():
    chunker = HierarchicalChunker(
        parent_chunk_size=100,
        parent_overlap=20,
        child_chunk_size=30,
        child_overlap=10
    )

    sample_text = (
        "The Constitution of India is the supreme law of India. "
        "The document lays down the framework that demarcates fundamental political code, "
        "structure, procedures, powers, and duties of government institutions and sets out "
        "fundamental rights, directive principles, and the duties of citizens. "
        "It is the longest written constitution of any country on earth. "
        "B. R. Ambedkar, chairman of the drafting committee, is widely considered to be its chief architect."
    )

    parents = chunker.chunk_document(
        doc_id="test_doc_001",
        text=sample_text,
        language="en",
        metadata={"category": "constitution"}
    )

    assert len(parents) >= 1
    p0 = parents[0]
    assert p0.doc_id == "test_doc_001"
    assert p0.parent_id == "test_doc_001_p0"
    assert len(p0.children) >= 1

    # Verify child chunk links
    for child in p0.children:
        assert child.parent_id == p0.parent_id
        assert child.doc_id == "test_doc_001"
        assert child.text in p0.text
        assert child.word_count > 0

def test_chunker_empty_input():
    chunker = HierarchicalChunker()
    parents = chunker.chunk_document("empty_doc", "")
    assert parents == []
