import os
import json
import time
from backend.app.rag.chunker import HierarchicalChunker
from backend.app.services.embedding_service import embedding_service
from backend.app.services.vector_store import vector_store

def build_hierarchical_index(dataset_path: str = "./data/msmarco_xi_sample.json"):
    print("=" * 60)
    print("BUILDING HIERARCHICAL VECTOR INDEX (MSMARCO-XI)")
    print("=" * 60)

    if not os.path.exists(dataset_path):
        print(f"[BuildIndex] Dataset not found at {dataset_path}. Generating sample first...")
        from scripts.download_msmarco import prepare_msmarco_sample
        prepare_msmarco_sample(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"[BuildIndex] Loaded {len(documents)} source documents from MSMARCO-XI sample.")

    # 1. Chunking Phase
    chunker = HierarchicalChunker(
        parent_chunk_size=350,
        parent_overlap=40,
        child_chunk_size=90,
        child_overlap=25
    )

    t0_chunk = time.perf_counter()
    all_parent_chunks = []
    all_child_texts = []

    for doc in documents:
        parents = chunker.chunk_document(
            doc_id=doc["doc_id"],
            text=doc["text"],
            language=doc.get("language", "en"),
            metadata={"title": doc.get("title", ""), "url": doc.get("url", "")}
        )
        for p in parents:
            all_parent_chunks.append(p)
            for c in p.children:
                all_child_texts.append(c.text)

    chunk_time = (time.perf_counter() - t0_chunk) * 1000.0
    print(f"[BuildIndex] Chunked into {len(all_parent_chunks)} Parent Passages and {len(all_child_texts)} Child Vectors in {chunk_time:.2f}ms.")

    # 2. Embedding Phase (FastEmbed ONNX)
    print(f"[BuildIndex] Generating ONNX embeddings for {len(all_child_texts)} child chunks...")
    t0_embed = time.perf_counter()
    embeddings = embedding_service.embed_documents(all_child_texts, batch_size=128)
    embed_time = (time.perf_counter() - t0_embed) * 1000.0
    print(f"[BuildIndex] Generated {len(embeddings)} embeddings in {embed_time:.2f}ms ({embed_time/max(1, len(embeddings)):.2f}ms per vector).")

    # 3. Insertion into Qdrant
    print(f"[BuildIndex] Upserting vectors and hierarchical metadata into Qdrant...")
    t0_upsert = time.perf_counter()
    total_inserted = vector_store.insert_parent_chunks(
        parent_chunks=all_parent_chunks,
        child_embeddings=embeddings,
        batch_size=256
    )
    upsert_time = (time.perf_counter() - t0_upsert) * 1000.0
    print(f"[BuildIndex] Successfully indexed {total_inserted} points in {upsert_time:.2f}ms.")

    # 4. Quick Verification Search
    test_query = "Who is the chief architect of the Constitution of India?"
    print(f"\n[BuildIndex] Running test verification query: '{test_query}'")
    q_vec = embedding_service.embed_query(test_query)
    hits = vector_store.search_children(q_vec, top_k=3)
    parents = vector_store.resolve_hierarchical_context(hits, max_parents=2)

    print(f"[BuildIndex] Top retrieved context:")
    for idx, p in enumerate(parents, 1):
        print(f"  [{idx}] Doc ID: {p['doc_id']} | Max Score: {p['max_child_score']:.4f}")
        print(f"      Text: {p['parent_text'][:120]}...\n")

    print(f"[SUCCESS] Hierarchical indexing complete and verified! Total points: {vector_store.count()}")

if __name__ == "__main__":
    build_hierarchical_index()
