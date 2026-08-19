import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
    Filter,
    FieldCondition,
    MatchValue
)
from ..config import settings
from ..rag.chunker import ChildChunk, ParentChunk

class VectorStoreService:
    """
    Qdrant Vector Database Service configured for sub-millisecond local HNSW retrieval.
    Stores Child chunks as vector points with Parent passage text in the payload.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or settings.QDRANT_STORAGE_PATH
        self.collection_name = settings.COLLECTION_NAME
        
        # Ensure data directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize local file-persisted Qdrant client
        self.client = QdrantClient(path=self.storage_path)
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the collection if it does not already exist."""
        collections = self.client.get_collections().collections
        existing_names = [c.name for c in collections]

        if self.collection_name not in existing_names:
            print(f"[VectorStore] Creating collection '{self.collection_name}' with HNSW indexing...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,               # Number of edges per node (balanced for speed & recall)
                    ef_construct=100,   # Construction search depth
                    full_scan_threshold=1000
                )
            )
            print(f"[VectorStore] Collection '{self.collection_name}' ready.")

    def insert_parent_chunks(
        self,
        parent_chunks: List[ParentChunk],
        child_embeddings: List[List[float]],
        batch_size: int = 256
    ) -> int:
        """
        Inserts child chunks with parent metadata into Qdrant.
        """
        points = []
        point_idx = 0

        for parent in parent_chunks:
            for child in parent.children:
                if point_idx >= len(child_embeddings):
                    break
                
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, child.child_id))
                payload = {
                    "child_id": child.child_id,
                    "parent_id": child.parent_id,
                    "doc_id": child.doc_id,
                    "child_text": child.text,
                    "parent_text": parent.text,
                    "language": child.language,
                    "word_count": child.word_count,
                    "metadata": child.metadata
                }
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=child_embeddings[point_idx],
                        payload=payload
                    )
                )
                point_idx += 1

        # Upload in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        return len(points)

    def search_children(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs high-speed cosine vector search over child chunks.
        """
        threshold = score_threshold if score_threshold is not None else settings.RELEVANCE_SCORE_THRESHOLD
        
        query_res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=threshold,
            with_payload=True
        )

        results = []
        for hit in query_res.points:
            payload = hit.payload or {}
            results.append({
                "score": round(hit.score, 4),
                "child_id": payload.get("child_id"),
                "parent_id": payload.get("parent_id"),
                "doc_id": payload.get("doc_id"),
                "child_text": payload.get("child_text"),
                "parent_text": payload.get("parent_text"),
                "language": payload.get("language", "en"),
                "metadata": payload.get("metadata", {})
            })

        return results

    def resolve_hierarchical_context(
        self,
        child_results: List[Dict[str, Any]],
        max_parents: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Groups retrieved child chunks by parent_id, deduplicates parent passages,
        and returns the top unique parent contexts with maximum relevance scores.
        """
        parent_map: Dict[str, Dict[str, Any]] = {}

        for hit in child_results:
            pid = hit["parent_id"]
            if pid not in parent_map:
                parent_map[pid] = {
                    "parent_id": pid,
                    "doc_id": hit["doc_id"],
                    "parent_text": hit["parent_text"],
                    "language": hit["language"],
                    "max_child_score": hit["score"],
                    "matched_children": [
                        {
                            "child_id": hit["child_id"],
                            "child_text": hit["child_text"],
                            "score": hit["score"]
                        }
                    ]
                }
            else:
                # Update max score and append matched child
                if hit["score"] > parent_map[pid]["max_child_score"]:
                    parent_map[pid]["max_child_score"] = hit["score"]
                parent_map[pid]["matched_children"].append({
                    "child_id": hit["child_id"],
                    "child_text": hit["child_text"],
                    "score": hit["score"]
                })

        # Sort parents by their best child match score
        sorted_parents = sorted(
            parent_map.values(),
            key=lambda x: x["max_child_score"],
            reverse=True
        )

        return sorted_parents[:max_parents]

    def count(self) -> int:
        """Returns total vectors stored in the collection."""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

vector_store = VectorStoreService()
