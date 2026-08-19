import time
from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest
from ..config import settings

class RerankerService:
    """
    FlashRank ONNX-based ultra-fast cross-encoder reranker (~5-12ms CPU latency).
    Operates conditionally to minimize unnecessary latency.
    """
    _instance = None
    _ranker = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RerankerService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("[RerankerService] Initializing FlashRank ONNX ranker (ms-marco-TinyBERT-L-2-v2)...")
        t0 = time.perf_counter()
        # ms-marco-TinyBERT-L-2-v2 is ultra lightweight (~4MB) with 5ms inference latency
        self._ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="./data/models")
        elapsed = (time.perf_counter() - t0) * 1000.0
        print(f"[RerankerService] Initialized in {elapsed:.2f}ms.")

    def rerank_passages(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Reranks retrieved candidate passages against the query using cross-encoder scoring.
        """
        if not passages or not settings.ENABLE_RERANKER:
            return passages[:top_n]

        # Prepare payload for FlashRank
        passages_for_ranker = []
        for idx, p in enumerate(passages):
            passages_for_ranker.append({
                "id": idx,
                "text": p.get("parent_text") or p.get("child_text", ""),
                "original_data": p
            })

        rerank_req = RerankRequest(query=query, passages=passages_for_ranker)
        ranked_results = self._ranker.rerank(rerank_req)

        output = []
        for item in ranked_results[:top_n]:
            orig = item["original_data"]
            orig["rerank_score"] = round(float(item["score"]), 4)
            output.append(orig)

        return output

reranker_service = RerankerService()
