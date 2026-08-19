import time
from typing import List, Union
import numpy as np
from fastembed import TextEmbedding
from ..config import settings

class EmbeddingService:
    """
    Ultra-low latency ONNX Embedding Generator using FastEmbed (bge-small-en-v1.5).
    Achieves sub-10ms query embedding times on standard CPU.
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        print(f"[EmbeddingService] Initializing FastEmbed ONNX model: {settings.EMBEDDING_MODEL_NAME}...")
        t0 = time.perf_counter()
        # FastEmbed runs ONNX runtime under the hood, perfectly tuned for latency
        self._model = TextEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)
        elapsed = (time.perf_counter() - t0) * 1000.0
        print(f"[EmbeddingService] Initialized in {elapsed:.2f}ms.")

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string for vector search.
        """
        generator = self._model.embed([f"query: {query}"])
        embedding = list(generator)[0].tolist()
        return embedding

    def embed_documents(self, documents: List[str], batch_size: int = 128) -> List[List[float]]:
        """
        Batch embeds document chunks for high-throughput indexing.
        """
        embeddings = []
        generator = self._model.embed(documents, batch_size=batch_size)
        for emb in generator:
            embeddings.append(emb.tolist())
        return embeddings

embedding_service = EmbeddingService()
