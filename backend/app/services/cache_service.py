import time
from typing import Dict, Any, Optional

class QueryCacheService:
    """
    In-memory LRU-style cache for repeated voice and text queries.
    Provides sub-millisecond instant retrieval for hot queries.
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds

    def _normalize(self, query: str) -> str:
        return query.strip().lower()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        key = self._normalize(query)
        entry = self.cache.get(key)
        if entry:
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["data"]
            else:
                del self.cache[key]
        return None

    def set(self, query: str, data: Dict[str, Any]):
        if len(self.cache) >= self.max_size:
            # Evict oldest
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

        key = self._normalize(query)
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }

    def clear(self):
        self.cache.clear()

cache_service = QueryCacheService()
