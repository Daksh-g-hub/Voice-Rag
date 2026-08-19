from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from ..config import settings

class RelevanceGuardResult(BaseModel):
    passed: bool
    confidence_score: float
    is_dataset_grounded: bool
    reason: Optional[str] = None

class RelevanceGuard:
    """
    Evaluates semantic overlap with MSMARCO-XI dataset and sets grounding flags.
    """
    def __init__(self, threshold: Optional[float] = None):
        self._custom_threshold = threshold

    @property
    def threshold(self) -> float:
        return self._custom_threshold if self._custom_threshold is not None else settings.RELEVANCE_SCORE_THRESHOLD

    def evaluate(self, retrieved_parents: List[Dict[str, Any]]) -> RelevanceGuardResult:
        if not retrieved_parents:
            return RelevanceGuardResult(
                passed=True,
                confidence_score=0.0,
                is_dataset_grounded=False,
                reason="No matching MSMARCO-XI passage found (using intelligent AI generation)."
            )

        top_score = retrieved_parents[0].get("max_child_score") or retrieved_parents[0].get("rerank_score", 0.0)
        is_grounded = float(top_score) >= self.threshold

        return RelevanceGuardResult(
            passed=True,
            confidence_score=float(top_score),
            is_dataset_grounded=is_grounded,
            reason="Retrieved relevant MSMARCO-XI context." if is_grounded else "Low dataset overlap (using intelligent synthesis)."
        )

relevance_guard = RelevanceGuard()
