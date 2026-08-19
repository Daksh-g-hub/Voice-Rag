import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel

class GroundingGuardResult(BaseModel):
    is_grounded: bool
    citations_found: List[str]
    valid_citations: List[str]
    invalid_citations: List[str]
    grounding_score: float
    notes: str

class GroundingGuard:
    """
    Post-Generation Citation & Grounding Guardrail (<2ms).
    Verifies that the LLM's response cites retrieved context correctly and avoids hallucinated source IDs.
    """
    CITATION_REGEX = re.compile(r'\[Source\s+(\d+)\]', re.IGNORECASE)

    def verify_grounding(
        self,
        generated_answer: str,
        retrieved_contexts: List[Dict[str, Any]]
    ) -> GroundingGuardResult:
        # Check if LLM explicitly gave the standard refusal
        if "I don't have enough reliable information" in generated_answer:
            return GroundingGuardResult(
                is_grounded=True,
                citations_found=[],
                valid_citations=[],
                invalid_citations=[],
                grounding_score=1.0,
                notes="Standard factual refusal verified."
            )

        # Find all [Source X] matches
        matches = self.CITATION_REGEX.findall(generated_answer)
        total_sources_available = len(retrieved_contexts)

        if not matches:
            # Answer generated without explicit citations
            return GroundingGuardResult(
                is_grounded=False,
                citations_found=[],
                valid_citations=[],
                invalid_citations=[],
                grounding_score=0.5,
                notes="Answer does not contain source citations."
            )

        valid_citations = []
        invalid_citations = []

        for m in matches:
            source_num = int(m)
            citation_str = f"[Source {source_num}]"
            if 1 <= source_num <= total_sources_available:
                if citation_str not in valid_citations:
                    valid_citations.append(citation_str)
            else:
                if citation_str not in invalid_citations:
                    invalid_citations.append(citation_str)

        is_grounded = (len(invalid_citations) == 0 and len(valid_citations) > 0)
        grounding_score = len(valid_citations) / (len(valid_citations) + len(invalid_citations)) if matches else 0.0

        notes = "All citations are strictly grounded." if is_grounded else f"Detected invalid citations: {invalid_citations}"

        return GroundingGuardResult(
            is_grounded=is_grounded,
            citations_found=[f"[Source {m}]" for m in matches],
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            grounding_score=round(grounding_score, 2),
            notes=notes
        )

grounding_guard = GroundingGuard()
